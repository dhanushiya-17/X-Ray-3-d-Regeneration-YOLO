"""
train_multiview_net.py
======================
Train / evaluate the multi-view femur network (scripts/multiview_net.py) on
samples from build_biplanar_dataset.py, respecting the saved subject split:
train subjects for fitting, validation subjects for checkpoint selection,
test subjects only with --evaluate test.

Examples
  # quick run at half resolution (128x48x48), AP + lateral
  python scripts/train_multiview_net.py --downsample 2 --epochs 300
  # single AP view model
  python scripts/train_multiview_net.py --views ap --downsample 2
  # full resolution on an 8 GB GPU (mixed precision, batch 1)
  python scripts/train_multiview_net.py --downsample 1 --batch-size 1
  # final held-out test, writes per-subject metrics CSV
  python scripts/train_multiview_net.py --evaluate test --checkpoint models/multiview_ap_lat_d2.pt
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

from femur_grid import ROOT, SPLIT_FILE, grid_spacing, load_split
from multiview_net import MultiViewFemurNet, dice_bce_loss

DATA_DIR = ROOT / "data" / "biplanar"
MODEL_DIR = ROOT / "models"


def normalize(img):
    hi = np.percentile(img, 99.5)
    return np.clip(img / hi, 0.0, 1.0).astype(np.float32) if hi > 0 else img.astype(np.float32)


def pool2d(img, f):
    if f == 1:
        return img
    x, y = img.shape[0] // f, img.shape[1] // f
    return img[:x * f, :y * f].reshape(x, f, y, f).mean(axis=(1, 3))


def pool3d(occ, f):
    if f == 1:
        return occ.astype(np.float32)
    x, y, z = (s // f for s in occ.shape)
    return (occ[:x * f, :y * f, :z * f].reshape(x, f, y, f, z, f).mean(axis=(1, 3, 5)) >= 0.5).astype(np.float32)


class BiplanarSet(torch.utils.data.Dataset):
    def __init__(self, files, factor, augment):
        self.items = []
        for path in files:
            d = np.load(path)
            self.items.append({"name": path.stem,
                               "ap": normalize(pool2d(d["ap"], factor)),
                               "lat": normalize(pool2d(d["lat"], factor)),
                               "occ": pool3d(d["occ"], factor)})
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def _aug(self, img, rng):
        # Radiograph appearance varies by exposure, processing and detector;
        # the 3D target does not, so only intensity is perturbed.
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-0.1, 0.1)
        img = np.clip(img, 0, 1) ** rng.uniform(0.6, 1.6)
        if rng.random() < 0.5:
            img = ndi.gaussian_filter(img, rng.uniform(0.3, 1.5))
        img = img + rng.normal(0, rng.uniform(0, 0.04), img.shape)
        return np.clip(img, 0, 1).astype(np.float32)

    def __getitem__(self, i):
        item = self.items[i]
        ap, lat = item["ap"], item["lat"]
        if self.augment:
            rng = np.random.default_rng()
            ap, lat = self._aug(ap, rng), self._aug(lat, rng)
        return {"name": item["name"], "ap": torch.from_numpy(ap)[None],
                "lat": torch.from_numpy(lat)[None], "occ": torch.from_numpy(item["occ"])[None]}


def split_files(data_dir, extra_dir=None):
    splits = load_split(SPLIT_FILE)
    out = {}
    for name in ("train", "validation", "test"):
        files = [data_dir / (s + ".npz") for s in splits.get(name, [])]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            print("WARNING: {} {} samples missing (e.g. {})".format(len(missing), name, missing[0]))
        out[name] = [f for f in files if f.exists()]
    if extra_dir:
        extra = sorted(Path(extra_dir).glob("*.npz"))
        print("Extra training samples:", len(extra))
        out["train"] += extra
    return out


def predict(model, batch, device, amp):
    images = {v: batch[v].to(device) for v in model.views}
    depth = batch["occ"].shape[3:]
    with torch.autocast(device_type=device.type, enabled=amp):
        return model(images, depth)


def evaluate(model, loader, device, amp, spacing, csv_path=None):
    from evaluate_single_view import metrics
    model.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            logits = predict(model, batch, device, amp).float()
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            gt = batch["occ"].numpy() > 0.5
            for k, name in enumerate(batch["name"]):
                p = ndi.binary_fill_holes(pred[k, 0])
                if not p.any():
                    rows.append({"subject": name, "dice": 0.0, "status": "EMPTY_PREDICTION"})
                    continue
                m = metrics(p, gt[k, 0], spacing, seed=42)
                rows.append({"subject": name, **{a: round(b, 4) for a, b in m.items()}, "status": "OK"})
    if csv_path:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "subject", k))
        with Path(csv_path).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return rows


def summarize(rows):
    ok = [r for r in rows if r["status"] == "OK"]
    out = {"n": len(rows), "failed": len(rows) - len(ok),
           "dice": float(np.mean([r["dice"] for r in rows])) if rows else 0.0}
    if ok:
        out["mean_surface_mm"] = float(np.mean([r["mean_symmetric_surface_distance_mm"] for r in ok]))
        out["hd95_mm"] = float(np.mean([r["hausdorff95_symmetric_mm"] for r in ok]))
    return out


def main():
    parser = argparse.ArgumentParser(description="Train the multi-view X-ray -> 3D femur network")
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--extra-train-dir", default=None, help="extra *.npz samples (training only)")
    parser.add_argument("--views", default="ap,lat", help="ap,lat or ap")
    parser.add_argument("--downsample", type=int, default=2, choices=(1, 2))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--evaluate", choices=("validation", "test"), default=None,
                        help="skip training and evaluate --checkpoint on this split")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    views = tuple(v.strip() for v in args.views.split(",") if v.strip())
    device = torch.device(args.device)
    amp = device.type == "cuda"
    spacing = tuple(s * args.downsample for s in grid_spacing())
    tag = "multiview_{}_d{}".format("_".join(views), args.downsample)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else MODEL_DIR / (tag + ".pt")
    files = split_files(Path(args.data_dir), args.extra_train_dir)

    if args.evaluate:
        state = torch.load(ckpt_path, map_location=device)
        cfg = state["config"]
        model = MultiViewFemurNet(views=cfg["views"]).to(device)
        model.load_state_dict(state["model"])
        spacing = tuple(s * cfg["downsample"] for s in grid_spacing())
        data = BiplanarSet(files[args.evaluate], cfg["downsample"], augment=False)
        loader = torch.utils.data.DataLoader(data, batch_size=1)
        out_csv = MODEL_DIR / "{}_{}_metrics.csv".format(ckpt_path.stem, args.evaluate)
        if out_csv.exists():
            raise FileExistsError(str(out_csv) + " exists; move it before re-running the evaluation")
        rows = evaluate(model, loader, device, amp, spacing, out_csv)
        print(args.evaluate, summarize(rows))
        print("Per-subject metrics:", out_csv)
        return

    if ckpt_path.exists():
        raise FileExistsError(str(ckpt_path) + " exists; pass a new --checkpoint path")
    train = BiplanarSet(files["train"], args.downsample, augment=True)
    val = BiplanarSet(files["validation"], args.downsample, augment=False)
    print("Train {}  validation {}  views {}  grid {}".format(
        len(train), len(val), views, tuple(train.items[0]["occ"].shape)))
    train_loader = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=1)

    # Mean training silhouettes on the full grid; the app uses them to place
    # an uploaded X-ray on the grid (scripts/multiview_net_backend.py).
    occ_sum, lengths = None, []
    for path in files["train"]:
        occ = np.load(path)["occ"].astype(np.float32)
        occ_sum = occ if occ_sum is None else occ_sum + occ
        rows = np.flatnonzero(occ.any(axis=(1, 2)))
        lengths.append((rows[-1] - rows[0]) * grid_spacing()[0])
    mean_occ = occ_sum / len(files["train"]) >= 0.5
    reference = {"ap": torch.from_numpy(mean_occ.any(axis=2)), "lat": torch.from_numpy(mean_occ.any(axis=1)),
                 "length_mm": float(np.mean(lengths))}

    model = MultiViewFemurNet(views=views).to(device)
    print("Parameters: {:.2f} M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_loader))
    scaler = torch.amp.GradScaler(enabled=amp)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best = -1.0
    log_path = ckpt_path.with_suffix(".log.csv")
    with log_path.open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(["epoch", "train_loss", "val_dice", "val_mean_surface_mm", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0, losses = time.time(), []
            for batch in train_loader:
                opt.zero_grad(set_to_none=True)
                logits = predict(model, batch, device, amp)
                loss = dice_bce_loss(logits.float(), batch["occ"].to(device))
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                losses.append(loss.item())
            row = [epoch, round(float(np.mean(losses)), 5), "", "", round(time.time() - t0, 1)]
            if epoch % args.val_every == 0 or epoch == args.epochs:
                s = summarize(evaluate(model, val_loader, device, amp, spacing))
                row[2], row[3] = round(s["dice"], 4), round(s.get("mean_surface_mm", float("nan")), 3)
                if s["dice"] > best:
                    best = s["dice"]
                    torch.save({"model": model.state_dict(), "epoch": epoch, "val": s, "reference": reference,
                                "config": {"views": views, "downsample": args.downsample}}, ckpt_path)
                print("epoch {:4d}  loss {:.4f}  val dice {:.4f}  surface {} mm{}".format(
                    epoch, row[1], s["dice"], row[3], "  *saved" if s["dice"] >= best else ""))
            writer.writerow(row)
            log.flush()
    print("Best validation Dice {:.4f}; checkpoint {}".format(best, ckpt_path))


if __name__ == "__main__":
    main()
