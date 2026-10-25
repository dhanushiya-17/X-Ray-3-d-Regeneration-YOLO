"""Start the BoneVision local web app.

    python serve.py                       # trained network (models/multiview_*_d2.pt) and/or models/ssm_highres.npz
    python serve.py --demo                # builds/uses SYNTHETIC demo models (UI testing only)
    python serve.py --model path/to/ssm.npz --device cuda --port 8000

Open http://127.0.0.1:8000. Uploads and results stay in this process's memory.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO_MODEL = ROOT / "models" / "ssm_demo_synthetic.npz"
DEMO_NET_DIR = ROOT / "models" / "demo"
DEMO_NETS = {"BONEVISION_NET_AP_LAT": DEMO_NET_DIR / "multiview_ap_lat_demo.pt", "BONEVISION_NET_AP": DEMO_NET_DIR / "multiview_ap_demo.pt"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=None, help="Shape model .npz (default models/ssm_highres.npz)")
    ap.add_argument("--demo", action="store_true", help="Use the synthetic demo shape model and networks, building them if missing")
    ap.add_argument("--net-ap-lat", type=Path, default=None, help="AP + lateral network checkpoint (default models/multiview_ap_lat_d2.pt)")
    ap.add_argument("--net-ap", type=Path, default=None, help="AP-only network checkpoint (default models/multiview_ap_d2.pt)")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--host", default="127.0.0.1", help="Keep 127.0.0.1 unless you need LAN access")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="Auto-reload on code changes (development)")
    args = ap.parse_args()
    if args.demo:
        if not DEMO_MODEL.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_model.py"), "--output", str(DEMO_MODEL)], check=True)
        args.model = DEMO_MODEL
        if not all(p.exists() for p in DEMO_NETS.values()):
            print("Training the synthetic demo networks (a few minutes on CPU, once)…")
            subprocess.run([sys.executable, str(ROOT / "scripts" / "make_demo_network.py"), "--output-dir", str(DEMO_NET_DIR)], check=True)
        args.net_ap_lat, args.net_ap = DEMO_NETS["BONEVISION_NET_AP_LAT"], DEMO_NETS["BONEVISION_NET_AP"]
    if args.net_ap_lat is not None:
        os.environ["BONEVISION_NET_AP_LAT"] = str(args.net_ap_lat.resolve())
    if args.net_ap is not None:
        os.environ["BONEVISION_NET_AP"] = str(args.net_ap.resolve())
    if args.model is not None:
        os.environ["BONEVISION_MODEL"] = str(args.model.resolve())
    os.environ["BONEVISION_DEVICE"] = args.device
    import uvicorn
    print(f"BoneVision → http://{args.host}:{args.port}")
    uvicorn.run("bonevision.server:app", host=args.host, port=args.port, reload=args.reload, app_dir=str(ROOT))


if __name__ == "__main__":
    main()
