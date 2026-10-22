"""Compare stored Pat001 DRRs with a fresh DiffDRR render without saving files."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffdrr.data import read
from diffdrr.drr import DRR


ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "Pat001"
ANGLES = range(0, 360, 45)


def main():
    ct_path = ROOT / "data" / "raw" / "Subjects" / SUBJECT / f"{SUBJECT}_femur_only.nrrd"
    drr_dir = ROOT / "data" / "drr_dataset" / SUBJECT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subject = read(ct_path, orientation="AP", center_volume=True)
    renderer = DRR(
        subject=subject,
        sdd=1020.0,
        height=512,
        width=512,
        delx=1.5,
        dely=1.5,
        renderer="siddon",
        patch_size=64,
    ).to(device)

    print(f"Device: {device}; patch size: 64")
    print("angle, correlation, normalized MAE")
    for angle in ANGLES:
        rotations = torch.tensor(
            [[np.deg2rad(angle), 0.0, 0.0]], dtype=torch.float32, device=device
        )
        translations = torch.tensor(
            [[0.0, 850.0, 0.0]], dtype=torch.float32, device=device
        )

        with torch.no_grad():
            rendered = renderer(
                rotations,
                translations,
                parameterization="euler_angles",
                convention="ZXY",
                degrees=False,
            )[0, 0].cpu().numpy()

        span = rendered.max() - rendered.min()
        if span <= 0:
            raise ValueError(f"Rendered view {angle} has no intensity range")
        rendered = (rendered - rendered.min()) / span

        image_path = drr_dir / f"view_{angle:03d}.png"
        with Image.open(image_path) as image:
            reference = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

        if rendered.shape != reference.shape:
            raise ValueError(f"Image dimensions differ at {image_path}")
        correlation = np.corrcoef(rendered.ravel(), reference.ravel())[0, 1]
        mae = np.mean(np.abs(rendered - reference))
        print(f"{angle:03d}, {correlation:.5f}, {mae:.5f}")
        if correlation < 0.999 or mae > 0.001:
            raise ValueError(
                f"Stored view {angle:03d} does not reproduce within tolerance"
            )


if __name__ == "__main__":
    main()
