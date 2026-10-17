import torch
import numpy as np
import matplotlib.pyplot as plt

from diffdrr.data import read
from diffdrr.drr import DRR


# ============================================================
# 1. Load cropped femur CT
# ============================================================

ct_path = "data/raw/Subjects/Pat001/Pat001_femur_only.nrrd"

subject = read(ct_path)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. Create DRR
# ============================================================

drr = DRR(
    subject=subject,
    sdd=1020.0,
    height=512,
    width=512,
    delx=1.5,
    dely=1.5,
    renderer="siddon",
).to(device)


# ============================================================
# 4. Test angles
# ============================================================

angles = [0, 45, 90]


# ============================================================
# 5. Generate views
# ============================================================

for angle in angles:

    # Azimuth angle
    alpha = np.deg2rad(angle)

    rotations = torch.tensor(
        [[alpha, 0.0, 0.0]],
        dtype=torch.float32,
        device=device
    )

    translations = torch.tensor(
        [[0.0, 850.0, 0.0]],
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():

        projection = drr(
            rotations,
            translations,
            parameterization="euler_angles",
            convention="ZXY",
            degrees=False,
        )

    projection = projection[0, 0].detach().cpu().numpy()

    # Normalize
    projection = (
        projection - projection.min()
    ) / (
        projection.max() - projection.min() + 1e-8
    )

    # Save
    output_path = f"drr/test_{angle:03d}.png"

    plt.figure(figsize=(6, 6))

    plt.imshow(
        projection,
        cmap="gray"
    )

    plt.axis("off")

    plt.tight_layout(pad=0)

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        pad_inches=0
    )

    plt.close()

    print(f"Saved: {output_path}")
