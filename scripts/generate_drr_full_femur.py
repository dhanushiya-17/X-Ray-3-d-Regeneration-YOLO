import torch
import matplotlib.pyplot as plt

from diffdrr.data import read
from diffdrr.drr import DRR


# --------------------------------------------------
# 1. Load CT
# --------------------------------------------------

ct_path = "data/raw/Subjects/Pat001/Pat001_femur_only.nrrd"

subject = read(ct_path)


# --------------------------------------------------
# 2. Select GPU
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# 3. Create DiffDRR
# --------------------------------------------------

drr = DRR(
    subject=subject,

    # Source-to-detector distance
    sdd=1020.0,

    # Keep resolution manageable for RTX 5060
    height=512,
    width=512,

    # Large physical field of view
    delx=1.5,
    dely=1.5,

    renderer="siddon",
).to(device)


# --------------------------------------------------
# 4. Camera pose
# --------------------------------------------------

rotations = torch.tensor(
    [[0.0, 0.0, 0.0]],
    device=device
)

translations = torch.tensor(
    [[0.0, 850.0, 0.0]],
    device=device
)


# --------------------------------------------------
# 5. Generate DRR
# --------------------------------------------------

with torch.no_grad():

    projection = drr(
        rotations,
        translations,
        parameterization="euler_angles",
        convention="ZXY",
        degrees=False,
    )


# --------------------------------------------------
# 6. Remove batch/channel dimensions
# --------------------------------------------------

projection = projection[0, 0].detach().cpu().numpy()


# --------------------------------------------------
# 7. Print information
# --------------------------------------------------

print("\nDRR generated successfully")

print("Shape:", projection.shape)

print("Minimum:", projection.min())

print("Maximum:", projection.max())


# --------------------------------------------------
# 8. Save image
# --------------------------------------------------

plt.figure(figsize=(8, 8))

plt.imshow(
    projection,
    cmap="gray"
)

plt.axis("off")

plt.tight_layout(pad=0)

plt.savefig(
    "drr/Pat001_full_femur_2mm.png",
    dpi=150,
    bbox_inches="tight",
    pad_inches=0
)

plt.show()

print("\nSaved:")
print("drr/Pat001_full_femur_2mm.png")
