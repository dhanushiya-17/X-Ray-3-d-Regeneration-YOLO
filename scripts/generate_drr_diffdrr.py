import torch
import matplotlib.pyplot as plt

from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.visualization import plot_drr


# ============================================================
# 1. Load CT
# ============================================================

ct_path = "data/raw/Subjects/Pat001/Pat001.nrrd"

subject = read(
    ct_path,
    center_volume=True,
)

print("CT loaded successfully")
print("Volume:", subject["volume"])
print("Density:", subject["density"])


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 3. Create DRR
# ============================================================

drr = DRR(
    subject=subject,

    # Source-to-detector distance
    sdd=1020.0,

    # Keep image small enough for 8 GB VRAM
    height=512,
    width=512,

    # Larger physical field of view
    delx=1.5,
    dely=1.5,

    renderer="siddon",
).to(device)

print("DRR created")
print("DRR device:", drr.device)

print(
    "Detector physical size:",
    drr.detector.width * drr.detector.delx,
    "mm ×",
    drr.detector.height * drr.detector.dely,
    "mm"
)


# ============================================================
# 4. Camera pose
# ============================================================

rotations = torch.tensor(
    [[0.0, 0.0, 0.0]],
    dtype=torch.float32,
    device=device,
)

translations = torch.tensor(
    [[0.0, 850.0, 0.0]],
    dtype=torch.float32,
    device=device,
)


print("Rotations:", rotations)
print("Translations:", translations)


# ============================================================
# 5. Generate DRR
# ============================================================

print("Generating DRR...")

with torch.no_grad():

    projection = drr(
        rotations,
        translations,
        parameterization="euler_angles",
        convention="ZXY",
        degrees=False,
    )


print("Raw DRR shape:", projection.shape)


# ============================================================
# 6. Save raw numerical DRR
# ============================================================

projection_cpu = (
    projection
    .squeeze()
    .detach()
    .cpu()
    .numpy()
)

print("Final DRR shape:", projection_cpu.shape)
print("Minimum:", projection_cpu.min())
print("Maximum:", projection_cpu.max())


# ============================================================
# 7. Save image
# ============================================================

output_path = "drr/Pat001_diffdrr.png"

plt.figure(figsize=(8, 8))

plt.imshow(
    projection_cpu,
    cmap="gray",
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight",
    pad_inches=0,
)

plt.close()


print()
print("==========================================")
print("DRR GENERATED SUCCESSFULLY")
print("==========================================")
print("Output:", output_path)
print("Image:", projection_cpu.shape)
print("Detector:", "768 mm × 768 mm")
