import torch
import matplotlib.pyplot as plt

from diffdrr.drr import DRR
from diffdrr.data import read


# ---------------------------------------------------------
# 1. Load the CT
# ---------------------------------------------------------

ct_path = "data/raw/Subjects/Pat001/Pat001.nrrd"

subject = read(ct_path)

print("CT loaded")
print("CT shape:", subject.data.shape)
print("Spacing:", subject.spacing)


# ---------------------------------------------------------
# 2. Use NVIDIA GPU if available
# ---------------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# 3. Create the DRR renderer
# ---------------------------------------------------------

drr = DRR(
    subject,
    sdd=1020.0,
    height=512,
    delx=0.7,
).to(device)


# ---------------------------------------------------------
# 4. Define X-ray view
# ---------------------------------------------------------

rotations = torch.tensor(
    [[0.0, 0.0, 0.0]],
    dtype=torch.float32,
    device=device,
)

translations = torch.tensor(
    [[0.0, 0.0, 0.0]],
    dtype=torch.float32,
    device=device,
)


# ---------------------------------------------------------
# 5. Generate DRR
# ---------------------------------------------------------

image = drr(
    rotations,
    translations,
    parameterization="euler_angles",
    convention="ZXY",
)


# ---------------------------------------------------------
# 6. Save DRR
# ---------------------------------------------------------

image = image.squeeze().detach().cpu().numpy()

plt.figure(figsize=(8, 8))
plt.imshow(image, cmap="gray")
plt.axis("off")
plt.tight_layout()

plt.savefig(
    "drr/Pat001_DRR.png",
    dpi=200,
    bbox_inches="tight",
    pad_inches=0,
)

plt.close()

print("Saved: drr/Pat001_DRR.png")
