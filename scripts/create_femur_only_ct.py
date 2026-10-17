import SimpleITK as sitk
import numpy as np


# ============================================================
# Paths
# ============================================================

ct_path = "data/raw/Subjects/Pat001/Pat001.nrrd"

mask_path = "data/raw/Subjects/Pat001/Pat001_femur_mask.nrrd"

output_path = "data/raw/Subjects/Pat001/Pat001_femur_only.nrrd"


# ============================================================
# Settings
# ============================================================

# Extra space around the femur.
# This prevents the crop from touching the femur surface.
MARGIN = 20


# ============================================================
# 1. Read CT and mask
# ============================================================

ct = sitk.ReadImage(ct_path)
mask = sitk.ReadImage(mask_path)

ct_array = sitk.GetArrayFromImage(ct)
mask_array = sitk.GetArrayFromImage(mask)


# ============================================================
# 2. Verify geometry
# ============================================================

if ct.GetSize() != mask.GetSize():
    raise ValueError("CT and mask sizes do not match!")

if ct.GetSpacing() != mask.GetSpacing():
    raise ValueError("CT and mask spacing do not match!")

print("CT and mask geometry match.")


# ============================================================
# 3. Find femur bounding box
# ============================================================

femur_indices = np.argwhere(mask_array == 1)

if len(femur_indices) == 0:
    raise ValueError("Femur mask is empty!")


# NumPy order is [Z, Y, X]
z_min, y_min, x_min = femur_indices.min(axis=0)
z_max, y_max, x_max = femur_indices.max(axis=0)

print("\nOriginal femur bounding box:")
print("Z:", z_min, "to", z_max)
print("Y:", y_min, "to", y_max)
print("X:", x_min, "to", x_max)


# ============================================================
# 4. Add margin around femur
# ============================================================

z_min = max(0, z_min - MARGIN)
y_min = max(0, y_min - MARGIN)
x_min = max(0, x_min - MARGIN)

z_max = min(ct_array.shape[0] - 1, z_max + MARGIN)
y_max = min(ct_array.shape[1] - 1, y_max + MARGIN)
x_max = min(ct_array.shape[2] - 1, x_max + MARGIN)


print("\nBounding box with margin:")
print("Z:", z_min, "to", z_max)
print("Y:", y_min, "to", y_max)
print("X:", x_min, "to", x_max)


# ============================================================
# 5. Crop CT and mask
# ============================================================

cropped_ct = ct_array[
    z_min:z_max + 1,
    y_min:y_max + 1,
    x_min:x_max + 1
]

cropped_mask = mask_array[
    z_min:z_max + 1,
    y_min:y_max + 1,
    x_min:x_max + 1
]


# ============================================================
# 6. Keep only femur voxels
# ============================================================

femur_only = np.where(
    cropped_mask == 1,
    cropped_ct,
    -1000
)


# ============================================================
# 7. Convert back to SimpleITK
# ============================================================

femur_ct = sitk.GetImageFromArray(femur_only)

femur_ct.SetSpacing(ct.GetSpacing())
femur_ct.SetDirection(ct.GetDirection())


# ============================================================
# 8. Correct the physical origin
# ============================================================

# The cropped image starts at this original CT index.
start_index = [int(x_min), int(y_min), int(z_min)]

new_origin = ct.TransformIndexToPhysicalPoint(start_index)

femur_ct.SetOrigin(new_origin)


# ============================================================
# 9. Save
# ============================================================

sitk.WriteImage(femur_ct, output_path)


# ============================================================
# 10. Print information
# ============================================================

print("\nFemur-only cropped CT saved:")
print(output_path)

print("\nOriginal CT size:")
print(ct.GetSize())

print("\nCropped CT size:")
print(femur_ct.GetSize())

print("\nOriginal spacing:")
print(ct.GetSpacing())

print("\nCropped origin:")
print(femur_ct.GetOrigin())

print("\nOriginal CT:")
print("Min:", ct_array.min())
print("Max:", ct_array.max())

print("\nFemur-only CT:")
print("Min:", femur_only.min())
print("Max:", femur_only.max())

print("\nFemur voxels:", np.count_nonzero(cropped_mask == 1))
