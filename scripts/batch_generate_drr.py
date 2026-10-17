import os
import struct
import numpy as np
import SimpleITK as sitk
import torch
import matplotlib.pyplot as plt

from diffdrr.data import read
from diffdrr.drr import DRR


# ============================================================
# SETTINGS
# ============================================================

DATA_ROOT = "data/raw/Subjects"
OUTPUT_ROOT = "data/drr_dataset"

# ------------------------------------------------------------
# TESTING ONLY
# ------------------------------------------------------------
# We are testing one subject first.
SUBJECTS = sorted(
    [
        name
        for name in os.listdir(DATA_ROOT)
        if name.startswith("Pat") and os.path.isdir(os.path.join(DATA_ROOT, name))
    ]
)

# ------------------------------------------------------------
# DRR geometry
# ------------------------------------------------------------

IMAGE_SIZE = 512
PIXEL_SIZE = 1.5
SDD = 1020.0

# ------------------------------------------------------------
# X-ray viewing angles
# ------------------------------------------------------------
# These are actual changes in viewing direction.
ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

# Camera translation
TRANSLATION = [0.0, 850.0, 0.0]

# Margin around femur when cropping
MARGIN = 20


# ============================================================
# CREATE FEMUR MASK
# ============================================================

def create_femur_mask(ct, stl_path):

    # --------------------------------------------------------
    # Read STL
    # --------------------------------------------------------

    stl_reader = vtk.vtkSTLReader()

    stl_reader.SetFileName(
        stl_path
    )

    stl_reader.Update()

    femur_mesh = stl_reader.GetOutput()

    # --------------------------------------------------------
    # CT geometry
    # --------------------------------------------------------

    size = ct.GetSize()

    spacing = ct.GetSpacing()

    origin = ct.GetOrigin()

    # --------------------------------------------------------
    # Create VTK image
    # --------------------------------------------------------

    image = vtk.vtkImageData()

    image.SetDimensions(
        size[0],
        size[1],
        size[2]
    )

    image.SetSpacing(
        spacing[0],
        spacing[1],
        spacing[2]
    )

    image.SetOrigin(
        origin[0],
        origin[1],
        origin[2]
    )

    image.AllocateScalars(
        vtk.VTK_UNSIGNED_CHAR,
        1
    )

    # --------------------------------------------------------
    # Initialize image to 1
    # --------------------------------------------------------
    # The stencil will change voxels outside the femur to 0.

    scalars = image.GetPointData().GetScalars()

    for i in range(
        scalars.GetNumberOfTuples()
    ):
        scalars.SetTuple1(
            i,
            1
        )

    # --------------------------------------------------------
    # Convert STL → image stencil
    # --------------------------------------------------------

    stencil = vtk.vtkPolyDataToImageStencil()

    stencil.SetInputData(
        femur_mesh
    )

    stencil.SetOutputOrigin(
        origin[0],
        origin[1],
        origin[2]
    )

    stencil.SetOutputSpacing(
        spacing[0],
        spacing[1],
        spacing[2]
    )

    stencil.SetOutputWholeExtent(
        image.GetExtent()
    )

    stencil.Update()

    # --------------------------------------------------------
    # Apply stencil
    # --------------------------------------------------------

    mask_filter = vtk.vtkImageStencil()

    mask_filter.SetInputData(
        image
    )

    mask_filter.SetStencilConnection(
        stencil.GetOutputPort()
    )

    mask_filter.ReverseStencilOff()

    mask_filter.SetBackgroundValue(
        0
    )

    mask_filter.Update()

    # --------------------------------------------------------
    # VTK → NumPy
    # --------------------------------------------------------

    vtk_array = (
        mask_filter
        .GetOutput()
        .GetPointData()
        .GetScalars()
    )

    mask_array = (
        vtk.util.numpy_support
        .vtk_to_numpy(vtk_array)
        .reshape(
            size[2],
            size[1],
            size[0]
        )
    )

    return mask_array


# ============================================================
# CREATE FEMUR-ONLY CT
# ============================================================

def create_femur_only_ct(
    ct,
    mask_array
):

    # --------------------------------------------------------
    # Convert CT to NumPy
    # --------------------------------------------------------

    ct_array = (
        sitk.GetArrayFromImage(ct)
    )

    # --------------------------------------------------------
    # Keep femur
    # --------------------------------------------------------
    # Femur voxels keep their original CT value.
    # Everything else becomes -1000 HU.

    femur_array = np.where(
        mask_array == 1,
        ct_array,
        -1000
    )

    # --------------------------------------------------------
    # NumPy → SimpleITK
    # --------------------------------------------------------

    femur_ct = (
        sitk.GetImageFromArray(
            femur_array
        )
    )

    # --------------------------------------------------------
    # Preserve CT geometry
    # --------------------------------------------------------

    femur_ct.SetSpacing(
        ct.GetSpacing()
    )

    femur_ct.SetOrigin(
        ct.GetOrigin()
    )

    femur_ct.SetDirection(
        ct.GetDirection()
    )

    return femur_ct


# ============================================================
# CROP FEMUR-ONLY CT
# ============================================================

def crop_femur_ct(
    ct,
    femur_ct,
    mask_array,
    margin=MARGIN
):

    # --------------------------------------------------------
    # Find all femur voxels
    # --------------------------------------------------------

    femur_indices = np.argwhere(
        mask_array == 1
    )

    if len(femur_indices) == 0:

        raise ValueError(
            "Femur mask is empty!"
        )

    # --------------------------------------------------------
    # NumPy order is [Z, Y, X]
    # --------------------------------------------------------

    z_min, y_min, x_min = (
        femur_indices.min(axis=0)
    )

    z_max, y_max, x_max = (
        femur_indices.max(axis=0)
    )

    # --------------------------------------------------------
    # Add margin
    # --------------------------------------------------------

    z_min = max(
        0,
        z_min - margin
    )

    y_min = max(
        0,
        y_min - margin
    )

    x_min = max(
        0,
        x_min - margin
    )

    z_max = min(
        femur_ct.GetSize()[2] - 1,
        z_max + margin
    )

    y_max = min(
        femur_ct.GetSize()[1] - 1,
        y_max + margin
    )

    x_max = min(
        femur_ct.GetSize()[0] - 1,
        x_max + margin
    )

    # --------------------------------------------------------
    # Print bounding box
    # --------------------------------------------------------

    print("\nFemur bounding box:")

    print(
        "Z:",
        z_min,
        "to",
        z_max
    )

    print(
        "Y:",
        y_min,
        "to",
        y_max
    )

    print(
        "X:",
        x_min,
        "to",
        x_max
    )

    # --------------------------------------------------------
    # Convert femur CT to NumPy
    # --------------------------------------------------------

    femur_array = (
        sitk.GetArrayFromImage(
            femur_ct
        )
    )

    # --------------------------------------------------------
    # Crop
    # --------------------------------------------------------

    cropped_array = femur_array[
        z_min:z_max + 1,
        y_min:y_max + 1,
        x_min:x_max + 1
    ]

    # --------------------------------------------------------
    # Convert back to SimpleITK
    # --------------------------------------------------------

    cropped_ct = (
        sitk.GetImageFromArray(
            cropped_array
        )
    )

    # --------------------------------------------------------
    # Preserve spacing and direction
    # --------------------------------------------------------

    cropped_ct.SetSpacing(
        femur_ct.GetSpacing()
    )

    cropped_ct.SetDirection(
        femur_ct.GetDirection()
    )

    # --------------------------------------------------------
    # Correct physical origin
    # --------------------------------------------------------
    # The cropped image starts at this location
    # in the original CT.

    start_index = [
        int(x_min),
        int(y_min),
        int(z_min)
    ]

    new_origin = (
        ct.TransformIndexToPhysicalPoint(
            start_index
        )
    )

    cropped_ct.SetOrigin(
        new_origin
    )

    # --------------------------------------------------------
    # Print cropped size
    # --------------------------------------------------------

    print(
        "Cropped size:",
        cropped_ct.GetSize()
    )

    return cropped_ct


# ============================================================
# GENERATE DRRs
# ============================================================

def generate_drrs(
    subject_id,
    femur_ct,
    output_dir
):

    # --------------------------------------------------------
    # Temporary CT path
    # --------------------------------------------------------

    temp_ct = os.path.join(
        output_dir,
        "_temporary_femur_ct.nrrd"
    )

    # --------------------------------------------------------
    # Save cropped CT temporarily
    # --------------------------------------------------------

    sitk.WriteImage(
        femur_ct,
        temp_ct
    )

    # --------------------------------------------------------
    # Read using DiffDRR
    # --------------------------------------------------------

    subject = read(
        temp_ct
    )

    # --------------------------------------------------------
    # Select GPU
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Create DiffDRR
    # --------------------------------------------------------

    drr = DRR(
        subject=subject,

        sdd=SDD,

        height=IMAGE_SIZE,

        width=IMAGE_SIZE,

        delx=PIXEL_SIZE,

        dely=PIXEL_SIZE,

        renderer="siddon",
    ).to(device)

    # --------------------------------------------------------
    # Generate each view
    # --------------------------------------------------------

    for angle in ANGLES:

        # ----------------------------------------------------
        # Change azimuth/viewing direction
        # ----------------------------------------------------

        rotations = torch.tensor(
            [
                [
                    np.deg2rad(angle),
                    0.0,
                    0.0
                ]
            ],
            dtype=torch.float32,
            device=device
        )

        # ----------------------------------------------------
        # Camera translation
        # ----------------------------------------------------

        translations = torch.tensor(
            [
                TRANSLATION
            ],
            dtype=torch.float32,
            device=device
        )

        # ----------------------------------------------------
        # Generate DRR
        # ----------------------------------------------------

        with torch.no_grad():

            projection = drr(
                rotations,
                translations,
                parameterization="euler_angles",
                convention="ZXY",
                degrees=False,
            )

        # ----------------------------------------------------
        # Remove batch/channel dimensions
        # ----------------------------------------------------

        projection = (
            projection[0, 0]
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # Normalize for PNG visualization
        # ----------------------------------------------------

        if projection.max() > 0:

            image = (
                projection - projection.min()
            ) / (
                projection.max()
                - projection.min()
            )

        else:

            image = projection

        # ----------------------------------------------------
        # Save image
        # ----------------------------------------------------

        output_file = os.path.join(
            output_dir,
            f"view_{angle:03d}.png"
        )

        plt.imsave(
            output_file,
            image,
            cmap="gray"
        )

        print(
            f"  Saved view {angle:03d}°"
        )

    # --------------------------------------------------------
    # Delete temporary CT
    # --------------------------------------------------------

    os.remove(
        temp_ct
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "========================================"
    )

    print(
        "BoneVision Batch DRR Generator"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_ROOT,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Process each subject
    # --------------------------------------------------------

    for subject_id in SUBJECTS:

        print(
            "\n----------------------------------------"
        )

        print(
            "Processing:",
            subject_id
        )

        print(
            "----------------------------------------"
        )

        # ----------------------------------------------------
        # Subject directory
        # ----------------------------------------------------

        subject_dir = os.path.join(
            DATA_ROOT,
            subject_id
        )

        # ----------------------------------------------------
        # CT path
        # ----------------------------------------------------

        ct_path = os.path.join(
            subject_dir,
            f"{subject_id}.nrrd"
        )

        # ----------------------------------------------------
        # STL path
        # ----------------------------------------------------

        stl_path = os.path.join(
            subject_dir,
            f"{subject_id}.stl"
        )

        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        if not os.path.exists(
            ct_path
        ):

            print(
                "ERROR: CT not found"
            )

            continue

        if not os.path.exists(
            stl_path
        ):

            print(
                "ERROR: STL not found"
            )

            continue

        # ----------------------------------------------------
        # Output directory
        # ----------------------------------------------------

        output_dir = os.path.join(
            OUTPUT_ROOT,
            subject_id
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # ----------------------------------------------------
        # Read CT
        # ----------------------------------------------------

        print(
            "Reading CT..."
        )

        ct = sitk.ReadImage(
            ct_path
        )

        # ----------------------------------------------------
        # Create femur mask
        # ----------------------------------------------------

        print(
            "Creating femur mask..."
        )

        mask = create_femur_mask(
            ct,
            stl_path
        )

        femur_voxels = (
            np.count_nonzero(mask)
        )

        print(
            "Femur voxels:",
            femur_voxels
        )

        # ----------------------------------------------------
        # Check mask
        # ----------------------------------------------------

        if femur_voxels == 0:

            print(
                "ERROR: Empty femur mask!"
            )

            continue

        # ----------------------------------------------------
        # Create femur-only CT
        # ----------------------------------------------------

        print(
            "Creating femur-only CT..."
        )

        femur_ct = (
            create_femur_only_ct(
                ct,
                mask
            )
        )

        # ----------------------------------------------------
        # Crop around femur
        # ----------------------------------------------------

        print(
            "Cropping around femur..."
        )

        femur_ct = crop_femur_ct(
            ct,
            femur_ct,
            mask,
            margin=MARGIN
        )

        # ----------------------------------------------------
        # Generate DRRs
        # ----------------------------------------------------

        print(
            "Generating DRRs..."
        )

        generate_drrs(
            subject_id,
            femur_ct,
            output_dir
        )

        # ----------------------------------------------------
        # Finished subject
        # ----------------------------------------------------

        print(
            f"{subject_id} COMPLETE"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "Batch processing finished"
    )

    print(
        "========================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import vtk
    import vtk.util.numpy_support

    main()
