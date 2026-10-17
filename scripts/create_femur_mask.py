import vtk
import vtk.util.numpy_support
import SimpleITK as sitk

# ============================================================
# Paths
# ============================================================

ct_path = "data/raw/Subjects/Pat001/Pat001.nrrd"
stl_path = "data/raw/Subjects/Pat001/Pat001.stl"

output_path = "data/raw/Subjects/Pat001/Pat001_femur_mask.nrrd"


# ============================================================
# 1. Read CT
# ============================================================

ct = sitk.ReadImage(ct_path)

print("CT size:     ", ct.GetSize())
print("CT spacing:  ", ct.GetSpacing())
print("CT origin:   ", ct.GetOrigin())
print("CT direction:", ct.GetDirection())


# ============================================================
# 2. Read STL
# ============================================================

stl_reader = vtk.vtkSTLReader()
stl_reader.SetFileName(stl_path)
stl_reader.Update()

femur_mesh = stl_reader.GetOutput()

print("\nSTL loaded")
print("Number of points:", femur_mesh.GetNumberOfPoints())
print("Number of cells: ", femur_mesh.GetNumberOfCells())


# ============================================================
# 3. Create VTK image with the SAME geometry as CT
# ============================================================

size = ct.GetSize()
spacing = ct.GetSpacing()
origin = ct.GetOrigin()


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

image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)


# ============================================================
# 4. Initialize the image to 1
# ============================================================

scalars = image.GetPointData().GetScalars()

for i in range(scalars.GetNumberOfTuples()):
    scalars.SetTuple1(i, 1)

# ============================================================
# 5. Convert STL surface into voxel mask
# ============================================================

stencil = vtk.vtkPolyDataToImageStencil()

stencil.SetInputData(femur_mesh)

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


# ============================================================
# 6. Fill voxels inside STL
# ============================================================

mask_filter = vtk.vtkImageStencil()

mask_filter.SetInputData(image)

mask_filter.SetStencilConnection(
    stencil.GetOutputPort()
)

mask_filter.ReverseStencilOff()

mask_filter.SetBackgroundValue(0)

mask_filter.Update()


# ============================================================
# 7. Convert VTK → SimpleITK
# ============================================================

mask = sitk.GetImageFromArray(
    vtk.util.numpy_support.vtk_to_numpy(
        mask_filter.GetOutput().GetPointData().GetScalars()
    ).reshape(
        size[2],
        size[1],
        size[0]
    )
)

mask.SetSpacing(ct.GetSpacing())
mask.SetOrigin(ct.GetOrigin())
mask.SetDirection(ct.GetDirection())


# ============================================================
# 8. Save
# ============================================================

sitk.WriteImage(mask, output_path)

print("\nMask saved:")
print(output_path)

print("\nMask size:")
print(mask.GetSize())

print("Mask spacing:")
print(mask.GetSpacing())
