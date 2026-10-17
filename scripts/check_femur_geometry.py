import struct
import numpy as np
import torch
from diffdrr.data import read

# --------------------------------------------------
# Paths
# --------------------------------------------------

ct_path = "data/raw/Subjects/Pat001/Pat001.nrrd"
stl_path = "data/raw/Subjects/Pat001/Pat001.stl"


# --------------------------------------------------
# Read binary STL
# --------------------------------------------------

def read_stl_vertices(filename):

    with open(filename, "rb") as f:
        header = f.read(80)
        n_triangles = struct.unpack("<I", f.read(4))[0]

        vertices = []

        for _ in range(n_triangles):

            # normal vector: 3 floats
            f.read(12)

            # three vertices
            for _ in range(3):
                x, y, z = struct.unpack("<fff", f.read(12))
                vertices.append([x, y, z])

            # attribute byte count
            f.read(2)

    return np.array(vertices)


# --------------------------------------------------
# Read femur STL
# --------------------------------------------------

vertices = read_stl_vertices(stl_path)

print("\n========== FEMUR STL ==========")

print("Number of vertices:", len(vertices))

minimum = vertices.min(axis=0)
maximum = vertices.max(axis=0)
center = (minimum + maximum) / 2
size = maximum - minimum

print("\nMinimum XYZ:")
print(minimum)

print("\nMaximum XYZ:")
print(maximum)

print("\nBounding-box center:")
print(center)

print("\nFemur dimensions (mm):")
print(size)


# --------------------------------------------------
# Read CT using DiffDRR
# --------------------------------------------------

subject = read(ct_path)

print("\n========== DIFFDRR CT ==========")

print("Volume shape:")
print(subject["volume"].data.shape)

print("\nVolume spacing:")
print(subject["volume"].spacing)

print("\nVolume affine:")
print(subject["volume"].affine)

print("\nVolume orientation:")
print(subject["volume"].orientation)

print("\nVolume origin:")
print(subject["volume"].origin)

print("\n================================")
