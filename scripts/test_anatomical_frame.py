
from pathlib import Path
import numpy as np


def read_landmarks(path):
    values = {}

    with open(path) as file:
        for line in file:
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            values[key.strip()] = float(value.strip())

    def point(prefix):
        return np.array([
            values[f"{prefix}X"],
            values[f"{prefix}Y"],
            values[f"{prefix}Z"],
        ], dtype=np.float64)

    return {
        "femurHead": point("femurHead"),
        "headDir": point("headDir"),
        "greatTroch": point("greatTroch"),
        "kneeCenter": point("kneeCenter"),
    }


def build_frame(landmarks):
    origin = landmarks["femurHead"]

    # X points from the femoral head origin toward headDir.
    x_axis = landmarks["headDir"] - origin
    x_axis /= np.linalg.norm(x_axis)

    # Remove the component of the greater-trochanter direction
    # that lies along X, making Y perpendicular to X.
    y_raw = landmarks["greatTroch"] - origin
    y_axis = y_raw - np.dot(y_raw, x_axis) * x_axis
    y_axis /= np.linalg.norm(y_axis)

    # Right-handed orthonormal frame.
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    # Columns of R are the local axes in CT coordinates.
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    return origin, rotation


path = Path("data/raw/Subjects/Pat001/Pat001_palp.inp")
landmarks = read_landmarks(path)
origin, rotation = build_frame(landmarks)

print("Origin in CT coordinates:", origin)
print("\nRotation matrix (columns = local axes):")
print(rotation)

print("\nR transpose R (should be identity):")
print(rotation.T @ rotation)

print("\nDeterminant (should be close to +1):")
print(np.linalg.det(rotation))

print("\nLandmarks in local coordinates:")
for name, point in landmarks.items():
    local = rotation.T @ (point - origin)
    print(f"{name:12s}: {local}")

print("\nHead origin maps to:", rotation.T @ (origin - origin))

