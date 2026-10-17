
from pathlib import Path
import numpy as np


ROOT = Path("data/raw/Subjects")
TOL = 1e-6


def read_landmarks(path):
    values = {}

    with path.open() as file:
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


def build_frame(points):
    origin = points["femurHead"]

    x = points["headDir"] - origin
    nx = np.linalg.norm(x)
    if nx < TOL:
        raise ValueError("Head direction has zero length")
    x /= nx

    y = points["greatTroch"] - origin
    y -= np.dot(y, x) * x
    ny = np.linalg.norm(y)
    if ny < TOL:
        raise ValueError("Trochanter direction is parallel to X")
    y /= ny

    z = np.cross(x, y)
    z /= np.linalg.norm(z)

    rotation = np.column_stack((x, y, z))
    return origin, rotation


subjects = sorted(p for p in ROOT.glob("Pat*") if p.is_dir())
failures = []
max_orthogonality_error = 0.0
min_determinant = float("inf")
max_determinant = -float("inf")

for subject in subjects:
    path = subject / f"{subject.name}_palp.inp"

    try:
        points = read_landmarks(path)
        if not all(np.isfinite(p).all() for p in points.values()):
            raise ValueError("Non-finite landmark coordinate")

        origin, rotation = build_frame(points)

        orth_error = np.max(
            np.abs(rotation.T @ rotation - np.eye(3))
        )
        determinant = np.linalg.det(rotation)

        max_orthogonality_error = max(
            max_orthogonality_error, orth_error
        )
        min_determinant = min(min_determinant, determinant)
        max_determinant = max(max_determinant, determinant)

        if orth_error > TOL or abs(determinant - 1.0) > TOL:
            raise ValueError(
                f"Invalid frame: orthogonality={orth_error}, "
                f"determinant={determinant}"
            )

    except Exception as exc:
        failures.append((subject.name, str(exc)))

print("Subjects checked:", len(subjects))
print("Failed frames:", len(failures))
print("Maximum orthogonality error:", max_orthogonality_error)
print("Determinant range:", min_determinant, "to", max_determinant)

if failures:
    print("\nFailures:")
    for subject, reason in failures:
        print(subject, ":", reason)
else:
    print("All anatomical frames passed.")
