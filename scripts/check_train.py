import csv
from pathlib import Path

ROOT = Path("../data/fractlas/FracAtlas")

# Read dataset.csv into a dictionary
metadata = {}
with open(ROOT / "dataset.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        metadata[row["image_id"]] = row

fractured = 0
non_fractured = 0

with open(ROOT / "Utilities" / "Fracture Split" / "train.csv", newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        image = row["image_id"]

        if metadata[image]["fractured"] == "1":
            fractured += 1
        else:
            non_fractured += 1

print(f"Fractured images    : {fractured}")
print(f"Non-fractured images: {non_fractured}")
print(f"Total               : {fractured + non_fractured}")