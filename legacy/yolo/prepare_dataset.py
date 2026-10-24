from pathlib import Path
import random
import shutil

# =====================================================
# Configuration
# =====================================================

# Make the train/val/test split reproducible
random.seed(42)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1

# =====================================================
# Dataset Paths
# =====================================================

DATASET_ROOT = Path("../data/fractlas/FracAtlas")
IMAGES_DIR = DATASET_ROOT / "images"
OUTPUT_DIR = Path("../datasets/fracatlas")
LABELS_DIR = DATASET_ROOT / "Annotations" / "YOLO"

# =====================================================
# Read Images
# =====================================================

fractured_images = list((IMAGES_DIR / "Fractured").glob("*.jpg"))
non_fractured_images = list((IMAGES_DIR / "Non_fractured").glob("*.jpg"))

# Safety check
if not fractured_images:
    raise RuntimeError("No fractured images found.")

if not non_fractured_images:
    raise RuntimeError("No non-fractured images found.")

# =====================================================
# Split Function
# =====================================================

def split_dataset(images):
    """
    Split a list of images into:
    - 80% Train
    - 10% Validation
    - 10% Test
    """

    random.shuffle(images)

    train_end = int(len(images) * TRAIN_RATIO)
    val_end = train_end + int(len(images) * VAL_RATIO)

    train = images[:train_end]
    val = images[train_end:val_end]
    test = images[val_end:]

    return train, val, test


# =====================================================
# Copy Images Function
# =====================================================

def copy_images(images, split, fractured):
    image_destination = OUTPUT_DIR / "images" / split
    label_destination = OUTPUT_DIR / "labels" / split

    for image in images:

        shutil.copy2(image, image_destination)

        label_file = label_destination / image.with_suffix(".txt").name

        if fractured:

            source_label = LABELS_DIR / image.with_suffix(".txt").name

            if source_label.exists():
                shutil.copy2(source_label, label_file)
            else:
                print(f"Missing label: {source_label.name}")

        else:

            label_file.touch()

    print(f"Finished copying {split} images.")


# =====================================================
# Verification Function
# =====================================================

def verify_split(split):
    """
    Confirms that every image in images/<split> has a matching
    file in labels/<split>, and that the counts line up.
    """

    image_files = list((OUTPUT_DIR / "images" / split).glob("*.jpg"))
    label_files = list((OUTPUT_DIR / "labels" / split).glob("*.txt"))

    n_images = len(image_files)
    n_labels = len(label_files)

    # Build a set of label filenames for fast lookup
    label_names = {label.name for label in label_files}

    # Find any image whose expected label file is missing
    unmatched = [
        img.name for img in image_files
        if img.with_suffix(".txt").name not in label_names
    ]

    status = "OK" if n_images == n_labels and not unmatched else "MISMATCH"

    print(f"{split:5s} -> images: {n_images:5d}  labels: {n_labels:5d}  [{status}]")

    if unmatched:
        print(f"  Images missing a label file ({len(unmatched)}):")
        for name in unmatched:
            print(f"    - {name}")

    return status == "OK"


# =====================================================
# Split Dataset
# =====================================================

train_fractured, val_fractured, test_fractured = split_dataset(fractured_images)

train_non_fractured, val_non_fractured, test_non_fractured = split_dataset(
    non_fractured_images
)

# =====================================================
# Recreate Output Directory
# =====================================================

if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for folder in ["images", "labels"]:
    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / folder / split).mkdir(parents=True, exist_ok=True)

# =====================================================
# Print Split Summary
# =====================================================

print("========== FRACTURED ==========")
print(f"Train : {len(train_fractured)}")
print(f"Val   : {len(val_fractured)}")
print(f"Test  : {len(test_fractured)}")
print(f"Total : {len(train_fractured) + len(val_fractured) + len(test_fractured)}")

print()

print("======= NON-FRACTURED ========")
print(f"Train : {len(train_non_fractured)}")
print(f"Val   : {len(val_non_fractured)}")
print(f"Test  : {len(test_non_fractured)}")
print(
    f"Total : {len(train_non_fractured) + len(val_non_fractured) + len(test_non_fractured)}"
)

print()

print("Output directory created at:")
print(OUTPUT_DIR)

# =====================================================
# Copy Images
# =====================================================

copy_images(train_fractured, "train", True)
copy_images(val_fractured, "val", True)
copy_images(test_fractured, "test", True)

copy_images(train_non_fractured, "train", False)
copy_images(val_non_fractured, "val", False)
copy_images(test_non_fractured, "test", False)

print("\nImage copying completed.")

# =====================================================
# Verify Split
# =====================================================

print("\n========== VERIFICATION ==========")

all_ok = True
for split in ["train", "val", "test"]:
    if not verify_split(split):
        all_ok = False

print("\nAll splits OK." if all_ok else "\nWARNING: mismatches found above.")