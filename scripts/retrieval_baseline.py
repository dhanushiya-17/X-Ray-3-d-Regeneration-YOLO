
from pathlib import Path
import csv

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = ROOT / "data" / "subject_split.csv"
DRR_ROOT = ROOT / "data" / "drr_dataset"
OUTPUT_DIR = ROOT / "reconstruction" / "retrieval_baseline"

ANGLES = list(range(0, 360, 45))


def load_projection(path):
    """Load an opaque grayscale DRR as a normalized float array."""
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L"), dtype=np.float32)

    # Convert 8-bit pixel values to [0, 1].
    return gray / 255.0


def normalized_correlation(image_a, image_b):
    """Calculate normalized cross-correlation of two projections."""
    a = image_a.ravel().astype(np.float64)
    b = image_b.ravel().astype(np.float64)

    a = a - a.mean()
    b = b - b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return float("-inf")

    return float(np.dot(a, b) / denominator)


def main():
    with SPLIT_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    train_subjects = sorted(
        row["subject"] for row in rows if row["split"] == "train"
    )
    val_subjects = sorted(
        row["subject"] for row in rows if row["split"] == "validation"
    )

    if not train_subjects or not val_subjects:
        raise ValueError("Training or validation subjects are missing.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    # Process validation subjects only.
    for val_subject in val_subjects:
        for angle in ANGLES:
            filename = f"view_{angle:03d}.png"
            query_path = DRR_ROOT / val_subject / filename

            query = load_projection(query_path)
            candidates = []

            for train_subject in train_subjects:
                candidate_path = DRR_ROOT / train_subject / filename
                candidate = load_projection(candidate_path)

                if candidate.shape != query.shape:
                    raise ValueError(
                        f"Image dimensions differ: {query_path}, "
                        f"{candidate_path}"
                    )

                score = normalized_correlation(query, candidate)
                candidates.append((score, train_subject))

            # Highest correlation is the closest projection match.
            candidates.sort(reverse=True)
            best_score, best_subject = candidates[0]

            results.append({
                "query_subject": val_subject,
                "angle_degrees": angle,
                "matched_train_subject": best_subject,
                "correlation": best_score,
                "rank_2_subject": (
                    candidates[1][1] if len(candidates) > 1 else ""
                ),
                "rank_2_correlation": (
                    candidates[1][0] if len(candidates) > 1 else ""
                ),
            })

        print(f"Processed validation subject: {val_subject}")

    output_csv = OUTPUT_DIR / "validation_matches.csv"

    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nRetrieval baseline completed.")
    print("Training templates:", len(train_subjects))
    print("Validation subjects:", len(val_subjects))
    print("Views per subject:", len(ANGLES))
    print("Total matches:", len(results))
    print("Results saved to:", output_csv)

    print("\nFirst five matches:")
    for result in results[:5]:
        print(
            result["query_subject"],
            f'view={result["angle_degrees"]}',
            "match=", result["matched_train_subject"],
            f'score={result["correlation"]:.4f}'
        )


if __name__ == "__main__":
    main()
