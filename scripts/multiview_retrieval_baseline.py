
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
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.float64) / 255.0


def normalized_correlation(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("-inf")

    return float(np.dot(a, b) / denominator)


def main():
    with SPLIT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    train_subjects = sorted(
        r["subject"] for r in rows if r["split"] == "train"
    )
    val_subjects = sorted(
        r["subject"] for r in rows if r["split"] == "validation"
    )

    if len(train_subjects) != 70 or len(val_subjects) != 15:
        raise ValueError("Unexpected training or validation split size.")

    # Cache the images so they are loaded only once.
    images = {}

    for subject in train_subjects + val_subjects:
        for angle in ANGLES:
            path = DRR_ROOT / subject / f"view_{angle:03d}.png"
            images[(subject, angle)] = load_projection(path)

    results = []

    for query_subject in val_subjects:
        scores = []

        for template_subject in train_subjects:
            angle_scores = []

            for angle in ANGLES:
                score = normalized_correlation(
                    images[(query_subject, angle)],
                    images[(template_subject, angle)],
                )
                angle_scores.append(score)

            scores.append((
                float(np.mean(angle_scores)),
                template_subject,
                angle_scores,
            ))

        scores.sort(key=lambda item: item[0], reverse=True)

        best_score, best_template, per_angle = scores[0]
        second_score, second_template, _ = scores[1]

        results.append({
            "query_subject": query_subject,
            "selected_template": best_template,
            "mean_correlation_all_views": best_score,
            "second_template": second_template,
            "second_template_mean_correlation": second_score,
            "view_scores": ";".join(
                f"{angle}:{score:.6f}"
                for angle, score in zip(ANGLES, per_angle)
            ),
        })

        print(
            f"{query_subject}: selected={best_template}, "
            f"mean correlation={best_score:.4f}, "
            f"runner-up={second_template} ({second_score:.4f})"
        )

    output_path = OUTPUT_DIR / "multiview_validation_matches.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nMulti-view retrieval completed.")
    print("Validation subjects:", len(results))
    print("Training candidates per subject:", len(train_subjects))
    print("Views combined per candidate:", len(ANGLES))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
