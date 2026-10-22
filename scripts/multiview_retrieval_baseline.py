
from pathlib import Path
import argparse
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
        return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def normalized_correlation(a, b):
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("-inf")

    return float(np.dot(a, b) / denominator)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-view synthetic-DRR template retrieval."
    )
    parser.add_argument(
        "--split", choices=("validation", "test"), default="validation",
        help="Subject-level query split; templates always come from training.",
    )
    args = parser.parse_args()

    with SPLIT_FILE.open(newline="") as f:
        rows = list(csv.DictReader(f))

    train_subjects = sorted(
        r["subject"] for r in rows if r["split"] == "train"
    )
    query_subjects = sorted(
        r["subject"] for r in rows if r["split"] == args.split
    )

    expected_queries = {"validation": 15, "test": 16}[args.split]
    if len(train_subjects) != 70 or len(query_subjects) != expected_queries:
        raise ValueError(f"Unexpected training or {args.split} split size.")

    # Load candidates one angle at a time. This keeps peak memory near 75 MB
    # and avoids rereading the same training images for every query subject.
    scores = np.empty(
        (len(query_subjects), len(train_subjects), len(ANGLES)),
        dtype=np.float32,
    )
    for angle_index, angle in enumerate(ANGLES):
        template_images = np.stack([
            load_projection(
                DRR_ROOT / template / f"view_{angle:03d}.png"
            ).ravel()
            for template in train_subjects
        ])
        template_images -= template_images.mean(axis=1, keepdims=True)
        template_norms = np.linalg.norm(template_images, axis=1)

        for query_index, query_subject in enumerate(query_subjects):
            query_path = DRR_ROOT / query_subject / f"view_{angle:03d}.png"
            query = load_projection(query_path).ravel()
            if query.size != template_images.shape[1]:
                raise ValueError(f"Image dimensions differ: {query_path}")
            query -= query.mean()
            query_norm = np.linalg.norm(query)
            denominators = template_norms * query_norm
            angle_scores = np.full(len(train_subjects), -np.inf, dtype=np.float32)
            np.divide(
                template_images @ query,
                denominators,
                out=angle_scores,
                where=denominators > 0,
            )
            scores[query_index, :, angle_index] = angle_scores

    results = []
    for query_index, query_subject in enumerate(query_subjects):
        scores_by_template = scores[query_index]
        mean_scores = scores_by_template.mean(axis=1)
        ranking = np.argsort(mean_scores)[::-1]
        best_index, second_index = int(ranking[0]), int(ranking[1])
        best_score = float(mean_scores[best_index])
        second_score = float(mean_scores[second_index])
        best_template = train_subjects[best_index]
        second_template = train_subjects[second_index]
        per_angle = scores_by_template[best_index].tolist()

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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"multiview_{args.split}_matches.csv"

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nMulti-view retrieval completed.")
    print(f"{args.split.capitalize()} subjects:", len(results))
    print("Training candidates per subject:", len(train_subjects))
    print("Views combined per candidate:", len(ANGLES))
    print("Saved to:", output_path)


if __name__ == "__main__":
    main()
