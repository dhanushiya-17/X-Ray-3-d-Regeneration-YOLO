"""
detect_fracture.py
==================
Run a trained YOLO fracture detector on X-ray images and save the annotated
results.

Only boxes scoring at or above --conf are kept (default 0.78). Overlapping
"fracture" boxes are merged by non-maximum suppression (--iou), so one
fracture shows as one box instead of a stack.

The threshold only hides weak detections; it does not change the scores the
model gives. If an image shows no boxes at 0.78, the model was less sure than
that. Lower --conf to see weaker detections.

Examples
  python scripts/detect_fracture.py --weights runs/detect/train/weights/best.pt --source wrist.png
  python scripts/detect_fracture.py --weights best.pt --source xrays/ --conf 0.5
"""

import argparse
from pathlib import Path

DEFAULT_CONF = 0.78
DEFAULT_IOU = 0.5


def main():
    parser = argparse.ArgumentParser(description="YOLO fracture detection on X-rays")
    parser.add_argument("--weights", required=True, help="trained YOLO weights (.pt)")
    parser.add_argument("--source", required=True, help="image file or folder")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF,
                        help="minimum confidence to keep a box (default %(default)s)")
    parser.add_argument("--iou", type=float, default=DEFAULT_IOU,
                        help="NMS IoU: overlapping boxes above this are merged (default %(default)s)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", default="runs/fracture_detect", help="output folder")
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        agnostic_nms=True,
        imgsz=args.imgsz,
        save=True,
        project=str(Path(args.out).parent),
        name=Path(args.out).name,
        exist_ok=True,
    )

    for r in results:
        name = Path(r.path).name
        if len(r.boxes) == 0:
            print(f"{name}: no detections at conf >= {args.conf}")
            continue
        for box in r.boxes:
            label = model.names[int(box.cls)]
            print(f"{name}: {label} {float(box.conf):.2f}")
    print("Annotated images saved in", args.out)


if __name__ == "__main__":
    main()
