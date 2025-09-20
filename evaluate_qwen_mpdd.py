# ##############################
# Evaluates Qwen on Multi-Pose Dog Dataset using manual annotations
# ##############################

# ------------------------------
# Settings
# ------------------------------

# Force disable FlashAttention and use math-based attention before any imports
import os

os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
os.environ["PYTORCH_SDP_ATTENTION"] = "math"

# Imports
from pathlib import Path
from PIL import Image, ImageDraw
import math
import random
import csv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

# Variables
IMAGE_DIR = "MPDD/pytorch/val"
GT_PATH = "MPDD/pytorch/val_annotations.csv"
NUM = -1  # Number of random images to test. Default = -1 (for all)
TRIES = 1  # Number of tries for valid outputs. Default = 1

# ------------------------------
# Helper functions
# ------------------------------

# Import function(s) from qwen_detector.py
from qwen_detector import load_model, prompt_model, read_output, get_bbox


# Store ground truth annotations into a dictionary
def read_annotations(gt_csv_path):
    gt_dict = {}
    with open(gt_csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Read each row
            img = row["image_name"]
            desc = row["description"]
            try:
                x1 = float(row["x1"]) if row["x1"] else None
                x2 = float(row["x2"]) if row["x2"] else None
                y1 = float(row["y1"]) if row["y1"] else None
                y2 = float(row["y2"]) if row["y2"] else None
            except Exception:
                x1 = x2 = y1 = y2 = None
            # Add to dictionary
            if all(v is not None for v in [x1, x2, y1, y2]):
                gt_dict[img] = {"label": desc, "bbox": [x1, y1, x2, y2]}
            else:
                gt_dict[img] = {"label": desc, "bbox": None}
    return gt_dict


# ------------------------------
# Main function
# ------------------------------


def main() -> None:

    # Load Model
    processor, model = load_model()

    # Load images
    gallery_dir = Path(IMAGE_DIR).resolve()
    image_paths = list(gallery_dir.glob("*.jpg"))

    # Load ground truth from val_annotations.csv
    gt_csv_path = Path(GT_PATH).resolve()
    gt_dict = read_annotations(gt_csv_path)

    # Filter images with valid annotations
    image_paths = [
        p for p in image_paths if gt_dict.get(p.name, {}).get("bbox") is not None
    ]

    # Select NUM random images for testing
    if NUM > 0:
        if len(image_paths) > NUM:
            image_paths = random.sample(image_paths, NUM)
        else:
            print(f"Only {len(image_paths)} images found, using all.")

    # Prepare output (Annotated) directory and clear it at the start
    annotated_dir = gallery_dir.parent / "Annotated"
    if annotated_dir.exists():
        for f in annotated_dir.glob("*"):
            if f.is_file():
                f.unlink()
    else:
        annotated_dir.mkdir(exist_ok=True)

    # Initialize results
    gt_bboxes = []
    pred_bboxes = []
    iou = []
    distances = []
    norm_distances = []

    # --- Loop through images ---
    for _, image_path in enumerate(image_paths):
        print(f"Processing {image_path.name} ...")
        image_uri = "file://" + image_path.as_posix()

        # Set label to description from CSV
        label = gt_dict.get(image_path.name, {}).get("label", "")

        # Get predicted bounding box
        pred_bbox = None
        for attempt in range(TRIES):
            # Prompt model
            raw_output = prompt_model(image_uri, label, processor, model)

            # Parse output
            detections = read_output(raw_output)

            """ # Print Output Payload
            payload = {
                "image": str(image_path),
                "raw": raw_output,
                "detections": detections,
            }
            print(json.dumps(payload, indent=2)) """

            pred_bbox = get_bbox(detections)  # Returns None if no bbox found

            # Check if valid bbox found
            if (
                pred_bbox is not None
                and isinstance(pred_bbox, (list, tuple))
                and len(pred_bbox) == 4
            ):
                # Clip bbox if needed
                x1, y1, x2, y2 = pred_bbox
                width, height = Image.open(image_path).size
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                break
            else:
                pred_bbox = None
                print(f"Try {attempt+1}/{TRIES} for {image_path.name}")
        pred_bboxes.append(pred_bbox)

        # Get GT bbox
        gt_bbox = None
        if gt_info := gt_dict.get(image_path.name, None):
            gt_bbox = gt_info["bbox"]
        gt_bboxes.append(gt_bbox)

        # Draw GT bbox (green) and model bbox (red) on image and save to Annotated folder
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        # Draw GT bbox in green
        if gt_bbox is not None:
            draw.rectangle(gt_bbox, outline="green", width=3)
        # Draw model bbox in red
        if pred_bbox is not None:
            # print("Drawing pred_bbox:", pred_bbox, type(pred_bbox))
            draw.rectangle(pred_bbox, outline="red", width=3)
        # Save annotated image
        annotated_dir.mkdir(exist_ok=True)
        stem = image_path.stem
        suffix = image_path.suffix
        custom_name = f"{stem}_ann{suffix}"
        output_path = annotated_dir / custom_name
        img.save(output_path)

        # --- Evaluation ---

        # Compute IoU if both bboxes exist
        iou_val = None
        if pred_bbox is not None and gt_bbox is not None:
            # Calculate Intersection
            ixmin = max(pred_bbox[0], gt_bbox[0])
            iymin = max(pred_bbox[1], gt_bbox[1])
            ixmax = min(pred_bbox[2], gt_bbox[2])
            iymax = min(pred_bbox[3], gt_bbox[3])
            iw = max(ixmax - ixmin, 0)
            ih = max(iymax - iymin, 0)
            inter = iw * ih
            # Calculate Union
            area_pred = max(pred_bbox[2] - pred_bbox[0], 0) * max(
                pred_bbox[3] - pred_bbox[1], 0
            )
            area_gt = max(gt_bbox[2] - gt_bbox[0], 0) * max(gt_bbox[3] - gt_bbox[1], 0)
            union = area_pred + area_gt - inter
            # Compute IoU
            iou_val = inter / union if union > 0 else 0.0
            print(f"Image: {image_path.name} | IoU (GT vs Prediction): {iou_val:.4f}")
        iou.append(iou_val)

        # Compute pixel distance between midpoints if both bboxes exist
        dist = None
        if pred_bbox is not None and gt_bbox is not None:
            gt_cx = (gt_bbox[0] + gt_bbox[2]) / 2
            gt_cy = (gt_bbox[1] + gt_bbox[3]) / 2
            pred_cx = (pred_bbox[0] + pred_bbox[2]) / 2
            pred_cy = (pred_bbox[1] + pred_bbox[3]) / 2
            dist = math.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2)
            print(
                f"Image: {image_path.name} | Distance between centres (GT vs Prediction): {dist:.2f} pixels"
            )
        distances.append(dist)

        # Compute normalised distance between midpoints if both bboxes exist
        norm_dist = None
        if dist is not None:
            with Image.open(image_path) as img:
                w, h = img.size
                norm_dist = dist / math.sqrt(w**2 + h**2) if w > 0 and h > 0 else None
                print(
                    f"Image: {image_path.name} | Normalized distance between centres (GT vs Prediction): {norm_dist:.2f}"
                )
        norm_distances.append(norm_dist)

    # --- Overall metrics ---
    print("--- Overall metrics ---")
    valid_preds = [p for p in pred_bboxes if p is not None]
    print(f"Total images: {len(image_paths)} | Total predictions: {len(valid_preds)}")

    # Mean IoU
    valid_ious = [v for v in iou if v is not None]
    if valid_ious:
        mean_iou = sum(valid_ious) / len(valid_ious)
        print(f"Mean IoU (GT vs Prediction): {mean_iou:.4f}")
    else:
        print("No valid IoU values to compute mean IoU.")

    # Mean pixel distance between midpoints
    valid_distances = [v for v in distances if v is not None]
    if valid_distances:
        mean_distance = sum(valid_distances) / len(valid_distances)
        print(
            f"Mean pixel distance between centres (GT vs Prediction): {mean_distance:.2f} pixels"
        )
    else:
        print("No valid distance values to compute mean distance.")

    # Mean normalised pixel distance between midpoints
    valid_norm_distances = [v for v in norm_distances if v is not None]
    if valid_norm_distances:
        mean_norm_distance = sum(valid_norm_distances) / len(valid_norm_distances)
        print(
            f"Mean normalised pixel distance between centres (GT vs Prediction): {mean_norm_distance:.2f}"
        )
    else:
        print("No valid distance values to compute mean normalised distance.")

    # --- COCO Evaluation ---

    # Constrct COCO-style JSON inputs
    categories = [{"id": 1, "name": "label"}]
    images = []
    annotations = []
    preds = []
    for idx, (image_path, gt_bbox, pred_bbox) in enumerate(
        zip(image_paths, gt_bboxes, pred_bboxes)
    ):
        with Image.open(image_path) as img:
            w, h = img.size
        images.append(
            {
                "id": idx,
                "file_name": image_path.name,
                "width": w,
                "height": h,
            }
        )
        if gt_bbox is not None:
            x1, y1, x2, y2 = gt_bbox
            annotations.append(
                {
                    "id": idx,
                    "image_id": idx,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": max(0, (x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
        if pred_bbox is not None:
            x1, y1, x2, y2 = pred_bbox
            preds.append(
                {
                    "image_id": idx,
                    "category_id": 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": 1.0,
                }
            )
    gt_json = {
        "info": {"description": "MPDD Manual Annotations", "version": "1.0"},
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    # Run COCOeval
    if len(preds) == 0:
        print("[Warning] No predictions found. Skipping COCOeval.")
        return

    # Suppress pycocotools print output
    class DummyFile(object):
        def write(self, x):
            pass

        def flush(self):
            pass

    _stdout = sys.stdout
    try:
        sys.stdout = DummyFile()
        cocoGt = COCO()
        cocoGt.dataset = gt_json
        cocoGt.createIndex()
        cocoDt = cocoGt.loadRes(preds)
        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    finally:
        sys.stdout = _stdout
    print("COCO Average Precision metrics:")
    # Only print first 3 metrics
    if (
        hasattr(cocoEval, "stats")
        and cocoEval.stats is not None
        and len(cocoEval.stats) >= 3
    ):
        print(f"AP@[.5:.95]: {cocoEval.stats[0]:.4f}")
        print(f"AP@0.5:      {cocoEval.stats[1]:.4f}")
        print(f"AP@0.75:     {cocoEval.stats[2]:.4f}")
    else:
        print(
            "[Warning] COCOeval could not compute AP metrics (not enough data or invalid input)."
        )


if __name__ == "__main__":
    main()
