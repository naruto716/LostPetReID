import json
import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TRANSFORMERS_NO_JAX", "1")

import torch
from PIL import Image
from transformers import AutoProcessor, GLIPModel


MODEL_ID = "microsoft/glip-large"
TEXT_QUERIES = ["dog face", "dog muzzle", "dog nose"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25


def main() -> None:
    image_path = Path("dog.jpg")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = GLIPModel.from_pretrained(MODEL_ID)
    model.eval()

    inputs = processor(images=image, text=TEXT_QUERIES, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    processed_results = processor.post_process_grounded_object_detection(
        outputs,
        inputs,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=target_sizes,
    )

    detections = []
    if processed_results:
        result = processed_results[0]
        labels = result["labels"]
        scores = result["scores"]
        boxes = result["boxes"]

        for label_idx, score, box in zip(labels, scores, boxes):
            label_index = int(label_idx)
            query = TEXT_QUERIES[label_index] if label_index < len(TEXT_QUERIES) else str(label_index)
            x_min, y_min, x_max, y_max = box.tolist()
            detections.append(
                {
                    "label": query,
                    "score": float(score),
                    "box": {
                        "xmin": float(x_min),
                        "ymin": float(y_min),
                        "xmax": float(x_max),
                        "ymax": float(y_max),
                    },
                }
            )

    print(json.dumps({"detections": detections}, indent=2))


if __name__ == "__main__":
    main()
