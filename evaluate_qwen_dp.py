# ------------------------------
# Evaluates Qwen on Dog-Pose Dataset
# ------------------------------

# Force disable FlashAttention and use math-based attention before any imports
import os

os.environ["FLASH_ATTENTION_FORCE_DISABLED"] = "1"
os.environ["PYTORCH_SDP_ATTENTION"] = "math"

# Imports from qwen_face.py
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import math
from PIL import ImageDraw
import random

# Variables from qwen_face.py
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
# Supported labels: "left front leg", "left back leg", "right front leg", "right back leg", "tail", "left ear", "right ear", "nose", "mouth"
LABEL = "nose"
PROMPT = (
    f"找到动物的身体部位，并用尽可能最小的矩形框将其紧密贴合，避免任何不必要的背景。\n"
    f"边界框坐标 (x1, y1, x2, y2) 必须完全在图像边界内: x1 >= 0, y1 >= 0, x2 <= 图像宽度, y2 <= 图像高度。请勿预测图像外的边界框。\n"
    f'返回一个 JSON 数组，格式如下：{{"label":"{LABEL}","bbox_2d":[x1,y1,x2,y2],"confidence": <0-1>}}。\n'
    f"注意：下面的示例仅供格式参考，实际输出请根据模型对检测结果的置信度合理填写 confidence 字段，并根据图片内容输出真实的检测结果。不要直接照搬示例。\n"
    f'示例：[{{"label":"{LABEL}","bbox_2d":[12,34,56,78], "confidence": 0.73}}]\n'
    f"如果未检测到目标，请返回空数组 []。\n"
    f"bbox_2d 应包含原始图像尺寸的绝对像素坐标。"
)

# Import function(s) from qwen_face.py
from qwen_face import annotate_image


# Modified load_model function
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    return processor, model


# Modified extract_json function - robust JSON extraction function (inlined)
def extract_json(raw_output: str):
    code_match = re.search(r"```(?:json)?\\s*(.*?)```", raw_output, re.S)
    candidate = code_match.group(1) if code_match else raw_output
    candidate = candidate.strip()

    # Try to extract JSON array or object by regex
    parsed = None
    for pattern in (r"\[.*\]", r"\{.*\}"):
        snippet = re.search(pattern, candidate, re.S)
        if not snippet:
            continue
        try:
            parsed = json.loads(snippet.group(0))
            break
        except json.JSONDecodeError:
            continue
    # If regex fails, try to parse the whole candidate
    if parsed is None:
        try:
            parsed = json.loads(candidate)
        except Exception:
            return []

    # Always wrap a single dict as a list
    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


# Draw keypoints from a YOLO-format label file onto the image.
def draw_keypoints_on_image(
    image_path: Path, label_path: Path, kpt_count: int = 24, radius: int = 6
) -> Image.Image:
    # Load image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    # Read label file
    with open(label_path, "r") as f:
        line = f.readline().strip()
        parts = line.split()
        # YOLO: class x_center y_center width height x1 y1 v1 x2 y2 v2 ...
        kpt_data = parts[5 : 5 + kpt_count * 3]
        keypoints = []
        for i in range(kpt_count):
            x = float(kpt_data[i * 3]) * w
            y = float(kpt_data[i * 3 + 1]) * h
            v = int(float(kpt_data[i * 3 + 2]))
            keypoints.append((x, y, v))

    # Standard keypoint names for Ultralytics Dog-Pose (24 keypoints)
    # User-specified mapping: index -> label
    kpt_names = [str(i) for i in range(kpt_count)]
    kpt_names[0] = "left front leg (lower)"
    kpt_names[1] = "left front leg (middle)"
    kpt_names[2] = "left front leg (upper)"
    kpt_names[3] = "left back leg (lower)"
    kpt_names[4] = "left back leg (middle)"
    kpt_names[5] = "left back leg (upper)"
    kpt_names[6] = "right front leg (lower)"
    kpt_names[7] = "right front leg (middle)"
    kpt_names[8] = "right front leg (upper)"
    kpt_names[9] = "right back leg (lower)"
    kpt_names[10] = "right back leg (middle)"
    kpt_names[11] = "right back leg (upper)"
    kpt_names[12] = "tail (base)"
    kpt_names[13] = "tail (end)"
    kpt_names[14] = "left ear (base)"
    kpt_names[15] = "right ear (base)"
    kpt_names[16] = "nose"
    kpt_names[17] = "mouth"
    kpt_names[18] = "left ear (end)"
    kpt_names[19] = "right ear (end)"
    kpt_names[20] = "20"
    kpt_names[21] = "21"
    kpt_names[22] = "22"
    kpt_names[23] = "23"

    # Draw keypoints
    palette = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "purple",
        "lime",
        "pink",
        "teal",
        "brown",
        "gold",
        "navy",
        "violet",
        "indigo",
        "maroon",
        "olive",
        "coral",
        "aqua",
        "orchid",
        "salmon",
        "turquoise",
        "gray",
    ]
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # Only draw visible keypoints
            color = palette[i % len(palette)]
            draw.ellipse(
                [(x - radius, y - radius), (x + radius, y + radius)],
                fill=color,
                outline="black",
                width=2,
            )
            # Draw label
            label = kpt_names[i] if i < len(kpt_names) else f"kp_{i+1}"
            draw.text((x + radius + 2, y - radius), label, fill=color, font=font)

    return img


# Map from simplified part name to list of keypoint indexes
SIMPLIFIED_GROUPS = {
    "left front leg": [0, 1, 2],
    "left back leg": [3, 4, 5],
    "right front leg": [6, 7, 8],
    "right back leg": [9, 10, 11],
    "tail": [12, 13],
    "left ear": [14, 18],
    "right ear": [15, 19],
    "nose": [16],
    "mouth": [17],
}


# Compute simplified keypoints as average of visible keypoints in each group
def compute_simplified_keypoints(keypoints):
    simplified = {}
    for part, idxs in SIMPLIFIED_GROUPS.items():
        pts = [(keypoints[i][0], keypoints[i][1]) for i in idxs if keypoints[i][2] > 0]
        if pts:
            x = sum(p[0] for p in pts) / len(pts)
            y = sum(p[1] for p in pts) / len(pts)
            simplified[part] = (x, y)
    return simplified


# Draw simplified keypoints as large colored dots with labels
def draw_simplified_keypoints_on_image(
    image_path: Path, label_path: Path, kpt_count: int = 24, radius: int = 10
) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    # Read label file
    with open(label_path, "r") as f:
        line = f.readline().strip()
        parts = line.split()
        kpt_data = parts[5 : 5 + kpt_count * 3]
        keypoints = []
        for i in range(kpt_count):
            x = float(kpt_data[i * 3]) * w
            y = float(kpt_data[i * 3 + 1]) * h
            v = int(float(kpt_data[i * 3 + 2]))
            keypoints.append((x, y, v))
    simplified = compute_simplified_keypoints(keypoints)
    palette = [
        "red",
        "green",
        "blue",
        "yellow",
        "magenta",
        "cyan",
        "orange",
        "purple",
        "lime",
    ]
    font = None
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 20)
    except Exception:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    for i, (part, (x, y)) in enumerate(simplified.items()):
        color = palette[i % len(palette)]
        draw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=color,
            outline="black",
            width=3,
        )
        draw.text((x + radius + 2, y - radius), part, fill=color, font=font)
    return img


def main() -> None:
    # Model and Processor Loading (once)
    processor, model = load_model()

    # Use Dog-Pose Dataset images
    dogpose_root = Path("dog-pose").resolve()
    images_dir = dogpose_root / "images" / "train"
    labels_dir = dogpose_root / "labels" / "train"
    image_paths = list(images_dir.glob("*.jpg"))
    # Prepare Annotated directory and clear it at the start
    annotated_dir = dogpose_root / "Annotated"
    if annotated_dir.exists():
        for f in annotated_dir.glob("*"):
            if f.is_file():
                f.unlink()
    else:
        annotated_dir.mkdir(exist_ok=True)
    # Select 5 random images for testing
    if len(image_paths) > 5:
        image_paths = random.sample(image_paths, 5)
    else:
        print(f"Only {len(image_paths)} images found, using all.")

    """ # --- Draw keypoints for the first 5 images ---
    for image_path in image_paths[:5]:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            print(f"No label for {image_path.name}, skipping.")
            continue
        img_with_kpts = draw_keypoints_on_image(image_path, label_path)
        out_path = annotated_dir / (image_path.stem + "_kpts.jpg")
        img_with_kpts.save(out_path)
        print(f"Saved keypoint visualization: {out_path}") """

    # --- Draw simplified keypoints for the first 5 images ---
    for image_path in image_paths[:5]:
        label_path = labels_dir / (image_path.stem + ".txt")
        if not label_path.exists():
            print(f"No label for {image_path.name}, skipping.")
            continue
        img_with_simple_kpts = draw_simplified_keypoints_on_image(
            image_path, label_path
        )
        out_path = annotated_dir / (image_path.stem + "_simplekpts.jpg")
        img_with_simple_kpts.save(out_path)
        print(f"Saved simplified keypoint visualization: {out_path}")

    # Loop through images
    for idx, image_path in enumerate(image_paths):
        print(f"Processing {image_path.name} ...")
        image_uri = "file://" + image_path.as_posix()

        # Build Prompt Messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_uri},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]

        # Apply Chat Template
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process Vision Info
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare Model Inputs
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        # Move Inputs to GPU (if available)
        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Model Inference
        output_ids = model.generate(**inputs, max_new_tokens=256)
        output_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
        raw_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        # Parse Model Output
        detections = extract_json(raw_output)

        # Annotate Image (if detections found) and save to Annotated folder
        annotated_path: Path | None = None
        if detections:
            # annotated_dir already defined above as dogpose_root / 'Annotated'
            annotated_dir.mkdir(exist_ok=True)
            # Save with custom name: original name + '_ann' before extension
            stem = image_path.stem
            suffix = image_path.suffix
            custom_name = f"{stem}_ann{suffix}"
            output_path = annotated_dir / custom_name
            annotated_path = annotate_image(image_path, detections)
            # Move/rename the annotated image to the Annotated folder with custom name
            if annotated_path.exists():
                annotated_path.replace(output_path)
            annotated_path = output_path

        # --- Compare simplified keypoint to Qwen bbox for LABEL ---
        pixel_errors = []
        norm_errors = []
        w, h = Image.open(image_path).size
        with open(labels_dir / (image_path.stem + ".txt"), "r") as f:
            line = f.readline().strip()
            parts = line.split()
            kpt_data = parts[5 : 5 + 24 * 3]
            keypoints = []
            for i in range(24):
                x = float(kpt_data[i * 3]) * w
                y = float(kpt_data[i * 3 + 1]) * h
                v = int(float(kpt_data[i * 3 + 2]))
                keypoints.append((x, y, v))
        simplified = compute_simplified_keypoints(keypoints)
        label_kpt = simplified.get(LABEL)
        if detections and label_kpt is not None:
            det = next(
                (
                    d
                    for d in detections
                    if d.get("label", "") == LABEL and "bbox_2d" in d
                ),
                None,
            )
            if det:
                x1, y1, x2, y2 = det["bbox_2d"]
                bbox_mid = ((x1 + x2) / 2, (y1 + y2) / 2)
                dx = bbox_mid[0] - label_kpt[0]
                dy = bbox_mid[1] - label_kpt[1]
                pixel_error = (dx**2 + dy**2) ** 0.5
                diag = (w**2 + h**2) ** 0.5
                norm_error = pixel_error / diag
                pixel_errors.append(pixel_error)
                norm_errors.append(norm_error)
                inside = (x1 <= label_kpt[0] <= x2) and (y1 <= label_kpt[1] <= y2)
                print(
                    f"Image: {image_path.name} | LABEL: {LABEL} | Keypoint inside bbox: {inside}"
                )
                # Store for summary
                if "inside_results" not in locals():
                    inside_results = []
                inside_results.append(inside)

        # Prepare Output Payload
        payload = {
            "image": str(image_path),
            "raw": raw_output,
            "detections": detections,
        }
        if annotated_path:
            payload["annotated_image"] = str(annotated_path)

        # Print Results
        print(json.dumps(payload, indent=2))

    # After all images, print overall proportion of True for inside_results
    if "inside_results" in locals() and inside_results:
        proportion = sum(inside_results) / len(inside_results)
        print(
            f"\nOverall proportion of keypoints inside bounding box: {proportion:.2%} ({sum(inside_results)}/{len(inside_results)})"
        )

    # Print mean pixel error and mean normalized error
    if pixel_errors and norm_errors:
        mean_pixel_error = sum(pixel_errors) / len(pixel_errors)
        mean_norm_error = sum(norm_errors) / len(norm_errors)
        print(f"Mean pixel error: {mean_pixel_error:.2f} pixels")
        print(f"Mean normalized error: {mean_norm_error:.4f}")


if __name__ == "__main__":
    main()
