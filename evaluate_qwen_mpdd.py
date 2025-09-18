# ------------------------------
# Evaluates Qwen on Multi-Pose Dog Dataset
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

# Variables from qwen_face.py
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
LABEL = "dog tail"
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


def main() -> None:

    # Model and Processor Loading (once)
    processor, model = load_model()

    # TEST
    # Directory containing gallery images
    import random

    gallery_dir = Path("MPDD/pytorch/gallery").resolve()
    image_paths = list(gallery_dir.glob("*.jpg"))
    # Prepare Annotated directory and clear it at the start
    annotated_dir = gallery_dir.parent / "Annotated"
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
            annotated_dir = gallery_dir.parent / "Annotated"
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


if __name__ == "__main__":
    main()
