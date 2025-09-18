import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
PROMPT = (
    "定位猫的胡须。\n"
    "仅返回有效的JSON格式，包含原始图像尺寸中的绝对像素坐标。\n"
    '使用此模式: {"label":"cat whiskers","bbox_2d":[x1,y1,x2,y2],"confidence": <0-1>}'
)
OUTPUT_IMAGE = "cat_whiskers_annotated.png"


def load_model():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return processor, model


def extract_json(raw_output: str) -> List[Dict[str, Any]]:
    """Return a list of detection objects parsed from the model response."""
    code_match = re.search(r"```(?:json)?\s*(.*?)```", raw_output, re.S)
    candidate = code_match.group(1) if code_match else raw_output
    candidate = candidate.strip()

    parsed: Any
    for pattern in (r"\[.*\]", r"\{.*\}"):
        snippet = re.search(pattern, candidate, re.S)
        if not snippet:
            continue
        try:
            parsed = json.loads(snippet.group(0))
            break
        except json.JSONDecodeError:
            continue
    else:
        return []

    if isinstance(parsed, dict):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    return []


def annotate_image(image_path: Path, detections: Iterable[Dict[str, Any]]) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    for detection in detections:
        bbox = detection.get("bbox_2d") or detection.get("box") or []
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            x0, y0, x1, y1 = map(int, bbox)
        except (TypeError, ValueError):
            continue

        label = detection.get("label", "object")
        conf = detection.get("confidence")
        caption = f"{label}"
        if isinstance(conf, (int, float)):
            caption += f" {conf:.2f}"

        draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
        text_bbox = draw.textbbox((x0, y0), caption, font=font)
        pad = 4
        box_coords = [
            text_bbox[0] - pad,
            max(text_bbox[1] - pad, 0),
            text_bbox[2] + pad,
            text_bbox[3] + pad,
        ]
        draw.rectangle(box_coords, fill="red")
        draw.text((box_coords[0] + pad, box_coords[1] + pad), caption, fill="white", font=font)

    output_path = image_path.with_name(OUTPUT_IMAGE)
    image.save(output_path)
    return output_path


def main() -> None:
    image_path = Path("cat.png").resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Missing image: {image_path}")

    processor, model = load_model()

    image_uri = "file://" + image_path.as_posix()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    output_ids = model.generate(**inputs, max_new_tokens=256)
    output_ids = output_ids[:, inputs["input_ids"].shape[-1] :]
    raw_output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]

    detections = extract_json(raw_output)

    annotated_path: Path | None = None
    if detections:
        annotated_path = annotate_image(image_path, detections)

    payload = {"raw": raw_output, "detections": detections}
    if annotated_path:
        payload["annotated_image"] = str(annotated_path)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
