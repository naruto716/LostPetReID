# ##############################
# Qwen detection model and utilities
# For example usage, see evaluate_qwen_mpdd.py
# ##############################

# Imports
import json
import re
import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
)
from qwen_vl_utils import process_vision_info


# Model version
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# Load model
def load_model():
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    return processor, model


# Prompt model with image and label
def prompt_model(image_uri, label, processor, model):
    prompt = (
        # Find animal body parts and tightly fit them with the smallest possible rectangular box, avoiding any unnecessary background.
        f"找到动物的身体部位，并用尽可能最小的矩形框将其紧密贴合，避免任何不必要的背景。"
        # The bounding box coordinates (x1, y1, x2, y2) must be completely within the image bounds: x1 >= 0, y1 >= 0, x2 <= image width, y2 <= image height. Do not project bounding boxes outside the image.
        f"边界框坐标 [x1, y1, x2, y2] 必须完全在图像边界内: x1 >= 0, y1 >= 0, x2 <= 图像宽度, y2 <= 图像高度。请勿预测图像外的边界框。"
        # Only return valid JSON arrays. All keys and values ​​must be enclosed in double quotes. No markdown, code blocks, line breaks, or tabs.
        f"请只返回有效的 JSON 数组，所有键和值都必须用英文双引号括起来，不能有任何 markdown、代码块、换行符或制表符。"
        # The output must fit entirely on one line, without extra spaces or formatting.
        f"输出必须全部在一行，不要有多余的空格或格式化。"
        # The output format is as follows: [{"bbox_2d":[x1,y1,x2,y2]}]
        f'格式如下：[{{"bbox_2d":[x1,y1,x2,y2]}}]'
        # Return the coordinates in an array of [x1, y1, x2, y2] format.
        f"输出必须全部在一行，不要有多余的空格或格式化。"
        # If the [body part] cannot be detected, give the location you think is most likely. Do not return an empty array.
        f"如果无法检测到 {label}，请给出你认为最有可能的位置。不要返回空数组。"
        # If the [body part] cannot be detected, return an empty array.
        # f"如果无法检测到 {label}，请返回空数组。"
        # Only detect the animal's [body part], do not detect other parts.
        f"只检测动物的 {label}，不要检测其他部位。"
        # The bbox_2d should contain absolute pixel coordinates of the original image size.
        f"bbox_2d 应包含原始图像尺寸的绝对像素坐标。"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_uri},
                {"type": "text", "text": prompt},
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
    output = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    return output


# Read output JSON from model response
def read_output(raw_output: str):
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


# Get bounding box from detections
def get_bbox(detections):
    pred_bbox = None
    for det_item in detections:
        if "bbox_2d" in det_item:
            pred_bbox = det_item["bbox_2d"]
            break
    return pred_bbox
