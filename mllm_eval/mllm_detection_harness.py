#!/usr/bin/env python3
"""Zero-shot MLLM detection harness -> COCO-format predictions.

Prompts a vision-language model for bounding boxes on each eval image and
writes a COCO detections JSON that `eval/coco_eval_report.py` can score, so
MLLM rows land in the same table as the fine-tuned detectors.

Scope and honesty notes (these go in the write-up verbatim):
  * Only box-capable models can be scored: GPT-4o / GPT-4o-mini, Gemini
    (2.0/2.5 family), Qwen2-VL / Qwen2.5-VL. Text-only medical MLLMs
    (LLaVA-Med, HuatuoGPT-Vision, RadFM) emit no structured localization and
    cannot be evaluated on mAP; report them as N/A rather than 0.
  * MLLMs do not produce calibrated ranked confidences. We request a
    per-box confidence in the JSON and fall back to 0.5; either way COCO
    mAP for MLLMs is an approximation and is labeled as such.
  * Every response is cached to disk (one JSON per image) so a run can be
    resumed and audited; temperature 0.

Providers:
    --provider openai   (OpenAI API; also Qwen via any OpenAI-compatible
                         endpoint using --base-url, e.g. vLLM or DashScope)
    --provider gemini   (Google Generative AI API)

Usage:
    python mllm_eval/mllm_detection_harness.py \
        --gt annotations/instances_test.json --images data/test/images \
        --provider openai --model gpt-4o-2024-11-20 \
        --out mllm_gpt4o_dets.json --cache-dir cache/gpt4o
"""
import argparse
import base64
import json
import os
import re
import time
from typing import Dict, List, Optional

PROMPT_TEMPLATE = """You are analyzing a dental radiograph (width={width}px, height={height}px).
Detect every instance of the following finding classes:

{class_list}

Return ONLY a JSON array. Each element:
{{"class": "<exact class name from the list>",
  "box_2d": [x_min, y_min, x_max, y_max],
  "confidence": <0.0-1.0>}}

Coordinates are absolute pixels in the {width}x{height} image. Include every
visible instance, even small or low-contrast ones. If none, return []."""


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json_array(text: str) -> Optional[list]:
    """Parse the first JSON array in the reply, tolerating code fences."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, list) else None
                except json.JSONDecodeError:
                    return None
    return None


def call_openai(model: str, prompt: str, image_b64: str,
                base_url: Optional[str]) -> str:
    from openai import OpenAI
    client = OpenAI(base_url=base_url) if base_url else OpenAI()
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
        ]}],
    )
    return resp.choices[0].message.content or ""


def call_gemini(model: str, prompt: str, image_path: str) -> str:
    from google import genai
    client = genai.Client()  # reads GEMINI_API_KEY
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    resp = client.models.generate_content(
        model=model,
        contents=[genai.types.Part.from_bytes(data=image_bytes,
                                              mime_type="image/jpeg"),
                  prompt],
        config=genai.types.GenerateContentConfig(temperature=0),
    )
    return resp.text or ""


def boxes_to_coco(raw: list, name_to_id: Dict[str, int], image_id: int,
                  width: int, height: int) -> List[dict]:
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("class", "")).strip()
        cat_id = name_to_id.get(cls) or name_to_id.get(cls.lower())
        box = item.get("box_2d")
        if cat_id is None or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in box]
        except (TypeError, ValueError):
            continue
        x0, x1 = sorted((max(0.0, min(x0, width)), max(0.0, min(x1, width))))
        y0, y1 = sorted((max(0.0, min(y0, height)), max(0.0, min(y1, height))))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        try:
            score = min(1.0, max(0.0, float(item.get("confidence", 0.5))))
        except (TypeError, ValueError):
            score = 0.5
        out.append({"image_id": image_id, "category_id": cat_id,
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "score": score})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True, help="COCO gt json for the eval split")
    ap.add_argument("--images", required=True)
    ap.add_argument("--provider", choices=["openai", "gemini"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None,
                    help="OpenAI-compatible endpoint (Qwen via vLLM/DashScope)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap image count (cost control for pilots)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between calls")
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)
    name_to_id = {}
    for c in gt["categories"]:
        name_to_id[c["name"]] = c["id"]
        name_to_id[c["name"].lower()] = c["id"]
    class_list = "\n".join("- " + c["name"] for c in gt["categories"])

    os.makedirs(args.cache_dir, exist_ok=True)
    images = gt["images"][: args.limit] if args.limit else gt["images"]

    dets, failures = [], 0
    for n, img in enumerate(images, 1):
        cache_path = os.path.join(args.cache_dir, "%d.json" % img["id"])
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                cached = json.load(f)
            dets.extend(boxes_to_coco(cached.get("boxes") or [], name_to_id,
                                      img["id"], img["width"], img["height"]))
            continue

        path = os.path.join(args.images, img["file_name"])
        prompt = PROMPT_TEMPLATE.format(width=img["width"], height=img["height"],
                                        class_list=class_list)
        try:
            if args.provider == "openai":
                text = call_openai(args.model, prompt, encode_image(path),
                                   args.base_url)
            else:
                text = call_gemini(args.model, prompt, path)
        except Exception as e:  # network/API errors: record and continue
            print("[%d/%d] ERROR %s: %s" % (n, len(images), img["file_name"], e))
            failures += 1
            continue

        boxes = extract_json_array(text)
        with open(cache_path, "w") as f:
            json.dump({"model": args.model, "raw": text, "boxes": boxes}, f)
        if boxes is None:
            print("[%d/%d] unparseable reply for %s" % (n, len(images), img["file_name"]))
            failures += 1
            continue
        dets.extend(boxes_to_coco(boxes, name_to_id, img["id"],
                                  img["width"], img["height"]))
        if n % 25 == 0:
            print("[%d/%d] %d detections so far" % (n, len(images), len(dets)))
        if args.sleep:
            time.sleep(args.sleep)

    with open(args.out, "w") as f:
        json.dump(dets, f)
    print("wrote %s: %d detections over %d images (%d failures/unparseable)"
          % (args.out, len(dets), len(images), failures))
    print("score it with eval/coco_eval_report.py --dt %s" % args.out)


if __name__ == "__main__":
    main()
