#!/usr/bin/env python3
"""Unit tests for the MLLM harness reply parsing and COCO conversion.

Run:  python tests/test_mllm_parsing.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mllm_eval.mllm_detection_harness import (  # noqa: E402
    extract_json_array, boxes_to_coco)

NAME_TO_ID = {"Caries": 1, "caries": 1, "Crown": 2, "crown": 2}


def main():
    # fenced reply with prose around it
    reply = """Here are the findings:
```json
[{"class": "Caries", "box_2d": [10, 20, 110, 90], "confidence": 0.8},
 {"class": "crown", "box_2d": [200, 50, 300, 150], "confidence": 0.6}]
```
Let me know if you need more."""
    boxes = extract_json_array(reply)
    assert boxes is not None and len(boxes) == 2

    dets = boxes_to_coco(boxes, NAME_TO_ID, image_id=7, width=640, height=480)
    assert len(dets) == 2
    assert dets[0]["bbox"] == [10.0, 20.0, 100.0, 70.0]        # xyxy -> xywh
    assert dets[0]["category_id"] == 1 and dets[0]["score"] == 0.8
    assert dets[1]["category_id"] == 2                          # case-insensitive

    # garbage tolerance: unknown class, malformed box, out-of-range coords,
    # swapped corners, non-numeric confidence
    messy = [
        {"class": "Dragon", "box_2d": [0, 0, 10, 10], "confidence": 0.9},
        {"class": "Caries", "box_2d": [50, 50], "confidence": 0.9},
        {"class": "Caries", "box_2d": [-40, -40, 700, 500], "confidence": 2.5},
        {"class": "Caries", "box_2d": [120, 90, 20, 30], "confidence": "high"},
        "not a dict",
    ]
    dets = boxes_to_coco(messy, NAME_TO_ID, image_id=1, width=640, height=480)
    assert len(dets) == 2
    assert dets[0]["bbox"] == [0.0, 0.0, 640.0, 480.0] and dets[0]["score"] == 1.0
    assert dets[1]["bbox"] == [20.0, 30.0, 100.0, 60.0] and dets[1]["score"] == 0.5

    # replies with no array, or broken JSON
    assert extract_json_array("I cannot identify any pathology.") is None
    assert extract_json_array("[{broken json") is None
    assert extract_json_array("[]") == []

    print("ALL CHECKS PASSED: MLLM reply parsing and COCO conversion robust")


if __name__ == "__main__":
    main()
