#!/usr/bin/env python
"""Read Braille from a photo and print the Latin text.

python scripts/infer.py --model models/brailly.onnx photo.jpg [--json out.json] [--vis out.jpg]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.pipeline import BraillyReader, ReaderConfig, draw_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", default="models/brailly.onnx")
    ap.add_argument("--json", default=None)
    ap.add_argument("--vis", default=None)
    ap.add_argument("--detect-page", action="store_true", help="crop and rectify the paper sheet first")
    ap.add_argument("--rotate-180", action="store_true", help="also try the page upside down")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--show-braille", action="store_true")
    args = ap.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"cannot read {args.image}")
    reader = BraillyReader(args.model, ReaderConfig(detect_page=args.detect_page,
                                                    try_rotate_180=args.rotate_180,
                                                    score_thr=args.score_thr))
    res = reader.read(img)
    if args.show_braille:
        for ln in res["lines"]:
            print(ln["braille"])
        print("-" * 40)
    print(res["text"])
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    if args.vis:
        cv2.imwrite(args.vis, draw_result(img, res))


if __name__ == "__main__":
    main()
