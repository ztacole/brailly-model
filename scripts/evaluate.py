#!/usr/bin/env python
"""Evaluate a model on annotated photos (Angelina CSV format) or synthetic pages.

Metrics
  det P/R/F1     : cell localisation (centre inside the GT cell)
  sym P/R/F1     : cell localised AND all 6 dots correct (the metric used for
                   optical Braille recognition in the Angelina paper)
  braille CER    : edit distance between page braille strings (reading order,
                   spaces and line breaks included) built from GT vs predicted cells
  text CER       : same on decoded Latin text (meaningful for Indonesian text only)

python scripts/evaluate.py --model models/brailly.pt --list data/AngelinaDataset/uploaded/test.txt
python scripts/evaluate.py --model models/brailly.pt --synthetic 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.data import load_real, read_image_list  # noqa: E402
from brailly.layout import build_layout  # noqa: E402
from brailly.metrics import gt_from_array, levenshtein, match_cells  # noqa: E402
from brailly.pipeline import BraillyReader, ReaderConfig  # noqa: E402
from brailly.synth import SynthConfig, render_sample  # noqa: E402


def iter_samples(args):
    if args.synthetic:
        rng = random.Random(12345)
        cfg = SynthConfig(out_h=1024, out_w=768, dot_px_min=6, dot_px_max=14, max_rotation=4,
                          p_page_edge=0.5, p_random_cells=0.0, p_negative=0.0)
        for i in range(args.synthetic):
            img, cells = render_sample(rng, cfg)
            yield f"synth_{i}", img, cells
    for lf in args.list:
        for p in read_image_list(lf):
            img, cells = load_real(p)
            yield p, img, cells


def f1(p, r):
    return 2 * p * r / max(1e-9, p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--list", nargs="*", default=[])
    ap.add_argument("--synthetic", type=int, default=0)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--save-vis", default=None, help="folder for visualisations")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    reader = BraillyReader(args.model, ReaderConfig(score_thr=args.score_thr))
    tot = dict(n_pred=0, n_gt=0, matched=0, correct=0, b_err=0, b_len=0, t_err=0, t_len=0)
    per = []
    t0 = time.time()
    for name, img, gt_arr in iter_samples(args):
        pred = reader.detect_cells(img)
        gt = gt_from_array(gt_arr)
        m, c = match_cells(pred, gt)
        lp, lg = build_layout(pred, reader.codec), build_layout(gt, reader.codec)
        be = levenshtein(lp.braille, lg.braille)
        te = levenshtein(lp.text, lg.text)
        tot["n_pred"] += len(pred); tot["n_gt"] += len(gt); tot["matched"] += m; tot["correct"] += c
        tot["b_err"] += be; tot["b_len"] += len(lg.braille); tot["t_err"] += te; tot["t_len"] += len(lg.text)
        per.append({"image": name, "gt": len(gt), "pred": len(pred), "sym_correct": c,
                    "braille_cer": be / max(1, len(lg.braille))})
        if args.save_vis:
            os.makedirs(args.save_vis, exist_ok=True)
            vis = img.copy()
            gtm = {(round(g.cx), round(g.cy)): g for g in gt}
            for p in pred:
                x0, y0, x1, y1 = map(int, p.box)
                cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 200, 0), 1)
            for g in gt:
                cv2.circle(vis, (int(g.cx), int(g.cy)), 2, (255, 0, 0), -1)
            cv2.imwrite(os.path.join(args.save_vis, os.path.basename(name) + ".jpg"), vis)
    dp = tot["matched"] / max(1, tot["n_pred"]); dr = tot["matched"] / max(1, tot["n_gt"])
    sp = tot["correct"] / max(1, tot["n_pred"]); sr = tot["correct"] / max(1, tot["n_gt"])
    res = {
        "images": len(per), "seconds_per_image": (time.time() - t0) / max(1, len(per)),
        "det_precision": dp, "det_recall": dr, "det_f1": f1(dp, dr),
        "sym_precision": sp, "sym_recall": sr, "sym_f1": f1(sp, sr),
        "pattern_acc_on_matched": tot["correct"] / max(1, tot["matched"]),
        "braille_cer": tot["b_err"] / max(1, tot["b_len"]),
        "text_cer": tot["t_err"] / max(1, tot["t_len"]),
    }
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"summary": res, "per_image": per}, f, indent=2)


if __name__ == "__main__":
    main()
