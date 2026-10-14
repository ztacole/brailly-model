"""Targets -> ideal logits -> decode -> layout must reproduce the page text."""

import random

import numpy as np

from brailly.braille import BrailleCodec
from brailly.data import encode_targets
from brailly.layout import build_layout
from brailly.metrics import gt_from_array
from brailly.postprocess import decode_output
from brailly.synth import SynthConfig, render_sample


def _ideal_logits(cells, h, w):
    heat, dots, dmask, reg, rmask = encode_targets(cells, h, w)
    out = np.zeros((11,) + heat.shape, np.float32)
    out[0] = np.where(heat >= 0.999, 5.0, np.where(heat > 0, -1.0, -8.0))
    out[1:7] = (dots * 2 - 1) * 6
    out[7:11] = reg
    return out


def test_decode_matches_ground_truth():
    rng = random.Random(3)
    cfg = SynthConfig(out_h=768, out_w=640, dot_px_min=8, dot_px_max=12, p_random_cells=0, p_negative=0,
                      max_rotation=3)
    codec = BrailleCodec()
    for _ in range(3):
        img, cells = render_sample(rng, cfg)
        assert img.shape == (768, 640, 3)
        pred = decode_output(_ideal_logits(cells, 768, 640))
        gt = gt_from_array(cells)
        assert abs(len(pred) - len(gt)) <= max(2, 0.02 * len(gt))
        lp, lg = build_layout(pred, codec), build_layout(gt, codec)
        assert lp.braille == lg.braille
