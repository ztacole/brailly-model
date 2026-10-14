"""End-to-end reader: camera photo -> Braille cells -> Latin paragraphs."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import List, Optional

import cv2
import numpy as np

from .braille import BrailleCodec, mask_to_dots
from .layout import Layout, build_layout
from .postprocess import Cell, decode_output
from .preprocess import find_page_quad, resize_pad, warp_page


@dataclass
class ReaderConfig:
    long_side: int = 1024          # first-pass input size
    target_cell_h: float = 36.0    # preferred cell height (px) at model input
    cell_h_tol: float = 1.35       # re-run when off by more than this factor
    max_side: int = 2560
    score_thr: float = 0.3
    detect_page: bool = False      # crop + rectify the sheet before reading
    try_rotate_180: bool = False   # pick best of 0/180 degrees
    dehyphenate: bool = False


class _TorchBackend:
    def __init__(self, path: str):
        import torch
        from .model import BraillyNet
        self.torch = torch
        self.model = BraillyNet(pretrained=False)
        sd = torch.load(path, map_location="cpu")
        self.model.load_state_dict(sd["model"] if "model" in sd else sd)
        self.model.eval()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            return self.model(self.torch.from_numpy(x)).numpy()


class _OnnxBackend:
    def __init__(self, path: str):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.name: x})[0]


class BraillyReader:
    def __init__(self, model_path: str, config: Optional[ReaderConfig] = None,
                 codec: Optional[BrailleCodec] = None):
        self.cfg = config or ReaderConfig()
        self.codec = codec or BrailleCodec()
        self.backend = _OnnxBackend(model_path) if model_path.endswith(".onnx") else _TorchBackend(model_path)

    # ------------------------------------------------------------------
    def detect(self, img_bgr: np.ndarray, scale: float) -> List[Cell]:
        x = resize_pad(img_bgr, scale)
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = self.backend(np.ascontiguousarray(rgb.transpose(2, 0, 1)[None]))[0]
        return decode_output(out, self.cfg.score_thr, scale=1.0 / scale)

    def detect_cells(self, img_bgr: np.ndarray) -> List[Cell]:
        """Two-pass multi-scale detection: auto-adjusts to the Braille size."""
        h, w = img_bgr.shape[:2]
        scale = self.cfg.long_side / max(h, w)
        cells = self.detect(img_bgr, scale)
        if len(cells) >= 5:
            med_h = _robust_cell_h(cells) * scale
            ratio = self.cfg.target_cell_h / max(med_h, 1e-3)
            if ratio > self.cfg.cell_h_tol or ratio < 1 / self.cfg.cell_h_tol:
                new_scale = min(scale * ratio, self.cfg.max_side / max(h, w))
                if abs(new_scale / scale - 1) > 0.15:
                    cells2 = self.detect(img_bgr, new_scale)
                    if len(cells2) >= 0.6 * len(cells):
                        cells = cells2
        return _size_consensus(cells)

    def read(self, img_bgr: np.ndarray) -> dict:
        page_quad = None
        if self.cfg.detect_page:
            page_quad = find_page_quad(img_bgr)
            if page_quad is not None:
                img_bgr, _ = warp_page(img_bgr, page_quad)
        cells = self.detect_cells(img_bgr)
        rotated = False
        if self.cfg.try_rotate_180 and cells:
            rot = cv2.rotate(img_bgr, cv2.ROTATE_180)
            cells_r = self.detect_cells(rot)
            if _quality(cells_r, self.codec) > _quality(cells, self.codec) * 1.05:
                cells, img_bgr, rotated = cells_r, rot, True
        layout = build_layout(cells, self.codec, dehyphenate=self.cfg.dehyphenate)
        return _to_dict(layout, cells, rotated, page_quad)


def _robust_cell_h(cells: List[Cell]) -> float:
    """Cell height supported by the most (confident) detections.

    Texture false positives tend to be small and low-confidence; real Braille
    on a page shares one size, so take the score-weighted mode of log(h).
    """
    hs = np.log(np.array([c.h for c in cells]))
    ws = np.array([c.score for c in cells]) ** 2
    bins = np.arange(hs.min() - 0.1, hs.max() + 0.2, 0.1)
    hist, edges = np.histogram(hs, bins=bins, weights=ws)
    hist = np.convolve(hist, [0.5, 1, 0.5], mode="same")
    k = int(np.argmax(hist))
    centre = (edges[k] + edges[k + 1]) / 2
    sel = np.abs(hs - centre) < 0.25
    return float(np.exp(np.median(hs[sel]))) if sel.any() else float(np.exp(np.median(hs)))


def _size_consensus(cells: List[Cell], tol: float = 0.4) -> List[Cell]:
    if len(cells) < 5:
        return cells
    ref = _robust_cell_h(cells)
    return [c for c in cells if abs(np.log(c.h / ref)) < tol]


def _quality(cells: List[Cell], codec: BrailleCodec) -> float:
    if not cells:
        return 0.0
    conf = np.mean([c.score * np.mean(np.abs(c.dot_probs - 0.5) * 2) for c in cells])
    valid = np.mean([(c.mask in codec.letter_of) or (c.mask in codec.punct_of) or
                     c.mask in (codec.cap, codec.num) for c in cells])
    return float(len(cells) * conf * valid)


def _to_dict(layout: Layout, cells: List[Cell], rotated: bool, page_quad) -> dict:
    return {
        "text": layout.text,
        "paragraphs": [{"kind": p.kind, "text": p.text} for p in layout.paragraphs],
        "lines": [{"braille": l.braille, "text": l.text} for l in layout.lines],
        "cells": [{"box": [round(v, 1) for v in c.box], "dots": mask_to_dots(c.mask),
                   "score": round(c.score, 3)} for c in cells],
        "meta": {"num_cells": len(cells), "angle_deg": round(layout.angle_deg, 2),
                 "cell_pitch_px": round(layout.pitch, 1), "rotated_180": rotated,
                 "page_quad": None if page_quad is None else np.round(page_quad, 1).tolist()},
    }


def draw_result(img_bgr: np.ndarray, result: dict) -> np.ndarray:
    vis = img_bgr.copy()
    for c in result["cells"]:
        x0, y0, x1, y1 = map(int, c["box"])
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 0, 255), 1)
    return vis
