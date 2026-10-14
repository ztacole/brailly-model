"""Camera-image preprocessing: page detection / perspective correction, resizing."""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def _order_quad(pts: np.ndarray) -> np.ndarray:
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    return np.float32([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]])


def find_page_quad(img_bgr: np.ndarray, min_area: float = 0.25) -> Optional[np.ndarray]:
    """Largest bright quadrilateral (the paper sheet). Returns TL,TR,BR,BL or None."""
    h, w = img_bgr.shape[:2]
    s = 600.0 / max(h, w)
    small = cv2.resize(img_bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    # Braille paper is usually the brightest large region: Otsu + morphology
    _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < min_area * small.shape[0] * small.shape[1] or area > 0.98 * small.shape[0] * small.shape[1]:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        approx = cv2.boxPoints(cv2.minAreaRect(c))
    return _order_quad(np.float32(approx) / s)


def warp_page(img_bgr: np.ndarray, quad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tl, tr, br, bl = quad
    W = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    H = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]])
    M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (W, H), flags=cv2.INTER_CUBIC), M


def resize_pad(img: np.ndarray, scale: float, multiple: int = 32) -> np.ndarray:
    """Resize by ``scale`` and pad (replicate) to a multiple of ``multiple``."""
    if abs(scale - 1.0) > 1e-3:
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=interp)
    h, w = img.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph or pw:
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
    return img
