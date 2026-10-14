#!/usr/bin/env python
"""Minimal HTTP API for the mobile/web app.

    uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
    curl -F "file=@photo.jpg" "http://localhost:8000/read?detect_page=true"

Env: BRAILLY_MODEL (default models/brailly.onnx)
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brailly.pipeline import BraillyReader, ReaderConfig  # noqa: E402

MODEL = os.environ.get("BRAILLY_MODEL", os.path.join(os.path.dirname(__file__), "..", "models", "brailly.onnx"))
app = FastAPI(title="Brailly", version="0.1.0")
_readers = {}


def _reader(detect_page: bool, rotate: bool) -> BraillyReader:
    key = (detect_page, rotate)
    if key not in _readers:
        _readers[key] = BraillyReader(MODEL, ReaderConfig(detect_page=detect_page, try_rotate_180=rotate))
    return _readers[key]


@app.get("/health")
def health():
    return {"status": "ok", "model": os.path.basename(MODEL)}


@app.post("/read")
async def read(file: UploadFile = File(...), detect_page: bool = False, rotate_180: bool = False,
               include_cells: bool = False):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "invalid image")
    res = _reader(detect_page, rotate_180).read(img)
    if not include_cells:
        res.pop("cells")
    return res
