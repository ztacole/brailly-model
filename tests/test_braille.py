import random

from brailly.braille import BrailleCodec, cells_to_unicode, dots_to_mask, mask_to_dots
from brailly.layout import build_layout
from brailly.metrics import GTCell
from brailly.textgen import random_paragraph

codec = BrailleCodec()


def test_dots_roundtrip():
    for m in range(64):
        assert dots_to_mask(mask_to_dots(m)) == m


def test_letters_unicode():
    assert cells_to_unicode(codec.encode("abc")) == "\u2801\u2803\u2809"


def test_roundtrip_sentences():
    cases = [
        "Saya membaca buku di perpustakaan.",
        "Pada tahun 1945 Indonesia merdeka!",
        "Apakah kamu sudah makan?",
        "Anak-anak bermain di taman; mereka senang.",
        "Harga 2,5 kilo adalah 12.500 rupiah.",
        "Ia berkata: \u201cAyo belajar\u201d dengan semangat.",
        "BRAILLE adalah huruf (titik timbul).",
        "Kelas 3a sudah pulang.",
    ]
    for t in cases:
        assert codec.decode(codec.encode(t)) == t, t


def test_roundtrip_random():
    rng = random.Random(0)
    for _ in range(200):
        t = random_paragraph(rng).replace('"', "")
        assert codec.decode(codec.encode(t)) == t


def _cells_for(lines, pitch=20.0, line_h=40.0, x0=10.0, indent=None):
    out = []
    for li, text in enumerate(lines):
        start = (indent[li] if indent else 0)
        for ci, m in enumerate(codec.encode(text)):
            if m is not None:
                out.append(GTCell(x0 + (ci + start) * pitch, 50 + li * line_h, 12, 19, m))
    return out


def test_layout_paragraphs():
    lines = ["Ini adalah kalimat pertama yang", "cukup panjang sekali.", "Paragraf kedua dimulai di sini."]
    cells = _cells_for(lines, indent=[2, 0, 2])
    lay = build_layout(cells, codec)
    assert [l.text for l in lay.lines] == lines
    assert [p.text for p in lay.paragraphs] == [
        "Ini adalah kalimat pertama yang cukup panjang sekali.", "Paragraf kedua dimulai di sini."]


def test_layout_rotated():
    import math
    cells = _cells_for(["satu dua tiga empat lima", "enam tujuh delapan"])
    a = math.radians(5)
    for c in cells:
        c.cx, c.cy = c.cx * math.cos(a) - c.cy * math.sin(a), c.cx * math.sin(a) + c.cy * math.cos(a)
    lay = build_layout(cells, codec)
    assert [l.text for l in lay.lines] == ["satu dua tiga empat lima", "enam tujuh delapan"]
