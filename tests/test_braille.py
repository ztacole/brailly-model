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


def test_punctuation_matches_international_standard():
    """Tanda baca mengikuti standar Braille internasional.

    Deret tanda baca diturunkan dari a-j yang digeser satu baris ke bawah
    (⠂⠆⠒⠲⠢⠖⠶⠦⠔⠴), sedangkan apostrof dan tanda hubung digeser dua posisi
    (⠄⠤). Rujukan: artikel "Braille" di Wikipedia, bagian Punctuation.
    https://en.wikipedia.org/wiki/Braille
    """
    standar = {
        ",": "2",      # koma
        ";": "23",     # titik koma
        ":": "25",     # titik dua
        ".": "256",    # titik
        "!": "235",    # tanda seru
        "?": "236",    # tanda tanya (sama dengan kutip buka)
        "(": "2356",   # kurung (buka dan tutup memakai pola sama)
        ")": "2356",
        "'": "3",      # apostrof
        "-": "36",     # tanda hubung
        "/": "34",     # garis miring
        "\u201c": "236",   # kutip buka
        "\u201d": "356",   # kutip tutup
    }
    for ch, dots in standar.items():
        assert codec.p.punctuation[ch] == dots, f"{ch}: {codec.p.punctuation[ch]} != {dots}"
    assert codec.p.capital == "6"      # tanda kapital
    assert codec.p.number == "3456"    # tanda angka


def test_shared_patterns_disambiguated_by_position():
    """⠦ dan ⠶ dipakai ganda; artinya ditentukan posisinya dalam kata."""
    assert codec.decode(codec.encode("Benar?")) == "Benar?"
    assert codec.decode(codec.encode("\u201cbenar\u201d")) == "\u201cbenar\u201d"
    assert codec.decode(codec.encode("(benar)")) == "(benar)"


def test_number_mode_and_letter_sign():
    # tanda angka aktif sampai tanda huruf atau spasi
    assert codec.decode(codec.encode("tahun 1945")) == "tahun 1945"
    assert codec.decode(codec.encode("12.500")) == "12.500"      # pemisah ribuan
    assert codec.decode(codec.encode("2,5")) == "2,5"            # desimal
    assert codec.decode(codec.encode("kelas 3a")) == "kelas 3a"  # angka diikuti huruf
    # titik di akhir angka adalah tanda baca, bukan pemisah ribuan
    assert codec.decode(codec.encode("ada 89.")) == "ada 89."
