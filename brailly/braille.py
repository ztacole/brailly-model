"""Braille cell <-> Latin text conversion (Braille Grade 1 / Indonesian profile).

A Braille cell is represented as an integer bit-mask in range 0..63:
bit 0 = dot 1, bit 1 = dot 2, ..., bit 5 = dot 6 (same convention as the
Angelina Braille Images Dataset).

    dot layout      1 o o 4
                    2 o o 5
                    3 o o 6

The default profile follows the Indonesian Braille system (based on
"Sistem Simbol Braille Indonesia", Grade 1 / uncontracted). Punctuation tables
are kept in a single dict so they can be adjusted if your source books use a
different convention.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence


def dots_to_mask(dots: str) -> int:
    """'1245' -> bit-mask."""
    m = 0
    for ch in dots:
        if ch not in "123456":
            raise ValueError(f"invalid dot '{ch}' in '{dots}'")
        m |= 1 << (int(ch) - 1)
    return m


def mask_to_dots(mask: int) -> str:
    return "".join(str(i + 1) for i in range(6) if mask & (1 << i))


def mask_to_unicode(mask: int) -> str:
    # Unicode Braille Patterns block uses the same bit order for dots 1-6.
    return chr(0x2800 + (mask & 0x3F))


def unicode_to_mask(ch: str) -> int:
    return (ord(ch) - 0x2800) & 0x3F


LETTERS: Dict[str, str] = {
    "a": "1", "b": "12", "c": "14", "d": "145", "e": "15", "f": "124",
    "g": "1245", "h": "125", "i": "24", "j": "245", "k": "13", "l": "123",
    "m": "134", "n": "1345", "o": "135", "p": "1234", "q": "12345",
    "r": "1235", "s": "234", "t": "2345", "u": "136", "v": "1236",
    "w": "2456", "x": "1346", "y": "13456", "z": "1356",
}

DIGIT_LETTERS = "jabcdefghi"  # index = digit (0 -> j, 1 -> a, ...)


@dataclass
class BrailleProfile:
    name: str = "id"
    capital: str = "6"            # tanda huruf kapital
    number: str = "3456"          # tanda angka
    letter: str = "56"            # tanda huruf (ends number mode)
    # punctuation: char -> dots. Order matters for the reverse map when two
    # characters share a pattern (resolved contextually in the decoder).
    punctuation: Dict[str, str] = field(default_factory=lambda: {
        ",": "2",
        ";": "23",
        ":": "25",
        ".": "256",
        "?": "236",
        "!": "235",
        "-": "36",
        "'": "3",
        "/": "34",
        "\u201c": "236",   # open quote  (same cell as '?')
        "\u201d": "356",   # close quote
        "(": "2356",
        ")": "2356",
        "*": "35",
    })
    # Separators allowed inside a number (keep number mode active).
    number_punct: Dict[str, str] = field(default_factory=lambda: {
        ",": "2",   # decimal comma
        ".": "3",   # thousands separator
    })


PROFILE_ID = BrailleProfile()


class BrailleCodec:
    def __init__(self, profile: BrailleProfile = PROFILE_ID, unknown: str = "unicode"):
        self.p = profile
        self.unknown = unknown  # "unicode" -> emit the braille glyph, otherwise literal string
        self.cap = dots_to_mask(profile.capital)
        self.num = dots_to_mask(profile.number)
        self.let = dots_to_mask(profile.letter)
        self.letter_of = {dots_to_mask(v): k for k, v in LETTERS.items()}
        self.mask_of_letter = {k: dots_to_mask(v) for k, v in LETTERS.items()}
        self.punct_mask = {k: dots_to_mask(v) for k, v in profile.punctuation.items()}
        self.num_punct_mask = {k: dots_to_mask(v) for k, v in profile.number_punct.items()}
        self.num_punct_of = {v: k for k, v in self.num_punct_mask.items()}
        self.punct_of: Dict[int, str] = {}
        for k, m in self.punct_mask.items():
            self.punct_of.setdefault(m, k)
        self.Q_OPEN = dots_to_mask(profile.punctuation["\u201c"])
        self.PAREN = dots_to_mask(profile.punctuation["("])

    # ------------------------------------------------------------------ encode
    def encode_word(self, word: str) -> List[int]:
        cells: List[int] = []
        if len(word) > 1 and word.isalpha() and word.isupper():
            cells += [self.cap, self.cap]
            word_caps = True
        else:
            word_caps = False
        in_num = False
        for idx, ch in enumerate(word):
            nxt = word[idx + 1] if idx + 1 < len(word) else ""
            if ch.isdigit():
                if not in_num:
                    cells.append(self.num)
                    in_num = True
                cells.append(self.mask_of_letter[DIGIT_LETTERS[int(ch)]])
                continue
            if in_num and ch in self.num_punct_mask and nxt.isdigit():
                cells.append(self.num_punct_mask[ch])
                continue
            if ch.lower() in self.mask_of_letter:
                low = ch.lower()
                if in_num and low in DIGIT_LETTERS:
                    cells.append(self.let)
                in_num = False
                if ch.isupper() and not word_caps:
                    cells.append(self.cap)
                cells.append(self.mask_of_letter[low])
                continue
            in_num = False
            if ch == '"':
                ch = "\u201c" if not cells else "\u201d"
            if ch in self.punct_mask:
                cells.append(self.punct_mask[ch])
            # silently drop unsupported characters
        return cells

    def encode(self, text: str) -> List[Optional[int]]:
        """Text -> list of cells, ``None`` = blank cell (space)."""
        out: List[Optional[int]] = []
        for i, w in enumerate(text.split()):
            if i:
                out.append(None)
            out += self.encode_word(w)
        return out

    # ------------------------------------------------------------------ decode
    def _unk(self, m: int) -> str:
        return mask_to_unicode(m) if self.unknown == "unicode" else self.unknown

    def decode_word(self, cells: Sequence[int]) -> str:
        out: List[str] = []
        in_num = False
        cap_next = False
        cap_word = False
        n = len(cells)
        i = 0
        while i < n:
            m = cells[i]
            if m == self.cap:
                if i + 1 < n and cells[i + 1] == self.cap:
                    cap_word = True
                    i += 2
                    continue
                if i + 1 < n:  # a lone trailing dot 6 is not a capital sign
                    cap_next = True
                    i += 1
                    continue
            if m == self.num:
                in_num = True
                i += 1
                continue
            if m == self.let and i + 1 < n:
                in_num = False
                i += 1
                continue
            if in_num:
                ch = self.letter_of.get(m)
                if ch is not None and ch in DIGIT_LETTERS:
                    out.append(str(DIGIT_LETTERS.index(ch)))
                    i += 1
                    continue
                if m in self.num_punct_of and i + 1 < n and self.letter_of.get(cells[i + 1], "") in DIGIT_LETTERS:
                    out.append(self.num_punct_of[m])
                    i += 1
                    continue
                in_num = False
            ch = self.letter_of.get(m)
            if ch is not None:
                if cap_next or cap_word:
                    ch = ch.upper()
                cap_next = False
                out.append(ch)
                i += 1
                continue
            cap_next = False
            # punctuation with contextual disambiguation
            letters_before = any(c.isalnum() for c in out)
            rest = cells[i + 1:]
            letters_after = any(c in self.letter_of or c == self.num for c in rest)
            if m == self.Q_OPEN:
                out.append("\u201c" if (not letters_before and letters_after) else "?")
            elif m == self.PAREN:
                out.append("(" if (not letters_before and letters_after) else ")")
            elif m in self.punct_of:
                out.append(self.punct_of[m])
            else:
                out.append(self._unk(m))
            i += 1
        return "".join(out)

    def decode(self, cells: Sequence[Optional[int]]) -> str:
        """Cells (``None`` = space) -> text."""
        words: List[str] = []
        cur: List[int] = []
        for m in list(cells) + [None]:
            if m is None or m == 0:
                if cur:
                    words.append(self.decode_word(cur))
                    cur = []
            else:
                cur.append(m)
        return " ".join(words)


def cells_to_unicode(cells: Sequence[Optional[int]]) -> str:
    return "".join(" " if m is None else mask_to_unicode(m) for m in cells)
