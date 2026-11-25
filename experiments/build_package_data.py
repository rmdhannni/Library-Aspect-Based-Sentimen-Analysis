"""Build the packaged data resources from the raw, un-committed sources.

Run from the repository root after placing the raw files under ``Data/raw``::

    python experiments/build_package_data.py

The script only derives data that we are allowed to redistribute:

* slang dictionary  -> kamus-alay (MIT) + NLP_bahasa_resources (MIT) + pujangga (Apache-2.0)
* stop words        -> NLP_bahasa_resources (MIT) + stopwords-iso (MIT)
* vocabulary        -> token frequencies from PRDECT-ID (CC BY 4.0)

Hand-authored resources (lexicons, negation, intensifiers, aspect vocabulary)
live directly in ``src/absa_library/data`` and are not touched here.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "raw" / "lexicon"
OUT = ROOT / "src" / "absa_library" / "data"

TOKEN_RE = re.compile(r"^[a-z]+$")
SPACED_RE = re.compile(r"^[a-z]+(?: [a-z]+)*$")

NEGATIONS = {
    "tidak", "tak", "bukan", "belum", "jangan", "kurang", "tanpa",
    "gak", "ga", "nggak", "ngga", "enggak", "engga", "tdk", "kagak",
    "ogah", "bukanlah", "takkan",
}

INTENSIFIERS = {
    "sangat", "banget", "amat", "sekali", "paling", "terlalu", "begitu",
    "bener", "benar", "super", "agak", "sedikit", "lumayan", "cukup",
    "hampir", "kebangetan",
}


def _clean_key(value: str) -> str:
    return value.strip().lower()


def _clean_value(value: str) -> str:
    value = re.sub(r"\s+", " ", value.strip().lower())
    return value


def build_slang() -> dict[str, str]:
    """Merge the three sources, preferring the more specific mapping."""
    slang: dict[str, str] = {}

    # 1. kamus-alay: the largest and most curated source.
    with (RAW / "kamus_alay.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = _clean_key(row.get("slang") or "")
            value = _clean_value(row.get("formal") or "")
            if not key or not value or not SPACED_RE.match(key) or not SPACED_RE.match(value):
                continue
            if key == value:
                continue
            slang.setdefault(key, value)

    # 2. NLP_bahasa_resources slang list.
    with (RAW / "slang_combined.json").open("r", encoding="utf-8") as handle:
        extra = json.load(handle)
    for raw_key, raw_value in extra.items():
        key = _clean_key(str(raw_key))
        value = _clean_value(str(raw_value))
        if not key or not value or not SPACED_RE.match(key) or not SPACED_RE.match(value):
            continue
        if key == value:
            continue
        slang.setdefault(key, value)

    # 3. pujangga formalization dictionary.
    with (RAW / "formalization_pujangga.txt").open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            key, value = _clean_key(parts[0]), _clean_value(parts[1])
            if not key or not value or not SPACED_RE.match(key) or not SPACED_RE.match(value):
                continue
            if key == value:
                continue
            slang.setdefault(key, value)

    # Guarantee the mappings used by the public examples and tests.
    mandatory = {
        "bgus": "bagus",
        "bgs": "bagus",
        "bgt": "banget",
        "bngt": "banget",
        "tp": "tapi",
        "tpi": "tapi",
        "gmn": "bagaimana",
        "gimana": "bagaimana",
        "sy": "saya",
        "gak": "tidak",
        "ga": "tidak",
        "nggak": "tidak",
        "enggak": "tidak",
        "tdk": "tidak",
        "udah": "sudah",
        "aja": "saja",
        "dgn": "dengan",
        "yg": "yang",
        "krn": "karena",
        "bkn": "bukan",
        "blm": "belum",
        "jgn": "jangan",
        "lemot": "lambat",
        "eror": "error",
        "mantul": "mantap",
    }
    for key, value in mandatory.items():
        slang[key] = value

    return dict(sorted(slang.items()))


def build_stopwords() -> list[str]:
    words: set[str] = set()
    for name in ("stopwords_combined.txt", "stopwords_iso.txt"):
        with (RAW / name).open("r", encoding="utf-8") as handle:
            for line in handle:
                word = line.strip().lower()
                if word and TOKEN_RE.match(word):
                    words.add(word)
    # Negations and intensifiers must survive stop-word removal.
    words -= NEGATIONS
    words -= INTENSIFIERS
    # Never drop words that carry sentiment or name an aspect.
    protected: set[str] = set()
    for name in ("lexicon_positive.txt", "lexicon_negative.txt", "aspect_vocab.txt"):
        path = OUT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            protected.update(word for word in line.split() if TOKEN_RE.match(word))
    if (OUT / "intensifiers.json").exists():
        protected.update(json.loads((OUT / "intensifiers.json").read_text(encoding="utf-8")))
    words -= protected
    return sorted(words)


def build_vocabulary() -> list[str]:
    import pandas as pd

    from absa_library.preprocessing import reduce_repeated_chars  # type: ignore

    slang = build_slang()
    # A slang key must never be a correction target: the slang stage runs
    # first, so leaving them in the vocabulary would "correct" typos back into
    # slang instead of the formal word.
    slang_keys = {key for key, value in slang.items() if key != value}

    frame = pd.read_csv(ROOT / "Data" / "raw" / "prdect_id.csv", usecols=["Customer Review"])
    counter: Counter[str] = Counter()
    for text in frame["Customer Review"].dropna().astype(str):
        lowered = reduce_repeated_chars(text.lower())
        for match in re.finditer(r"[a-z]+", lowered):
            token = match.group()
            if 3 <= len(token) <= 20:
                counter[token] += 1

    vocab = {
        word for word, count in counter.items() if count >= 2 and word not in slang_keys
    }

    # Add formal words, lexicon entries and domain nouns so that the corrector
    # never lacks a valid target for a common word.
    for value in slang.values():
        vocab.update(value.split())

    for name in ("lexicon_positive.txt", "lexicon_negative.txt", "negation.txt", "aspect_vocab.txt"):
        path = OUT / name
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            vocab.update(word for word in line.split() if TOKEN_RE.match(word))

    if (OUT / "intensifiers.json").exists():
        vocab.update(json.loads((OUT / "intensifiers.json").read_text(encoding="utf-8")))

    vocab = {word for word in vocab if TOKEN_RE.match(word)}
    # Write most frequent words first: the corrector breaks distance ties by rank.
    ranked = [word for word, _ in counter.most_common() if word in vocab]
    extras = sorted(vocab - set(ranked))
    return ranked + extras


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    slang = build_slang()
    (OUT / "slang_dict.json").write_text(
        json.dumps(slang, ensure_ascii=False, indent=0, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    stopwords = build_stopwords()
    (OUT / "stopwords_id.txt").write_text("\n".join(stopwords) + "\n", encoding="utf-8")

    vocab = build_vocabulary()
    (OUT / "vocab_id.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")

    print(f"slang_dict.json      {len(slang):>6} entries")
    print(f"stopwords_id.txt     {len(stopwords):>6} entries")
    print(f"vocab_id.txt         {len(vocab):>6} entries")


if __name__ == "__main__":
    main()
