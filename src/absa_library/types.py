"""Shared internal types and numeric helpers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

LABELS: tuple[str, str, str] = ("Positif", "Negatif", "Netral")

_LABEL_ALIASES = {
    "positif": "Positif",
    "positive": "Positif",
    "pos": "Positif",
    "1": "Positif",
    "negatif": "Negatif",
    "negative": "Negatif",
    "neg": "Negatif",
    "-1": "Negatif",
    "netral": "Netral",
    "neutral": "Netral",
    "neu": "Netral",
    "0": "Netral",
}


def normalize_label(label: object) -> str:
    """Map a label from any common spelling to the canonical Indonesian label."""
    key = str(label).strip().lower()
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]
    raise ValueError(f"Label sentimen tidak dikenal: {label!r}")


@dataclass(frozen=True)
class AnalysisUnit:
    """A single (text, aspect, segment) tuple ready to be classified."""

    text: str
    aspect: str
    segment: str
    start: int = 0
    end: int = 0
    source: str = "auto"


@dataclass
class Prediction:
    """Uniform engine output (FR-ENG-03)."""

    label: str
    confidence: float
    scores: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.label = normalize_label(self.label)
        self.confidence = float(min(1.0, max(0.0, self.confidence)))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "scores": {k: round(float(v), 6) for k, v in self.scores.items()},
        }

    @classmethod
    def neutral(cls, confidence: float = 0.5) -> Prediction:
        return cls("Netral", confidence, {"Netral": confidence})


def softmax(logits: Sequence[float]) -> list[float]:
    """Numerically stable softmax."""
    if not logits:
        return []
    highest = max(logits)
    exps = [math.exp(value - highest) for value in logits]
    total = sum(exps)
    return [value / total for value in exps]


def argmax(values: Sequence[float]) -> int:
    """Return the index of the largest value (first wins on ties)."""
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index
