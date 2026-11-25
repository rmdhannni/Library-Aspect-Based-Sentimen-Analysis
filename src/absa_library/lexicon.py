"""Sentiment lexicon loading and lookup."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from .exceptions import InvalidInputError, ResourceError
from .resources import load_lines


def _parse_weighted_lines(lines: Iterable[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", "\t").split()
        if len(parts) == 1:
            word, raw_weight = parts[0], "1"
        elif len(parts) >= 2:
            word, raw_weight = parts[0], parts[1]
        else:  # pragma: no cover - defensive
            continue
        try:
            weight = float(raw_weight)
        except ValueError:
            continue
        scores[word.lower()] = weight
    return scores


class SentimentLexicon:
    """A mapping from word to a polarity score in ``[-1.0, 1.0]``.

    The packaged lexicons store integer weights from 1 to 5. External lexicons
    such as InSet store signed weights from -5 to +5; both are normalised to the
    ``[-1, 1]`` range automatically.
    """

    def __init__(
        self,
        positive: Mapping[str, float] | None = None,
        negative: Mapping[str, float] | None = None,
        *,
        path: str | Path | None = None,
        scores: Mapping[str, float] | None = None,
    ) -> None:
        self._scores: dict[str, float] = {}
        if scores is not None:
            self._scores.update({str(word).lower(): float(score) for word, score in scores.items()})
        elif path is not None:
            self.load(path)
        else:
            if positive is None:
                positive = _parse_weighted_lines(load_lines("lexicon_positive.txt"))
            if negative is None:
                negative = _parse_weighted_lines(load_lines("lexicon_negative.txt"))
            for word, weight in positive.items():
                self._scores[word.lower()] = self._normalise(weight, default_sign=1.0)
            for word, weight in negative.items():
                self._scores[word.lower()] = self._normalise(weight, default_sign=-1.0)

    @staticmethod
    def _normalise(weight: float, *, default_sign: float | None) -> float:
        """Scale a raw weight into ``[-1, 1]`` using the file's polarity role.

        ``default_sign`` is ``+1`` for the positive file, ``-1`` for the
        negative file and ``None`` when the file already carries its own sign
        (for example InSet's ``-5 … +5`` weights).
        """
        magnitude = abs(float(weight))
        if magnitude == 0:
            return 0.0
        if magnitude > 1:
            magnitude /= 5.0
        magnitude = min(magnitude, 1.0)
        if default_sign is None:
            return magnitude if weight > 0 else -magnitude
        return magnitude if default_sign > 0 else -magnitude

    def load(self, path: str | Path) -> SentimentLexicon:
        """Load an external lexicon (for example InSet's ``positive.tsv``)."""
        path = Path(path)
        if not path.exists():
            raise ResourceError(f"Berkas leksikon tidak ditemukan: {path}")
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        self._scores.update(
            {
                word: self._normalise(weight, default_sign=None)
                for word, weight in _parse_weighted_lines(lines).items()
            }
        )
        return self

    def add(self, word: str, score: float) -> None:
        """Add or replace one word's score."""
        if not word:
            raise InvalidInputError("Kata leksikon tidak boleh kosong.")
        self._scores[word.lower()] = max(-1.0, min(1.0, float(score)))

    def score(self, word: str) -> float:
        """Return the polarity of ``word`` (0.0 when unknown)."""
        return self._scores.get(word.lower(), 0.0)

    def polarity(self, word: str) -> str:
        """Return ``Positif``/``Negatif``/``Netral`` for a single word."""
        score = self.score(word)
        if score > 0:
            return "Positif"
        if score < 0:
            return "Negatif"
        return "Netral"

    def __contains__(self, word: object) -> bool:
        return str(word).lower() in self._scores

    def as_dict(self) -> dict[str, float]:
        """Return a copy of the underlying word → score mapping."""
        return dict(self._scores)

    def __len__(self) -> int:
        return len(self._scores)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"SentimentLexicon(words={len(self._scores)})"


def load_lexicon(path: str | Path) -> SentimentLexicon:
    """Convenience wrapper to load an external lexicon file."""
    return SentimentLexicon(path=path)
