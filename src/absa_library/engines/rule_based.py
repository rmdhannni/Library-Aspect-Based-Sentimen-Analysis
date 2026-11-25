"""Rule-based sentiment engine backed by a polarity lexicon (FR-RB-01 … 06).

Scoring works on a token window around the aspect term:

``score += polarity * intensifier * negation * distance_weight``

The distance weight penalises opinion terms that are far from the aspect, so a
mixed clause such as "barang bagus tapi kurir lambat" still assigns the right
polarity to each aspect when clause splitting is not available.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..exceptions import InvalidInputError
from ..lexicon import SentimentLexicon
from ..resources import load_json, load_lines
from ..types import AnalysisUnit, Prediction
from .base import BaseEngine

_TOKEN_RE = re.compile(r"[A-Za-z]+")

DEFAULT_NEUTRAL_MARGIN = 0.3
DEFAULT_TEMPERATURE = 0.25
DEFAULT_NEGATION_FACTOR = 1.2
DEFAULT_DISTANCE_DECAY = 0.25
DEFAULT_NEGATION_WINDOW = 3
DEFAULT_INTENSIFIER_WINDOW = 2


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 60.0)))
    exponent = math.exp(max(value, -60.0))
    return exponent / (1.0 + exponent)



class RuleBasedEngine(BaseEngine):
    """Lexicon + negation + intensifier engine that needs no training.

    Parameters
    ----------
    lexicon:
        Polarity lexicon. The packaged Indonesian lexicon is used when omitted.
    intensifiers:
        Mapping of token to multiplier (values above 1 strengthen, below 1
        weaken the opinion).
    negation_words:
        Tokens that flip the polarity of the following opinion term.
    neutral_margin:
        Absolute score below which the result is classified as ``Netral``.
    temperature:
        Softmax temperature used to turn the score into probabilities. Lower
        values produce sharper confidence scores.
    negation_factor:
        Extra multiplier applied when an opinion is negated.
    distance_decay:
        How strongly the distance between aspect and opinion reduces a score.
    """

    name = "rule-based"

    def __init__(
        self,
        *,
        lexicon: SentimentLexicon | None = None,
        intensifiers: Mapping[str, float] | None = None,
        negation_words: Sequence[str] | None = None,
        neutral_margin: float = DEFAULT_NEUTRAL_MARGIN,
        temperature: float = DEFAULT_TEMPERATURE,
        negation_factor: float = DEFAULT_NEGATION_FACTOR,
        distance_decay: float = DEFAULT_DISTANCE_DECAY,
        negation_window: int = DEFAULT_NEGATION_WINDOW,
        intensifier_window: int = DEFAULT_INTENSIFIER_WINDOW,
        use_stanza: bool = False,
        **_: Any,
    ) -> None:
        super().__init__()
        self.lexicon = lexicon or SentimentLexicon()
        self.intensifiers = (
            dict(intensifiers)
            if intensifiers is not None
            else {word.lower(): float(value) for word, value in load_json("intensifiers.json").items()}
        )
        self.negation_words = (
            frozenset(word.lower() for word in negation_words)
            if negation_words is not None
            else frozenset(load_lines("negation.txt"))
        )
        if not 0.0 <= neutral_margin < 1.0:
            raise InvalidInputError("neutral_margin harus berada pada rentang [0, 1).")
        if temperature <= 0:
            raise InvalidInputError("temperature harus lebih besar dari 0.")
        self.neutral_margin = float(neutral_margin)
        self.temperature = float(temperature)
        self.negation_factor = float(negation_factor)
        self.distance_decay = max(0.0, float(distance_decay))
        self.negation_window = max(1, int(negation_window))
        self.intensifier_window = max(1, int(intensifier_window))
        self.use_stanza = bool(use_stanza)
        self._stanza_available: bool | None = None

    # -- BaseEngine --------------------------------------------------------
    def predict(self, X: Sequence[AnalysisUnit] | Sequence[str] | str) -> list[Prediction]:
        units = self.as_units(X)
        predictions: list[Prediction] = []
        for unit in units:
            text = unit.segment or unit.text
            score = self.score(text, unit.aspect)
            predictions.append(self._to_prediction(score))
        return predictions

    # -- scoring -----------------------------------------------------------
    def score(self, text: str, aspect: str = "UMUM") -> float:
        """Return the raw polarity score of ``text`` for ``aspect``."""
        if not isinstance(text, str):
            raise InvalidInputError("Teks rule-based harus berupa str.")
        tokens = [token.lower() for token in _TOKEN_RE.findall(text)]
        if not tokens:
            return 0.0
        aspect_positions = self._aspect_positions(tokens, aspect)
        total = 0.0
        for index, token in enumerate(tokens):
            polarity = self.lexicon.score(token)
            if polarity == 0.0:
                continue
            value = polarity
            value *= self._intensifier_multiplier(tokens, index)
            if self._is_negated(tokens, index):
                value = -value * self.negation_factor
            value *= self._distance_weight(index, aspect_positions)
            total += value
        return total

    def probabilities(self, text: str, aspect: str = "UMUM") -> dict[str, float]:
        """Return the three-class probability distribution for ``text``."""
        return self.probabilities_from_score(self.score(text, aspect))

    def _to_prediction(self, score: float) -> Prediction:
        scores = self.probabilities_from_score(score)
        label = self._label_for_score(score)
        return Prediction(label=label, confidence=scores[label], scores=scores)

    def _label_for_score(self, score: float) -> str:
        """Return the class implied by the raw score and the neutral margin."""
        if score >= self.neutral_margin:
            return "Positif"
        if score <= -self.neutral_margin:
            return "Negatif"
        return "Netral"

    def probabilities_from_score(self, score: float) -> dict[str, float]:
        """Convert a raw score into a proper three-class distribution.

        Positive and negative evidence are mapped with a shifted logistic
        curve; the remaining mass belongs to the neutral class. The result is
        always a valid distribution in ``[0, 1]`` that sums to one.
        """
        slope = max(self.temperature, 1e-6)
        positive = _sigmoid((score - self.neutral_margin) / slope)
        negative = _sigmoid((-score - self.neutral_margin) / slope)
        neutral = max(0.0, 1.0 - positive - negative)
        total = positive + negative + neutral
        if total <= 0:  # pragma: no cover - defensive
            return {"Positif": 0.0, "Negatif": 0.0, "Netral": 1.0}
        return {
            "Positif": positive / total,
            "Negatif": negative / total,
            "Netral": neutral / total,
        }

    # -- rule helpers ------------------------------------------------------
    def _intensifier_multiplier(self, tokens: Sequence[str], index: int) -> float:
        multiplier = 1.0
        start = max(0, index - self.intensifier_window)
        end = min(len(tokens), index + self.intensifier_window + 1)
        for position in range(start, end):
            if position == index:
                continue
            value = self.intensifiers.get(tokens[position])
            if value is not None:
                multiplier *= float(value)
        return multiplier

    def _is_negated(self, tokens: Sequence[str], index: int) -> bool:
        start = max(0, index - self.negation_window)
        negations = sum(1 for token in tokens[start:index] if token in self.negation_words)
        return negations % 2 == 1

    def _aspect_positions(self, tokens: Sequence[str], aspect: str) -> list[int]:
        key = (aspect or "").strip().lower()
        if not key or key == "umum":
            return []
        parts = key.split()
        positions: list[int] = []
        for index in range(0, len(tokens) - len(parts) + 1):
            if tokens[index : index + len(parts)] == parts:
                positions.append(index)
        if positions:
            return positions
        # The aspect label may be a stem ("kurir" for "kurirnya").
        for index, token in enumerate(tokens):
            stripped = token[:-3] if token.endswith("nya") and len(token) > 5 else token
            if token == key or stripped == key:
                positions.append(index)
        return positions

    def _distance_weight(self, index: int, aspects: Sequence[int]) -> float:
        if not aspects or self.distance_decay == 0.0:
            return 1.0
        distance = min(abs(index - position) for position in aspects)
        return 1.0 / (1.0 + self.distance_decay * distance)

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "lexicon_scores": dict(self.lexicon.as_dict()),
            "intensifiers": dict(self.intensifiers),
            "negation_words": sorted(self.negation_words),
            "neutral_margin": self.neutral_margin,
            "temperature": self.temperature,
            "negation_factor": self.negation_factor,
            "distance_decay": self.distance_decay,
            "negation_window": self.negation_window,
            "intensifier_window": self.intensifier_window,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> RuleBasedEngine:
        self.lexicon = SentimentLexicon(scores=state["lexicon_scores"])
        self.intensifiers = {str(k): float(v) for k, v in state.get("intensifiers", {}).items()}
        self.negation_words = frozenset(state.get("negation_words", self.negation_words))
        self.neutral_margin = float(state.get("neutral_margin", self.neutral_margin))
        self.temperature = float(state.get("temperature", self.temperature))
        self.negation_factor = float(state.get("negation_factor", self.negation_factor))
        self.distance_decay = float(state.get("distance_decay", self.distance_decay))
        self.negation_window = int(state.get("negation_window", self.negation_window))
        self.intensifier_window = int(state.get("intensifier_window", self.intensifier_window))
        self._fitted = True
        return self

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        super().save(path)
        joblib.dump(self.state_dict(), path / "model.joblib")

    def load(self, path: str | Path) -> RuleBasedEngine:
        import joblib

        state = joblib.load(Path(path) / "model.joblib")
        return self.load_state_dict(state)
