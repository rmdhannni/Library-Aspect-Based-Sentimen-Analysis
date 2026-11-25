"""Complement Naive Bayes engine (FR-ML-02)."""

from __future__ import annotations

from typing import Any

from .sklearn_base import SklearnEngine


class NaiveBayesEngine(SklearnEngine):
    """Naive Bayes classifier over TF-IDF features.

    ``ComplementNB`` is preferred over ``MultinomialNB`` because it is more
    robust on the imbalanced sentiment distributions typical of review data.
    """

    name = "naive-bayes"
    pretrained_filename = "naive_bayes.joblib"

    def __init__(self, *, alpha: float = 0.3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if alpha <= 0:
            from ..exceptions import InvalidInputError

            raise InvalidInputError("alpha harus lebih besar dari 0.")
        self.alpha = float(alpha)

    def _build_classifier(self) -> Any:
        from sklearn.naive_bayes import ComplementNB

        return ComplementNB(alpha=self.alpha)

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["params"]["alpha"] = self.alpha
        return state

    def load_state_dict(self, state: dict) -> NaiveBayesEngine:
        super().load_state_dict(state)
        self.alpha = float(state.get("params", {}).get("alpha", self.alpha))
        return self
