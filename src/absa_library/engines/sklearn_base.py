"""Shared TF-IDF + scikit-learn base for the classical ML engines."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ..exceptions import EngineNotFittedError, InvalidInputError
from ..resources import data_path
from ..types import LABELS, AnalysisUnit, Prediction, argmax, normalize_label
from .base import BaseEngine

PRETRAINED_DIR = "pretrained"


class SklearnEngine(BaseEngine):
    """Base class for engines built from a TF-IDF vectoriser and a classifier.

    Subclasses only need to implement :meth:`_build_classifier`. Both engines
    follow the same contract, which keeps :class:`AspectAnalyzer` agnostic to
    the underlying algorithm (FR-ENG-04).
    """

    requires_training = True
    pretrained_filename: str | None = None

    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        max_features: int | None = 20000,
        min_df: int = 1,
        sublinear_tf: bool = True,
        neutral_threshold: float = 0.45,
        random_state: int = 42,
        logger: Any | None = None,
    ) -> None:
        super().__init__(logger=logger)
        if not 0.0 <= neutral_threshold <= 1.0:
            raise InvalidInputError("neutral_threshold harus berada pada rentang [0, 1].")
        self.ngram_range = (int(ngram_range[0]), int(ngram_range[1]))
        self.max_features = max_features
        self.min_df = int(min_df)
        self.sublinear_tf = bool(sublinear_tf)
        self.neutral_threshold = float(neutral_threshold)
        self.random_state = int(random_state)
        self.vectorizer: Any | None = None
        self.classifier: Any | None = None
        self.classes_: list[str] = []

    # -- training ----------------------------------------------------------
    def train(self, X: Sequence[AnalysisUnit] | Sequence[str], y: Iterable[object]) -> SklearnEngine:
        units = self.as_units(X)
        labels = self.normalise_labels(y)
        if len(units) != len(labels):
            raise InvalidInputError(
                f"Jumlah data ({len(units)}) dan label ({len(labels)}) harus sama."
            )
        if not units:
            raise InvalidInputError("Data training tidak boleh kosong.")
        texts = [unit.segment or unit.text for unit in units]
        if len(set(labels)) < 2:
            raise InvalidInputError("Training membutuhkan minimal dua kelas sentimen.")

        from sklearn.feature_extraction.text import TfidfVectorizer

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        features = vectorizer.fit_transform(texts)
        classifier = self._build_classifier()
        classifier.fit(features, labels)

        self.vectorizer = vectorizer
        self.classifier = classifier
        self.classes_ = [normalize_label(label) for label in classifier.classes_]
        self._fitted = True
        self.logger.info(
            "Engine %s dilatih pada %d contoh dengan %d fitur.",
            self.name,
            len(labels),
            features.shape[1],
        )
        return self

    # -- prediction --------------------------------------------------------
    def predict(self, X: Sequence[AnalysisUnit] | Sequence[str] | str) -> list[Prediction]:
        units = self.as_units(X)
        if not self._fitted:
            self._load_pretrained()
        if not self._fitted or self.vectorizer is None or self.classifier is None:
            raise EngineNotFittedError(
                f"Engine {self.name!r} belum dilatih. Panggil fit()/train() atau muat model "
                "pretrained dengan load()."
            )
        texts = [unit.segment or unit.text for unit in units]
        if not texts:
            return []
        features = self.vectorizer.transform(texts)
        probabilities = self.classifier.predict_proba(features)
        return [self._row_to_prediction(row) for row in probabilities]

    def _row_to_prediction(self, row: Sequence[float]) -> Prediction:
        scores = {
            label: float(probability) for label, probability in zip(self.classes_, row)
        }
        label = self.classes_[argmax(row)]
        confidence = scores[label]
        if label != "Netral" and "Netral" in scores:
            neutral = scores["Netral"]
            if neutral >= self.neutral_threshold:
                label = "Netral"
                confidence = neutral
        return Prediction(label=label, confidence=confidence, scores=scores)

    def _build_classifier(self) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "classes": list(self.classes_),
            "params": {
                "ngram_range": self.ngram_range,
                "max_features": self.max_features,
                "min_df": self.min_df,
                "sublinear_tf": self.sublinear_tf,
                "neutral_threshold": self.neutral_threshold,
                "random_state": self.random_state,
            },
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> SklearnEngine:
        self.vectorizer = state["vectorizer"]
        self.classifier = state["classifier"]
        self.classes_ = [normalize_label(label) for label in state.get("classes", LABELS)]
        params = state.get("params", {})
        self.ngram_range = tuple(params.get("ngram_range", self.ngram_range))
        self.max_features = params.get("max_features", self.max_features)
        self.min_df = int(params.get("min_df", self.min_df))
        self.sublinear_tf = bool(params.get("sublinear_tf", self.sublinear_tf))
        self.neutral_threshold = float(params.get("neutral_threshold", self.neutral_threshold))
        self.random_state = int(params.get("random_state", self.random_state))
        self._fitted = self.vectorizer is not None and self.classifier is not None
        return self

    def save(self, path: str | Path) -> None:
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        super().save(path)
        joblib.dump(self.state_dict(), path / "model.joblib")

    def load(self, path: str | Path) -> SklearnEngine:
        import joblib

        state = joblib.load(Path(path) / "model.joblib")
        return self.load_state_dict(state)

    def _load_pretrained(self) -> None:
        """Load the packaged model for this engine, if one was shipped."""
        if not self.pretrained_filename:
            return
        try:
            path = data_path(f"{PRETRAINED_DIR}/{self.pretrained_filename}")
        except Exception:  # ResourceError - model not shipped with this build
            self.logger.debug("Model pretrained %s tidak tersedia.", self.name)
            return
        import joblib

        state = joblib.load(path)
        self.load_state_dict(state)
        self.logger.info("Model pretrained %s dimuat dari %s.", self.name, path.name)

    # -- metadata ----------------------------------------------------------
    def metadata(self) -> dict:
        """Return a JSON-serialisable description of the engine."""
        return {
            "engine": self.name,
            "fitted": self._fitted,
            "classes": list(self.classes_),
            "params": {
                "ngram_range": list(self.ngram_range),
                "max_features": self.max_features,
                "min_df": self.min_df,
                "neutral_threshold": self.neutral_threshold,
            },
        }

    def save_metadata(self, path: str | Path) -> None:
        """Write :meth:`metadata` next to a saved model."""
        Path(path).write_text(json.dumps(self.metadata(), indent=2), encoding="utf-8")
