"""Linear SVM engine with probability calibration (FR-ML-03)."""

from __future__ import annotations

from typing import Any

from ..exceptions import InvalidInputError
from .sklearn_base import SklearnEngine


class SVMEngine(SklearnEngine):
    """Support Vector Machine over TF-IDF features.

    ``LinearSVC`` does not expose probabilities, so it is wrapped in
    :class:`~sklearn.calibration.CalibratedClassifierCV` (sigmoid/Platt scaling)
    to obtain calibrated confidence scores.
    """

    name = "svm"
    pretrained_filename = "svm.joblib"

    def __init__(self, *, C: float = 1.0, calibration_cv: int = 3, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if C <= 0:
            raise InvalidInputError("C harus lebih besar dari 0.")
        if calibration_cv < 2:
            raise InvalidInputError("calibration_cv minimal 2.")
        self.C = float(C)
        self.calibration_cv = int(calibration_cv)

    def _build_classifier(self) -> Any:
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        base = LinearSVC(C=self.C, random_state=self.random_state, max_iter=5000)
        return CalibratedClassifierCV(estimator=base, cv=self.calibration_cv, method="sigmoid")

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["params"]["C"] = self.C
        state["params"]["calibration_cv"] = self.calibration_cv
        return state

    def load_state_dict(self, state: dict) -> SVMEngine:
        super().load_state_dict(state)
        params = state.get("params", {})
        self.C = float(params.get("C", self.C))
        self.calibration_cv = int(params.get("calibration_cv", self.calibration_cv))
        return self
