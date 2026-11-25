"""Strategy interface shared by every classification engine."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

from ..exceptions import EngineNotFittedError, InvalidInputError
from ..logger import Logger
from ..types import AnalysisUnit, Prediction, normalize_label


class BaseEngine(ABC):
    """Abstract base class for sentiment engines (FR-ENG-01).

    Every concrete engine implements :meth:`predict` and returns a list of
    :class:`~absa_library.types.Prediction`, one per input unit, so that the
    facade can treat all algorithms uniformly (FR-ENG-02, FR-ENG-03).

    Parameters
    ----------
    logger:
        Optional :class:`~absa_library.logger.Logger`. A namespaced logger is
        created when omitted.
    """

    #: Canonical registry name of the engine.
    name: str = "base"
    #: Whether :meth:`train` must be called before :meth:`predict`.
    requires_training: bool = False

    def __init__(self, logger: Logger | None = None) -> None:
        self.logger = logger or Logger(f"absa_library.engines.{self.name}")
        self._fitted = False

    # -- public API --------------------------------------------------------
    @property
    def is_fitted(self) -> bool:
        """Return ``True`` when the engine is ready to predict."""
        return self._fitted

    @abstractmethod
    def predict(self, X: Sequence[AnalysisUnit] | Sequence[str] | str) -> list[Prediction]:
        """Classify every unit and return one prediction per unit."""

    def train(self, X: Sequence[AnalysisUnit] | Sequence[str], y: Iterable[object]) -> BaseEngine:
        """Fit the engine on labelled data and return ``self``."""
        raise NotImplementedError(f"Engine {self.name!r} tidak mendukung training.")

    def save(self, path: str | Path) -> None:
        """Persist the engine under ``path`` (a directory)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "meta.json").write_text(
            json.dumps({"engine": self.name}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> BaseEngine:
        """Restore a previously saved engine in place."""
        return self

    def state_dict(self) -> dict:
        """Return the picklable state of the engine."""
        return {}

    def load_state_dict(self, state: dict) -> BaseEngine:
        """Restore the engine from :meth:`state_dict` output."""
        return self

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r}, fitted={self._fitted})"

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def as_units(X: Sequence[AnalysisUnit] | Sequence[str] | str) -> list[AnalysisUnit]:
        """Normalise ``str`` / ``list[str]`` / units input into a list of units."""
        if isinstance(X, str):
            return [AnalysisUnit(text=X, aspect="UMUM", segment=X)]
        if isinstance(X, AnalysisUnit):
            return [X]
        if X is None:
            raise InvalidInputError("Input engine tidak boleh None.")
        units: list[AnalysisUnit] = []
        for item in X:
            if isinstance(item, AnalysisUnit):
                units.append(item)
            elif isinstance(item, str):
                units.append(AnalysisUnit(text=item, aspect="UMUM", segment=item))
            else:
                raise InvalidInputError(
                    "Setiap elemen input engine harus str atau AnalysisUnit."
                )
        return units

    @staticmethod
    def normalise_labels(y: Iterable[object]) -> list[str]:
        """Convert arbitrary label spellings to canonical Indonesian labels."""
        labels: list[str] = []
        for value in y:
            try:
                labels.append(normalize_label(value))
            except ValueError as exc:
                raise InvalidInputError(str(exc)) from exc
        return labels

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise EngineNotFittedError(
                f"Engine {self.name!r} belum dilatih. Panggil train()/fit() terlebih dahulu."
            )
