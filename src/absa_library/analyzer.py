"""Facade that ties preprocessing, aspect extraction, engines and evaluation together."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .aspects import AspectExtractor, fallback_mention
from .engines import (
    BaseEngine,
    available_engines,
    create_engine,
    load_engine,
    normalize_engine_name,
)
from .evaluator import Evaluator
from .exceptions import AbsaError, EngineNotFittedError, InvalidInputError
from .logger import Logger
from .preprocessing import PreprocessConfig, TextPreprocessor
from .types import AnalysisUnit, Prediction, normalize_label
from .version import __version__

OUTPUT_COLUMNS = ("text", "aspect", "sentiment", "confidence_score")
MAX_TEXT_LENGTH = 10_000
MAX_BATCH_SIZE = 1_000
CONFIG_FILENAME = "analyzer.json"

_ALPHA_RE = re.compile(r"[A-Za-z]")


class AspectAnalyzer:
    """High-level API for aspect-based sentiment analysis (FR-API-01 … 11).

    Parameters
    ----------
    engine:
        ``"rule-based"`` (default), ``"naive-bayes"``, ``"svm"`` or
        ``"indobert"``. Aliases such as ``"nb"`` or ``"bert"`` are accepted.
    aspects:
        ``"auto"`` (default) for open-domain extraction, a mapping
        ``{label: [keywords]}`` for dictionary mode, or ``None``/``"none"`` to
        disable aspect extraction.
    preprocess:
        ``True`` for the default pipeline, ``False`` to skip it, or a dict of
        :class:`~absa_library.preprocessing.PreprocessConfig` overrides.
    output_format:
        ``"dict"`` (default) or ``"dataframe"``.
    on_error:
        ``"raise"`` (default) or ``"skip"`` for partially invalid batches.
    engine_type:
        Alias of ``engine`` kept for compatibility with the proposal diagram.
    **engine_kwargs:
        Forwarded to the engine constructor.

    Examples
    --------
    >>> from absa_library import AspectAnalyzer
    >>> AspectAnalyzer().analyze("barang bagus tapi kurir lama")[0]["aspect"]
    'barang'
    """

    def __init__(
        self,
        engine: str = "rule-based",
        aspects: str | Mapping[str, Iterable[str]] | None = "auto",
        preprocess: bool | dict | PreprocessConfig = True,
        output_format: str = "dict",
        on_error: str = "raise",
        engine_type: str | None = None,
        **engine_kwargs: Any,
    ) -> None:
        if engine_type is not None:
            if engine != "rule-based" and engine != engine_type:
                raise InvalidInputError(
                    "Parameter 'engine' dan 'engine_type' saling bertentangan."
                )
            engine = engine_type
        self.logger = Logger("absa_library.analyzer")
        self.engine_name = normalize_engine_name(engine)
        self._engine_kwargs = dict(engine_kwargs)
        self.engine: BaseEngine = create_engine(self.engine_name, **engine_kwargs)

        self.preprocess_config = PreprocessConfig.from_value(preprocess)
        self.preprocessor = TextPreprocessor(self.preprocess_config)
        self.aspects = aspects
        self.extractor = AspectExtractor(aspects, preprocessor=self.preprocessor)
        self.evaluator = Evaluator()
        self.output_format = self._validate_output_format(output_format)
        self.on_error = self._validate_on_error(on_error)

    # -- properties --------------------------------------------------------
    @property
    def engine_type(self) -> str:
        """Alias of :attr:`engine_name` for compatibility with the proposal."""
        return self.engine_name

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        mode = self.extractor.mode
        return (
            f"AspectAnalyzer(engine={self.engine_name!r}, aspects={mode!r}, "
            f"output_format={self.output_format!r})"
        )

    # -- analysis ----------------------------------------------------------
    def analyze(self, texts: str | Sequence[str]):
        """Analyse one text or a batch and return one row per detected aspect."""
        items = self._as_text_list(texts)
        rows: list[dict] = []
        for text in items:
            try:
                self._validate_text(text)
                units = self._build_units(text)
                predictions = self.engine.predict(units)
                rows.extend(self._rows(text, units, predictions))
            except AbsaError as exc:
                if self.on_error != "skip":
                    raise
                self.logger.warning("Melewati teks tidak valid: %s", exc)
        return self._format(rows)

    def analyze_aspect(self, text: str, aspect: str | None = None) -> Prediction:
        """Classify the sentiment of one explicit aspect (or the whole sentence)."""
        self._validate_text(text)
        return self._predict_for_aspect(text, aspect)

    # -- training ----------------------------------------------------------
    def fit(self, X: Sequence[str], y: Sequence[object]) -> AspectAnalyzer:
        """Train the active engine on labelled texts (FR-API-08)."""
        if not self.engine.requires_training:
            self.logger.warning(
                "Engine %s tidak memerlukan training; fit() diabaikan.", self.engine_name
            )
            return self
        texts = self._as_text_list(X, validate=True)
        if len(texts) != len(y):
            raise InvalidInputError(
                f"Jumlah teks ({len(texts)}) dan label ({len(y)}) harus sama."
            )
        units: list[AnalysisUnit] = []
        labels: list[str] = []
        for text, expected in zip(texts, y):
            aspect, label = self._parse_expected(expected)
            units.append(self._unit_for(text, aspect))
            labels.append(label)
        self.engine.train(units, labels)
        return self

    train = fit

    # -- evaluation --------------------------------------------------------
    def evaluate(self, x_test: Sequence[str], y_test: Sequence[object]) -> dict:
        """Evaluate the active engine and return accuracy/precision/recall/F1."""
        texts = self._as_text_list(x_test, validate=True)
        if len(texts) != len(y_test):
            raise InvalidInputError(
                f"Jumlah teks uji ({len(texts)}) dan label ({len(y_test)}) harus sama."
            )
        gold: list[str] = []
        predicted: list[str] = []
        with self.evaluator.timer() as box:
            for text, expected in zip(texts, y_test):
                aspect, label = self._parse_expected(expected)
                gold.append(label)
                predicted.append(self._predict_for_aspect(text, aspect).label)
        return self.evaluator.evaluate(
            gold, predicted, elapsed_seconds=box.get("elapsed_seconds")
        )

    def compare_engines(
        self,
        x_test: Sequence[str],
        y_test: Sequence[object],
        engines: Sequence[str] | None = None,
    ) -> list[dict]:
        """Evaluate several engines on the same split (accuracy and time)."""
        names = list(engines) if engines is not None else ["rule-based", "naive-bayes", "svm"]
        results: list[dict] = []
        for name in names:
            canonical = normalize_engine_name(name)
            kwargs = self._engine_kwargs if canonical == self.engine_name else {}
            candidate = AspectAnalyzer(
                engine=canonical,
                aspects=self.aspects,
                preprocess=self.preprocess_config,
                on_error=self.on_error,
                **kwargs,
            )
            try:
                report = candidate.evaluate(x_test, y_test)
            except EngineNotFittedError as exc:
                self.logger.warning("Engine %s dilewati: %s", canonical, exc)
                results.append({"engine": canonical, "error": str(exc)})
                continue
            results.append(
                {
                    "engine": canonical,
                    "accuracy": report["accuracy"],
                    "precision": report["precision"],
                    "recall": report["recall"],
                    "f1_score": report["f1_score"],
                    "elapsed_seconds": report.get("elapsed_seconds", 0.0),
                }
            )
        return results

    # -- runtime configuration ---------------------------------------------
    def set_engine(self, engine: str, **engine_kwargs: Any) -> None:
        """Switch the active engine without rebuilding the facade (FR-API-06)."""
        canonical = normalize_engine_name(engine)
        self.engine = create_engine(canonical, **engine_kwargs)
        self.engine_name = canonical
        self._engine_kwargs = dict(engine_kwargs)

    def set_output_format(self, output_format: str) -> None:
        """Change the output format between ``"dict"`` and ``"dataframe"``."""
        self.output_format = self._validate_output_format(output_format)

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Persist the facade configuration and the fitted engine."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        aspects = self._aspects_spec()
        config = {
            "version": __version__,
            "engine": self.engine_name,
            "aspects": aspects,
            "preprocess": asdict(self.preprocess_config),
            "output_format": self.output_format,
            "on_error": self.on_error,
        }
        (path / CONFIG_FILENAME).write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.engine.save(path / "engine")

    @classmethod
    def load(cls, path: str | Path) -> AspectAnalyzer:
        """Restore a facade saved with :meth:`save`."""
        path = Path(path)
        config_path = path / CONFIG_FILENAME
        if not config_path.exists():
            raise InvalidInputError(f"Konfigurasi analyzer tidak ditemukan di {path}.")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        analyzer = cls(
            engine=config.get("engine", "rule-based"),
            aspects=config.get("aspects", "auto"),
            preprocess=config.get("preprocess", True),
            output_format=config.get("output_format", "dict"),
            on_error=config.get("on_error", "raise"),
        )
        analyzer.engine = load_engine(path / "engine")
        return analyzer

    # -- internals ---------------------------------------------------------
    def _aspects_spec(self):
        if self.extractor.mode == "none":
            return None
        if self.extractor.mode == "auto":
            return "auto"
        return {
            label: [" ".join(phrase) for phrase in phrases]
            for label, phrases in self.extractor.dictionary.items()
        }

    def _build_units(self, text: str) -> list[AnalysisUnit]:
        normalized = self.preprocessor.normalize(text)
        if not normalized:
            raise InvalidInputError(
                f"Teks tidak mengandung kata yang bisa dianalisis: {text[:40]!r}"
            )
        mentions = self.extractor.extract(normalized)
        if not mentions:
            mentions = [fallback_mention(normalized)]
        units: list[AnalysisUnit] = []
        for mention in mentions:
            segment = self.extractor.segment(normalized, mention)
            processed = self.preprocessor.process_normalized(segment) or segment
            units.append(
                AnalysisUnit(
                    text=text,
                    aspect=mention.aspect,
                    segment=processed,
                    start=mention.start,
                    end=mention.end,
                    source=mention.source,
                )
            )
        return units

    def _unit_for(self, text: str, aspect: str | None) -> AnalysisUnit:
        self._validate_text(text)
        normalized = self.preprocessor.normalize(text)
        if not normalized:
            raise InvalidInputError(f"Teks tidak mengandung kata: {text[:40]!r}")
        label = aspect or "UMUM"
        if aspect and aspect.strip().lower() != "umum":
            segment = self.extractor.segment_for_label(normalized, aspect)
        else:
            segment = normalized
        processed = self.preprocessor.process_normalized(segment) or segment
        return AnalysisUnit(text=text, aspect=label, segment=processed)

    def _predict_for_aspect(self, text: str, aspect: str | None) -> Prediction:
        unit = self._unit_for(text, aspect)
        predictions = self.engine.predict([unit])
        if not predictions:  # pragma: no cover - defensive
            raise AbsaError("Engine tidak mengembalikan prediksi.")
        return predictions[0]

    @staticmethod
    def _rows(
        text: str,
        units: Sequence[AnalysisUnit],
        predictions: Sequence[Prediction],
    ) -> list[dict]:
        rows = []
        for unit, prediction in zip(units, predictions):
            rows.append(
                {
                    "text": text,
                    "aspect": unit.aspect,
                    "sentiment": prediction.label,
                    "confidence_score": round(float(prediction.confidence), 4),
                }
            )
        return rows

    def _format(self, rows: list[dict]):
        if self.output_format == "dataframe":
            import pandas as pd

            return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
        return rows

    @staticmethod
    def _as_text_list(texts: str | Sequence[str], *, validate: bool = False) -> list[str]:
        if isinstance(texts, str):
            items: list[str] = [texts]
        elif texts is None:
            raise InvalidInputError("Input teks tidak boleh None.")
        elif isinstance(texts, (list, tuple)):
            items = list(texts)
        else:
            raise InvalidInputError(
                "Input harus berupa str atau list[str], bukan "
                f"{type(texts).__name__}."
            )
        if not items:
            raise InvalidInputError("Daftar teks tidak boleh kosong.")
        if len(items) > MAX_BATCH_SIZE:
            raise InvalidInputError(
                f"Batch maksimal {MAX_BATCH_SIZE} teks, diterima {len(items)}."
            )
        if validate:
            for item in items:
                AspectAnalyzer._validate_text(item)
        return items

    @staticmethod
    def _validate_text(text: object) -> None:
        if not isinstance(text, str):
            raise InvalidInputError(
                f"Teks harus berupa str, bukan {type(text).__name__}."
            )
        if not text.strip():
            raise InvalidInputError("Teks tidak boleh kosong.")
        if len(text) > MAX_TEXT_LENGTH:
            raise InvalidInputError(
                f"Teks terlalu panjang ({len(text)} karakter, maksimal {MAX_TEXT_LENGTH})."
            )
        if not _ALPHA_RE.search(text):
            raise InvalidInputError("Teks harus mengandung minimal satu huruf.")

    @staticmethod
    def _validate_output_format(value: str) -> str:
        key = str(value).strip().lower()
        aliases = {"list": "dict", "dictionary": "dict", "pd": "dataframe", "df": "dataframe"}
        key = aliases.get(key, key)
        if key not in {"dict", "dataframe"}:
            raise InvalidInputError("output_format harus 'dict' atau 'dataframe'.")
        return key

    @staticmethod
    def _validate_on_error(value: str) -> str:
        key = str(value).strip().lower()
        if key not in {"raise", "skip"}:
            raise InvalidInputError("on_error harus 'raise' atau 'skip'.")
        return key

    @staticmethod
    def _parse_expected(expected: object) -> tuple[str | None, str]:
        aspect: str | None = None
        label: object = expected
        if isinstance(expected, Mapping):
            aspect = expected.get("aspect")
            label = expected.get("sentiment", expected.get("label"))
        elif isinstance(expected, (tuple, list)) and len(expected) == 2:
            aspect, label = expected
        try:
            normalized = normalize_label(label)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
        return (str(aspect) if aspect else None), normalized


__all__ = ["AspectAnalyzer", "available_engines"]
