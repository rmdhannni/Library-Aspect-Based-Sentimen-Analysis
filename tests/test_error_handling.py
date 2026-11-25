"""Error hierarchy, safe-state and logging tests (FR-ERR-01 … 08)."""

from __future__ import annotations

import logging

import pytest

from absa_library import (
    AbsaError,
    AspectAnalyzer,
    BaseEngine,
    EngineNotFittedError,
    InvalidInputError,
    ModelLoadError,
    ModelManager,
    ResourceError,
    UnsupportedEngineError,
    register_engine,
)


class TestExceptionHierarchy:
    def test_all_errors_share_base(self) -> None:
        for error in (
            InvalidInputError("x"),
            EngineNotFittedError("x"),
            ModelLoadError("x"),
            ResourceError("x"),
        ):
            assert isinstance(error, AbsaError)

    def test_detail_is_rendered(self) -> None:
        error = AbsaError("gagal", detail="kode 500")
        assert "gagal" in str(error)
        assert "kode 500" in str(error)
        assert error.detail == "kode 500"

    def test_unsupported_engine_lists_choices(self) -> None:
        error = UnsupportedEngineError("x", ["svm", "rule-based"])
        assert "svm" in str(error)
        assert error.engine == "x"
        assert error.valid == ("rule-based", "svm")

    def test_value_error_compatibility(self) -> None:
        assert isinstance(InvalidInputError("x"), ValueError)
        assert isinstance(EngineNotFittedError("x"), RuntimeError)


class BoomEngine(BaseEngine):
    name = "boom"

    def predict(self, X):
        raise AbsaError("engine meledak")


class TestEngineFailure:
    def test_error_propagates_by_default(self) -> None:
        register_engine("boom", BoomEngine)
        analyzer = AspectAnalyzer(engine="boom")
        with pytest.raises(AbsaError):
            analyzer.analyze("barang bagus")

    def test_skip_mode_survives_engine_failure(self, caplog) -> None:
        register_engine("boom", BoomEngine)
        analyzer = AspectAnalyzer(engine="boom", on_error="skip")
        with caplog.at_level(logging.WARNING, logger="absa_library.analyzer"):
            assert analyzer.analyze("barang bagus") == []
        assert caplog.records

    def test_facade_remains_usable_after_engine_failure(self) -> None:
        register_engine("boom", BoomEngine)
        analyzer = AspectAnalyzer(engine="boom", on_error="skip")
        analyzer.analyze("barang bagus")
        analyzer.set_engine("rule-based")
        assert analyzer.analyze("barang bagus")[0]["sentiment"] == "Positif"


class TestModelFailure:
    def test_indobert_failure_is_logged_and_safe(self, caplog) -> None:
        class FailingModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("hub mati")

        manager = ModelManager(backend=_backend(FailingModel))
        with caplog.at_level(logging.ERROR, logger="absa_library.model_manager"):
            with pytest.raises(ModelLoadError):
                manager.load_model()
        assert manager.is_loaded() is False
        assert caplog.records

    def test_analyzer_skip_handles_model_failure(self, caplog) -> None:
        class FailingModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("hub mati")

        manager = ModelManager(backend=_backend(FailingModel))
        analyzer = AspectAnalyzer(
            engine="indobert", on_error="skip", model_manager=manager
        )
        with caplog.at_level(logging.ERROR):
            assert analyzer.analyze("barang bagus") == []


class TestDuplicateSessions:
    def test_repeated_calls_keep_working(self) -> None:
        analyzer = AspectAnalyzer()
        for _ in range(3):
            assert analyzer.analyze("barang bagus")

    def test_skip_keeps_valid_items(self) -> None:
        analyzer = AspectAnalyzer(on_error="skip")
        rows = analyzer.analyze([None, "barang bagus", 12])
        assert len(rows) == 1


def _backend(model_class):
    from types import SimpleNamespace

    class Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("tokenizer tidak boleh dimuat")

    return SimpleNamespace(
        torch=SimpleNamespace(no_grad=lambda: None),
        AutoModelForSequenceClassification=model_class,
        AutoTokenizer=Tokenizer,
    )
