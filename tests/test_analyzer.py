"""Tests for the AspectAnalyzer facade (FR-API-01 … 11)."""

from __future__ import annotations

import logging

import pytest

from absa_library import (
    AspectAnalyzer,
    BaseEngine,
    EngineNotFittedError,
    InvalidInputError,
    RuleBasedEngine,
    SVMEngine,
    __version__,
    available_engines,
    register_engine,
)
from absa_library.types import AnalysisUnit, Prediction

TEXT = "barang bagus tapi kurir lama"


class TestConstruction:
    def test_defaults(self) -> None:
        analyzer = AspectAnalyzer()
        assert isinstance(analyzer.engine, RuleBasedEngine)
        assert analyzer.engine_name == "rule-based"
        assert analyzer.output_format == "dict"
        assert analyzer.on_error == "raise"

    def test_engine_alias(self) -> None:
        analyzer = AspectAnalyzer(engine="nb")
        assert analyzer.engine_name == "naive-bayes"

    def test_engine_type_alias(self) -> None:
        analyzer = AspectAnalyzer(engine_type="svm")
        assert analyzer.engine_name == "svm"
        assert analyzer.engine_type == "svm"

    def test_conflicting_engine_arguments(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer(engine="svm", engine_type="rule-based")

    def test_engine_kwargs_are_forwarded(self) -> None:
        analyzer = AspectAnalyzer(engine="rule-based", neutral_margin=0.5)
        assert analyzer.engine.neutral_margin == 0.5

    def test_invalid_output_format(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer(output_format="excel")

    def test_invalid_on_error(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer(on_error="ignore")

    def test_repr(self) -> None:
        assert "AspectAnalyzer" in repr(AspectAnalyzer())

    def test_version_is_exposed(self) -> None:
        assert __version__ == "0.1.0"

    def test_available_engines(self) -> None:
        assert "indobert" in available_engines()


class TestAnalyze:
    def test_string_input(self) -> None:
        rows = AspectAnalyzer().analyze(TEXT)
        assert isinstance(rows, list)
        assert {row["aspect"] for row in rows} == {"barang", "kurir"}
        assert set(rows[0]) == {"text", "aspect", "sentiment", "confidence_score"}

    def test_list_input(self) -> None:
        rows = AspectAnalyzer().analyze(["barang bagus", "kurir lama"])
        assert len(rows) == 2

    def test_acceptance_example(self) -> None:
        rows = AspectAnalyzer().analyze("bgus bgt tp lamaaa!!")
        assert len(rows) == 1
        assert rows[0]["aspect"] == "UMUM"
        assert rows[0]["sentiment"] == "Positif"

    def test_mixed_sentiment_is_independent(self) -> None:
        rows = {row["aspect"]: row["sentiment"] for row in AspectAnalyzer().analyze(TEXT)}
        assert rows["barang"] == "Positif"
        assert rows["kurir"] == "Negatif"

    def test_dataframe_output(self) -> None:
        analyzer = AspectAnalyzer(output_format="dataframe")
        frame = analyzer.analyze(TEXT)
        assert list(frame.columns) == [
            "text",
            "aspect",
            "sentiment",
            "confidence_score",
        ]
        assert len(frame) == 2

    def test_dataframe_alias(self) -> None:
        assert AspectAnalyzer(output_format="df").output_format == "dataframe"

    def test_empty_result_dataframe(self) -> None:
        analyzer = AspectAnalyzer(output_format="dataframe", aspects=None)
        frame = analyzer.analyze("wow banget")
        assert list(frame.columns) == [
            "text",
            "aspect",
            "sentiment",
            "confidence_score",
        ]

    def test_output_format_switch(self) -> None:
        analyzer = AspectAnalyzer()
        analyzer.set_output_format("dataframe")
        assert analyzer.output_format == "dataframe"
        with pytest.raises(InvalidInputError):
            analyzer.set_output_format("xml")

    def test_analyze_aspect(self) -> None:
        analyzer = AspectAnalyzer()
        prediction = analyzer.analyze_aspect(TEXT, "kurir")
        assert prediction.label == "Negatif"
        assert analyzer.analyze_aspect("barang bagus").label == "Positif"

    def test_aspects_none(self) -> None:
        rows = AspectAnalyzer(aspects=None).analyze("barang bagus")
        assert rows[0]["aspect"] == "UMUM"

    def test_aspects_dictionary(self) -> None:
        analyzer = AspectAnalyzer(
            aspects={"Produk": ["barang"], "Layanan": ["kurir"]}
        )
        rows = analyzer.analyze(TEXT)
        assert {row["aspect"] for row in rows} == {"Produk", "Layanan"}

    def test_custom_preprocess_config(self) -> None:
        analyzer = AspectAnalyzer(preprocess={"slang": False})
        assert analyzer.preprocess_config.slang is False

    def test_preprocess_disabled(self) -> None:
        analyzer = AspectAnalyzer(preprocess=False)
        assert analyzer.preprocess_config.slang is False
        assert analyzer.analyze("barang bagus")[0]["sentiment"] == "Positif"


class TestValidation:
    @pytest.mark.parametrize("bad", [None, 123, 4.5, {"text": "x"}])
    def test_wrong_type(self, bad) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze(bad)

    @pytest.mark.parametrize("bad", ["", "   ", "!!!", "12345"])
    def test_invalid_text(self, bad) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze(bad)

    def test_empty_batch(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze([])

    def test_batch_with_non_string(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze(["barang bagus", 42])

    def test_text_too_long(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze("a" * 10_001)

    def test_batch_too_large(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze(["barang bagus"] * 1_001)

    def test_skip_mode(self, caplog) -> None:
        analyzer = AspectAnalyzer(on_error="skip")
        with caplog.at_level(logging.WARNING, logger="absa_library.analyzer"):
            rows = analyzer.analyze(["", "barang bagus"])
        assert len(rows) == 1
        assert caplog.records

    def test_analyzer_stays_usable_after_error(self) -> None:
        analyzer = AspectAnalyzer()
        with pytest.raises(InvalidInputError):
            analyzer.analyze("")
        assert analyzer.analyze("barang bagus")[0]["sentiment"] == "Positif"

    def test_analyze_aspect_validates(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().analyze_aspect("", "barang")


class TestEngineSwitching:
    def test_set_engine(self) -> None:
        analyzer = AspectAnalyzer()
        analyzer.set_engine("svm")
        assert analyzer.engine_name == "svm"
        assert isinstance(analyzer.engine, SVMEngine)

    def test_set_engine_unknown(self) -> None:
        from absa_library import UnsupportedEngineError

        with pytest.raises(UnsupportedEngineError):
            AspectAnalyzer().set_engine("transformer-ajaib")

    def test_set_engine_keeps_facade(self) -> None:
        analyzer = AspectAnalyzer()
        analyzer.set_engine("rule-based", neutral_margin=0.2)
        assert analyzer.engine.neutral_margin == 0.2
        assert analyzer.analyze("barang bagus")[0]["sentiment"] == "Positif"


class TestFitAndEvaluate:
    def test_fit_rule_based_is_noop(self, caplog) -> None:
        analyzer = AspectAnalyzer()
        with caplog.at_level(logging.WARNING, logger="absa_library.analyzer"):
            assert analyzer.fit(["barang bagus"], ["Positif"]) is analyzer
        assert caplog.records

    def test_fit_naive_bayes(self, ml_dataset) -> None:
        texts, labels = ml_dataset
        analyzer = AspectAnalyzer(engine="naive-bayes")
        assert analyzer.fit(texts, labels) is analyzer
        assert analyzer.engine.is_fitted is True
        assert analyzer.analyze("barang bagus")[0]["sentiment"] in {
            "Positif",
            "Negatif",
            "Netral",
        }

    def test_fit_with_aspect_pairs(self, ml_dataset) -> None:
        texts, labels = ml_dataset
        analyzer = AspectAnalyzer(engine="naive-bayes")
        pairs = [("barang", label) for label in labels]
        analyzer.fit(texts, pairs)
        assert analyzer.engine.is_fitted is True

    def test_fit_length_mismatch(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer(engine="naive-bayes").fit(["barang bagus"], [])

    def test_fit_invalid_label(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer(engine="naive-bayes").fit(
                ["barang bagus", "barang lama"], ["Positif", "Entah"]
            )

    def test_train_alias(self, ml_dataset) -> None:
        texts, labels = ml_dataset
        analyzer = AspectAnalyzer(engine="naive-bayes")
        assert analyzer.train(texts, labels) is analyzer

    def test_evaluate_plain_labels(self) -> None:
        analyzer = AspectAnalyzer()
        report = analyzer.evaluate(
            ["barang bagus", "pelayanan jelek", "aplikasi biasa"],
            ["Positif", "Negatif", "Netral"],
        )
        assert report["accuracy"] == 1.0
        assert "elapsed_seconds" in report

    def test_evaluate_aspect_pairs(self) -> None:
        analyzer = AspectAnalyzer()
        report = analyzer.evaluate(
            [TEXT, TEXT],
            [("barang", "Positif"), ("kurir", "Negatif")],
        )
        assert report["samples"] == 2
        assert report["accuracy"] == 1.0

    def test_evaluate_dict_labels(self) -> None:
        analyzer = AspectAnalyzer()
        report = analyzer.evaluate(
            [TEXT, TEXT],
            [
                {"aspect": "barang", "sentiment": "Positif"},
                {"aspect": "kurir", "sentiment": "Negatif"},
            ],
        )
        assert report["accuracy"] == 1.0

    def test_evaluate_length_mismatch(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().evaluate(["barang bagus"], [])

    def test_evaluate_invalid_label(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer().evaluate(["barang bagus"], ["Entah"])

    def test_compare_engines(self) -> None:
        texts = ["barang bagus", "kurir lama", "harga murah", "aplikasi error"]
        labels = ["Positif", "Negatif", "Positif", "Negatif"]
        results = AspectAnalyzer().compare_engines(texts, labels)
        assert [row["engine"] for row in results] == [
            "rule-based",
            "naive-bayes",
            "svm",
        ]
        for row in results:
            assert 0.0 <= row["accuracy"] <= 1.0
            assert "elapsed_seconds" in row

    def test_compare_engines_selection(self) -> None:
        results = AspectAnalyzer().compare_engines(
            ["barang bagus"], ["Positif"], engines=["rule-based"]
        )
        assert len(results) == 1

    def test_compare_engines_skips_unfitted(self) -> None:
        class UnfittedEngine(BaseEngine):
            name = "unfitted"
            requires_training = True

            def predict(self, X):
                raise EngineNotFittedError("belum dilatih")

        register_engine("unfitted", UnfittedEngine)
        results = AspectAnalyzer().compare_engines(
            ["barang bagus"], ["Positif"], engines=["unfitted"]
        )
        assert "error" in results[0]


class TestPersistence:
    def test_rule_based_roundtrip(self, tmp_path) -> None:
        analyzer = AspectAnalyzer()
        analyzer.save(tmp_path / "model")
        restored = AspectAnalyzer.load(tmp_path / "model")
        assert restored.engine_name == "rule-based"
        assert restored.analyze(TEXT) == analyzer.analyze(TEXT)

    def test_dictionary_aspects_roundtrip(self, tmp_path) -> None:
        analyzer = AspectAnalyzer(aspects={"Produk": ["barang"]})
        analyzer.save(tmp_path / "model")
        restored = AspectAnalyzer.load(tmp_path / "model")
        assert restored.analyze(TEXT)[0]["aspect"] == "Produk"

    def test_none_aspects_roundtrip(self, tmp_path) -> None:
        analyzer = AspectAnalyzer(aspects=None)
        analyzer.save(tmp_path / "model")
        restored = AspectAnalyzer.load(tmp_path / "model")
        assert restored.aspects is None

    def test_ml_roundtrip(self, tmp_path, ml_dataset) -> None:
        texts, labels = ml_dataset
        analyzer = AspectAnalyzer(engine="naive-bayes")
        analyzer.fit(texts, labels)
        analyzer.save(tmp_path / "model")
        restored = AspectAnalyzer.load(tmp_path / "model")
        assert restored.engine_name == "naive-bayes"
        assert restored.engine.is_fitted is True

    def test_load_missing(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            AspectAnalyzer.load(tmp_path / "nope")

    def test_save_preserves_output_format(self, tmp_path) -> None:
        analyzer = AspectAnalyzer(output_format="dataframe")
        analyzer.save(tmp_path / "model")
        restored = AspectAnalyzer.load(tmp_path / "model")
        assert restored.output_format == "dataframe"


class TestPredictionObject:
    def test_prediction_defaults(self) -> None:
        prediction = Prediction.neutral()
        assert prediction.label == "Netral"
        assert prediction.to_dict()["label"] == "Netral"

    def test_analysis_unit_defaults(self) -> None:
        unit = AnalysisUnit(text="a", aspect="b", segment="c")
        assert unit.start == 0
        assert unit.source == "auto"
