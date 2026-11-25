"""Tests for the Naive Bayes and SVM engines and the engine registry."""

from __future__ import annotations

import json

import pytest

from absa_library import (
    EngineNotFittedError,
    NaiveBayesEngine,
    SentimentLexicon,
    SVMEngine,
    UnsupportedEngineError,
    available_engines,
    create_engine,
    register_engine,
)
from absa_library.engines import load_engine, normalize_engine_name
from absa_library.engines.base import BaseEngine
from absa_library.engines.sklearn_base import SklearnEngine
from absa_library.exceptions import InvalidInputError
from absa_library.types import AnalysisUnit, Prediction


class TestTraining:
    @pytest.mark.parametrize("engine_fixture", ["nb_engine", "svm_engine"])
    def test_predict_returns_valid_predictions(self, engine_fixture, request) -> None:
        engine = request.getfixturevalue(engine_fixture)
        predictions = engine.predict(["barang bagus", "barang lama", "barang biasa"])
        assert len(predictions) == 3
        for prediction in predictions:
            assert isinstance(prediction, Prediction)
            assert prediction.label in {"Positif", "Negatif", "Netral"}
            assert 0.0 <= prediction.confidence <= 1.0
            assert sum(prediction.scores.values()) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("engine_fixture", ["nb_engine", "svm_engine"])
    def test_learns_simple_dataset(self, engine_fixture, request, ml_dataset) -> None:
        engine = request.getfixturevalue(engine_fixture)
        texts, labels = ml_dataset
        predicted = [prediction.label for prediction in engine.predict(texts)]
        accuracy = sum(a == b for a, b in zip(predicted, labels)) / len(labels)
        assert accuracy > 0.8

    def test_accepts_analysis_units(self, nb_engine: NaiveBayesEngine) -> None:
        unit = AnalysisUnit(text="x", aspect="barang", segment="barang bagus")
        assert nb_engine.predict([unit])[0].label == "Positif"

    def test_train_returns_self(self, ml_dataset) -> None:
        texts, labels = ml_dataset
        engine = NaiveBayesEngine()
        assert engine.train(texts, labels) is engine
        assert engine.is_fitted is True

    def test_length_mismatch(self, ml_dataset) -> None:
        texts, labels = ml_dataset
        with pytest.raises(InvalidInputError):
            NaiveBayesEngine().train(texts, labels[:-1])

    def test_empty_training_data(self) -> None:
        with pytest.raises(InvalidInputError):
            NaiveBayesEngine().train([], [])

    def test_single_class(self) -> None:
        with pytest.raises(InvalidInputError):
            NaiveBayesEngine().train(["bagus", "bagus"], ["Positif", "Positif"])

    def test_invalid_labels(self, ml_dataset) -> None:
        texts, labels = ml_dataset
        with pytest.raises(InvalidInputError):
            NaiveBayesEngine().train(texts, ["entah"] * len(labels))

    def test_unsupported_kwargs(self) -> None:
        with pytest.raises(TypeError):
            NaiveBayesEngine(unknown_option=True)

    def test_invalid_parameters(self) -> None:
        with pytest.raises(InvalidInputError):
            NaiveBayesEngine(alpha=0)
        with pytest.raises(InvalidInputError):
            SVMEngine(C=0)
        with pytest.raises(InvalidInputError):
            SVMEngine(calibration_cv=1)
        with pytest.raises(InvalidInputError):
            NaiveBayesEngine(neutral_threshold=2)


class TestNotFitted:
    def test_raises_without_pretrained(self, tmp_path, monkeypatch) -> None:
        engine = NaiveBayesEngine()
        engine.pretrained_filename = None
        with pytest.raises(EngineNotFittedError):
            engine.predict(["barang bagus"])

    def test_missing_pretrained_file_is_ignored(self, monkeypatch) -> None:
        engine = SVMEngine()
        engine.pretrained_filename = "missing_file.joblib"
        with pytest.raises(EngineNotFittedError):
            engine.predict(["barang bagus"])

    def test_bundled_pretrained_model_loads(self) -> None:
        engine = NaiveBayesEngine()
        predictions = engine.predict(["barang bagus", "pelayanan buruk"])
        assert len(predictions) == 2
        assert engine.is_fitted is True

    def test_empty_input_after_fit(self, nb_engine: NaiveBayesEngine) -> None:
        assert nb_engine.predict([]) == []


class TestPersistence:
    @pytest.mark.parametrize("engine_fixture", ["nb_engine", "svm_engine"])
    def test_save_load_roundtrip(self, engine_fixture, request, tmp_path) -> None:
        engine = request.getfixturevalue(engine_fixture)
        target = tmp_path / "engine"
        engine.save(target)
        meta = json.loads((target / "meta.json").read_text(encoding="utf-8"))
        assert meta["engine"] == engine.name
        clone = type(engine)()
        clone.load(target)
        assert clone.predict(["barang bagus"])[0].label == "Positif"

    def test_state_dict_includes_params(self, nb_engine: NaiveBayesEngine) -> None:
        state = nb_engine.state_dict()
        assert state["params"]["alpha"] == 0.3
        assert state["classes"]

    def test_svm_state_dict(self, svm_engine: SVMEngine) -> None:
        state = svm_engine.state_dict()
        assert state["params"]["C"] == 1.0
        assert state["params"]["calibration_cv"] == 3

    def test_metadata(self, nb_engine: NaiveBayesEngine, tmp_path) -> None:
        target = tmp_path / "meta.json"
        nb_engine.save_metadata(target)
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["engine"] == "naive-bayes"
        assert payload["fitted"] is True

    def test_load_engine_from_directory(self, nb_engine: NaiveBayesEngine, tmp_path) -> None:
        nb_engine.save(tmp_path / "engine")
        restored = load_engine(tmp_path / "engine")
        assert isinstance(restored, NaiveBayesEngine)
        assert restored.predict(["barang bagus"])[0].label == "Positif"

    def test_load_engine_missing_metadata(self, tmp_path) -> None:
        from absa_library.exceptions import ModelLoadError

        with pytest.raises(ModelLoadError):
            load_engine(tmp_path)


class TestRegistry:
    def test_available_engines(self) -> None:
        assert {
            "rule-based",
            "naive-bayes",
            "svm",
            "indobert",
        } <= set(available_engines())

    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("rule_based", "rule-based"),
            ("NB", "naive-bayes"),
            ("naivebayes", "naive-bayes"),
            ("bert", "indobert"),
            (" SVM ", "svm"),
            ("support-vector-machine", "svm"),
        ],
    )
    def test_aliases(self, alias, expected) -> None:
        assert normalize_engine_name(alias) == expected

    def test_unknown_engine(self) -> None:
        with pytest.raises(UnsupportedEngineError) as info:
            normalize_engine_name("jaringan-syaraf")
        assert "rule-based" in str(info.value)

    def test_non_string_engine(self) -> None:
        with pytest.raises(UnsupportedEngineError):
            normalize_engine_name(None)  # type: ignore[arg-type]

    def test_create_engine(self) -> None:
        assert isinstance(create_engine("nb"), NaiveBayesEngine)

    def test_register_custom_engine(self) -> None:
        class DummyEngine(BaseEngine):
            name = "dummy"

            def predict(self, X):
                return [Prediction.neutral() for _ in BaseEngine.as_units(X)]

        register_engine("dummy-engine", DummyEngine, aliases=["dummy"])
        assert isinstance(create_engine("dummy"), DummyEngine)
        assert "dummy-engine" in available_engines()

    def test_register_invalid_class(self) -> None:
        with pytest.raises(TypeError):
            register_engine("bad", object)  # type: ignore[arg-type]

    def test_register_empty_name(self) -> None:
        with pytest.raises(ValueError):
            register_engine("  ", NaiveBayesEngine)


class TestNeutralThreshold:
    def test_threshold_promotes_neutral(self, nb_engine: NaiveBayesEngine, monkeypatch) -> None:
        engine = NaiveBayesEngine(neutral_threshold=0.1)
        engine.vectorizer = nb_engine.vectorizer
        engine.classifier = nb_engine.classifier
        engine.classes_ = nb_engine.classes_
        engine._fitted = True
        prediction = engine.predict(["barang bagus"])[0]
        assert prediction.label in {"Positif", "Negatif", "Netral"}

    def test_row_conversion(self) -> None:
        engine = NaiveBayesEngine()
        engine.classes_ = ["Positif", "Negatif", "Netral"]
        prediction = engine._row_to_prediction([0.2, 0.7, 0.1])
        assert prediction.label == "Negatif"

    def test_row_conversion_neutral(self) -> None:
        engine = NaiveBayesEngine(neutral_threshold=0.6)
        engine.classes_ = ["Positif", "Negatif", "Netral"]
        prediction = engine._row_to_prediction([0.3, 0.1, 0.6])
        assert prediction.label == "Netral"

    def test_two_class_model_never_neutral(self) -> None:
        engine = NaiveBayesEngine()
        engine.classes_ = ["Positif", "Negatif"]
        prediction = engine._row_to_prediction([0.7, 0.3])
        assert prediction.label == "Positif"


class TestSklearnBase:
    def test_build_classifier_is_abstract(self) -> None:
        class Incomplete(SklearnEngine):
            pass

        with pytest.raises(NotImplementedError):
            Incomplete()._build_classifier()

    def test_state_restores_parameters(self) -> None:
        engine = NaiveBayesEngine(max_features=500, min_df=2)
        state = engine.state_dict()
        clone = NaiveBayesEngine()
        clone.load_state_dict(state)
        assert clone.max_features == 500
        assert clone.min_df == 2
        assert clone.is_fitted is False

    def test_state_classes_are_normalised(self) -> None:
        engine = NaiveBayesEngine()
        engine.load_state_dict(
            {"vectorizer": object(), "classifier": object(), "classes": ["positive"]}
        )
        assert engine.classes_ == ["Positif"]

    def test_resource_helper_returns_path(self) -> None:
        from absa_library.resources import data_path

        assert data_path("pretrained/naive_bayes.joblib").suffix == ".joblib"


def test_lexicon_argument_is_unused_by_ml_engines() -> None:
    lexicon = SentimentLexicon(positive={"bagus": 1})
    with pytest.raises(TypeError):
        NaiveBayesEngine(lexicon=lexicon)  # type: ignore[call-arg]
