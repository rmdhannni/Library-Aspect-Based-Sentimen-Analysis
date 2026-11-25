"""Tests for the rule-based engine (FR-RB-01 … 06)."""

from __future__ import annotations

import pytest

from absa_library import RuleBasedEngine, SentimentLexicon
from absa_library.exceptions import InvalidInputError
from absa_library.types import AnalysisUnit


class TestPrediction:
    def test_positive(self, rule_engine: RuleBasedEngine) -> None:
        prediction = rule_engine.predict("barang bagus")[0]
        assert prediction.label == "Positif"
        assert 0.0 <= prediction.confidence <= 1.0
        assert set(prediction.scores) == {"Positif", "Negatif", "Netral"}

    def test_negative(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.predict("barang lama")[0].label == "Negatif"

    def test_neutral(self, rule_engine: RuleBasedEngine) -> None:
        prediction = rule_engine.predict("barang biasa")[0]
        assert prediction.label == "Netral"

    def test_scores_sum_to_one(self, rule_engine: RuleBasedEngine) -> None:
        scores = rule_engine.predict("barang bagus")[0].scores
        assert sum(scores.values()) == pytest.approx(1.0)

    def test_string_and_list_inputs(self, rule_engine: RuleBasedEngine) -> None:
        assert len(rule_engine.predict("barang bagus")) == 1
        assert len(rule_engine.predict(["barang bagus", "barang lama"])) == 2
        unit = AnalysisUnit(text="x", aspect="barang", segment="barang bagus")
        assert len(rule_engine.predict([unit])) == 1

    def test_invalid_input(self, rule_engine: RuleBasedEngine) -> None:
        with pytest.raises(InvalidInputError):
            rule_engine.predict(None)  # type: ignore[arg-type]
        with pytest.raises(InvalidInputError):
            rule_engine.predict([1])  # type: ignore[list-item]
        with pytest.raises(InvalidInputError):
            rule_engine.score(None)  # type: ignore[arg-type]

    def test_empty_text(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.predict("")[0].label == "Netral"


class TestRules:
    def test_negation_flips(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.predict("tidak bagus")[0].label == "Negatif"
        assert rule_engine.score("tidak bagus") < 0

    def test_double_negation_flips_back(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.score("bukan tidak bagus") > 0

    def test_negation_window_limits_scope(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.score("tidak satu dua tiga empat bagus") > 0

    def test_intensifier_strengthens(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.score("sangat bagus") > rule_engine.score("bagus")

    def test_diminisher_weakens(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.score("agak bagus") < rule_engine.score("bagus")

    def test_distance_reduces_score(self, rule_engine: RuleBasedEngine) -> None:
        close = rule_engine.score("barang bagus", "barang")
        far = rule_engine.score("barang satu dua tiga bagus", "barang")
        assert far < close

    def test_aspect_stemming(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.score("kurirnya lama", "kurir") < 0
        assert rule_engine.score("kurirnya lama", "barang") < 0

    def test_unknown_aspect_uses_no_distance(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.score("bagus", "UMUM") == pytest.approx(0.8)

    def test_distance_decay_zero(self) -> None:
        engine = RuleBasedEngine(
            lexicon=SentimentLexicon(positive={"bagus": 1}),
            distance_decay=0.0,
            negation_words=set(),
            intensifiers={},
        )
        assert engine.score("barang satu dua bagus", "barang") == 1.0

    def test_threshold_boundary(self) -> None:
        engine = RuleBasedEngine(
            lexicon=SentimentLexicon(positive={"pas": 0.3}),
            neutral_margin=0.3,
            negation_words=set(),
            intensifiers={},
        )
        assert engine.predict("pas")[0].label == "Positif"
        assert engine.probabilities_from_score(0.0)["Netral"] > 0.5

    def test_negative_score_uses_stable_sigmoid(self, rule_engine: RuleBasedEngine) -> None:
        probabilities = rule_engine.probabilities_from_score(-5.0)
        assert probabilities["Negatif"] == pytest.approx(1.0, abs=1e-4)

    def test_probabilities_method(self, rule_engine: RuleBasedEngine) -> None:
        assert sum(rule_engine.probabilities("bagus").values()) == pytest.approx(1.0)


class TestConfiguration:
    def test_invalid_margin(self) -> None:
        with pytest.raises(InvalidInputError):
            RuleBasedEngine(neutral_margin=1.0)
        with pytest.raises(InvalidInputError):
            RuleBasedEngine(neutral_margin=-0.1)

    def test_invalid_temperature(self) -> None:
        with pytest.raises(InvalidInputError):
            RuleBasedEngine(temperature=0)

    def test_unknown_kwargs_are_ignored(self) -> None:
        engine = RuleBasedEngine(verbose=True, something=1)
        assert engine.name == "rule-based"

    def test_is_fitted_property(self, rule_engine: RuleBasedEngine) -> None:
        assert rule_engine.is_fitted is False


class TestPersistence:
    def test_state_roundtrip(self, rule_engine: RuleBasedEngine) -> None:
        state = rule_engine.state_dict()
        clone = RuleBasedEngine(lexicon=SentimentLexicon(positive={"x": 1}))
        clone.load_state_dict(state)
        assert clone.score("bagus") == rule_engine.score("bagus")
        assert clone.negation_words == rule_engine.negation_words

    def test_save_and_load(self, rule_engine: RuleBasedEngine, tmp_path) -> None:
        target = tmp_path / "engine"
        rule_engine.save(target)
        assert (target / "meta.json").exists()
        clone = RuleBasedEngine()
        clone.load(target)
        assert clone.predict("barang bagus")[0].label == "Positif"

    def test_state_contains_configuration(self, rule_engine: RuleBasedEngine) -> None:
        state = rule_engine.state_dict()
        assert state["negation_factor"] == 1.2
        assert state["negation_window"] == 3
        assert state["intensifier_window"] == 2


class TestBaseHelpers:
    def test_as_units(self) -> None:
        from absa_library.engines.base import BaseEngine

        assert BaseEngine.as_units("halo")[0].aspect == "UMUM"
        unit = AnalysisUnit(text="a", aspect="b", segment="c")
        assert BaseEngine.as_units(unit) == [unit]
        with pytest.raises(InvalidInputError):
            BaseEngine.as_units(None)

    def test_normalise_labels(self) -> None:
        from absa_library.engines.base import BaseEngine

        assert BaseEngine.normalise_labels(["positive", "NEG", 0]) == [
            "Positif",
            "Negatif",
            "Netral",
        ]
        with pytest.raises(InvalidInputError):
            BaseEngine.normalise_labels(["entah"])

    def test_train_not_supported(self, rule_engine: RuleBasedEngine) -> None:
        with pytest.raises(NotImplementedError):
            rule_engine.train(["x"], ["Positif"])

    def test_repr(self, rule_engine: RuleBasedEngine) -> None:
        assert "RuleBasedEngine" in repr(rule_engine)
