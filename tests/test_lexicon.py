"""Tests for the sentiment lexicon helpers."""

from __future__ import annotations

import pytest

from absa_library import SentimentLexicon, load_lexicon
from absa_library.exceptions import InvalidInputError, ResourceError


class TestPackagedLexicon:
    def test_default_lexicon_is_populated(self) -> None:
        lexicon = SentimentLexicon()
        assert len(lexicon) > 250
        assert lexicon.score("bagus") > 0
        assert lexicon.score("jelek") < 0

    def test_scores_are_normalised(self) -> None:
        lexicon = SentimentLexicon()
        assert all(-1.0 <= score <= 1.0 for score in lexicon.as_dict().values())

    def test_polarity(self) -> None:
        lexicon = SentimentLexicon()
        assert lexicon.polarity("bagus") == "Positif"
        assert lexicon.polarity("rusak") == "Negatif"
        assert lexicon.polarity("tidakdiketahui") == "Netral"


class TestConstruction:
    def test_explicit_mappings(self) -> None:
        lexicon = SentimentLexicon(positive={"bagus": 5}, negative={"jelek": 4})
        assert lexicon.score("bagus") == 1.0
        assert lexicon.score("jelek") == pytest.approx(-0.8)

    def test_sign_is_fixed_when_file_uses_other_convention(self) -> None:
        lexicon = SentimentLexicon(positive={"salah": -3}, negative={"aneh": 3})
        assert lexicon.score("salah") == pytest.approx(0.6)
        assert lexicon.score("aneh") == pytest.approx(-0.6)

    def test_zero_weight(self) -> None:
        lexicon = SentimentLexicon(positive={"netral": 0})
        assert lexicon.score("netral") == 0.0

    def test_scores_mapping(self) -> None:
        lexicon = SentimentLexicon(scores={"Bagus": 0.5})
        assert lexicon.score("bagus") == 0.5

    def test_symbols(self) -> None:
        lexicon = SentimentLexicon(positive={"bagus": 1}, negative={})
        assert "bagus" in lexicon
        assert "jelek" not in lexicon
        assert "SentimentLexicon" in repr(lexicon)


class TestFileLoading:
    def test_load_external_file(self, tmp_path) -> None:
        path = tmp_path / "lexicon.tsv"
        path.write_text(
            "# komentar\n\nbagus\t5\njelek\t-5\nnol\t0\naneh,2\n", encoding="utf-8"
        )
        lexicon = SentimentLexicon(path=path)
        assert lexicon.score("bagus") == 1.0
        assert lexicon.score("jelek") == -1.0
        assert lexicon.score("nol") == 0.0
        assert lexicon.score("aneh") == 0.4

    def test_load_lexicon_helper(self, tmp_path) -> None:
        path = tmp_path / "lexicon.txt"
        path.write_text("bagus 3\n", encoding="utf-8")
        assert load_lexicon(path).score("bagus") == pytest.approx(0.6)

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(ResourceError):
            SentimentLexicon(path=tmp_path / "missing.txt")

    def test_invalid_weight_is_ignored(self, tmp_path) -> None:
        path = tmp_path / "lexicon.txt"
        path.write_text("bagus banyak\njelek -2\n", encoding="utf-8")
        lexicon = SentimentLexicon(path=path)
        assert lexicon.score("bagus") == 0.0
        assert lexicon.score("jelek") == pytest.approx(-0.4)


class TestMutation:
    def test_add_and_clamp(self) -> None:
        lexicon = SentimentLexicon(positive={"bagus": 1})
        lexicon.add("Mantap", 5)
        assert lexicon.score("mantap") == 1.0
        lexicon.add("buruk", -5)
        assert lexicon.score("buruk") == -1.0

    def test_add_empty_word(self) -> None:
        with pytest.raises(InvalidInputError):
            SentimentLexicon(positive={"bagus": 1}).add("", 0.5)

    def test_load_returns_self(self, tmp_path) -> None:
        path = tmp_path / "extra.txt"
        path.write_text("keren 2\n", encoding="utf-8")
        lexicon = SentimentLexicon(positive={"bagus": 1})
        assert lexicon.load(path) is lexicon
        assert lexicon.score("keren") == pytest.approx(0.4)

    def test_as_dict_is_a_copy(self) -> None:
        lexicon = SentimentLexicon(positive={"bagus": 1})
        snapshot = lexicon.as_dict()
        snapshot["bagus"] = -1
        assert lexicon.score("bagus") == 1.0
