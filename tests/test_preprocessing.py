"""Tests for the preprocessing pipeline (FR-PRE-01 … 14)."""

from __future__ import annotations

import pytest

from absa_library import PreprocessConfig, TextPreprocessor
from absa_library.exceptions import InvalidInputError
from absa_library.preprocessing import (
    damerau_levenshtein,
    deletion_variants,
    reduce_repeated_chars,
)


class TestPreprocessConfig:
    def test_defaults(self) -> None:
        config = PreprocessConfig.from_value(None)
        assert config.cleaning is True
        assert config.max_edit_distance == 2

    def test_true_and_instance(self) -> None:
        assert PreprocessConfig.from_value(True).slang is True
        original = PreprocessConfig(stopwords=False)
        assert PreprocessConfig.from_value(original) is original

    def test_false_disables_everything(self) -> None:
        config = PreprocessConfig.from_value(False)
        assert config.cleaning is False
        assert config.case_folding is False
        assert config.stopwords is False

    def test_dict_overrides(self) -> None:
        config = PreprocessConfig.from_value({"slang": False, "max_tokens": 10})
        assert config.slang is False
        assert config.max_tokens == 10

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(InvalidInputError):
            PreprocessConfig.from_value({"tidak_ada": 1})

    def test_wrong_type_raises(self) -> None:
        with pytest.raises(InvalidInputError):
            PreprocessConfig.from_value(123)  # type: ignore[arg-type]


class TestDamerauLevenshtein:
    def test_identical(self) -> None:
        assert damerau_levenshtein("bagus", "bagus") == 0

    def test_empty(self) -> None:
        assert damerau_levenshtein("", "abc") == 3
        assert damerau_levenshtein("abc", "") == 3
        assert damerau_levenshtein("", "") == 0

    def test_substitution_and_insertion(self) -> None:
        assert damerau_levenshtein("bagus", "bagos") == 1
        assert damerau_levenshtein("bagus", "baguss") == 1

    def test_transposition(self) -> None:
        assert damerau_levenshtein("bgaus", "bagus") == 1

    def test_max_distance_short_circuit(self) -> None:
        assert damerau_levenshtein("kucing", "anjing", max_distance=1) == 2
        assert damerau_levenshtein("abcd", "wxyz", max_distance=2) == 3

    def test_length_gap_short_circuit(self) -> None:
        assert damerau_levenshtein("a", "abcdef", max_distance=1) == 2

    def test_deletion_variants(self) -> None:
        variants = deletion_variants("abc", 1)
        assert "abc" in variants
        assert "ab" in variants
        assert "bc" in variants
        assert len(deletion_variants("abc", 2)) > len(variants)


class TestCleaning:
    def test_removes_urls_mentions_hashtags_numbers(self, preprocessor: TextPreprocessor) -> None:
        cleaned = preprocessor.clean_text(
            "Cek https://toko.id @seller #promo harga 100rb!!!"
        )
        assert "http" not in cleaned
        assert "@" not in cleaned
        assert "#" not in cleaned
        assert "100" not in cleaned
        assert cleaned == "cek promo harga rb"

    def test_case_folding(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean_text("BAGUS Sekali") == "bagus sekali"

    def test_non_string_raises(self, preprocessor: TextPreprocessor) -> None:
        with pytest.raises(InvalidInputError):
            preprocessor.clean_text(None)  # type: ignore[arg-type]

    def test_reduce_repeated(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.reduce_repeated("bagussss") == "bagus"
        assert preprocessor.reduce_repeated("lamaaa!!") == "lama!!"
        assert reduce_repeated_chars("mantapppp") == "mantap"

    def test_tokenize(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.tokenize("barang bagus, harga 10rb") == [
            "barang",
            "bagus",
            "harga",
            "rb",
        ]

    def test_tokenize_with_spans(self, preprocessor: TextPreprocessor) -> None:
        spans = preprocessor.tokenize_with_spans("barang bagus")
        assert spans == [("barang", 0, 6), ("bagus", 7, 12)]


class TestSlang:
    def test_expands_dictionary(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.normalize_slang(["bgus", "bgt", "tp"]) == [
            "bagus",
            "banget",
            "tapi",
        ]

    def test_strips_nya_suffix(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.normalize_slang(["bgusnya"]) == ["bagus"]

    def test_unknown_token_kept(self) -> None:
        processor = TextPreprocessor(slang_dict={"bgus": "bagus"}, vocabulary=[])
        assert processor.normalize_slang(["zzz"]) == ["zzz"]

    def test_expansion_can_produce_multiple_tokens(self) -> None:
        processor = TextPreprocessor(slang_dict={"bgt": "banget sekali"}, vocabulary=[])
        assert processor.normalize_slang(["bgt"]) == ["banget", "sekali"]


class TestSpellCorrection:
    def test_transposition_is_corrected(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.spell_correction(["bgaus"]) == ["bagus"]

    def test_insertion_is_corrected(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.spell_correction(["aplikasii"]) == ["aplikasi"]

    def test_unknown_word_is_kept(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.spell_correction(["zzzzzz"]) == ["zzzzzz"]

    def test_short_token_is_skipped(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.spell_correction(["bgt"]) == ["bgt"]

    def test_known_word_is_untouched(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.spell_correction(["bagus"]) == ["bagus"]

    def test_negation_and_intensifier_are_untouched(
        self, preprocessor: TextPreprocessor
    ) -> None:
        assert preprocessor.spell_correction(["tidak"]) == ["tidak"]

    def test_cache_is_reused(self, preprocessor: TextPreprocessor) -> None:
        preprocessor.spell_correction(["bgaus"])
        assert "bgaus" in preprocessor._corrections
        assert preprocessor.spell_correction(["bgaus"]) == ["bagus"]

    def test_disabled_correction(self) -> None:
        processor = TextPreprocessor(
            {"spell_correction": False}, vocabulary=["bagus"]
        )
        assert processor.spell_correction(["bgaus"]) == ["bgaus"]

    def test_very_long_token_is_skipped(self, preprocessor: TextPreprocessor) -> None:
        token = "a" * 40
        assert preprocessor.spell_correction([token]) == [token]

    def test_slang_key_is_never_a_target(self) -> None:
        processor = TextPreprocessor(
            vocabulary=["bagus", "bgus"], slang_dict={"bgus": "bagus"}
        )
        assert processor._correct_token("bgaus") == "bagus"

    def test_rank_breaks_ties(self) -> None:
        processor = TextPreprocessor(vocabulary=["harga", "haram"])
        assert processor.spell_correction(["harag"]) == ["harga"]


class TestStopwords:
    def test_removes_stopwords(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.remove_stopwords(["yang", "bagus"]) == ["bagus"]

    def test_keeps_negation(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.remove_stopwords(["tidak", "bagus"]) == ["tidak", "bagus"]

    def test_keeps_intensifier(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.remove_stopwords(["sangat", "bagus"]) == [
            "sangat",
            "bagus",
        ]


class TestPipeline:
    def test_acceptance_example(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.normalize("bgus bgt tp lamaaa!!") == "bagus banget tapi lama"

    def test_normalize_keeps_stopwords(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.normalize("barang yang bagus") == "barang yang bagus"

    def test_process_removes_stopwords(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.process("barang yang bagus") == "barang bagus"

    def test_process_keeps_negation(self, preprocessor: TextPreprocessor) -> None:
        assert "tidak" in preprocessor.process("tidak bagus")

    def test_process_normalized(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.process_normalized("barang yang bagus") == "barang bagus"

    def test_max_tokens(self) -> None:
        processor = TextPreprocessor(
            {"max_tokens": 3}, vocabulary=[], slang_dict={}, stopwords=[]
        )
        assert processor.process("satu dua tiga empat lima") == "satu dua tiga"

    def test_process_tokens(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.process_tokens("bgus bgt barang")
        assert tokens == ["bagus", "banget", "barang"]

    def test_batch(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.process_batch(["bgus bgt", "brang bagus"])
        assert result == ["bagus banget", "barang bagus"]

    def test_disabled_slang(self) -> None:
        processor = TextPreprocessor(
            {"slang": False, "spell_correction": False}, vocabulary=["bagus"]
        )
        assert processor.normalize("bgus") == "bgus"

    def test_disabled_repeat_reduction(self) -> None:
        processor = TextPreprocessor(
            {"reduce_repeated": False},
            vocabulary=["bagus"],
            slang_dict={},
        )
        assert processor.normalize("bagussss") == "bagussss"

    def test_case_folding_disabled(self) -> None:
        processor = TextPreprocessor({"case_folding": False}, vocabulary=["bagus"])
        assert processor.normalize("Bagus") == "Bagus"
