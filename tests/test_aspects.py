"""Tests for aspect extraction and clause segmentation."""

from __future__ import annotations

import pytest

from absa_library import AspectExtractor, TextPreprocessor
from absa_library.aspects import fallback_mention
from absa_library.exceptions import InvalidInputError


class TestModes:
    def test_auto_detects_nouns_next_to_opinions(self) -> None:
        extractor = AspectExtractor()
        mentions = extractor.extract("barang bagus tapi kurir lama")
        aspects = [mention.aspect for mention in mentions]
        assert aspects == ["barang", "kurir"]
        assert all(mention.source == "auto" for mention in mentions)

    def test_auto_uses_vocabulary_without_opinion(self) -> None:
        extractor = AspectExtractor()
        mentions = extractor.extract("pengiriman tiba di rumah")
        found = {mention.aspect: mention.source for mention in mentions}
        assert found.get("pengiriman") == "vocab"

    def test_auto_returns_empty_without_candidates(self) -> None:
        extractor = AspectExtractor()
        assert extractor.extract("wow banget sih") == []

    def test_dictionary_mode_single_and_multiword(self) -> None:
        extractor = AspectExtractor(
            {"Harga": ["harga", "harga murah"], "Layanan": ["pelayanan"]}
        )
        mentions = extractor.extract("harga murah dan pelayanan ramah")
        assert {mention.aspect for mention in mentions} == {"Harga", "Layanan"}
        assert all(mention.source == "dictionary" for mention in mentions)

    def test_dictionary_mode_accepts_list(self) -> None:
        extractor = AspectExtractor(["Harga", "Layanan"])
        assert extractor.mode == "dictionary"
        mentions = extractor.extract("harga layanan")
        assert {mention.aspect for mention in mentions} == {"Harga", "Layanan"}

    def test_none_mode(self) -> None:
        assert AspectExtractor(None).mode == "none"
        assert AspectExtractor("none").extract("barang bagus") == []
        assert AspectExtractor("NONE").extract("barang bagus") == []

    def test_invalid_mode(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectExtractor("kamus")
        with pytest.raises(InvalidInputError):
            AspectExtractor(123)  # type: ignore[arg-type]

    def test_non_string_input(self) -> None:
        with pytest.raises(InvalidInputError):
            AspectExtractor().extract(None)  # type: ignore[arg-type]

    def test_empty_text(self) -> None:
        assert AspectExtractor().extract("") == []
        assert AspectExtractor().extract("!!!") == []

    def test_synonyms(self) -> None:
        extractor = AspectExtractor({"Layanan": ["kurir"]})
        extractor.synonyms = {"kurir": "Layanan"}
        extractor.mode = "auto"
        mentions = extractor.extract("kurir lama")
        assert mentions[0].aspect == "Layanan"

    def test_max_aspects(self) -> None:
        extractor = AspectExtractor(max_aspects=1)
        mentions = extractor.extract("barang bagus kurir lama harga murah")
        assert len(mentions) == 1

    def test_mention_payload(self) -> None:
        extractor = AspectExtractor()
        mention = extractor.extract("barang bagus")[0]
        assert mention.surface == "barang"
        assert mention.start == 0
        assert mention.end == 6


class TestSegmentation:
    def test_segment_returns_clause(self) -> None:
        extractor = AspectExtractor()
        text = "barang bagus tapi kurir lama"
        mention = extractor.extract(text)[1]
        assert extractor.segment(text, mention) == "tapi kurir lama"

    def test_clause_ranges(self) -> None:
        extractor = AspectExtractor()
        tokens = extractor._tokenize("barang bagus tapi kurir lama")
        assert extractor.clause_ranges(tokens) == [(0, 2), (2, 5)]
        assert extractor.clause_ranges([]) == []

    def test_segment_truncates_long_clause(self) -> None:
        extractor = AspectExtractor()
        words = ["kata"] * 60
        words[50] = "bagus"
        words[0] = "barang"
        text = " ".join(words)
        mention = extractor.extract(text)[0]
        segment = extractor.segment(text, mention)
        assert len(segment.split()) < 60

    def test_segment_unknown_position(self) -> None:
        extractor = AspectExtractor()
        mention = fallback_mention("barang bagus")
        assert extractor.segment("barang bagus", mention) == "barang bagus"

    def test_segment_for_label_uses_extracted_mention(self) -> None:
        extractor = AspectExtractor()
        text = "barang bagus tapi kurir lama"
        assert extractor.segment_for_label(text, "kurir") == "tapi kurir lama"

    def test_segment_for_label_finds_raw_token(self) -> None:
        extractor = AspectExtractor(None)
        text = "harga murah sekali"
        assert extractor.segment_for_label(text, "harga") == "harga murah sekali"

    def test_segment_for_label_falls_back_to_full_text(self) -> None:
        extractor = AspectExtractor(None)
        assert extractor.segment_for_label("barang bagus", "produk") == "barang bagus"


class TestHelpers:
    def test_fallback_mention(self) -> None:
        mention = fallback_mention("barang bagus")
        assert mention.aspect == "UMUM"
        assert mention.source == "fallback"
        assert (mention.start, mention.end) == (0, 12)

    def test_phrase_tokens(self) -> None:
        assert AspectExtractor._phrase_tokens("Harga Murah!") == ("harga", "murah")

    def test_preprocessor_injection(self) -> None:
        preprocessor = TextPreprocessor(vocabulary=["bagus"])
        extractor = AspectExtractor(preprocessor=preprocessor)
        assert extractor.preprocessor is preprocessor

    def test_vocab_and_window(self) -> None:
        extractor = AspectExtractor(aspect_vocab=["barang"], window=1)
        assert extractor.window == 1
        assert extractor.extract("barang bagus")[0].aspect == "barang"
