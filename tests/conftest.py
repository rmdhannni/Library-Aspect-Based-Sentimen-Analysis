"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest

from absa_library import (
    NaiveBayesEngine,
    RuleBasedEngine,
    SentimentLexicon,
    SVMEngine,
    TextPreprocessor,
)

SMALL_VOCAB = [
    "bagus",
    "kualitas",
    "harga",
    "barang",
    "lama",
    "kurir",
    "aplikasi",
    "error",
    "murah",
    "pelayanan",
    "memuaskan",
    "biasa",
    "pengiriman",
    "cepat",
]


@pytest.fixture()
def preprocessor() -> TextPreprocessor:
    """Preprocessor with a tiny vocabulary so tests stay deterministic."""
    return TextPreprocessor(vocabulary=SMALL_VOCAB)


@pytest.fixture()
def lexicon() -> SentimentLexicon:
    return SentimentLexicon(
        positive={"bagus": 0.8, "murah": 0.6, "cepat": 0.6, "memuaskan": 0.8},
        negative={"lama": 0.6, "error": 0.8, "mahal": 0.6},
    )


@pytest.fixture()
def rule_engine(lexicon: SentimentLexicon) -> RuleBasedEngine:
    return RuleBasedEngine(
        lexicon=lexicon,
        intensifiers={"sangat": 1.5, "banget": 1.5, "agak": 0.7},
        negation_words={"tidak", "bukan", "kurang"},
    )


POSITIVE_TEXTS = [
    "barang bagus",
    "kualitas bagus",
    "harga murah",
    "pelayanan memuaskan",
    "pengiriman cepat",
    "produk bagus sekali",
    "barang murah",
    "pelayanan bagus",
]
NEGATIVE_TEXTS = [
    "barang lama",
    "kualitas error",
    "harga mahal",
    "pelayanan error",
    "pengiriman lama",
    "produk error sekali",
    "barang mahal",
    "pelayanan lama",
]
NEUTRAL_TEXTS = [
    "barang biasa",
    "kualitas biasa",
    "harga biasa",
    "pelayanan biasa",
    "pengiriman biasa",
    "produk biasa",
    "barang cukup",
    "pelayanan cukup",
]


@pytest.fixture(scope="session")
def ml_dataset() -> tuple[list[str], list[str]]:
    texts = POSITIVE_TEXTS + NEGATIVE_TEXTS + NEUTRAL_TEXTS
    labels = ["Positif"] * 8 + ["Negatif"] * 8 + ["Netral"] * 8
    return texts, labels


@pytest.fixture(scope="session")
def nb_engine(ml_dataset) -> NaiveBayesEngine:
    texts, labels = ml_dataset
    return NaiveBayesEngine().train(texts, labels)


@pytest.fixture(scope="session")
def svm_engine(ml_dataset) -> SVMEngine:
    texts, labels = ml_dataset
    return SVMEngine().train(texts, labels)
