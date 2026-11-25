"""Aspect-Based Sentiment Analysis for informal Indonesian text.

Quick start
-----------
>>> from absa_library import AspectAnalyzer
>>> analyzer = AspectAnalyzer()
>>> analyzer.analyze("barang bagus tapi kurir lama")
[{'text': 'barang bagus tapi kurir lama', 'aspect': 'barang', 'sentiment': 'Positif', ...},
 {'text': 'barang bagus tapi kurir lama', 'aspect': 'kurir', 'sentiment': 'Negatif', ...}]
"""

from __future__ import annotations

import logging

from .analyzer import AspectAnalyzer, available_engines
from .aspects import AspectExtractor, AspectMention
from .engines import (
    BaseEngine,
    IndoBERTEngine,
    NaiveBayesEngine,
    RuleBasedEngine,
    SVMEngine,
    create_engine,
    register_engine,
)
from .evaluator import Evaluator
from .exceptions import (
    AbsaError,
    EngineNotFittedError,
    InvalidInputError,
    ModelLoadError,
    ResourceError,
    UnsupportedEngineError,
)
from .lexicon import SentimentLexicon, load_lexicon
from .logger import Logger, configure_logging, get_logger
from .model_manager import ModelManager
from .preprocessing import PreprocessConfig, TextPreprocessor, damerau_levenshtein
from .version import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__all__ = [
    "AbsaError",
    "AspectAnalyzer",
    "AspectExtractor",
    "AspectMention",
    "BaseEngine",
    "EngineNotFittedError",
    "Evaluator",
    "IndoBERTEngine",
    "InvalidInputError",
    "Logger",
    "ModelLoadError",
    "ModelManager",
    "NaiveBayesEngine",
    "PreprocessConfig",
    "ResourceError",
    "RuleBasedEngine",
    "SVMEngine",
    "SentimentLexicon",
    "TextPreprocessor",
    "UnsupportedEngineError",
    "__version__",
    "available_engines",
    "configure_logging",
    "create_engine",
    "damerau_levenshtein",
    "get_logger",
    "load_lexicon",
    "register_engine",
]
