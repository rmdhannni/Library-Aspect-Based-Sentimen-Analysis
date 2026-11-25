"""Engine registry with name normalisation and a public factory (FR-ENG-04)."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from ..exceptions import UnsupportedEngineError
from .base import BaseEngine
from .indobert import IndoBERTEngine
from .naive_bayes import NaiveBayesEngine
from .rule_based import RuleBasedEngine
from .svm import SVMEngine

_ENGINE_REGISTRY: dict[str, type[BaseEngine]] = {
    RuleBasedEngine.name: RuleBasedEngine,
    NaiveBayesEngine.name: NaiveBayesEngine,
    SVMEngine.name: SVMEngine,
    IndoBERTEngine.name: IndoBERTEngine,
}

_ALIASES: dict[str, str] = {
    "rule_based": "rule-based",
    "rulebased": "rule-based",
    "rule": "rule-based",
    "rb": "rule-based",
    "lexicon": "rule-based",
    "naive_bayes": "naive-bayes",
    "naivebayes": "naive-bayes",
    "nb": "naive-bayes",
    "bayes": "naive-bayes",
    "linear-svm": "svm",
    "linearsvm": "svm",
    "support-vector-machine": "svm",
    "bert": "indobert",
    "indobert-base": "indobert",
    "indo-bert": "indobert",
}


def normalize_engine_name(name: str) -> str:
    """Map an alias to the canonical engine name.

    Raises
    ------
    UnsupportedEngineError
        When ``name`` is unknown. The message lists valid engines (FR-ERR-08).
    """
    if not isinstance(name, str):
        raise UnsupportedEngineError(str(name), _ENGINE_REGISTRY)
    key = name.strip().lower().replace(" ", "-")
    key = _ALIASES.get(key, key)
    if key not in _ENGINE_REGISTRY:
        raise UnsupportedEngineError(name, _ENGINE_REGISTRY)
    return key


def available_engines() -> tuple[str, ...]:
    """Return the canonical engine names in registration order."""
    return tuple(_ENGINE_REGISTRY)


def register_engine(
    name: str,
    engine_class: type[BaseEngine],
    aliases: Iterable[str] = (),
) -> None:
    """Register a custom engine so ``AspectAnalyzer`` can use it by name."""
    if not isinstance(engine_class, type) or not issubclass(engine_class, BaseEngine):
        raise TypeError("engine_class harus subclass dari BaseEngine.")
    key = name.strip().lower()
    if not key:
        raise ValueError("Nama engine tidak boleh kosong.")
    _ENGINE_REGISTRY[key] = engine_class
    for alias in aliases:
        _ALIASES[alias.strip().lower()] = key


def create_engine(name: str, **kwargs) -> BaseEngine:
    """Instantiate an engine by canonical name or alias."""
    canonical = normalize_engine_name(name)
    return _ENGINE_REGISTRY[canonical](**kwargs)


def load_engine(path: str | Path) -> BaseEngine:
    """Load a saved engine from a directory written by ``BaseEngine.save``."""
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.exists():
        from ..exceptions import ModelLoadError

        raise ModelLoadError(f"Metadata engine tidak ditemukan di {path}.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    engine = create_engine(meta.get("engine", "rule-based"))
    return engine.load(path)


__all__ = [
    "BaseEngine",
    "IndoBERTEngine",
    "NaiveBayesEngine",
    "RuleBasedEngine",
    "SVMEngine",
    "available_engines",
    "create_engine",
    "load_engine",
    "normalize_engine_name",
    "register_engine",
]
