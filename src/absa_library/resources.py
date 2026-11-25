"""Helpers to load packaged data resources exactly once.

Every loader is memoised, so the dictionaries are read from disk a single time
per process (FR-PRE-14).
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from .exceptions import ResourceError

DATA_DIR = Path(__file__).resolve().parent / "data"


def data_path(name: str) -> Path:
    """Return the absolute path of a packaged data file."""
    path = DATA_DIR / name
    if not path.exists():
        raise ResourceError(f"Berkas data paket tidak ditemukan: {name}")
    return path


@cache
def load_lines(name: str) -> tuple[str, ...]:
    """Load a text file as a tuple of non-empty, stripped lines."""
    path = data_path(name)
    try:
        with path.open("r", encoding="utf-8") as handle:
            return tuple(
                line.strip()
                for line in handle
                if line.strip() and not line.lstrip().startswith("#")
            )
    except OSError as exc:  # pragma: no cover - defensive
        raise ResourceError(f"Gagal membaca berkas data: {name}") from exc


@cache
def load_json(name: str) -> dict:
    """Load a JSON object from the package data directory."""
    path = data_path(name)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:  # pragma: no cover - defensive
        raise ResourceError(f"Gagal membaca berkas JSON: {name}") from exc
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ResourceError(f"Berkas JSON harus berisi objek: {name}")
    return data


def clear_cache() -> None:
    """Clear every resource cache. Useful in tests."""
    load_lines.cache_clear()
    load_json.cache_clear()
