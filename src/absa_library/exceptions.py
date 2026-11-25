"""Exception hierarchy for :mod:`absa_library`.

All library-specific errors inherit from :class:`AbsaError` so users can catch a
single base class when they do not care about the specific failure mode.
"""

from __future__ import annotations

from collections.abc import Iterable


class AbsaError(Exception):
    """Base class for every error raised by :mod:`absa_library`."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        self.message = message
        self.detail = detail
        super().__init__(message if detail is None else f"{message} ({detail})")


class InvalidInputError(AbsaError, ValueError):
    """Raised when user input is empty, of the wrong type, or malformed."""


class UnsupportedEngineError(AbsaError, ValueError):
    """Raised when an unknown engine name is requested."""

    def __init__(self, engine: str, valid: Iterable[str]) -> None:
        self.engine = engine
        self.valid = tuple(sorted(valid))
        super().__init__(
            f"Engine {engine!r} tidak dikenal. Engine yang tersedia: {', '.join(self.valid)}."
        )


class EngineNotFittedError(AbsaError, RuntimeError):
    """Raised when a trainable engine is used before being fitted."""


class ModelLoadError(AbsaError, RuntimeError):
    """Raised when a model or tokenizer cannot be loaded."""


class ResourceError(AbsaError, RuntimeError):
    """Raised when a packaged data resource is missing or unreadable."""
