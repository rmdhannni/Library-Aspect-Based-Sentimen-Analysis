"""Thin wrapper around the standard :mod:`logging` module.

The library never configures the root logger. A :class:`logging.NullHandler` is
attached by :mod:`absa_library` so importing the package stays silent until the
user configures logging.
"""

from __future__ import annotations

import logging
from typing import Any

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DEFAULT_LOGGER_NAME = "absa_library"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    name: str = DEFAULT_LOGGER_NAME,
    fmt: str = DEFAULT_FORMAT,
) -> logging.Logger:
    """Configure a console handler for the library logger.

    Parameters
    ----------
    level:
        Logging level, either a :mod:`logging` constant or a level name.
    name:
        Logger name to configure.
    fmt:
        Format string passed to :class:`logging.Formatter`.

    Returns
    -------
    logging.Logger
        The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    return logger


class Logger:
    """Small facade with ``info``/``error``/``warning``/``debug`` helpers."""

    def __init__(
        self,
        name: str = DEFAULT_LOGGER_NAME,
        level: int | str | None = None,
        handler: logging.Handler | None = None,
    ) -> None:
        self.name = name
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            self._logger.addHandler(logging.NullHandler())
        if level is not None:
            self._logger.setLevel(level)
        if handler is not None:
            if handler.formatter is None:
                handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
            self._logger.addHandler(handler)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.error(message, *args, **kwargs)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.debug(message, *args, **kwargs)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.exception(message, *args, **kwargs)

    @property
    def logger(self) -> logging.Logger:
        """Return the underlying :class:`logging.Logger`."""
        return self._logger


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> Logger:
    """Return a :class:`Logger` for ``name``."""
    return Logger(name)
