"""Tests for packaged resources and logging helpers."""

from __future__ import annotations

import logging

import pytest

from absa_library import Logger, ResourceError, configure_logging, get_logger, resources


class TestResources:
    def test_data_path_exists(self) -> None:
        assert resources.data_path("lexicon_positive.txt").exists()

    def test_data_path_missing(self) -> None:
        with pytest.raises(ResourceError):
            resources.data_path("does_not_exist.txt")

    def test_load_lines_skips_comments(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "words.txt").write_text("# komentar\nbagus\n\njelek\n", encoding="utf-8")
        monkeypatch.setattr(resources, "DATA_DIR", tmp_path)
        resources.clear_cache()
        assert resources.load_lines("words.txt") == ("bagus", "jelek")

    def test_load_json(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "data.json").write_text('{"a": 1}', encoding="utf-8")
        monkeypatch.setattr(resources, "DATA_DIR", tmp_path)
        resources.clear_cache()
        assert resources.load_json("data.json") == {"a": 1}

    def test_load_json_invalid(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "data.json").write_text("{not json", encoding="utf-8")
        monkeypatch.setattr(resources, "DATA_DIR", tmp_path)
        resources.clear_cache()
        with pytest.raises(ResourceError):
            resources.load_json("data.json")

    def test_load_json_must_be_object(self, tmp_path, monkeypatch) -> None:
        (tmp_path / "data.json").write_text("[1, 2]", encoding="utf-8")
        monkeypatch.setattr(resources, "DATA_DIR", tmp_path)
        resources.clear_cache()
        with pytest.raises(ResourceError):
            resources.load_json("data.json")

    def test_cache_and_clear(self) -> None:
        first = resources.load_lines("stopwords_id.txt")
        assert resources.load_lines("stopwords_id.txt") is first
        resources.clear_cache()
        second = resources.load_lines("stopwords_id.txt")
        assert second == first
        assert second is not first


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestLogger:
    def test_levels(self) -> None:
        handler = _Capture()
        logger = Logger("absa_library.test_logger", level=logging.DEBUG, handler=handler)
        logger.debug("debug %s", "x")
        logger.info("info %s", "y")
        logger.warning("warn")
        logger.error("error")
        try:
            raise ValueError("boom")
        except ValueError:
            logger.exception("exception")
        messages = [record.getMessage() for record in handler.records]
        assert "debug x" in messages
        assert "info y" in messages
        assert "warn" in messages
        assert "error" in messages
        assert "exception" in messages

    def test_underlying_logger(self) -> None:
        logger = Logger("absa_library.test_underlying")
        assert logger.logger.name == "absa_library.test_underlying"

    def test_handler_gets_formatter(self) -> None:
        handler = _Capture()
        Logger("absa_library.test_formatter", handler=handler)
        assert handler.formatter is not None

    def test_configure_logging_is_idempotent(self) -> None:
        name = "absa_library.test_configure"
        configure_logging("DEBUG", name=name)
        configure_logging("INFO", name=name)
        handlers = logging.getLogger(name).handlers
        assert len([h for h in handlers if isinstance(h, logging.StreamHandler)]) == 1

    def test_get_logger(self) -> None:
        logger = get_logger("absa_library.test_get")
        assert isinstance(logger, Logger)
        assert logger.name == "absa_library.test_get"
