"""Tests for metric computation and report serialisation (FR-EVA-01 … 06)."""

from __future__ import annotations

import json

import pytest

from absa_library import Evaluator
from absa_library.exceptions import InvalidInputError

Y_TRUE = ["Positif", "Positif", "Negatif", "Netral"]
Y_PRED = ["Positif", "Negatif", "Negatif", "Netral"]


class TestMetrics:
    def test_reference_values(self) -> None:
        report = Evaluator().calculate_metrics(Y_TRUE, Y_PRED)
        assert report["accuracy"] == pytest.approx(0.75)
        assert report["precision"] == pytest.approx(0.8333, abs=1e-4)
        assert report["recall"] == pytest.approx(0.8333, abs=1e-4)
        assert report["f1_score"] == pytest.approx(0.7778, abs=1e-4)
        assert report["samples"] == 4

    def test_per_class_and_confusion_matrix(self) -> None:
        report = Evaluator().calculate_metrics(Y_TRUE, Y_PRED)
        assert set(report["labels"]) == {"Positif", "Negatif", "Netral"}
        assert report["per_class"]["Positif"]["support"] == 2
        assert len(report["confusion_matrix"]) == 3

    def test_label_aliases_are_normalised(self) -> None:
        report = Evaluator().calculate_metrics(["positive", 1], ["Positif", "Positif"])
        assert report["accuracy"] == 1.0

    def test_pairs_and_dicts(self) -> None:
        report = Evaluator().calculate_metrics(
            [("Harga", "Positif"), {"aspect": "Layanan", "sentiment": "Negatif"}],
            ["Positif", {"label": "neg"}],
        )
        assert report["samples"] == 2
        assert report["accuracy"] == 1.0

    def test_length_mismatch(self) -> None:
        with pytest.raises(InvalidInputError):
            Evaluator().calculate_metrics(["Positif"], [])

    def test_empty(self) -> None:
        with pytest.raises(InvalidInputError):
            Evaluator().calculate_metrics([], [])

    def test_invalid_label(self) -> None:
        with pytest.raises(InvalidInputError):
            Evaluator().calculate_metrics(["Entah"], ["Positif"])

    def test_evaluate_stores_report(self) -> None:
        evaluator = Evaluator()
        report = evaluator.evaluate(Y_TRUE, Y_PRED, elapsed_seconds=0.5)
        assert report["elapsed_seconds"] == 0.5
        assert evaluator.last_report is report


class TestTiming:
    def test_timer(self) -> None:
        evaluator = Evaluator()
        with evaluator.timer() as box:
            pass
        assert box["elapsed_seconds"] >= 0.0


class TestReports:
    def test_save_json(self, tmp_path) -> None:
        evaluator = Evaluator()
        evaluator.evaluate(Y_TRUE, Y_PRED, elapsed_seconds=0.1)
        target = evaluator.save_report(tmp_path / "report.json")
        payload = json.loads(target.read_text(encoding="utf-8"))
        assert payload["accuracy"] == pytest.approx(0.75)

    def test_save_csv(self, tmp_path) -> None:
        evaluator = Evaluator()
        evaluator.evaluate(Y_TRUE, Y_PRED)
        target = evaluator.save_report(tmp_path / "report.csv")
        content = target.read_text(encoding="utf-8")
        assert "metric,value" in content
        assert "accuracy" in content
        assert "Positif.f1_score" in content

    def test_save_markdown(self, tmp_path) -> None:
        evaluator = Evaluator()
        evaluator.evaluate(Y_TRUE, Y_PRED, elapsed_seconds=0.2)
        target = evaluator.save_report(tmp_path / "report.md")
        content = target.read_text(encoding="utf-8")
        assert "# Laporan Evaluasi ABSA" in content
        assert "F1-score (macro)" in content
        assert "Waktu eksekusi" in content
        assert "| Kelas |" in content

    def test_save_without_report(self, tmp_path) -> None:
        with pytest.raises(InvalidInputError):
            Evaluator().save_report(tmp_path / "report.json")

    def test_save_invalid_suffix(self, tmp_path) -> None:
        evaluator = Evaluator()
        evaluator.evaluate(Y_TRUE, Y_PRED)
        with pytest.raises(InvalidInputError):
            evaluator.save_report(tmp_path / "report.txt")

    def test_markdown_without_optional_sections(self) -> None:
        markdown = Evaluator.to_markdown({"accuracy": 1.0, "samples": 1})
        assert "Accuracy" in markdown
        assert "Waktu eksekusi" not in markdown

    def test_save_explicit_report(self, tmp_path) -> None:
        target = Evaluator().save_report(tmp_path / "custom.json", {"accuracy": 1.0})
        assert json.loads(target.read_text(encoding="utf-8"))["accuracy"] == 1.0
