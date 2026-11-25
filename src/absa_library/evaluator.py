"""Metric computation and report serialisation (FR-EVA-01 … 06)."""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

from .exceptions import InvalidInputError
from .types import normalize_label


class Evaluator:
    """Compute standard classification metrics and store the latest report."""

    def __init__(self) -> None:
        self.last_report: dict | None = None

    # -- metrics -----------------------------------------------------------
    def calculate_metrics(
        self,
        y_true: Sequence[object],
        y_pred: Sequence[object],
    ) -> dict:
        """Return accuracy, precision, recall, F1 and the confusion matrix.

        Precision, recall and F1 use macro averaging, which is the metric to
        report when sentiment classes are imbalanced (R7).
        """
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_fscore_support,
            precision_score,
            recall_score,
        )

        if len(y_true) != len(y_pred):
            raise InvalidInputError(
                f"Panjang y_true ({len(y_true)}) dan y_pred ({len(y_pred)}) harus sama."
            )
        if not y_true:
            raise InvalidInputError("Data evaluasi tidak boleh kosong.")

        truth = [self._as_label(value) for value in y_true]
        predicted = [self._as_label(value) for value in y_pred]
        labels = sorted(set(truth) | set(predicted))

        precision, recall, f1, support = precision_recall_fscore_support(
            truth, predicted, labels=labels, zero_division=0
        )
        per_class = {}
        for index, label in enumerate(labels):
            per_class[label] = {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1_score": float(f1[index]),
                "support": int(support[index]),
            }

        return {
            "accuracy": float(accuracy_score(truth, predicted)),
            "precision": float(precision_score(truth, predicted, average="macro", zero_division=0)),
            "recall": float(recall_score(truth, predicted, average="macro", zero_division=0)),
            "f1_score": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(truth, predicted, average="weighted", zero_division=0)),
            "samples": len(truth),
            "labels": labels,
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        }

    def evaluate(
        self,
        y_true: Sequence[object],
        y_pred: Sequence[object],
        *,
        elapsed_seconds: float | None = None,
    ) -> dict:
        """Compute metrics, attach the elapsed time and remember the report."""
        report = self.calculate_metrics(y_true, y_pred)
        if elapsed_seconds is not None:
            report["elapsed_seconds"] = float(elapsed_seconds)
        self.last_report = report
        return report

    # -- timing ------------------------------------------------------------
    @contextmanager
    def timer(self) -> Iterator[dict]:
        """Measure wall-clock time of a block (FR-EVA-03)."""
        box: dict[str, float] = {}
        started = time.perf_counter()
        try:
            yield box
        finally:
            box["elapsed_seconds"] = time.perf_counter() - started

    # -- reporting ---------------------------------------------------------
    def save_report(self, path: str | Path, report: dict | None = None) -> Path:
        """Write the report as JSON, CSV or Markdown based on the suffix."""
        data = report if report is not None else self.last_report
        if data is None:
            raise InvalidInputError("Belum ada laporan evaluasi untuk disimpan.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".json":
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        elif suffix == ".csv":
            self._write_csv(path, data)
        elif suffix in {".md", ".markdown"}:
            path.write_text(self.to_markdown(data), encoding="utf-8")
        else:
            raise InvalidInputError("Format laporan harus .json, .csv, atau .md.")
        return path

    @staticmethod
    def to_markdown(report: dict) -> str:
        """Render a report as a Markdown table with summary metrics."""
        lines = [
            "# Laporan Evaluasi ABSA",
            "",
            "| Metrik | Nilai |",
            "|---|---|",
            f"| Accuracy | {report.get('accuracy', 0.0):.4f} |",
            f"| Precision (macro) | {report.get('precision', 0.0):.4f} |",
            f"| Recall (macro) | {report.get('recall', 0.0):.4f} |",
            f"| F1-score (macro) | {report.get('f1_score', 0.0):.4f} |",
            f"| F1-score (weighted) | {report.get('weighted_f1', 0.0):.4f} |",
            f"| Jumlah sampel | {report.get('samples', 0)} |",
        ]
        if "elapsed_seconds" in report:
            lines.append(f"| Waktu eksekusi (s) | {report['elapsed_seconds']:.4f} |")
        per_class = report.get("per_class") or {}
        if per_class:
            lines += [
                "",
                "| Kelas | Precision | Recall | F1 | Support |",
                "|---|---|---|---|---|",
            ]
            for label, values in per_class.items():
                lines.append(
                    f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
                    f"{values['f1_score']:.4f} | {values['support']} |"
                )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _write_csv(path: Path, report: dict) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["metric", "value"])
            for key in ("accuracy", "precision", "recall", "f1_score", "weighted_f1", "samples"):
                writer.writerow([key, report.get(key, "")])
            writer.writerow(["elapsed_seconds", report.get("elapsed_seconds", "")])
            per_class = report.get("per_class") or {}
            for label, values in per_class.items():
                writer.writerow([f"{label}.precision", values["precision"]])
                writer.writerow([f"{label}.recall", values["recall"]])
                writer.writerow([f"{label}.f1_score", values["f1_score"]])

    @staticmethod
    def _as_label(value: object) -> str:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            value = value[1]
        if isinstance(value, dict):
            value = value.get("sentiment", value.get("label"))
        try:
            return normalize_label(value)
        except ValueError as exc:
            raise InvalidInputError(str(exc)) from exc
