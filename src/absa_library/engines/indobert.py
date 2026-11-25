"""IndoBERT sentence-pair classifier with lazy loading (FR-BERT-01 … 10)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from ..exceptions import InvalidInputError, ModelLoadError
from ..model_manager import DEFAULT_MODEL_NAME, ModelManager
from ..types import LABELS, AnalysisUnit, Prediction, argmax, normalize_label, softmax
from .base import BaseEngine

META_FILENAME = "absa_meta.json"


class IndoBERTEngine(BaseEngine):
    """Aspect-aware BERT classifier.

    Each input is encoded as the sentence pair ``[CLS] aspect [SEP] text [SEP]``
    (FR-BERT-01). Weights are downloaded on first use through
    :class:`~absa_library.model_manager.ModelManager`, and the optional
    ``torch``/``transformers`` dependencies are imported only at that moment.

    Parameters
    ----------
    model_name:
        Hugging Face checkpoint. Defaults to ``indolem/indobert-base-uncased``.
    max_length, batch_size:
        Tokenisation and inference batch settings.
    labels:
        Ordered class labels. Must match the fine-tuned head.
    model_manager:
        Injection point for tests and for pre-configured managers.
    """

    name = "indobert"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        max_length: int = 128,
        batch_size: int = 16,
        device: str | None = None,
        labels: Sequence[str] = LABELS,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        learning_rate: float = 2e-5,
        epochs: int = 3,
        model_manager: ModelManager | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        if max_length <= 0:
            raise InvalidInputError("max_length harus lebih besar dari 0.")
        if batch_size <= 0:
            raise InvalidInputError("batch_size harus lebih besar dari 0.")
        self.model_name = model_name
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.labels = [normalize_label(label) for label in labels]
        self.learning_rate = float(learning_rate)
        self.epochs = max(1, int(epochs))
        self.model_manager = model_manager or ModelManager(
            model_name=model_name,
            device=device,
            num_labels=len(self.labels),
            local_files_only=local_files_only,
            cache_dir=cache_dir,
            logger=self.logger,
        )

    @property
    def device(self) -> str:
        return self.model_manager.resolve_device()

    # -- prediction --------------------------------------------------------
    def predict(self, X: Sequence[AnalysisUnit] | Sequence[str] | str) -> list[Prediction]:
        units = self.as_units(X)
        if not units:
            return []
        model, tokenizer = self.model_manager.ensure_loaded()
        backend = self.model_manager.backend()
        predictions: list[Prediction] = []
        for start in range(0, len(units), self.batch_size):
            batch = units[start : start + self.batch_size]
            pairs = [(unit.aspect or "UMUM", unit.segment or unit.text) for unit in batch]
            encoded = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = self._to_device(dict(encoded))
            model.eval()
            context = backend.torch.no_grad()
            with context:
                outputs = model(**inputs)
            rows = _to_rows(outputs.logits)
            for row in rows:
                predictions.append(self._row_to_prediction(row))
        return predictions

    def _to_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        device = self.device
        if device == "cpu":
            return inputs
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _row_to_prediction(self, row: Sequence[float]) -> Prediction:
        weights = softmax([float(value) for value in row])
        labels = self.labels if len(self.labels) == len(weights) else LABELS
        scores = {label: weight for label, weight in zip(labels, weights)}
        label = labels[argmax(weights)]
        return Prediction(label=label, confidence=scores[label], scores=scores)

    # -- training ----------------------------------------------------------
    def train(
        self,
        X: Sequence[AnalysisUnit] | Sequence[str],
        y: Iterable[object],
    ) -> IndoBERTEngine:
        """Fine-tune the classification head on labelled sentence pairs (FR-BERT-04)."""
        units = self.as_units(X)
        labels = self.normalise_labels(y)
        if len(units) != len(labels):
            raise InvalidInputError(
                f"Jumlah data ({len(units)}) dan label ({len(labels)}) harus sama."
            )
        if not units:
            raise InvalidInputError("Data training tidak boleh kosong.")
        label_to_id = {label: index for index, label in enumerate(self.labels)}
        missing = sorted(set(labels) - set(label_to_id))
        if missing:
            raise InvalidInputError(
                "Label berikut tidak dikenal oleh head IndoBERT: " + ", ".join(missing)
            )

        model, tokenizer = self.model_manager.ensure_loaded()
        backend = self.model_manager.backend()
        torch = backend.torch

        pairs = [(unit.aspect or "UMUM", unit.segment or unit.text) for unit in units]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors=None,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask", [[1] * len(row) for row in input_ids])
        label_ids = [label_to_id[label] for label in labels]

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(input_ids),
            torch.tensor(attention_mask),
            torch.tensor(label_ids),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)

        for epoch in range(1, self.epochs + 1):
            loss = run_training_epoch(model, loader, optimizer)
            self.logger.info("IndoBERT epoch %d/%d - loss %.4f", epoch, self.epochs, loss)

        self._fitted = True
        return self

    # -- persistence -------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "labels": list(self.labels),
        }

    def load_state_dict(self, state: dict) -> IndoBERTEngine:
        self.model_name = state.get("model_name", self.model_name)
        self.max_length = int(state.get("max_length", self.max_length))
        self.batch_size = int(state.get("batch_size", self.batch_size))
        self.labels = [normalize_label(label) for label in state.get("labels", self.labels)]
        return self

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        super().save(path)
        metadata = self.state_dict()
        if self.model_manager.is_loaded():
            self.model_manager.save(path / "model")
            metadata["model_dir"] = "model"
        (path / META_FILENAME).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: str | Path) -> IndoBERTEngine:
        path = Path(path)
        meta_path = path / META_FILENAME
        if not meta_path.exists():
            raise ModelLoadError(f"Metadata IndoBERT tidak ditemukan di {path}.")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        self.load_state_dict(metadata)
        model_dir = metadata.get("model_dir")
        if model_dir:
            self.model_manager = ModelManager(
                model_name=str(path / model_dir),
                device=self.model_manager.device,
                num_labels=len(self.labels),
                local_files_only=True,
                backend=self.model_manager.injected_backend,
                logger=self.logger,
            )
        return self


def run_training_epoch(model: Any, loader: Any, optimizer: Any) -> float:
    """Run one optimisation epoch and return the mean batch loss.

    The helper only relies on duck-typed methods (``zero_grad``, ``backward``,
    ``step``) so it can be unit-tested with lightweight fakes instead of torch.
    """
    model.train()
    total = 0.0
    batches = 0
    for input_ids, attention_mask, labels in loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def _to_rows(logits: Any) -> list[list[float]]:
    if hasattr(logits, "tolist"):
        values = logits.tolist()
    else:  # pragma: no cover - exercised only by exotic backends
        values = list(logits)
    if values and isinstance(values[0], (int, float)):
        return [list(values)]
    return [list(row) for row in values]
