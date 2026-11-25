"""Tests for IndoBERT and the lazy ModelManager, using lightweight fakes."""

from __future__ import annotations

import logging
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from absa_library import IndoBERTEngine, ModelManager
from absa_library.engines.indobert import IndoBERTEngine as EngineFromModule
from absa_library.engines.indobert import _to_rows, run_training_epoch
from absa_library.exceptions import InvalidInputError, ModelLoadError
from absa_library.types import AnalysisUnit


class FakeTensor:
    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data

    def to(self, device):
        self.device = device
        return self


class FakeLoss:
    def __init__(self, value: float) -> None:
        self.value = value
        self.backward_called = False

    def backward(self) -> None:
        self.backward_called = True

    def item(self) -> float:
        return self.value


class FakeOutput:
    def __init__(self, logits=None, loss=None) -> None:
        self.logits = logits
        self.loss = loss


class FakeModel:
    def __init__(self, logits) -> None:
        self._logits = logits
        self.mode = None
        self.saved_to = None

    def eval(self) -> None:
        self.mode = "eval"

    def train(self) -> None:
        self.mode = "train"

    def parameters(self):
        return ["param"]

    def __call__(self, **kwargs):
        if "labels" in kwargs:
            return FakeOutput(loss=FakeLoss(0.25))
        size = len(kwargs["input_ids"])
        return FakeOutput(logits=FakeTensor([self._logits for _ in range(size)]))

    def save_pretrained(self, directory) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "config.json").write_text("{}", encoding="utf-8")
        self.saved_to = directory


class FakeTokenizer:
    def __init__(self, pieces=None) -> None:
        self.pieces = pieces or [[1, 2, 3]]

    def __call__(self, pairs, **kwargs):
        rows = self.pieces * len(pairs)
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}

    def save_pretrained(self, directory) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeTensorDataset:
    def __init__(self, *tensors) -> None:
        self.tensors = tensors
        self.length = len(tensors[0]) if tensors else 0

    def __iter__(self):
        for index in range(self.length):
            yield tuple(tensor[index] for tensor in self.tensors)

    def __len__(self) -> int:
        return self.length


class FakeDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        items = list(self.dataset)
        for start in range(0, len(items), self.batch_size):
            chunk = items[start : start + self.batch_size]
            yield tuple(list(column) for column in zip(*chunk))


class FakeOptimizer:
    def __init__(self, parameters, lr=1e-5) -> None:
        self.parameters = parameters
        self.lr = lr
        self.steps = 0

    def zero_grad(self) -> None:
        self.steps += 0

    def step(self) -> None:
        self.steps += 1


def make_backend(logits=None, model_error=None, tokenizer_error=None):
    logits = logits or [0.1, 0.2, 0.7]

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if model_error is not None:
                raise model_error
            return FakeModel(logits)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            if tokenizer_error is not None:
                raise tokenizer_error
            return FakeTokenizer()

    torch = SimpleNamespace(
        no_grad=lambda: nullcontext(),
        cuda=SimpleNamespace(is_available=lambda: False),
        tensor=lambda value: value,
        utils=SimpleNamespace(
            data=SimpleNamespace(
                TensorDataset=FakeTensorDataset, DataLoader=FakeDataLoader
            )
        ),
        optim=SimpleNamespace(AdamW=FakeOptimizer),
    )
    return SimpleNamespace(
        torch=torch,
        AutoModelForSequenceClassification=FakeAutoModel,
        AutoTokenizer=FakeAutoTokenizer,
    )


class TestModelManager:
    def test_lazy_loading(self) -> None:
        manager = ModelManager(backend=make_backend())
        assert manager.is_loaded() is False
        assert manager.model is None
        model = manager.load_model()
        assert manager.is_loaded() is True
        assert manager.load_model() is model

    def test_tokenizer_is_cached(self) -> None:
        manager = ModelManager(backend=make_backend())
        tokenizer = manager.load_tokenizer()
        assert manager.tokenizer is tokenizer
        assert manager.load_tokenizer() is tokenizer

    def test_ensure_loaded(self) -> None:
        manager = ModelManager(backend=make_backend())
        model, tokenizer = manager.ensure_loaded()
        assert model is not None and tokenizer is not None

    def test_unload(self) -> None:
        manager = ModelManager(backend=make_backend())
        manager.ensure_loaded()
        manager.unload()
        assert manager.is_loaded() is False
        assert manager.tokenizer is None

    def test_device_defaults_to_cpu(self) -> None:
        manager = ModelManager(backend=make_backend())
        assert manager.resolve_device() == "cpu"

    def test_explicit_device_wins(self) -> None:
        assert ModelManager(backend=make_backend(), device="cuda").resolve_device() == "cuda"

    def test_model_failure_sets_safe_state(self, caplog) -> None:
        manager = ModelManager(backend=make_backend(model_error=RuntimeError("offline")))
        with caplog.at_level(logging.ERROR, logger="absa_library.model_manager"):
            with pytest.raises(ModelLoadError):
                manager.load_model()
        assert manager.is_loaded() is False
        assert any("Gagal memuat model" in record.message for record in caplog.records)

    def test_tokenizer_failure(self, caplog) -> None:
        manager = ModelManager(backend=make_backend(tokenizer_error=RuntimeError("offline")))
        with caplog.at_level(logging.ERROR, logger="absa_library.model_manager"):
            with pytest.raises(ModelLoadError):
                manager.load_tokenizer()
        assert manager.tokenizer is None
        assert any("tokenizer" in record.message for record in caplog.records)

    def test_save_requires_loaded_model(self, tmp_path) -> None:
        with pytest.raises(ModelLoadError):
            ModelManager(backend=make_backend()).save(tmp_path)

    def test_save_writes_files(self, tmp_path) -> None:
        manager = ModelManager(backend=make_backend())
        manager.ensure_loaded()
        manager.save(tmp_path / "model")
        assert (tmp_path / "model" / "config.json").exists()
        assert (tmp_path / "model" / "tokenizer.json").exists()

    def test_missing_optional_dependencies(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "transformers", None)
        manager = ModelManager()
        with pytest.raises(ModelLoadError) as info:
            manager.load_model()
        assert "indobert" in str(info.value)

    def test_backend_property(self) -> None:
        backend = make_backend()
        assert ModelManager(backend=backend).backend() is backend


class TestIndoBERT:
    def test_predict(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend([0.1, 0.2, 0.7])))
        predictions = engine.predict(["barang bagus", "barang lama"])
        assert len(predictions) == 2
        assert predictions[0].label == "Netral"
        assert sum(predictions[0].scores.values()) == pytest.approx(1.0)

    def test_predict_empty(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        assert engine.predict([]) == []

    def test_predict_uses_aspect_pairs(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        unit = AnalysisUnit(text="x", aspect="barang", segment="bagus")
        assert engine.predict([unit])[0].label == "Netral"

    def test_two_labels(self) -> None:
        engine = IndoBERTEngine(
            labels=["Positif", "Negatif"],
            model_manager=ModelManager(backend=make_backend([0.8, 0.2])),
        )
        prediction = engine.predict(["barang bagus"])[0]
        assert prediction.label == "Positif"
        assert set(prediction.scores) == {"Positif", "Negatif"}

    def test_train_sets_fitted(self) -> None:
        engine = IndoBERTEngine(
            model_manager=ModelManager(backend=make_backend()), epochs=2
        )
        assert engine.train(["barang bagus", "barang lama"], ["Positif", "Negatif"]) is engine
        assert engine.is_fitted is True

    def test_train_length_mismatch(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        with pytest.raises(InvalidInputError):
            engine.train(["a", "b"], ["Positif"])

    def test_train_empty(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        with pytest.raises(InvalidInputError):
            engine.train([], [])

    def test_train_unknown_label(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        with pytest.raises(InvalidInputError):
            engine.train(["barang bagus"], ["Entah"])

    def test_invalid_configuration(self) -> None:
        with pytest.raises(InvalidInputError):
            IndoBERTEngine(max_length=0)
        with pytest.raises(InvalidInputError):
            IndoBERTEngine(batch_size=0)

    def test_device_property(self) -> None:
        engine = IndoBERTEngine(
            model_manager=ModelManager(backend=make_backend(), device="cpu")
        )
        assert engine.device == "cpu"

    def test_state_roundtrip(self) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        state = engine.state_dict()
        clone = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        clone.load_state_dict(state)
        assert clone.max_length == 128
        assert clone.labels == ["Positif", "Negatif", "Netral"]

    def test_save_without_model(self, tmp_path) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        engine.save(tmp_path / "bert")
        assert (tmp_path / "bert" / "absa_meta.json").exists()
        assert not (tmp_path / "bert" / "model").exists()

    def test_save_with_model_and_reload(self, tmp_path) -> None:
        engine = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        engine.predict(["barang bagus"])
        engine.save(tmp_path / "bert")
        assert (tmp_path / "bert" / "model" / "config.json").exists()
        restored = IndoBERTEngine(model_manager=ModelManager(backend=make_backend()))
        restored.load(tmp_path / "bert")
        assert restored.model_manager.local_files_only is True
        assert restored.predict(["barang bagus"])[0].label == "Netral"

    def test_load_missing_metadata(self, tmp_path) -> None:
        with pytest.raises(ModelLoadError):
            IndoBERTEngine().load(tmp_path)

    def test_module_alias(self) -> None:
        assert EngineFromModule is IndoBERTEngine


class TestTrainingHelpers:
    def test_run_training_epoch(self) -> None:
        model = FakeModel([0.1, 0.2, 0.7])
        loader = FakeDataLoader(FakeTensorDataset([[1]], [[1]], [0]), batch_size=1)
        optimizer = FakeOptimizer(["p"])
        loss = run_training_epoch(model, loader, optimizer)
        assert loss == pytest.approx(0.25)
        assert optimizer.steps == 1
        assert model.mode == "train"

    def test_run_training_epoch_empty(self) -> None:
        assert run_training_epoch(FakeModel([1]), FakeDataLoader([]), FakeOptimizer([])) == 0.0

    def test_to_rows(self) -> None:
        assert _to_rows([0.1, 0.2]) == [[0.1, 0.2]]
        assert _to_rows(FakeTensor([[0.3, 0.7]])) == [[0.3, 0.7]]
