"""Lazy loading and safe state management for the IndoBERT checkpoint (FR-BERT-03 … 07)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .exceptions import ModelLoadError
from .logger import Logger

DEFAULT_MODEL_NAME = "indolem/indobert-base-uncased"

_INSTALL_HINT = (
    "Dependensi IndoBERT belum terpasang. Jalankan: "
    "pip install \"absa-library[indobert]\""
)


class ModelManager:
    """Download and cache a Hugging Face model/tokenizer on first use.

    The heavy imports (``torch``/``transformers``) only happen inside
    :meth:`_get_backend`, so importing :mod:`absa_library` never pulls them in
    (NFR-05). Any failure while loading is logged and leaves the manager in a
    safe, unloaded state (FR-BERT-06).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        num_labels: int = 3,
        local_files_only: bool = False,
        cache_dir: str | Path | None = None,
        backend: Any | None = None,
        logger: Logger | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.num_labels = int(num_labels)
        self.local_files_only = bool(local_files_only)
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.logger = logger or Logger("absa_library.model_manager")
        self._backend = backend
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    # -- state -------------------------------------------------------------
    def is_loaded(self) -> bool:
        """Return ``True`` when both the model and tokenizer are in memory."""
        return self._model is not None

    @property
    def model(self) -> Any | None:
        """Return the loaded model (``None`` before the first call)."""
        return self._model

    @property
    def tokenizer(self) -> Any | None:
        """Return the loaded tokenizer (``None`` before the first call)."""
        return self._tokenizer

    @property
    def injected_backend(self) -> Any | None:
        """Return the backend passed to the constructor, if any."""
        return self._backend

    def backend(self) -> Any:
        """Return (and memoise) the deep-learning backend."""
        return self._get_backend()

    def resolve_device(self) -> str:
        """Pick ``cuda`` when available, otherwise ``cpu`` (FR-BERT-07)."""
        if self.device:
            return self.device
        try:
            backend = self._get_backend()
            cuda = getattr(getattr(backend, "torch", None), "cuda", None)
            if cuda is not None and cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        except ModelLoadError:  # pragma: no cover - depends on optional deps
            self.device = "cpu"
        return self.device

    # -- loading -----------------------------------------------------------
    def load_model(self, model_name: str | None = None) -> Any:
        """Load the sequence-classification model, caching it in memory."""
        if self._model is not None:
            return self._model
        backend = self._get_backend()
        name = model_name or self.model_name
        try:
            model = backend.AutoModelForSequenceClassification.from_pretrained(
                name,
                num_labels=self.num_labels,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            self.logger.error("Gagal memuat model IndoBERT %s: %s", name, exc)
            self._model = None  # safe state
            raise ModelLoadError(
                f"Gagal memuat model IndoBERT dari {name!r}. "
                "Periksa koneksi internet atau gunakan model lokal."
            ) from exc
        self.model_name = name
        self._model = model
        self.logger.info("Model IndoBERT %s dimuat.", name)
        return model

    def load_tokenizer(self, model_name: str | None = None) -> Any:
        """Load the matching tokenizer, caching it in memory."""
        if self._tokenizer is not None:
            return self._tokenizer
        backend = self._get_backend()
        name = model_name or self.model_name
        try:
            tokenizer = backend.AutoTokenizer.from_pretrained(
                name,
                local_files_only=self.local_files_only,
                cache_dir=self.cache_dir,
            )
        except Exception as exc:
            self.logger.error("Gagal memuat tokenizer IndoBERT %s: %s", name, exc)
            self._tokenizer = None  # safe state
            raise ModelLoadError(
                f"Gagal memuat tokenizer IndoBERT dari {name!r}. "
                "Periksa koneksi internet atau gunakan model lokal."
            ) from exc
        self._tokenizer = tokenizer
        return tokenizer

    def ensure_loaded(self, model_name: str | None = None) -> tuple[Any, Any]:
        """Load both model and tokenizer (or return the cached pair)."""
        return self.load_model(model_name), self.load_tokenizer(model_name)

    def unload(self) -> None:
        """Drop the cached model and tokenizer."""
        self._model = None
        self._tokenizer = None

    def save(self, directory: str | Path) -> None:
        """Persist the loaded model and tokenizer to ``directory``."""
        if self._model is None or self._tokenizer is None:
            raise ModelLoadError("Tidak ada model yang dimuat untuk disimpan.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(directory)
        self._tokenizer.save_pretrained(directory)

    # -- internals ---------------------------------------------------------
    def _get_backend(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ModelLoadError(_INSTALL_HINT) from exc
        self._backend = SimpleNamespace(
            torch=torch,
            AutoModelForSequenceClassification=AutoModelForSequenceClassification,
            AutoTokenizer=AutoTokenizer,
        )
        return self._backend
