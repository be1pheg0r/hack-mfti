from __future__ import annotations

"""Lightweight loader for the Sber HF model.

Задача этого модуля — дать маленький и понятный API для загрузки модели и
токенизатора из локального cache, а при необходимости — инициировать скачивание
с Hugging Face в `model/`.

Это отдельный слой от feature extractor-а, чтобы в будущем сервер мог просто
вызвать `SberHFModelLoader.load()` и получить готовый bundle без знания деталей
про кэш, локальные пути и fallback на remote repository.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike, get_model_dpath
from ..constants import DEFAULT_SBER_HF_MODEL_DIRNAME, DEFAULT_SBER_HF_MODEL_REPO_ID
from ..utils.hf_retrieve import retrieve_hf_model


@dataclass(slots=True)
class SberModelBundle:
    """Готовый bundle с моделью, токенизатором и runtime-контекстом."""

    model: Any
    tokenizer: Any
    device: torch.device
    source_path: Path


class SberHFModelLoader:
    """Минимальный loader для causal LM из HF cache.

    Loader держит только нужные параметры и умеет:
    - находить локальный snapshot модели;
        - при необходимости скачивать модель в `model/`;
    - загружать модель, токенизатор и bundle целиком.
    """

    def __init__(
        self,
        repo_id: str = DEFAULT_SBER_HF_MODEL_REPO_ID,
        cache_root: PathLike | None = None,
        revision: str | None = None,
        force_download: bool = False,
        trust_remote_code: bool = True,
    ) -> None:
        self.repo_id: str = repo_id.strip()
        if not self.repo_id:
            raise ValueError("repo_id не может быть пустым")

        self.cache_root: Path = Path(cache_root) if cache_root is not None else Path(get_model_dpath())
        self.revision: str | None = revision
        self.force_download: bool = force_download
        self.trust_remote_code: bool = trust_remote_code
        self._cached_source_path: Path | None = None

    def _default_target_dir(self) -> Path:
        """Возвращает директорию, куда должен попасть snapshot модели."""
        return self.cache_root / DEFAULT_SBER_HF_MODEL_DIRNAME

    def resolve_source(self) -> Path:
        """Возвращает локальный путь до модели, скачивая ее при необходимости."""
        if self._cached_source_path is not None and self._cached_source_path.exists() and not self.force_download:
            return self._cached_source_path

        logger.info("Проверяю локальный snapshot модели: %s", self.repo_id)
        source_path: Path = retrieve_hf_model(
            repo_id=self.repo_id,
            target_root=self.cache_root,
            revision=self.revision,
            force_download=self.force_download,
        )
        self._cached_source_path = source_path
        logger.info("Модель доступна локально: %s", source_path)
        return source_path

    def _resolve_runtime(self) -> tuple[torch.device, torch.dtype, dict[str, Any]]:
        """Подбирает device/dtype/device_map под доступное железо."""
        if torch.cuda.is_available():
            gpu_count: int = torch.cuda.device_count()
            torch.backends.cuda.matmul.allow_tf32 = True
            compute_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            logger.info("Доступно GPU: %s, использую dtype=%s и device_map=auto", gpu_count, compute_dtype)
            return torch.device("cuda:0"), compute_dtype, {"device_map": "auto"}

        logger.info("GPU не обнаружены, использую CPU")
        return torch.device("cpu"), torch.float32, {}

    def load_tokenizer(self) -> Any:
        """Загружает токенизатор и настраивает его для decoder-only inference."""
        source_path: Path = self.resolve_source()
        logger.info("Загружаю токенизатор из %s", source_path)
        tokenizer = AutoTokenizer.from_pretrained(str(source_path), trust_remote_code=self.trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(self) -> Any:
        """Загружает causal LM из локального snapshot-а."""
        source_path: Path = self.resolve_source()
        device, dtype, extra_kwargs = self._resolve_runtime()
        logger.info("Загружаю модель из %s", source_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(source_path),
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            **extra_kwargs,
        )
        if device.type == "cpu":
            model = model.to(device)
        model.eval()
        logger.info("Модель загружена и переведена в eval mode на device=%s", device)
        return model

    def load(self) -> SberModelBundle:
        """Загружает модель и токенизатор одним вызовом."""
        source_path: Path = self.resolve_source()
        device, dtype, extra_kwargs = self._resolve_runtime()

        logger.info("Гружу bundle для %s", source_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(source_path),
            torch_dtype=dtype,
            trust_remote_code=self.trust_remote_code,
            **extra_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(source_path), trust_remote_code=self.trust_remote_code)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        if device.type == "cpu":
            model = model.to(device)
        model.eval()
        logger.info("Bundle готов: model=%s, tokenizer=%s, device=%s", model.__class__.__name__, tokenizer.__class__.__name__, device)
        return SberModelBundle(
            model=model,
            tokenizer=tokenizer,
            device=device,
            source_path=source_path,
        )


__all__ = ["SberHFModelLoader", "SberModelBundle"]



