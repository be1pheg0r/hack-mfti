from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
from pydantic import BaseModel, ConfigDict, field_validator
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

from common.configs import load_pydantic_config
from common.logger import SBER_NLI_INFERENCE_LOGGER as logger
from common.paths import PathLike, get_model_dpath
from ..constants import (
    DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE,
    DEFAULT_SBER_HF_MODEL_DIRNAME,
    DEFAULT_SBER_HF_MODEL_REPO_ID,
    DEFAULT_SBER_HF_NLI_BATCH_SIZE,
    DEFAULT_SBER_HF_NLI_CONFIG_FPATH,
    DEFAULT_SBER_HF_NLI_HALLUCINATION_THRESHOLD,
    DEFAULT_SBER_HF_NLI_MAX_LENGTH,
    DEFAULT_SBER_HF_NLI_POSITIVE_CLASS_INDEX,
)
from ..utils.hf_retrieve import retrieve_hf_model


class HFNLIClfConfig(BaseModel):
    """Конфигурация Hugging Face NLI-классификатора для детекции галлюцинаций.

    Attributes:
        repo_id: Hugging Face repo id модели.
        cache_root: Локальная директория для snapshot модели.
        revision: Опциональная ревизия HF-репозитория.
        force_download: Принудительная перезагрузка snapshot.
        trust_remote_code: Разрешение на remote code в HF.
        max_length: Максимальная длина токенизированной пары.
        batch_size: Размер батча для инференса.
        hallucination_threshold: Порог для бинарного предсказания галлюцинации.
        positive_class_index: Индекс положительного класса "галлюцинация".
        compute_dtype: Тип вычислений для загрузки модели (float32/float16/bfloat16).
    """

    model_config = ConfigDict(frozen=True)

    repo_id: str = DEFAULT_SBER_HF_MODEL_REPO_ID
    cache_root: PathLike | None = None
    revision: str | None = None
    force_download: bool = False
    trust_remote_code: bool = True
    max_length: int = DEFAULT_SBER_HF_NLI_MAX_LENGTH
    batch_size: int = DEFAULT_SBER_HF_NLI_BATCH_SIZE
    hallucination_threshold: float = DEFAULT_SBER_HF_NLI_HALLUCINATION_THRESHOLD
    positive_class_index: int = DEFAULT_SBER_HF_NLI_POSITIVE_CLASS_INDEX
    compute_dtype: str = DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("repo_id не может быть пустым")
        return normalized

    @field_validator("max_length", "batch_size")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Числовой параметр должен быть положительным")
        return value

    @field_validator("hallucination_threshold")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("hallucination_threshold должен быть в диапазоне [0, 1]")
        return value

    @field_validator("positive_class_index")
    @classmethod
    def validate_positive_class_index(cls, value: int) -> int:
        if value < 0:
            raise ValueError("positive_class_index должен быть неотрицательным")
        return value

    @field_validator("compute_dtype")
    @classmethod
    def validate_compute_dtype(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        allowed: set[str] = {"float32", "float16", "bfloat16"}
        if normalized not in allowed:
            raise ValueError("compute_dtype должен быть одним из: float32, float16, bfloat16")
        return normalized

    @classmethod
    def from_yaml(cls, fpath: PathLike) -> HFNLIClfConfig:
        """Создает конфигурацию из YAML-файла."""
        return load_pydantic_config(config_cls=cls, fpath=fpath, section_name="hf_nli_clf")

    @classmethod
    def from_default_yaml(cls) -> HFNLIClfConfig:
        """Создает конфигурацию из штатного YAML-файла."""
        return cls.from_yaml(fpath=DEFAULT_SBER_HF_NLI_CONFIG_FPATH)


@dataclass(slots=True)
class HFNLIClfBundle:
    """Готовый bundle с моделью и runtime-контекстом."""

    model: Any
    tokenizer: Any
    device: torch.device
    source_path: Path


class HFNLIClf:
    """Классификатор пары `correct_answer`/`model_answer` на галлюцинацию."""

    def __init__(self, config: HFNLIClfConfig | None = None) -> None:
        self.config: HFNLIClfConfig = config or HFNLIClfConfig.from_default_yaml()
        self.cache_root: Path = Path(self.config.cache_root) if self.config.cache_root is not None else Path(get_model_dpath())
        self._cached_source_path: Path | None = None
        self._bundle: HFNLIClfBundle | None = None

    def _default_target_dir(self) -> Path:
        return self.cache_root / DEFAULT_SBER_HF_MODEL_DIRNAME

    def resolve_source(self) -> Path:
        if self._cached_source_path is not None and self._cached_source_path.exists() and not self.config.force_download:
            return self._cached_source_path

        source_path: Path = retrieve_hf_model(
            repo_id=self.config.repo_id,
            target_root=self.cache_root,
            revision=self.config.revision,
            force_download=self.config.force_download,
        )
        self._cached_source_path = source_path
        logger.info("HFNLIClf snapshot готов: %s", source_path)
        return source_path

    def download_checkpoint(self) -> Path:
        """Скачивает/обновляет локальный checkpoint и возвращает путь к нему."""
        return self.resolve_source()

    def _resolve_runtime(self) -> tuple[torch.device, torch.dtype, dict[str, Any]]:
        dtype_map: dict[str, torch.dtype] = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        compute_dtype: torch.dtype = dtype_map[self.config.compute_dtype]

        if torch.cuda.is_available():
            gpu_count: int = torch.cuda.device_count()
            torch.backends.cuda.matmul.allow_tf32 = True
            logger.info("HFNLIClf: GPU=%s, dtype=%s, device=cuda:0", gpu_count, compute_dtype)
            # Для sequence-classification инференса используем один device,
            # чтобы избежать конфликтов input/model тензоров между cuda:0/cuda:1.
            return torch.device("cuda:0"), compute_dtype, {}

        if compute_dtype != torch.float32:
            logger.info("HFNLIClf: CPU runtime поддерживает только float32, переключаюсь с %s", self.config.compute_dtype)
            compute_dtype = torch.float32

        logger.info("HFNLIClf: GPU не обнаружены, использую CPU")
        return torch.device("cpu"), compute_dtype, {}

    @contextmanager
    def _suppress_transformers_progress_bar(self) -> Iterator[None]:
        # Подавляем только локальный progress bar загрузки весов, затем
        # возвращаем исходное состояние, чтобы не влиять на другие части пайплайна.
        prev_enabled: bool | None = None
        if hasattr(hf_logging, "is_progress_bar_enabled"):
            prev_enabled = bool(hf_logging.is_progress_bar_enabled())
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()
        try:
            yield
        finally:
            if prev_enabled is True and hasattr(hf_logging, "enable_progress_bar"):
                hf_logging.enable_progress_bar()
            elif prev_enabled is False and hasattr(hf_logging, "disable_progress_bar"):
                hf_logging.disable_progress_bar()

    def load(self) -> HFNLIClfBundle:
        source_path: Path = self.download_checkpoint()
        device, dtype, extra_kwargs = self._resolve_runtime()

        tokenizer: Any = self.load_tokenizer(source_path=source_path)
        model: Any = self.load_model(
            source_path=source_path,
            dtype=dtype,
            extra_kwargs=extra_kwargs,
            device=device,
        )

        if device.type == "cpu":
            model = model.to(device)
        model.eval()

        bundle: HFNLIClfBundle = HFNLIClfBundle(
            model=model,
            tokenizer=tokenizer,
            device=device,
            source_path=source_path,
        )
        self._bundle = bundle
        return bundle

    def load_tokenizer(self, source_path: Path | None = None) -> Any:
        """Загружает токенизатор для пары текстов premise-hypothesis."""
        path: Path = source_path if source_path is not None else self.download_checkpoint()
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=self.config.trust_remote_code)
        # Для паритета с notebook оставляем стандартный right padding.
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def load_model(
        self,
        source_path: Path | None = None,
        dtype: torch.dtype | None = None,
        extra_kwargs: dict[str, Any] | None = None,
        device: torch.device | None = None,
    ) -> Any:
        """Загружает модель sequence-classification из checkpoint."""
        path: Path = source_path if source_path is not None else self.download_checkpoint()
        target_device: torch.device | None = device
        if dtype is None or extra_kwargs is None or target_device is None:
            runtime_device, runtime_dtype, runtime_kwargs = self._resolve_runtime()
            target_dtype: torch.dtype = runtime_dtype if dtype is None else dtype
            target_kwargs: dict[str, Any] = runtime_kwargs if extra_kwargs is None else extra_kwargs
            if target_device is None:
                target_device = runtime_device
        else:
            target_dtype = dtype
            target_kwargs = extra_kwargs

        with self._suppress_transformers_progress_bar():
            model = AutoModelForSequenceClassification.from_pretrained(
                str(path),
                dtype=target_dtype,
                trust_remote_code=self.config.trust_remote_code,
                **target_kwargs,
            )
        if not target_kwargs.get("device_map") and target_device is not None:
            model = model.to(target_device)
        model.eval()
        return model

    def _ensure_bundle(self) -> HFNLIClfBundle:
        if self._bundle is not None:
            return self._bundle
        return self.load()

    def _validate_pairs(self, premises: Sequence[str], hypotheses: Sequence[str]) -> None:
        if len(premises) != len(hypotheses):
            raise ValueError("premises и hypotheses должны быть одинаковой длины")

    def predict_logits(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> torch.Tensor:
        """Возвращает логиты модели для батча текстовых пар."""
        self._validate_pairs(premises=premises, hypotheses=hypotheses)
        if not premises:
            return torch.empty((0, 2), dtype=torch.float32)

        bundle: HFNLIClfBundle = self._ensure_bundle()
        size: int = batch_size if batch_size is not None else self.config.batch_size
        length: int = max_length if max_length is not None else self.config.max_length

        logits_chunks: list[torch.Tensor] = []
        for start in range(0, len(premises), size):
            batch_premises: list[str] = [str(value) for value in premises[start : start + size]]
            batch_hypotheses: list[str] = [str(value) for value in hypotheses[start : start + size]]
            encoded = bundle.tokenizer(
                batch_premises,
                batch_hypotheses,
                truncation=True,
                max_length=length,
                padding="max_length",
                return_tensors="pt",
            ).to(bundle.device)

            with torch.inference_mode():
                if bundle.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        batch_logits = bundle.model(**encoded).logits
                else:
                    batch_logits = bundle.model(**encoded).logits
                batch_logits = batch_logits.float().detach().cpu()
            logits_chunks.append(batch_logits)

        return torch.cat(logits_chunks, dim=0)

    def predict_hallucination_proba(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[float]:
        """Возвращает вероятность класса "галлюцинация" для каждой пары."""
        logits: torch.Tensor = self.predict_logits(
            premises=premises,
            hypotheses=hypotheses,
            batch_size=batch_size,
            max_length=max_length,
        )
        if logits.numel() == 0:
            return []
        probs: torch.Tensor = torch.softmax(logits, dim=-1)
        positive_index: int = self.config.positive_class_index
        if positive_index >= int(probs.shape[1]):
            raise ValueError("positive_class_index выходит за число классов модели")
        return [float(value) for value in probs[:, positive_index].tolist()]

    def predict_entailment_proba(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[float]:
        """Возвращает вероятность non-hallucination как `1 - hallucination`."""
        hallucination_probs: list[float] = self.predict_hallucination_proba(
            premises=premises,
            hypotheses=hypotheses,
            batch_size=batch_size,
            max_length=max_length,
        )
        return [1.0 - value for value in hallucination_probs]

    def predict_is_hallucination(
        self,
        premises: Sequence[str],
        hypotheses: Sequence[str],
        threshold: float | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
    ) -> list[int]:
        """Возвращает бинарное предсказание галлюцинации (0/1) по argmax."""
        _ = threshold  # Совместимость сигнатуры со старыми вызовами.
        logits: torch.Tensor = self.predict_logits(
            premises=premises,
            hypotheses=hypotheses,
            batch_size=batch_size,
            max_length=max_length,
        )
        if logits.numel() == 0:
            return []
        predicted_indices: list[int] = [int(value) for value in logits.argmax(dim=-1).tolist()]
        positive_index: int = self.config.positive_class_index
        return [int(index == positive_index) for index in predicted_indices]


__all__ = ["HFNLIClf", "HFNLIClfBundle", "HFNLIClfConfig"]


