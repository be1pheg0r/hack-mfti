from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import *

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.files import read_yaml
from common.logger import LoggerConfig, setup_logger
from common.paths import get_avito_configs_dpath, get_checkpoints_dpath, PathLike


DEFAULT_ENCODER_CONFIG_FPATH: Path = get_avito_configs_dpath() / "frida_base_config.yaml"


class SentenceTransformerLike(Protocol):
    """Протокол модели sentence-transformers для типизации."""

    def encode(
        self,
        sentences: list[str],
        *,
        prompt: str,
        batch_size: int,
        normalize_embeddings: bool,
        convert_to_numpy: bool,
        show_progress_bar: bool,
    ) -> Any:
        """Возвращает эмбеддинги для списка предложений."""


class EncoderConfig(BaseModel):
    """Конфигурация энкодера текста.

    Attributes:
        config_path: Путь к YAML-конфигу энкодера.
        local_model_path: Относительный путь до директории модели внутри checkpoints.
        hub_model_name: Имя модели на HF Hub.
        prompt: Префикс, который передается в encode.
        batch_size: Размер батча при инференсе.
        normalize_embeddings: Нормализовать ли векторы эмбеддингов.
        show_progress_bar: Показывать ли progress bar в encode.
        device: Целевое устройство для модели.
    """

    model_config = ConfigDict(frozen=True)

    config_path: Optional[PathLike] = Field(default=None, exclude=True)
    local_model_path: Optional[str]= None
    hub_model_name: Optional[str]= None
    prompt: Optional[str]
    batch_size: Optional[int]
    normalize_embeddings: bool = False
    show_progress_bar: bool = False
    device: Optional[str]= None

    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, value: Any) -> Any:
        """Собирает конфиг из YAML и явных аргументов.

        Args:
            value: Входные данные pydantic-модели.

        Returns:
            Словарь параметров для инициализации модели.
        """
        raw_data: Any = value
        if isinstance(raw_data, (str, Path)):
            raw_data = {"config_path": raw_data}

        if not isinstance(raw_data, dict):
            return raw_data

        # Поддержка legacy-поля model_name как alias для hub_model_name.
        if "hub_model_name" not in raw_data and raw_data.get("model_name") is not None:
            raw_data["hub_model_name"] = raw_data.get("model_name")

        config_path: PathLike | None = raw_data.get("config_path")
        if config_path is None:
            return raw_data

        yaml_raw: Any = read_yaml(config_path)
        if yaml_raw is None:
            raise ValueError(f"Конфиг энкодера пустой: {config_path}")
        if not isinstance(yaml_raw, dict):
            raise ValueError(f"Ожидается словарь в YAML-конфиге: {config_path}")

        explicit_values: dict[str, Any] = {
            key: item
            for key, item in raw_data.items()
            if key != "config_path" and item is not None
        }
        merged_values: dict[str, Any] = {**yaml_raw, **explicit_values}
        return merged_values

    @classmethod
    def from_default_yaml(cls) -> EncoderConfig:
        """Создает конфиг из дефолтного YAML-файла для Avito.

        Returns:
            Инициализированная конфигурация энкодера.
        """
        return cls.model_validate({"config_path": DEFAULT_ENCODER_CONFIG_FPATH})

    @field_validator("local_model_path")
    @classmethod
    def validate_local_model_path(cls, value: str | None) -> str | None:
        """Проверяет корректность относительного пути к локальной модели."""
        if value is None:
            return None
        normalized_value: str = value.strip()
        if not normalized_value:
            return None
        if Path(normalized_value).is_absolute():
            raise ValueError("local_model_path должен быть относительным путем внутри checkpoints.")
        return normalized_value

    @field_validator("hub_model_name")
    @classmethod
    def validate_hub_model_name(cls, value: str | None) -> str | None:
        """Проверяет корректность имени модели на HF Hub."""
        if value is None:
            return None
        normalized_value: str = value.strip()
        return normalized_value or None

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Проверяет и нормализует prompt для encode."""
        if not value:
            raise ValueError("prompt не может быть пустым.")
        return value

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        """Проверяет размер батча."""
        if value <= 0:
            raise ValueError("batch_size должен быть положительным.")
        return value

    @model_validator(mode="after")
    def validate_model_source(self) -> EncoderConfig:
        """Проверяет, что задан хотя бы один источник модели."""
        if self.local_model_path is None and self.hub_model_name is None:
            raise ValueError("Нужно задать local_model_path или hub_model_name.")
        return self

    def resolve_local_model_dir(self) -> Path | None:
        """Возвращает путь к локальной директории модели внутри checkpoints."""
        if self.local_model_path is None:
            return None
        return get_checkpoints_dpath() / self.local_model_path


class BaseEncoder(ABC):
    """Базовый интерфейс текстовых энкодеров."""

    def __init__(self, config: EncoderConfig) -> None:
        self.config: EncoderConfig = config

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        """Преобразует тексты в эмбеддинги."""


class SentenceTransformerEncoder(BaseEncoder):
    """Энкодер на базе sentence-transformers.

    Для instruct-моделей prompt передается в `encode` через аргумент `prompt`.
    """

    def __init__(
        self,
        config: EncoderConfig,
        backend_model: SentenceTransformerLike | None = None,
    ) -> None:
        super().__init__(config=config)
        self._logger = setup_logger(
            LoggerConfig(
                name="avito-embeddings",
                level="INFO",
                prefix="[AVITO/EMBEDDINGS] ",
            )
        )
        self._model: SentenceTransformerLike = backend_model or self._build_model(config=config)

    def _build_model(self, config: EncoderConfig) -> SentenceTransformerLike:
        """Создает инстанс sentence-transformers модели с fallback-логикой."""
        model_class: type[Any] = self._load_sentence_transformer_class()
        errors: list[str] = []

        local_model_dir: Path | None = config.resolve_local_model_dir()
        if local_model_dir is not None:
            if local_model_dir.exists() and local_model_dir.is_dir():
                try:
                    self._logger.info(f"Пробую загрузить локальную модель: {local_model_dir}")
                    local_model: SentenceTransformerLike = model_class(
                        model_name_or_path=str(local_model_dir),
                        device=config.device,
                    )
                    return local_model
                except Exception as error:
                    errors.append(f"Локальная модель '{local_model_dir}': {error}")
            else:
                self._logger.warning(f"Локальная модель не найдена или не директория: {local_model_dir}")

        if config.hub_model_name is not None:
            try:
                self._logger.info(f"Пробую загрузить модель из HF Hub: {config.hub_model_name}")
                hub_model: SentenceTransformerLike = model_class(
                    model_name_or_path=config.hub_model_name,
                    device=config.device,
                )
                return hub_model
            except Exception as error:
                errors.append(f"HF Hub модель '{config.hub_model_name}': {error}")

        details: str = "; ".join(errors) if errors else "Оба источника модели недоступны."
        raise RuntimeError(f"Не удалось загрузить sentence-transformers модель. {details}")

    @staticmethod
    def _load_sentence_transformer_class() -> type[Any]:
        """Ленивая загрузка класса SentenceTransformer."""
        module: Any = __import__("sentence_transformers", fromlist=["SentenceTransformer"])
        return getattr(module, "SentenceTransformer")

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Преобразует список текстов в эмбеддинги.

        Args:
            texts: Список входных текстов.

        Returns:
            Матрица эмбеддингов размерности `N x D`.
        """
        if not texts:
            raise ValueError("texts не может быть пустым списком.")

        prepared_texts: list[str] = [text.strip() for text in texts if text.strip()]
        if not prepared_texts:
            raise ValueError("После нормализации не осталось валидных текстов.")

        self._logger.info(f"Запуск encode для {len(prepared_texts)} текстов")
        embeddings_raw: Any = self._model.encode(
            sentences=prepared_texts,
            prompt=self.config.prompt,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.config.show_progress_bar,
        )

        if hasattr(embeddings_raw, "tolist"):
            embeddings: list[list[float]] = embeddings_raw.tolist()
        else:
            embeddings = [list(vector) for vector in embeddings_raw]

        return embeddings


if __name__ == "__main__":
    config = EncoderConfig.from_default_yaml()
    encoder = SentenceTransformerEncoder(config=config)
    sample_texts = ["Обезьяна обожралась бананов и сдохла", "Обезьяна обожралась бананов и умерла навсегда"]
    embeddings = encoder.encode(sample_texts)

    similarity = sum(a * b for a, b in zip(embeddings[0], embeddings[1])) / (
        (sum(a * a for a in embeddings[0]) ** 0.5) * (sum(b * b for b in embeddings[1]) ** 0.5)
    )
    print(f"Cosine similarity: {similarity:.4f}")