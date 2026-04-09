from __future__ import annotations

from pathlib import Path
from typing import *

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.files import read_yaml
from common.mistral import MistralCallConfig
from common.paths import get_avito_configs_dpath, get_avito_gitignore_dpath

DEFAULT_SHOULD_SPLIT_GRAPH_CONFIG_FPATH = Path(get_avito_configs_dpath()) / "should_split_graph.yaml"


def _validate_positive_int(value: int, field_names: str) -> int:
    if value <= 0:
        raise ValueError(f"{field_names} должны быть положительными")
    return value


def _validate_non_negative_float(value: float, field_name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{field_name} должна быть неотрицательной")
    return float(value)


class DataConfig(BaseModel):
    """Пути к данным для пайплайна should-split.

    Attributes:
        mc_map_filename: CSV-словарь микрокатегорий с keyphrases.
        markup_filename: JSON-разметка объявлений.
        rag_markup_path: Явный путь к JSON-разметке для RAG (абсолютный или относительный к avito/data).
    """

    mc_map_filename: str = "rnc_mic_key_phrases.csv"
    markup_filename: str = "rnc_dataset_markup.json"
    rag_markup_path: str | None = None


class EncoderConfig(BaseModel):
    """Конфигурация эмбеддингового энкодера.

    Attributes:
        model_name: HuggingFace id модели энкодера.
    """

    model_name: str = "sergeyzh/rubert-mini-frida"


class VectorDbConfig(BaseModel):
    """Конфигурация векторной БД и retrieval-слоя.

    Attributes:
        backend: Тип retrieval backend (in_memory/chroma).
        collection_name: Имя коллекции векторной БД.
        persist_dir: Директория хранения Chroma индекса.
        search_k: Количество кандидатов для retrieval.
        balanced_k: Размер сбалансированного top-k для промпта.
        persist: Сохранять ли индекс на диск.
    """

    backend: Literal["in_memory", "chroma"] = "in_memory"
    collection_name: str = "avito_descriptions"
    persist_dir: str = Field(
        default_factory=lambda: str(Path(get_avito_gitignore_dpath()) / "chroma")
    )
    search_k: int = 20
    balanced_k: int = 6
    persist: bool = False

    @field_validator("search_k", "balanced_k")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        return _validate_positive_int(value, "search_k и balanced_k")


class MistralInferenceConfig(BaseModel):
    """Конфигурация вызова Mistral для стадии rag_split.

    Attributes:
        enabled: Если False, LLM-стадия возвращает решение префильтра.
        models_list: Список моделей в порядке приоритета.
        timeout: Таймаут одного запроса.
        max_attempts_per_call: Число попыток внутри safe_call.
        reasoning_effort: Параметр reasoning_effort для chat.complete.
        temperature: Температура генерации.
    """

    enabled: bool = True
    models_list: list[str] = Field(default_factory=lambda: ["mistral-medium-latest"])
    timeout: int = 240
    max_attempts_per_call: int = 1
    reasoning_effort: str = "hard"
    temperature: float = 0.0

    @field_validator("models_list")
    @classmethod
    def validate_models_list(cls, value: list[str]) -> list[str]:
        normalized = [model.strip() for model in value if model.strip()]
        if not normalized:
            raise ValueError("models_list не может быть пустым")
        return normalized

    @field_validator("timeout", "max_attempts_per_call")
    @classmethod
    def validate_positive(cls, value: int) -> int:
        return _validate_positive_int(value, "timeout и max_attempts_per_call")

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        return _validate_non_negative_float(value, "temperature")

    def to_call_config(self) -> MistralCallConfig:
        """Преобразует конфиг в формат common.mistral."""
        return MistralCallConfig(
            models_list=self.models_list,
            timeout=self.timeout,
            max_attempts_per_call=self.max_attempts_per_call,
        )


class GraphConfig(BaseModel):
    """Конфигурация графа should-split.

    Attributes:
        verbose: Включить ли подробные логи стадий.
        use_rag: Использовать retrieval-примеры при формировании LLM-промптов.
        enable_categorization: Запускать ли categorize-узел после rag_split.
        enable_drafts: Переходить ли в draft-ветку при shouldSplit=True.
        draft_stub_mode: Режим заглушки для draft-узла.
    """

    verbose: bool = True
    use_rag: bool = True
    enable_categorization: bool = True
    enable_drafts: bool = True
    draft_stub_mode: Literal["metadata_only", "raise", "generate"] = "generate"


class DraftGenerationConfig(BaseModel):
    """Конфигурация генерации черновиков объявлений.

    Attributes:
        enabled: Включена ли генерация черновиков.
        temperature: Температура генерации для draft-узла.
        reasoning_effort: Уровень reasoning_effort для draft-запросов.
    """

    enabled: bool = True
    temperature: float = 0.2
    reasoning_effort: str = "medium"

    @field_validator("temperature")
    @classmethod
    def validate_temperature_draft(cls, value: float) -> float:
        return _validate_non_negative_float(value, "temperature")


class ShouldSplitGraphConfig(BaseModel):
    """Корневой конфиг пайплайна should-split графа.

    Attributes:
        config_path: YAML путь для загрузки конфига.
        data: Пути к данным.
        encoder: Конфиг энкодера.
        vector_db: Конфиг retrieval/векторной БД.
        mistral: Конфиг Mistral.
        graph: Конфиг исполнения графа.
        drafts: Конфиг генерации черновиков.
    """

    model_config = ConfigDict(frozen=True)

    config_path: str | Path | None = Field(default=None, exclude=True)
    data: DataConfig = Field(default_factory=DataConfig)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    vector_db: VectorDbConfig = Field(default_factory=VectorDbConfig)
    mistral: MistralInferenceConfig = Field(default_factory=MistralInferenceConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    drafts: DraftGenerationConfig = Field(default_factory=DraftGenerationConfig)

    @model_validator(mode="before")
    @classmethod
    def load_yaml_config(cls, value: Any) -> Any:
        if isinstance(value, (str, Path)):
            value = {"config_path": value}

        if not isinstance(value, dict):
            return value

        config_path: str | Path | None = value.get("config_path")
        if config_path is None:
            return value

        yaml_data = read_yaml(config_path)
        if yaml_data is None:
            raise ValueError(f"Конфиг-файл пустой: {config_path}")
        if not isinstance(yaml_data, dict):
            raise ValueError(f"Ожидается YAML-словарь в корне: {config_path}")

        explicit_values = {
            key: item
            for key, item in value.items()
            if key != "config_path" and item is not None
        }
        return {**yaml_data, **explicit_values}

    @classmethod
    def from_default_yaml(cls) -> ShouldSplitGraphConfig:
        """Создает конфиг из avito/configs/should_split_graph.yaml или дефолтов."""
        if DEFAULT_SHOULD_SPLIT_GRAPH_CONFIG_FPATH.exists():
            return cls.model_validate({"config_path": DEFAULT_SHOULD_SPLIT_GRAPH_CONFIG_FPATH})
        return cls()
