from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.files import read_yaml


DEFAULT_AVITO_CONFIG_FPATH = Path(__file__).resolve().parent / "config.yaml"


class ShouldSplitTrainingConfig(BaseModel):
    """Общая конфигурация процесса обучения shouldSplit.

    Attributes:
        random_state: Seed для воспроизводимого обучения.
        categorical_features: Имена категориальных признаков.
        numeric_imputer_fill_value: Значение заполнения для числового imputer.
        merge_train_test_for_fit: Обучать ли модели на объединении train+test.
        optuna_n_trials: Число trial в Optuna для тюнинга лучшей архитектуры.
        optuna_timeout_sec: Ограничение времени Optuna-тюнинга в секундах.
        objective_metric: Название оптимизируемой метрики.
    """

    random_state: int = 42
    categorical_features: list[str] = Field(default_factory=lambda: ["source_mc_id", "case_type"])
    numeric_imputer_fill_value: float = 0.0
    merge_train_test_for_fit: bool = True
    optuna_n_trials: int = 25
    optuna_timeout_sec: int | None = None
    objective_metric: str = "ratio_abs_delta"

    @field_validator("random_state")
    @classmethod
    def validate_random_state(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_state должен быть неотрицательным")
        return value

    @field_validator("optuna_n_trials")
    @classmethod
    def validate_optuna_n_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("optuna_n_trials должен быть положительным")
        return value

    @field_validator("objective_metric")
    @classmethod
    def validate_objective_metric(cls, value: str) -> str:
        if value.strip() != "ratio_abs_delta":
            raise ValueError("Поддерживается только метрика ratio_abs_delta")
        return value


class ShouldSplitCliDefaultsConfig(BaseModel):
    """Имена файлов по умолчанию для CLI-режима обучения.

    Attributes:
        dataset_filename: Имя CSV-датасета в avito/data.
        artifact_filename: Имя файла артефакта модели в checkpoints.
        report_filename: Имя файла JSON-отчета в checkpoints.
    """

    dataset_filename: str = "rnc_dataset.csv"
    artifact_filename: str = "avito_should_split_model.joblib"
    report_filename: str = "avito_should_split_report.json"


class ShouldSplitCaseConfig(BaseModel):
    """Корневой конфиг-блок для shouldSplit.

    Attributes:
        include_extra_text_features: Считать ли дополнительные текстовые признаки.
        training: Настройки процесса обучения.
        cli_defaults: Имена файлов по умолчанию для CLI.
    """

    include_extra_text_features: bool = True
    training: ShouldSplitTrainingConfig = Field(default_factory=ShouldSplitTrainingConfig)
    cli_defaults: ShouldSplitCliDefaultsConfig = Field(default_factory=ShouldSplitCliDefaultsConfig)


class MicrocategoriesMistralInferenceConfig(BaseModel):
    """Конфигурация Mistral-бэкенда для инференса микрокатегорий."""

    enabled: bool = False
    models_list: list[str] = Field(default_factory=lambda: ["mistral-small-latest"])
    timeout: int = 120
    max_attempts_per_call: int = 2
    temperature: float = 0.0
    top_p: float = 1.0

    @field_validator("models_list")
    @classmethod
    def validate_models_list(cls, value: list[str]) -> list[str]:
        normalized = [model.strip() for model in value if model.strip()]
        if not normalized:
            raise ValueError("models_list не может быть пустым")
        return normalized

    @field_validator("timeout")
    @classmethod
    def validate_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("timeout должен быть положительным")
        return value

    @field_validator("max_attempts_per_call")
    @classmethod
    def validate_max_attempts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_attempts_per_call должен быть положительным")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("temperature должна быть неотрицательной")
        return float(value)

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float) -> float:
        if not 0.0 < value <= 1.0:
            raise ValueError("top_p должен быть в диапазоне (0, 1]")
        return float(value)


class MicrocategoriesCaseConfig(BaseModel):
    """Корневой конфиг-блок для подсистемы микрокатегорий."""

    mistral_inference: MicrocategoriesMistralInferenceConfig = Field(
        default_factory=MicrocategoriesMistralInferenceConfig
    )


class AvitoCaseConfig(BaseModel):
    """Верхнеуровневая конфигурация кейса Avito.

    Attributes:
        config_path: Опциональный путь к YAML-конфигу.
        should_split: Конфиг подсистемы shouldSplit.
        microcategories: Конфиг подсистемы микрокатегорий.
    """

    model_config = ConfigDict(frozen=True)

    config_path: str | Path | None = Field(default=None, exclude=True)
    should_split: ShouldSplitCaseConfig = Field(default_factory=ShouldSplitCaseConfig)
    microcategories: MicrocategoriesCaseConfig = Field(default_factory=MicrocategoriesCaseConfig)

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
        merged_values = {**yaml_data, **explicit_values}
        return merged_values

    @classmethod
    def from_default_yaml(cls) -> AvitoCaseConfig:
        """Создает конфиг с дефолтными значениями.

        Если файл avito/config.yaml существует, загружает его.
        Если нет — использует встроенные дефолты.

        Returns:
            Распарсенная и провалидированная конфигурация кейса.
        """
        if DEFAULT_AVITO_CONFIG_FPATH.exists():
            return cls.model_validate({"config_path": DEFAULT_AVITO_CONFIG_FPATH})
        return cls()