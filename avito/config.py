from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.files import read_yaml


DEFAULT_AVITO_CONFIG_FPATH = Path(__file__).resolve().parent / "config.yaml"


class ShouldSplitTrainingConfig(BaseModel):
    """General process configuration for shouldSplit training.

    Attributes:
        random_state: Random seed for reproducible training.
        categorical_features: Categorical feature names.
        numeric_imputer_fill_value: Fill value for numeric imputer.
        merge_train_test_for_fit: Whether to fit models on merged train+test split.
        optuna_n_trials: Number of Optuna trials for best-architecture tuning.
        optuna_timeout_sec: Optional timeout for Optuna tuning in seconds.
        objective_metric: Name of optimization metric.
    """

    random_state: int = 42
    categorical_features: list[str] = Field(default_factory=lambda: ["source_mc_id"])
    numeric_imputer_fill_value: float = 0.0
    merge_train_test_for_fit: bool = True
    optuna_n_trials: int = 25
    optuna_timeout_sec: int | None = None
    objective_metric: str = "ratio_abs_delta"

    @field_validator("random_state")
    @classmethod
    def validate_random_state(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_state must be non-negative")
        return value

    @field_validator("optuna_n_trials")
    @classmethod
    def validate_optuna_n_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("optuna_n_trials must be positive")
        return value

    @field_validator("objective_metric")
    @classmethod
    def validate_objective_metric(cls, value: str) -> str:
        if value.strip() != "ratio_abs_delta":
            raise ValueError("Only ratio_abs_delta objective metric is supported")
        return value


class ShouldSplitCliDefaultsConfig(BaseModel):
    """Default file names for train CLI artifacts.

    Attributes:
        dataset_filename: Dataset CSV file name in avito/data.
        artifact_filename: Model artifact file name in checkpoints.
        report_filename: Metrics report file name in checkpoints.
    """

    dataset_filename: str = "rnc_dataset.csv"
    artifact_filename: str = "avito_should_split_model.joblib"
    report_filename: str = "avito_should_split_report.json"


class ShouldSplitCaseConfig(BaseModel):
    """Root config block for shouldSplit case settings.

    Attributes:
        include_extra_text_features: Whether to compute extra text features.
        training: Training pipeline hyperparameters and model settings.
        cli_defaults: Default file names for CLI mode.
    """

    include_extra_text_features: bool = True
    training: ShouldSplitTrainingConfig = Field(default_factory=ShouldSplitTrainingConfig)
    cli_defaults: ShouldSplitCliDefaultsConfig = Field(default_factory=ShouldSplitCliDefaultsConfig)


class AvitoCaseConfig(BaseModel):
    """Top-level Avito case configuration.

    Attributes:
        config_path: Optional source path for YAML loading.
        should_split: shouldSplit subsystem config.
    """

    model_config = ConfigDict(frozen=True)

    config_path: str | Path | None = Field(default=None, exclude=True)
    should_split: ShouldSplitCaseConfig = Field(default_factory=ShouldSplitCaseConfig)

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
            raise ValueError(f"Config file is empty: {config_path}")
        if not isinstance(yaml_data, dict):
            raise ValueError(f"Expected YAML mapping at root: {config_path}")

        explicit_values = {
            key: item
            for key, item in value.items()
            if key != "config_path" and item is not None
        }
        merged_values = {**yaml_data, **explicit_values}
        return merged_values

    @classmethod
    def from_default_yaml(cls) -> AvitoCaseConfig:
        """Create config from default avito/config.yaml file.

        Returns:
            Parsed and validated case configuration.
        """
        return cls.model_validate({"config_path": DEFAULT_AVITO_CONFIG_FPATH})