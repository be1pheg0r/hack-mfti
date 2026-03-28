from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from common.files import read_yaml


DEFAULT_AVITO_CONFIG_FPATH = Path(__file__).resolve().parent / "config.yaml"


class LogisticRegressionConfig(BaseModel):
    """Hyperparameters for Logistic Regression.

    Attributes:
        enabled: Whether this model participates in comparison.
        max_iter: Maximum number of optimization iterations.
        class_weight: Class weighting strategy for imbalanced data.
    """

    enabled: bool = True
    max_iter: int = 1200
    class_weight: str = "balanced"

    @field_validator("max_iter")
    @classmethod
    def validate_max_iter(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_iter must be positive")
        return value


class RandomForestConfig(BaseModel):
    """Hyperparameters for Random Forest.

    Attributes:
        enabled: Whether this model participates in comparison.
        n_estimators: Number of trees in the forest.
        class_weight: Class weighting strategy for imbalanced data.
    """

    enabled: bool = True
    n_estimators: int = 400
    class_weight: Literal["balanced", "balanced_subsample"] | None = "balanced_subsample"

    @field_validator("n_estimators")
    @classmethod
    def validate_n_estimators(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_estimators must be positive")
        return value


class HistGradientBoostingConfig(BaseModel):
    """Hyperparameters for HistGradientBoostingClassifier.

    Attributes:
        enabled: Whether this model participates in comparison.
        max_depth: Maximum depth of individual trees.
        max_iter: Number of boosting iterations.
        learning_rate: Boosting learning rate.
    """

    enabled: bool = True
    max_depth: int = 8
    max_iter: int = 300
    learning_rate: float = 0.05

    @field_validator("max_depth", "max_iter")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_depth and max_iter must be positive")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("learning_rate must be positive")
        return value


class CatBoostConfig(BaseModel):
    """Hyperparameters for CatBoostClassifier.

    Attributes:
        enabled: Whether this model participates in comparison.
        iterations: Number of boosting iterations.
        learning_rate: Boosting learning rate.
        depth: Tree depth.
        auto_class_weights: Class weighting mode.
    """

    enabled: bool = True
    iterations: int = 400
    learning_rate: float = 0.05
    depth: int = 6
    auto_class_weights: Literal["Balanced", "SqrtBalanced"] | None = "Balanced"

    @field_validator("iterations", "depth")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("iterations and depth must be positive")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("learning_rate must be positive")
        return value


class ShouldSplitTrainingConfig(BaseModel):
    """Configuration for shouldSplit training pipeline.

    Attributes:
        random_state: Random seed for reproducible training.
        primary_metric: Metric used for best-model selection.
        categorical_features: Categorical feature names.
        numeric_imputer_fill_value: Fill value for numeric imputer.
        logistic_regression: Logistic Regression settings.
        random_forest: Random Forest settings.
        hist_gradient_boosting: Hist Gradient Boosting settings.
        catboost: CatBoost settings.
    """

    random_state: int = 42
    primary_metric: str = "accuracy"
    categorical_features: list[str] = Field(default_factory=lambda: ["source_mc_id"])
    numeric_imputer_fill_value: float = 0.0
    logistic_regression: LogisticRegressionConfig = Field(default_factory=LogisticRegressionConfig)
    random_forest: RandomForestConfig = Field(default_factory=RandomForestConfig)
    hist_gradient_boosting: HistGradientBoostingConfig = Field(default_factory=HistGradientBoostingConfig)
    catboost: CatBoostConfig = Field(default_factory=CatBoostConfig)

    @field_validator("random_state")
    @classmethod
    def validate_random_state(cls, value: int) -> int:
        if value < 0:
            raise ValueError("random_state must be non-negative")
        return value

    @field_validator("primary_metric")
    @classmethod
    def validate_primary_metric(cls, value: str) -> str:
        if value.strip() != "accuracy":
            raise ValueError("Only accuracy is supported as primary_metric for now")
        return value

    @model_validator(mode="after")
    def validate_models_enabled(self) -> ShouldSplitTrainingConfig:
        if not (
            self.logistic_regression.enabled
            or self.random_forest.enabled
            or self.hist_gradient_boosting.enabled
            or self.catboost.enabled
        ):
            raise ValueError("At least one model must be enabled")
        return self


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