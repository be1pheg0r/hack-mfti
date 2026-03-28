from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from avito.config import ShouldSplitTrainingConfig
from avito.constants import GT_SHOULD_SPLIT_RATIO
from avito.features import ShouldSplitFeatureConfig, TextEncoderLike, build_training_matrix
from common.files import read_yaml


@dataclass(frozen=True)
class ShouldSplitModelConfigPaths:
    """Paths to per-architecture model configs.

    Attributes:
        logistic_regression: Path to Logistic Regression config.
        random_forest: Path to Random Forest config.
        hist_gradient_boosting: Path to Hist Gradient Boosting config.
        catboost: Path to CatBoost config.
        xgboost: Path to XGBoost config.
        lightgbm: Path to LightGBM config.
    """

    logistic_regression: Path
    random_forest: Path
    hist_gradient_boosting: Path
    catboost: Path
    xgboost: Path
    lightgbm: Path


def get_should_split_model_config_paths() -> ShouldSplitModelConfigPaths:
    """Return absolute paths for per-architecture shouldSplit configs."""
    config_dir = Path(__file__).resolve().parent / "configs" / "should_split"
    return ShouldSplitModelConfigPaths(
        logistic_regression=config_dir / "logistic_regression.yaml",
        random_forest=config_dir / "random_forest.yaml",
        hist_gradient_boosting=config_dir / "hist_gradient_boosting.yaml",
        catboost=config_dir / "catboost.yaml",
        xgboost=config_dir / "xgboost.yaml",
        lightgbm=config_dir / "lightgbm.yaml",
    )


class TuneParamConfig(BaseModel):
    """Single tunable parameter config for Optuna.

    Attributes:
        type: Parameter type for suggestion.
        low: Lower bound for numeric params.
        high: Upper bound for numeric params.
        choices: Categorical options.
        log: Whether to use log scale for numeric suggestions.
        step: Optional step for numeric suggestions.
    """

    type: Literal["int", "float", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    choices: list[str | int | float | bool] | None = None
    log: bool = False
    step: float | int | None = None

    @field_validator("choices")
    @classmethod
    def validate_choices(
        cls,
        value: list[str | int | float | bool] | None,
        info: Any,
    ) -> list[str | int | float | bool] | None:
        if info.data.get("type") == "categorical":
            if value is None or len(value) == 0:
                raise ValueError("choices must be provided for categorical tune param")
        return value


class ModelArchitectureConfig(BaseModel):
    """Config for one candidate architecture.

    Attributes:
        enabled: Whether architecture participates in model selection.
        params: Base estimator parameters.
        for_tune: Tunable params for Optuna.
    """

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    for_tune: dict[str, TuneParamConfig] = Field(default_factory=dict)


class TrainedShouldSplitModel(BaseModel):
    """Training result for shouldSplit model selection.

    Attributes:
        model_name: Name of selected architecture.
        pipeline: Fitted sklearn pipeline.
        gt_should_split_ratio: Ground-truth positive ratio used for optimization.
        model_should_split_ratio: Predicted positive ratio on validation split.
        ratio_delta: Signed difference gt_should_split_ratio - model_should_split_ratio.
        ratio_abs_delta: Absolute difference for optimization.
        tuned_params: Params sampled by Optuna for best architecture.
        model_comparison_records: Validation metric records for all architectures.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    pipeline: Pipeline
    gt_should_split_ratio: float
    model_should_split_ratio: float
    ratio_delta: float
    ratio_abs_delta: float
    tuned_params: dict[str, Any] = Field(default_factory=dict)
    model_comparison_records: list[dict[str, float | str]] = Field(default_factory=list)


def _build_preprocessor(feature_frame: pd.DataFrame, config: ShouldSplitTrainingConfig) -> ColumnTransformer:
    categorical_features = config.categorical_features
    numeric_features = [column for column in feature_frame.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=config.numeric_imputer_fill_value),
            ),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def _build_architecture_map(
    paths: ShouldSplitModelConfigPaths,
) -> dict[str, Path]:
    return {
        "logistic_regression": paths.logistic_regression,
        "random_forest": paths.random_forest,
        "hist_gradient_boosting": paths.hist_gradient_boosting,
        "catboost": paths.catboost,
        "xgboost": paths.xgboost,
        "lightgbm": paths.lightgbm,
    }


def _load_architecture_configs() -> dict[str, ModelArchitectureConfig]:
    paths = get_should_split_model_config_paths()
    architecture_files = _build_architecture_map(paths)

    configs: dict[str, ModelArchitectureConfig] = {}
    for architecture_name, config_path in architecture_files.items():
        raw_data = read_yaml(config_path)
        if raw_data is None:
            raise ValueError(f"Architecture config is empty: {config_path}")
        if not isinstance(raw_data, dict):
            raise ValueError(f"Architecture config must be mapping: {config_path}")
        configs[architecture_name] = ModelArchitectureConfig.model_validate(raw_data)

    if not any(config.enabled for config in configs.values()):
        raise ValueError("At least one architecture must be enabled in model config files")

    return configs


def _make_estimator(
    architecture_name: str,
    *,
    params: dict[str, Any],
    random_state: int,
) -> Any:
    effective_params = dict(params)

    if architecture_name == "logistic_regression":
        effective_params.setdefault("random_state", random_state)
        return LogisticRegression(**effective_params)

    if architecture_name == "random_forest":
        effective_params.setdefault("random_state", random_state)
        effective_params.setdefault("n_jobs", -1)
        return RandomForestClassifier(**effective_params)

    if architecture_name == "hist_gradient_boosting":
        effective_params.setdefault("random_state", random_state)
        return HistGradientBoostingClassifier(**effective_params)

    if architecture_name == "catboost":
        effective_params.setdefault("random_seed", random_state)
        effective_params.setdefault("verbose", False)
        effective_params.setdefault("loss_function", "Logloss")
        return CatBoostClassifier(**effective_params)

    if architecture_name == "xgboost":
        effective_params.setdefault("random_state", random_state)
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("eval_metric", "logloss")
        return XGBClassifier(**effective_params)

    if architecture_name == "lightgbm":
        effective_params.setdefault("random_state", random_state)
        effective_params.setdefault("n_jobs", -1)
        return LGBMClassifier(**effective_params)

    raise ValueError(f"Unknown architecture: {architecture_name}")


def _sample_tune_params(
    trial: optuna.trial.BaseTrial,
    architecture_name: str,
    tune_config: dict[str, TuneParamConfig],
) -> dict[str, Any]:
    sampled_params: dict[str, Any] = {}

    for param_name, tune_param in tune_config.items():
        trial_param_name = f"{architecture_name}__{param_name}"

        if tune_param.type == "categorical":
            choices = tune_param.choices
            if choices is None:
                raise ValueError(f"choices must be provided for categorical tune param: {param_name}")
            choices_tuple: tuple[str | int | float | bool, ...] = tuple(choices)
            sampled_params[param_name] = trial.suggest_categorical(trial_param_name, choices_tuple)
            continue

        if tune_param.low is None or tune_param.high is None:
            raise ValueError(f"low/high must be provided for numeric tune param: {param_name}")

        if tune_param.type == "int":
            sampled_params[param_name] = trial.suggest_int(
                trial_param_name,
                int(tune_param.low),
                int(tune_param.high),
                step=int(tune_param.step) if tune_param.step is not None else 1,
                log=tune_param.log,
            )
            continue

        sampled_params[param_name] = trial.suggest_float(
            trial_param_name,
            float(tune_param.low),
            float(tune_param.high),
            step=float(tune_param.step) if tune_param.step is not None else None,
            log=tune_param.log,
        )

    return sampled_params


def _compute_ratio_metrics(y_pred: np.ndarray) -> tuple[float, float, float]:
    y_pred_bool = np.asarray(y_pred, dtype=bool)
    model_ratio = float(np.mean(y_pred_bool))
    ratio_delta = float(GT_SHOULD_SPLIT_RATIO - model_ratio)
    ratio_abs_delta = float(abs(ratio_delta))
    return model_ratio, ratio_delta, ratio_abs_delta


def _split_frame(
    X: pd.DataFrame,
    y: pd.Series,
    split: pd.Series,
    *,
    merge_train_test_for_fit: bool,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_mask = split.eq("train")
    test_mask = split.eq("test")
    val_mask = split.eq("val")

    if not train_mask.any() or not test_mask.any() or not val_mask.any():
        raise ValueError("Expected non-empty split groups train/val/test")

    fit_mask = (train_mask | test_mask) if merge_train_test_for_fit else train_mask
    return (
        X.loc[fit_mask],
        y.loc[fit_mask],
        X.loc[val_mask],
        y.loc[val_mask],
    )


def train_should_split_models(
    df: pd.DataFrame,
    *,
    include_embeddings: bool = False,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
    training_config: ShouldSplitTrainingConfig | None = None,
) -> TrainedShouldSplitModel:
    """Train, compare, and tune shouldSplit architectures.

    Args:
        df: Input dataset with split labels and target.
        include_embeddings: Whether to append embedding features.
        encoder: Encoder for embedding extraction.
        feature_config: Feature extraction configuration.
        training_config: General process configuration.

    Returns:
        Pydantic result with best architecture trained pipeline and ratio metric values.
    """
    effective_training_config = training_config or ShouldSplitTrainingConfig()
    architecture_configs = _load_architecture_configs()

    X, y, split = build_training_matrix(
        df=df,
        include_embeddings=include_embeddings,
        encoder=encoder,
        config=feature_config,
    )
    X_fit, y_fit, X_val, _ = _split_frame(
        X=X,
        y=y,
        split=split,
        merge_train_test_for_fit=effective_training_config.merge_train_test_for_fit,
    )

    preprocessor = _build_preprocessor(X, config=effective_training_config)

    model_rows: list[dict[str, float | str]] = []
    best_model_name: str | None = None
    best_base_params: dict[str, Any] | None = None
    best_ratio_abs_delta = float("inf")

    for architecture_name, architecture_config in architecture_configs.items():
        if not architecture_config.enabled:
            continue

        estimator = _make_estimator(
            architecture_name,
            params=architecture_config.params,
            random_state=effective_training_config.random_state,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_fit, y_fit)

        val_pred = np.asarray(pipeline.predict(X_val))
        model_ratio, ratio_delta, ratio_abs_delta = _compute_ratio_metrics(val_pred)

        model_rows.append(
            {
                "model": architecture_name,
                "gt_should_split_ratio": float(GT_SHOULD_SPLIT_RATIO),
                "model_should_split_ratio": model_ratio,
                "ratio_delta": ratio_delta,
                "ratio_abs_delta": ratio_abs_delta,
            }
        )

        if ratio_abs_delta < best_ratio_abs_delta:
            best_ratio_abs_delta = ratio_abs_delta
            best_model_name = architecture_name
            best_base_params = dict(architecture_config.params)

    if best_model_name is None or best_base_params is None:
        raise RuntimeError("Failed to choose the best shouldSplit architecture")

    best_arch_config = architecture_configs[best_model_name]

    tuned_params: dict[str, Any] = {}
    if best_arch_config.for_tune:
        def objective(trial: optuna.Trial) -> float:
            sampled_params = _sample_tune_params(
                trial=trial,
                architecture_name=best_model_name,
                tune_config=best_arch_config.for_tune,
            )
            merged_params = {**best_base_params, **sampled_params}
            estimator = _make_estimator(
                best_model_name,
                params=merged_params,
                random_state=effective_training_config.random_state,
            )
            pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            pipeline.fit(X_fit, y_fit)
            val_pred = np.asarray(pipeline.predict(X_val))
            _, _, ratio_abs_delta = _compute_ratio_metrics(val_pred)
            return ratio_abs_delta

        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=effective_training_config.optuna_n_trials,
            timeout=effective_training_config.optuna_timeout_sec,
            show_progress_bar=False,
        )

        tuned_params = {
            param_name: value
            for prefixed_name, value in study.best_trial.params.items()
            for param_name in [prefixed_name.replace(f"{best_model_name}__", "", 1)]
        }

    final_params = {**best_base_params, **tuned_params}
    final_estimator = _make_estimator(
        best_model_name,
        params=final_params,
        random_state=effective_training_config.random_state,
    )
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", final_estimator)])
    final_pipeline.fit(X_fit, y_fit)

    final_val_pred = np.asarray(final_pipeline.predict(X_val))
    final_model_ratio, final_ratio_delta, final_ratio_abs_delta = _compute_ratio_metrics(final_val_pred)

    comparison_records_raw = (
        pd.DataFrame(model_rows)
        .sort_values(by=effective_training_config.objective_metric, ascending=True)
        .reset_index(drop=True)
        .to_dict(orient="records")
    )
    comparison_records: list[dict[str, float | str]] = [
        {str(key): value for key, value in row.items()}
        for row in comparison_records_raw
    ]

    return TrainedShouldSplitModel(
        model_name=best_model_name,
        pipeline=final_pipeline,
        gt_should_split_ratio=float(GT_SHOULD_SPLIT_RATIO),
        model_should_split_ratio=final_model_ratio,
        ratio_delta=final_ratio_delta,
        ratio_abs_delta=final_ratio_abs_delta,
        tuned_params=tuned_params,
        model_comparison_records=comparison_records,
    )
