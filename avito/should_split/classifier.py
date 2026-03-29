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
from avito.should_split.features import ShouldSplitFeatureConfig, TextEncoderLike, build_training_matrix
from common.files import read_yaml
from common.logger import (
    AVITO_SHOULD_SPLIT_LOGGER as logger,
    fit_model_with_progress,
    log_block_separator,
    predict_with_clean_warnings,
)


@dataclass(frozen=True)
class ShouldSplitModelConfigPaths:
    """Пути к конфигам архитектур shouldSplit.

    Attributes:
        logistic_regression: Путь к конфигу Logistic Regression.
        random_forest: Путь к конфигу Random Forest.
        hist_gradient_boosting: Путь к конфигу Hist Gradient Boosting.
        catboost: Путь к конфигу CatBoost.
        xgboost: Путь к конфигу XGBoost.
        lightgbm: Путь к конфигу LightGBM.
    """

    logistic_regression: Path
    random_forest: Path
    hist_gradient_boosting: Path
    catboost: Path
    xgboost: Path
    lightgbm: Path


def get_should_split_model_config_paths() -> ShouldSplitModelConfigPaths:
    """Возвращает абсолютные пути к конфигам архитектур shouldSplit."""
    config_dir = Path(__file__).resolve().parent.parent / "configs" / "should_split"
    return ShouldSplitModelConfigPaths(
        logistic_regression=config_dir / "logistic_regression.yaml",
        random_forest=config_dir / "random_forest.yaml",
        hist_gradient_boosting=config_dir / "hist_gradient_boosting.yaml",
        catboost=config_dir / "catboost.yaml",
        xgboost=config_dir / "xgboost.yaml",
        lightgbm=config_dir / "lightgbm.yaml",
    )


class TuneParamConfig(BaseModel):
    """Конфигурация одного тюнимого параметра для Optuna.

    Attributes:
        type: Тип параметра для suggest-функций.
        low: Нижняя граница для числовых параметров.
        high: Верхняя граница для числовых параметров.
        choices: Варианты для категориальных параметров.
        log: Использовать ли логарифмическую шкалу.
        step: Шаг для числовых параметров.
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
    """Конфиг одной кандидатной архитектуры.

    Attributes:
        enabled: Участвует ли архитектура в сравнении.
        params: Базовые параметры оценщика.
        for_tune: Параметры, доступные для тюнинга через Optuna.
    """

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    for_tune: dict[str, TuneParamConfig] = Field(default_factory=dict)


class TrainedShouldSplitModel(BaseModel):
    """Результат обучения и выбора модели shouldSplit.

    Attributes:
        model_name: Название выбранной архитектуры.
        pipeline: Обученный sklearn pipeline.
        gt_should_split_ratio: Целевая доля положительного класса.
        model_should_split_ratio: Доля положительного класса в предсказаниях на валидации.
        ratio_delta: Разница gt_should_split_ratio - model_should_split_ratio.
        ratio_abs_delta: Абсолютная разница, используемая как метрика.
        tuned_params: Лучшие параметры после Optuna-тюнинга.
        model_comparison_records: Записи метрик по всем архитектурам.
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
            raise ValueError(f"Пустой конфиг архитектуры: {config_path}")
        if not isinstance(raw_data, dict):
            raise ValueError(f"Конфиг архитектуры должен быть словарем: {config_path}")
        configs[architecture_name] = ModelArchitectureConfig.model_validate(raw_data)

    if not any(config.enabled for config in configs.values()):
        raise ValueError("Хотя бы одна архитектура должна быть включена в model-конфигах")

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
        effective_params.setdefault("verbose", 100)  # Логировать каждые 100 итераций
        effective_params.setdefault("loss_function", "Logloss")
        return CatBoostClassifier(**effective_params)

    if architecture_name == "xgboost":
        effective_params.setdefault("random_state", random_state)
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("eval_metric", "logloss")
        effective_params.setdefault("verbosity", 1)  # Логировать прогресс обучения
        return XGBClassifier(**effective_params)

    if architecture_name == "lightgbm":
        effective_params.setdefault("random_state", random_state)
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("verbosity", 0)  # Логировать прогресс обучения
        return LGBMClassifier(**effective_params)

    raise ValueError(f"Неизвестная архитектура: {architecture_name}")


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
                raise ValueError(f"Для категориального параметра нужно задать choices: {param_name}")
            choices_tuple: tuple[str | int | float | bool, ...] = tuple(choices)
            sampled_params[param_name] = trial.suggest_categorical(trial_param_name, choices_tuple)
            continue

        if tune_param.low is None or tune_param.high is None:
            raise ValueError(f"Для числового параметра нужно задать low/high: {param_name}")

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
        raise ValueError("Ожидаются непустые группы split: train/test/val")

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
    include_embeddings: bool = True,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
    training_config: ShouldSplitTrainingConfig | None = None,
) -> TrainedShouldSplitModel:
    """Обучает, сравнивает и тюнит архитектуры shouldSplit.

    Args:
        df: Входной датасет с таргетом и split-колонкой.
        include_embeddings: Добавлять ли эмбеддинговые признаки.
        encoder: Энкодер для генерации эмбеддингов.
        feature_config: Конфиг извлечения признаков.
        training_config: Общий конфиг процесса обучения.

    Returns:
        Pydantic-результат с лучшей архитектурой и значениями ratio-метрики.
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
    embedding_feature_count = int(sum(1 for column in X.columns if str(column).startswith("embedding_")))
    enabled_models_count = sum(1 for cfg in architecture_configs.values() if cfg.enabled)
    log_block_separator(logger)
    logger.info("Старт сравнения архитектур shouldSplit")
    logger.info(f"Кандидатов в сравнении: {enabled_models_count}")
    logger.info(f"Целевая метрика: {effective_training_config.objective_metric}")
    logger.info(f"Эмбеддинговых признаков в матрице: {embedding_feature_count}")
    log_block_separator(logger)

    model_rows: list[dict[str, float | str]] = []
    best_model_name: str | None = None
    best_base_params: dict[str, Any] | None = None
    best_ratio_abs_delta = float("inf")

    enabled_items = [
        (name, cfg)
        for name, cfg in architecture_configs.items()
        if cfg.enabled
    ]

    for architecture_index, (architecture_name, architecture_config) in enumerate(enabled_items, start=1):
        log_block_separator(logger)
        logger.info(f"[{architecture_index}/{len(enabled_items)}] Архитектура: {architecture_name}")
        logger.info(f"Базовые параметры: {architecture_config.params}")

        estimator = _make_estimator(
            architecture_name,
            params=architecture_config.params,
            random_state=effective_training_config.random_state,
        )
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        logger.info(f"Обучение началось...")
        fit_model_with_progress(pipeline, X_fit, y_fit, X_val)
        logger.info(f"Обучение завершено.")

        val_pred = predict_with_clean_warnings(pipeline, X_val)
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

        logger.info("Результат на валидации:")
        logger.info(f"  gt_should_split_ratio    = {GT_SHOULD_SPLIT_RATIO:.4f}")
        logger.info(f"  model_should_split_ratio = {model_ratio:.4f}")
        logger.info(f"  ratio_delta              = {ratio_delta:.6f}")
        logger.info(f"  ratio_abs_delta          = {ratio_abs_delta:.6f}")

    if best_model_name is None or best_base_params is None:
        raise RuntimeError("Не удалось выбрать лучшую архитектуру shouldSplit")

    logger.info("Итог базового сравнения:")
    logger.info(
        f"Лучшая архитектура до тюнинга: {best_model_name}; "
        f"ratio_abs_delta={best_ratio_abs_delta:.6f}"
    )

    best_arch_config = architecture_configs[best_model_name]

    tuned_params: dict[str, Any] = {}
    if best_arch_config.for_tune:
        log_block_separator(logger)
        logger.info("Запуск Optuna-тюнинга")
        logger.info(f"Архитектура: {best_model_name}")
        logger.info(f"n_trials: {effective_training_config.optuna_n_trials}")
        logger.info(f"timeout: {effective_training_config.optuna_timeout_sec}")
        log_block_separator(logger)

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
            try:
                fit_model_with_progress(pipeline, X_fit, y_fit, X_val)
            except Exception as fit_error:
                logger.warning(f"Ошибка обучения во время trial: {fit_error}")
                return float("inf")
            val_pred = predict_with_clean_warnings(pipeline, X_val)
            _, _, ratio_abs_delta = _compute_ratio_metrics(val_pred)
            return ratio_abs_delta

        def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            logger.info(
                "Optuna trial завершен: "
                f"trial={trial.number}, value={trial.value:.6f}, best={study.best_value:.6f}"
            )
            logger.info(f"  params={trial.params}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize")
        study.optimize(
            objective,
            n_trials=effective_training_config.optuna_n_trials,
            timeout=effective_training_config.optuna_timeout_sec,
            show_progress_bar=False,
            callbacks=[_trial_callback],
        )

        tuned_params = {
            param_name: value
            for prefixed_name, value in study.best_trial.params.items()
            for param_name in [prefixed_name.replace(f"{best_model_name}__", "", 1)]
        }
        log_block_separator(logger)
        logger.info(f"Лучшие параметры после Optuna для {best_model_name}:")
        logger.info(f"  tuned_params={tuned_params}")
        logger.info(f"  best_value={study.best_value:.6f}")
        log_block_separator(logger)
    else:
        logger.info(f"Для архитектуры {best_model_name} не задан for_tune, тюнинг пропущен")

    final_params = {**best_base_params, **tuned_params}
    final_estimator = _make_estimator(
        best_model_name,
        params=final_params,
        random_state=effective_training_config.random_state,
    )
    final_pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", final_estimator)])
    logger.info(f"Финальное обучение модели {best_model_name} на объединенных train+test...")
    fit_model_with_progress(final_pipeline, X_fit, y_fit, X_val)
    logger.info(f"Финальное обучение завершено.")

    final_val_pred = predict_with_clean_warnings(final_pipeline, X_val)
    final_model_ratio, final_ratio_delta, final_ratio_abs_delta = _compute_ratio_metrics(final_val_pred)

    log_block_separator(logger)
    logger.info(f"Финальная модель: {best_model_name}")
    logger.info(f"  gt_should_split_ratio    = {GT_SHOULD_SPLIT_RATIO:.4f}")
    logger.info(f"  model_should_split_ratio = {final_model_ratio:.4f}")
    logger.info(f"  ratio_delta              = {final_ratio_delta:.6f}")
    logger.info(f"  ratio_abs_delta          = {final_ratio_abs_delta:.6f}")
    log_block_separator(logger)

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

    logger.info(f"Рейтинг архитектур по {effective_training_config.objective_metric}:")
    for rank, row in enumerate(comparison_records, start=1):
        logger.info(
            f"  #{rank} {row['model']}: "
            f"ratio_abs_delta={float(row['ratio_abs_delta']):.6f}, "
            f"model_ratio={float(row['model_should_split_ratio']):.4f}, "
            f"ratio_delta={float(row['ratio_delta']):.6f}"
        )
    log_block_separator(logger)

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
