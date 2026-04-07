from __future__ import annotations

import json
import pickle
import shutil
import warnings
from datetime import datetime
from pathlib import Path
from typing import *

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rapidfuzz import fuzz
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from common.logger import INFERENCE_LOGGER, SBER_TRAIN_LOGGER
from common.paths import (
    PathLike,
    get_data_bench_dpath,
    get_data_raw_dpath,
    get_project_root,
    get_public_checkpoints_dpath,
)


TRAIN_LOGGER = SBER_TRAIN_LOGGER
INFER_LOGGER = INFERENCE_LOGGER
PROBE_LAYERS: list[int] = [0, 4, 8, 12, 16, 20, 24, 25]


class FeatureGroupFlags(BaseModel):
    """Флаги включения групп признаков.

    Attributes:
        uncertainty: uncertainty-группа.
        internal_scalars: internal scalars-группа.
        probe_vec: probe vector-группа.
        attention_entropy: attention entropy-группа.
        entropy_drops: entropy drops-группа.
        moe_routing: MoE routing-группа.
        text_features: статистические text features.
        tfidf: TF-IDF признаки.
    """

    model_config = ConfigDict(frozen=True)

    uncertainty: bool = True
    internal_scalars: bool = False
    probe_vec: bool = False
    attention_entropy: bool = False
    entropy_drops: bool = False
    moe_routing: bool = False
    text_features: bool = True
    tfidf: bool = True

    @model_validator(mode="after")
    def validate_any_enabled(self) -> FeatureGroupFlags:
        if not any(self.model_dump().values()):
            raise ValueError("At least one feature-group flag must be True")
        return self


class OptunaTuningConfig(BaseModel):
    """Конфигурация небольшого Optuna-тюнинга.

    Attributes:
        enabled: Включить подбор гиперпараметров.
        n_trials: Число trial для подбора параметров модели.
        threshold_trials: Число trial для подбора порога лучшей модели.
        timeout_sec: Опциональный таймаут одного study в секундах.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    n_trials: int = 15
    threshold_trials: int = 30
    timeout_sec: int | None = None

    @field_validator("n_trials", "threshold_trials")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("n_trials и threshold_trials должны быть положительными")
        return value

    @field_validator("timeout_sec")
    @classmethod
    def validate_timeout(cls, value: int | None) -> int | None:
        if value is None:
            return value
        if value <= 0:
            raise ValueError("timeout_sec должен быть положительным")
        return value


class TabularTrainConfig(BaseModel):
    """Конфиг обучения tabular-модели.

    Attributes:
        train_csv: CSV train-выборки.
        val_csv: CSV validation-выборки.
        target_col: Колонка таргета.
        run_name: Имя запуска, если None генерируется автоматически.
        output_root: Корень для сохранения артефактов.
        feature_flags: Флаги групп признаков.
        pca_n_components: Число PCA-компонент для probe_vec.
        tfidf_max_features: Размер словаря TF-IDF.
        tfidf_n_components: Число PCA-компонент для TF-IDF.
        scaling: Применять z-score scaling.
        under_sampling: Применять random undersampling.
        oversampling: Применять SMOTE oversampling.
        bad_features: Список признаков для удаления до обучения.
        plot_feature_distributions: Строить графики распределений.
        feature_plot_batch_size: Размер батча графиков.
        feature_plot_dir: Директория для графиков.
        model_variants: Список вариантов архитектур для сравнения.
        optuna: Настройки небольшого Optuna-подбора.
        ignore_warnings_during_training: Подавлять предупреждения во время fit.
        random_seed: Seed.
        iterations: CatBoost iterations.
        learning_rate: CatBoost learning rate.
        depth: CatBoost depth.
        l2_leaf_reg: CatBoost l2.
        subsample: CatBoost subsample.
        colsample_bylevel: CatBoost colsample.
        min_data_in_leaf: CatBoost min_data_in_leaf.
        early_stopping_rounds: CatBoost early stopping.
    """

    model_config = ConfigDict(frozen=True)

    train_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / "merged_features_with_judge_scores.csv")
    val_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged_mapped.csv")
    target_col: str = "is_hallucination"
    run_name: str | None = None
    output_root: PathLike = Field(default_factory=lambda: Path(get_public_checkpoints_dpath()) / "sber_tabular")
    feature_flags: FeatureGroupFlags = Field(default_factory=FeatureGroupFlags)

    pca_n_components: int | None = 128
    tfidf_max_features: int = 1000
    tfidf_n_components: int | None = 64
    scaling: bool = False
    under_sampling: bool = True
    oversampling: bool = False
    bad_features: list[str] = Field(default_factory=list)

    plot_feature_distributions: bool = True
    feature_plot_batch_size: int = 24
    feature_plot_dir: PathLike | None = None

    model_variants: list[ModelVariantConfig] = Field(
        default_factory=lambda: [
            ModelVariantConfig(name="catboost_base", architecture="catboost"),
            ModelVariantConfig(name="xgboost_base", architecture="xgboost"),
            ModelVariantConfig(name="lightgbm_base", architecture="lightgbm"),
            ModelVariantConfig(name="logreg_base", architecture="logreg"),
        ]
    )
    optuna: OptunaTuningConfig = Field(default_factory=OptunaTuningConfig)
    ignore_warnings_during_training: bool = True

    random_seed: int = 42
    iterations: int = 1000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 1.0
    subsample: float = 0.8
    colsample_bylevel: float = 0.8
    min_data_in_leaf: int = 20
    early_stopping_rounds: int = 50

    @field_validator("target_col")
    @classmethod
    def validate_target_col(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("target_col must not be empty")
        return normalized

    @field_validator("tfidf_max_features", "feature_plot_batch_size", "iterations", "depth", "min_data_in_leaf", "early_stopping_rounds")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Integer parameter must be positive")
        return value

    @field_validator("learning_rate", "l2_leaf_reg", "subsample", "colsample_bylevel")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("Float parameter must be positive")
        return value

    @model_validator(mode="after")
    def validate_sampling(self) -> TabularTrainConfig:
        if self.under_sampling and self.oversampling:
            raise ValueError("under_sampling and oversampling cannot be enabled simultaneously")
        if not any(variant.enabled for variant in self.model_variants):
            raise ValueError("At least one model variant must be enabled")
        return self


class ModelVariantConfig(BaseModel):
    """Описание варианта архитектуры для model selection.

    Attributes:
        name: Человекочитаемый идентификатор варианта.
        architecture: Название архитектуры (`catboost`/`xgboost`/`lightgbm`/`logreg`).
        enabled: Включен ли вариант в текущем запуске.
        params: Дополнительные гиперпараметры модели.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    architecture: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "architecture")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        if not normalized:
            raise ValueError("Model variant fields must not be empty")
        return normalized

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, value: str) -> str:
        allowed: set[str] = {"catboost", "xgboost", "lightgbm", "logreg"}
        if value not in allowed:
            raise ValueError(f"Unsupported architecture: {value}")
        return value


class TabularInferenceConfig(BaseModel):
    """Конфиг инференса tabular-модели.

    Attributes:
        checkpoint_dir: Директория с артефактами обучения.
        input_csv: Входной CSV для скоринга.
        output_csv: Путь для сохранения предсказаний.
        threshold: Опциональный override порога.
    """

    model_config = ConfigDict(frozen=True)

    checkpoint_dir: PathLike
    input_csv: PathLike
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "tabular_inference.csv")
    threshold: float | None = None

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return value


class TrainResult(BaseModel):
    """Результат обучения.

    Attributes:
        checkpoint_dir: Путь к сохраненным артефактам.
        selected_feature_names: Список признаков после preprocessing.
        selected_groups: Выбранные группы признаков.
        best_threshold: Лучший порог по F1 на validation.
        f1: F1 на validation.
        average_precision: AP на validation.
        best_architecture: Лучшая архитектура по average_precision.
        validation_metrics: Набор метрик лучшей модели.
    """

    model_config = ConfigDict(frozen=True)

    checkpoint_dir: str
    selected_feature_names: list[str]
    selected_groups: list[str]
    best_threshold: float
    f1: float
    average_precision: float
    best_architecture: str
    validation_metrics: dict[str, float]


class FeatureSchema:
    """Схема и группировка признаков."""

    uncertainty_map: list[str] = [
        "mean_log_prob",
        "min_log_prob",
        "max_log_prob",
        "std_log_prob",
        "mean_entropy",
        "min_entropy",
        "max_entropy",
        "std_entropy",
        "n_answer_tokens",
        "first_token_log_prob",
        "mean_top1_prob",
        "mean_top5_prob_sum",
    ]

    internal_scalars_map: list[str] = [
        f"layer_{layer}_{suffix}"
        for layer in PROBE_LAYERS
        for suffix in ["prompt_last_token_norm", "answer_hidden_norm_mean", "logit_lens_entropy"]
    ]

    probe_vec_map: list[str] = [f"probe_vec_{index}" for index in range(1536)]

    attention_entropy_map: list[str] = [
        f"attn_layer_{layer}_{suffix}"
        for layer in PROBE_LAYERS
        for suffix in ["entropy_mean", "entropy_max", "entropy_std"]
    ]

    entropy_drops_map: list[str] = [
        f"entropy_drop_{PROBE_LAYERS[index]}_to_{PROBE_LAYERS[index + 1]}"
        for index in range(len(PROBE_LAYERS) - 1)
    ]

    moe_routing_map: list[str] = [
        "moe_mean_top_prob_mean",
        "moe_mean_top_prob_std",
        "moe_std_top_prob_mean",
        "moe_std_top_prob_std",
        "moe_mean_entropy_mean",
        "moe_mean_entropy_std",
        "moe_std_entropy_mean",
        "moe_std_entropy_std",
        "moe_active_ratio_mean",
        "moe_active_ratio_std",
    ]

    text_features_map: list[str] = [
        "query_token_count",
        "answer_token_count",
        "query_stopword_count",
        "answer_stopword_count",
        "query_digit_count",
        "answer_digit_count",
        "query_punctuation_count",
        "answer_punctuation_count",
        "query_answer_token_diff",
        "query_answer_fuzz_partial_ratio",
    ]

    @classmethod
    def build_groups(cls, tfidf_feature_names: list[str]) -> dict[str, list[str]]:
        return {
            "uncertainty": cls.uncertainty_map,
            "internal_scalars": cls.internal_scalars_map,
            "probe_vec": cls.probe_vec_map,
            "attention_entropy": cls.attention_entropy_map,
            "entropy_drops": cls.entropy_drops_map,
            "moe_routing": cls.moe_routing_map,
            "text_features": cls.text_features_map,
            "tfidf": tfidf_feature_names,
        }


class TabularPreprocessor:
    """Пайплайн подготовки табличных признаков для train/inference."""

    def __init__(self, config: TabularTrainConfig) -> None:
        self.config: TabularTrainConfig = config
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.pca_probe: PCA | None = None
        self.pca_tfidf: PCA | None = None
        self.scaler: Any = None
        self.tfidf_feature_names: list[str] = []
        self.selected_feature_names: list[str] = []
        self.selected_groups: list[str] = []

    def _count_tokens(self, text: str) -> int:
        return len(str(text).split())

    def _count_stopwords(self, text: str) -> int:
        stop_words: set[str] = {
            "и",
            "в",
            "во",
            "не",
            "что",
            "он",
            "на",
            "я",
            "с",
            "со",
            "как",
            "а",
            "то",
            "все",
            "она",
        }
        return sum(1 for token in str(text).lower().split() if token in stop_words)

    def _add_text_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result: pd.DataFrame = dataframe.copy()
        if "query" not in result.columns or "model_answer" not in result.columns:
            return result

        result["query_token_count"] = result["query"].astype("string").fillna("").map(self._count_tokens)
        result["answer_token_count"] = result["model_answer"].astype("string").fillna("").map(self._count_tokens)
        result["query_stopword_count"] = result["query"].astype("string").fillna("").map(self._count_stopwords)
        result["answer_stopword_count"] = result["model_answer"].astype("string").fillna("").map(self._count_stopwords)

        result["query_digit_count"] = result["query"].astype("string").fillna("").str.count(r"\d")
        result["answer_digit_count"] = result["model_answer"].astype("string").fillna("").str.count(r"\d")
        result["query_punctuation_count"] = result["query"].astype("string").fillna("").str.count(r"[^\w\s]")
        result["answer_punctuation_count"] = result["model_answer"].astype("string").fillna("").str.count(r"[^\w\s]")
        result["query_answer_token_diff"] = result["query_token_count"] - result["answer_token_count"]

        result["query_answer_fuzz_partial_ratio"] = result.apply(
            lambda row: float(
                fuzz.partial_ratio(
                    str(row["query"]).lower(),
                    str(row["model_answer"]).lower(),
                )
            ),
            axis=1,
        )
        return result

    def _fit_tfidf(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.feature_flags.tfidf:
            return train_df, val_df
        if "model_answer" not in train_df.columns or "model_answer" not in val_df.columns:
            return train_df, val_df

        self.tfidf_vectorizer = TfidfVectorizer(max_features=self.config.tfidf_max_features)
        train_tfidf = self.tfidf_vectorizer.fit_transform(train_df["model_answer"].astype("string").fillna(""))
        val_tfidf = self.tfidf_vectorizer.transform(val_df["model_answer"].astype("string").fillna(""))

        self.tfidf_feature_names = [f"tfidf_{index}" for index in range(train_tfidf.shape[1])]
        train_tfidf_df = pd.DataFrame(train_tfidf.toarray(), columns=self.tfidf_feature_names, index=train_df.index)
        val_tfidf_df = pd.DataFrame(val_tfidf.toarray(), columns=self.tfidf_feature_names, index=val_df.index)
        return pd.concat([train_df, train_tfidf_df], axis=1), pd.concat([val_df, val_tfidf_df], axis=1)

    def _apply_tfidf(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.tfidf_vectorizer is None or not self.tfidf_feature_names:
            return dataframe
        if "model_answer" not in dataframe.columns:
            return dataframe
        transformed = self.tfidf_vectorizer.transform(dataframe["model_answer"].astype("string").fillna(""))
        tfidf_df = pd.DataFrame(transformed.toarray(), columns=self.tfidf_feature_names, index=dataframe.index)
        return pd.concat([dataframe, tfidf_df], axis=1)

    def _coerce_numeric_columns(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        result: pd.DataFrame = dataframe.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns.difference(["sample_id"])
        for column_name in numeric_cols:
            result[column_name] = pd.to_numeric(result[column_name], errors="coerce")
        return result

    def _drop_bad_features(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not self.config.bad_features:
            return train_df, val_df
        train_drop = [name for name in self.config.bad_features if name in train_df.columns]
        val_drop = [name for name in self.config.bad_features if name in val_df.columns]
        if train_drop:
            train_df = train_df.drop(columns=train_drop)
        if val_drop:
            val_df = val_df.drop(columns=val_drop)
        return train_df, val_df

    def _resolve_selected_features(self, common_features: set[str]) -> list[str]:
        groups = FeatureSchema.build_groups(tfidf_feature_names=self.tfidf_feature_names)
        selected_groups = [
            group_name
            for group_name, enabled in self.config.feature_flags.model_dump().items()
            if enabled and group_name in groups
        ]
        selected_features: list[str] = []
        for group_name in selected_groups:
            selected_features.extend([feature for feature in groups[group_name] if feature in common_features])
        if not selected_features:
            raise ValueError("No selected features were found in both train and validation datasets")
        self.selected_groups = selected_groups
        return selected_features

    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
        prepared_train: pd.DataFrame = self._add_text_features(train_df)
        prepared_val: pd.DataFrame = self._add_text_features(val_df)
        prepared_train, prepared_val = self._fit_tfidf(prepared_train, prepared_val)
        prepared_train, prepared_val = self._drop_bad_features(prepared_train, prepared_val)
        prepared_train = self._coerce_numeric_columns(prepared_train)
        prepared_val = self._coerce_numeric_columns(prepared_val)

        common_cols: set[str] = set(prepared_train.columns).intersection(set(prepared_val.columns))
        self.selected_feature_names = self._resolve_selected_features(common_cols)

        selected_probe = [name for name in FeatureSchema.probe_vec_map if name in self.selected_feature_names]
        if self.config.pca_n_components is not None and selected_probe:
            pca_components: int = min(self.config.pca_n_components, len(selected_probe), len(prepared_train))
            self.pca_probe = PCA(n_components=pca_components, random_state=self.config.random_seed)
            train_probe = self.pca_probe.fit_transform(prepared_train[selected_probe].fillna(0.0).values)
            val_probe = self.pca_probe.transform(prepared_val[selected_probe].fillna(0.0).values)
            pca_names: list[str] = [f"pca_probe_vec_{index}" for index in range(pca_components)]
            prepared_train = prepared_train.drop(columns=selected_probe)
            prepared_val = prepared_val.drop(columns=selected_probe)
            prepared_train[pca_names] = train_probe
            prepared_val[pca_names] = val_probe
            self.selected_feature_names = [name for name in self.selected_feature_names if name not in selected_probe] + pca_names

        tfidf_cols = [name for name in self.tfidf_feature_names if name in self.selected_feature_names]
        if self.config.tfidf_n_components is not None and tfidf_cols:
            pca_components = min(self.config.tfidf_n_components, len(tfidf_cols), len(prepared_train))
            self.pca_tfidf = PCA(n_components=pca_components, random_state=self.config.random_seed)
            train_tfidf = self.pca_tfidf.fit_transform(prepared_train[tfidf_cols].fillna(0.0).values)
            val_tfidf = self.pca_tfidf.transform(prepared_val[tfidf_cols].fillna(0.0).values)
            tfidf_pca_names: list[str] = [f"pca_tfidf_{index}" for index in range(pca_components)]
            prepared_train = prepared_train.drop(columns=tfidf_cols)
            prepared_val = prepared_val.drop(columns=tfidf_cols)
            prepared_train[tfidf_pca_names] = train_tfidf
            prepared_val[tfidf_pca_names] = val_tfidf
            self.selected_feature_names = [name for name in self.selected_feature_names if name not in tfidf_cols] + tfidf_pca_names

        if self.config.scaling:
            from sklearn.preprocessing import StandardScaler

            self.scaler = StandardScaler()
            prepared_train[self.selected_feature_names] = self.scaler.fit_transform(
                prepared_train[self.selected_feature_names].fillna(0.0)
            )
            prepared_val[self.selected_feature_names] = self.scaler.transform(
                prepared_val[self.selected_feature_names].fillna(0.0)
            )

        return prepared_train, prepared_val, self.selected_feature_names

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        prepared: pd.DataFrame = self._add_text_features(dataframe)
        prepared = self._apply_tfidf(prepared)
        prepared = self._coerce_numeric_columns(prepared)

        if self.pca_probe is not None:
            source_cols = [name for name in FeatureSchema.probe_vec_map if name in prepared.columns]
            if source_cols:
                transformed = self.pca_probe.transform(prepared[source_cols].fillna(0.0).values)
                pca_names: list[str] = [f"pca_probe_vec_{index}" for index in range(transformed.shape[1])]
                prepared = prepared.drop(columns=source_cols)
                prepared[pca_names] = transformed

        if self.pca_tfidf is not None and self.tfidf_feature_names:
            source_cols = [name for name in self.tfidf_feature_names if name in prepared.columns]
            if source_cols:
                transformed = self.pca_tfidf.transform(prepared[source_cols].fillna(0.0).values)
                pca_names = [f"pca_tfidf_{index}" for index in range(transformed.shape[1])]
                prepared = prepared.drop(columns=source_cols)
                prepared[pca_names] = transformed

        missing = [name for name in self.selected_feature_names if name not in prepared.columns]
        if missing:
            raise ValueError(f"Missing features for inference: {missing[:10]}")

        if self.scaler is not None:
            prepared[self.selected_feature_names] = self.scaler.transform(prepared[self.selected_feature_names].fillna(0.0))

        return prepared


class TabularHallucinationTrainer:
    """Обучает CatBoost-модель и сохраняет bundle для инференса."""

    def __init__(self, config: TabularTrainConfig) -> None:
        self.config: TabularTrainConfig = config

    def _resolve_checkpoint_dir(self) -> Path:
        run_name: str = self.config.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
        output_root: Path = Path(self.config.output_root).expanduser()
        if not output_root.is_absolute():
            output_root = Path(get_project_root()) / output_root
        checkpoint_dir: Path = output_root / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def _best_threshold(self, y_true: pd.Series, y_probs: np.ndarray) -> tuple[float, float]:
        thresholds = np.linspace(0.0, 1.0, num=101)
        f1_values: list[float] = []
        for threshold in thresholds:
            preds = (y_probs >= threshold).astype(int)
            f1_values.append(float(f1_score(y_true, preds)))
        best_idx: int = int(np.argmax(f1_values))
        return float(thresholds[best_idx]), float(f1_values[best_idx])

    def _fit_model(
        self,
        model: Any,
        variant: ModelVariantConfig,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Any:
        with warnings.catch_warnings():
            if self.config.ignore_warnings_during_training:
                warnings.filterwarnings("ignore")
            if variant.architecture == "catboost":
                model.fit(x_train, y_train, eval_set=(x_val, y_val))
            else:
                model.fit(x_train, y_train)
        return model

    def _sample_optuna_params(self, trial: Any, architecture: str) -> dict[str, Any]:
        if architecture == "catboost":
            return {
                "iterations": int(trial.suggest_int("iterations", 300, 1500)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
                "depth": int(trial.suggest_int("depth", 4, 10)),
                "l2_leaf_reg": float(trial.suggest_float("l2_leaf_reg", 1e-2, 10.0, log=True)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bylevel": float(trial.suggest_float("colsample_bylevel", 0.6, 1.0)),
                "min_data_in_leaf": int(trial.suggest_int("min_data_in_leaf", 5, 100)),
            }
        if architecture == "xgboost":
            return {
                "n_estimators": int(trial.suggest_int("n_estimators", 200, 1200)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
                "max_depth": int(trial.suggest_int("max_depth", 3, 10)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.6, 1.0)),
                "reg_lambda": float(trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)),
            }
        if architecture == "lightgbm":
            return {
                "n_estimators": int(trial.suggest_int("n_estimators", 200, 1200)),
                "learning_rate": float(trial.suggest_float("learning_rate", 0.01, 0.2, log=True)),
                "max_depth": int(trial.suggest_int("max_depth", 3, 12)),
                "num_leaves": int(trial.suggest_int("num_leaves", 16, 128)),
                "subsample": float(trial.suggest_float("subsample", 0.6, 1.0)),
                "colsample_bytree": float(trial.suggest_float("colsample_bytree", 0.6, 1.0)),
                "verbose": -1,
            }
        return {
            "C": float(trial.suggest_float("C", 1e-3, 20.0, log=True)),
            "solver": str(trial.suggest_categorical("solver", ["lbfgs", "liblinear"])),
            "max_iter": int(trial.suggest_int("max_iter", 1000, 5000)),
        }

    def _tune_best_variant_with_optuna(
        self,
        variant: ModelVariantConfig,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
        scale_pos_weight: float,
    ) -> tuple[Any, dict[str, Any], float, dict[str, float]] | None:
        try:
            import optuna
        except ImportError:
            TRAIN_LOGGER.warning("Optuna не установлена. Тюнинг пропущен.")
            return None

        TRAIN_LOGGER.info(
            "Запускаю Optuna-подбор параметров для варианта '%s' по метрике ROC AUC. trials=%s",
            variant.name,
            self.config.optuna.n_trials,
        )

        def objective(trial: Any) -> float:
            sampled_params = self._sample_optuna_params(trial=trial, architecture=variant.architecture)
            tuned_variant = ModelVariantConfig(
                name=variant.name,
                architecture=variant.architecture,
                enabled=True,
                params={**variant.params, **sampled_params},
            )
            model = self._build_estimator(variant=tuned_variant, scale_pos_weight=scale_pos_weight)
            model = self._fit_model(
                model=model,
                variant=tuned_variant,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
            )
            scores = self._predict_scores(model=model, features_df=x_val)
            try:
                return float(roc_auc_score(y_val, scores))
            except ValueError:
                return 0.0

        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=self.config.optuna.n_trials,
            timeout=self.config.optuna.timeout_sec,
            show_progress_bar=False,
        )
        best_params: dict[str, Any] = dict(study.best_params)

        tuned_variant = ModelVariantConfig(
            name=variant.name,
            architecture=variant.architecture,
            enabled=True,
            params={**variant.params, **best_params},
        )
        tuned_model = self._build_estimator(variant=tuned_variant, scale_pos_weight=scale_pos_weight)
        tuned_model = self._fit_model(
            model=tuned_model,
            variant=tuned_variant,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
        )
        tuned_scores = self._predict_scores(model=tuned_model, features_df=x_val)

        def threshold_objective(trial: Any) -> float:
            threshold: float = float(trial.suggest_float("threshold", 0.01, 0.99))
            preds = (tuned_scores >= threshold).astype(int)
            return float(f1_score(y_val, preds, zero_division=0))

        threshold_study = optuna.create_study(direction="maximize")
        threshold_study.optimize(
            threshold_objective,
            n_trials=self.config.optuna.threshold_trials,
            timeout=self.config.optuna.timeout_sec,
            show_progress_bar=False,
        )
        best_threshold: float = float(threshold_study.best_params["threshold"])
        tuned_metrics = self._compute_validation_metrics(y_true=y_val, y_scores=tuned_scores, threshold=best_threshold)
        TRAIN_LOGGER.info(
            "Optuna завершен для '%s': ROC_AUC=%.6f, AP=%.6f, F1=%.6f, threshold=%.4f",
            variant.name,
            tuned_metrics.get("roc_auc", float("nan")),
            tuned_metrics["average_precision"],
            tuned_metrics["f1"],
            best_threshold,
        )
        return tuned_model, best_params, best_threshold, tuned_metrics

    def _build_estimator(self, variant: ModelVariantConfig, scale_pos_weight: float) -> Any:
        if variant.architecture == "catboost":
            model_params: dict[str, Any] = {
                "iterations": self.config.iterations,
                "learning_rate": self.config.learning_rate,
                "depth": self.config.depth,
                "l2_leaf_reg": self.config.l2_leaf_reg,
                "subsample": self.config.subsample,
                "colsample_bylevel": self.config.colsample_bylevel,
                "min_data_in_leaf": self.config.min_data_in_leaf,
                "early_stopping_rounds": self.config.early_stopping_rounds,
                "verbose": False,
                "random_seed": self.config.random_seed,
                "scale_pos_weight": scale_pos_weight,
            }
            model_params.update(variant.params)
            return cb.CatBoostClassifier(**model_params)

        if variant.architecture == "xgboost":
            try:
                import xgboost as xgb
            except ImportError as error:
                raise RuntimeError("xgboost is not installed") from error
            model_params = {
                "n_estimators": min(self.config.iterations, 400),
                "learning_rate": self.config.learning_rate,
                "max_depth": self.config.depth,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bylevel,
                "random_state": self.config.random_seed,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "scale_pos_weight": scale_pos_weight,
            }
            model_params.update(variant.params)
            return xgb.XGBClassifier(**model_params)

        if variant.architecture == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as error:
                raise RuntimeError("lightgbm is not installed") from error
            model_params = {
                "n_estimators": min(self.config.iterations, 400),
                "learning_rate": self.config.learning_rate,
                "max_depth": self.config.depth,
                "subsample": self.config.subsample,
                "colsample_bytree": self.config.colsample_bylevel,
                "random_state": self.config.random_seed,
                "objective": "binary",
                "class_weight": None,
                "verbose": -1,
            }
            model_params.update(variant.params)
            return lgb.LGBMClassifier(**model_params)

        from sklearn.linear_model import LogisticRegression

        model_params = {
            "max_iter": 5000,
            "random_state": self.config.random_seed,
            "class_weight": "balanced",
        }
        model_params.update(variant.params)
        return LogisticRegression(**model_params)

    def _predict_scores(self, model: Any, features_df: pd.DataFrame) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features_df)
            if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                return proba[:, 1]
            return np.array(proba).reshape(-1)

        if hasattr(model, "decision_function"):
            decision = np.array(model.decision_function(features_df), dtype=float).reshape(-1)
            return 1.0 / (1.0 + np.exp(-decision))

        raise ValueError("Model does not support probability scoring")

    def _compute_validation_metrics(
        self,
        y_true: pd.Series,
        y_scores: np.ndarray,
        threshold: float,
    ) -> dict[str, float]:
        y_pred: np.ndarray = (y_scores >= threshold).astype(int)
        metrics: dict[str, float] = {
            "average_precision": float(average_precision_score(y_true, y_scores)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
            "threshold": float(threshold),
        }
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_scores))
        except ValueError:
            metrics["roc_auc"] = float("nan")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        metrics.update({"tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp)})
        return metrics

    def _update_latest_alias(self, output_root: Path, checkpoint_dir: Path) -> Path:
        latest_path: Path = output_root / "latest"
        if latest_path.exists() or latest_path.is_symlink():
            if latest_path.is_symlink() or latest_path.is_file():
                latest_path.unlink()
            else:
                shutil.rmtree(latest_path)

        try:
            latest_path.symlink_to(checkpoint_dir, target_is_directory=True)
            TRAIN_LOGGER.info("Обновлена ссылка latest: %s -> %s", latest_path, checkpoint_dir)
        except OSError as error:
            TRAIN_LOGGER.warning("Не удалось создать symlink для latest (%s). Использую резервное копирование директории.", error)
            shutil.copytree(checkpoint_dir, latest_path)
            TRAIN_LOGGER.info("Обновлена резервная директория latest: %s", latest_path)
        return latest_path

    def _plot_feature_distributions(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        feature_names: list[str],
        checkpoint_dir: Path,
    ) -> None:
        if not self.config.plot_feature_distributions or not feature_names:
            return

        plot_dir: Path
        if self.config.feature_plot_dir is None:
            plot_dir = checkpoint_dir / "feature_plots"
        else:
            plot_dir = Path(self.config.feature_plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

        train_plot_df = train_df[feature_names].copy()
        train_plot_df["split"] = "train"
        val_plot_df = val_df[feature_names].copy()
        val_plot_df["split"] = "validation"
        combined_plot_df = pd.concat([train_plot_df, val_plot_df], axis=0, ignore_index=True)

        batch_size: int = self.config.feature_plot_batch_size
        for batch_start in range(0, len(feature_names), batch_size):
            batch_features = feature_names[batch_start : batch_start + batch_size]
            n_cols = 4
            n_rows = int(np.ceil(len(batch_features) / n_cols))
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, max(4 * n_rows, 5)))
            axes = np.array(axes).reshape(-1)

            for axis_index, feature_name in enumerate(batch_features):
                axis = axes[axis_index]
                sns.histplot(
                    data=combined_plot_df,
                    x=feature_name,
                    hue="split",
                    bins=30,
                    kde=True,
                    stat="density",
                    common_norm=False,
                    ax=axis,
                )
                axis.set_title(feature_name)

            for axis in axes[len(batch_features) :]:
                axis.axis("off")

            fig.suptitle(
                f"Feature distributions train vs validation [{batch_start + 1}-{batch_start + len(batch_features)}]",
                y=1.02,
            )
            fig.tight_layout()
            figure_path = plot_dir / f"feature_distribution_{batch_start + 1}_{batch_start + len(batch_features)}.png"
            fig.savefig(figure_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

    def train(self) -> TrainResult:
        checkpoint_dir = self._resolve_checkpoint_dir()
        train_df = pd.read_csv(self.config.train_csv)
        val_df = pd.read_csv(self.config.val_csv)
        TRAIN_LOGGER.info("Датасеты загружены. train=%s, val=%s", train_df.shape, val_df.shape)
        if "comment" in val_df.columns:
            val_df = val_df.drop(columns=["comment"])
            TRAIN_LOGGER.info("Из валидационного датасета удалена колонка 'comment'")

        if self.config.target_col not in train_df.columns or self.config.target_col not in val_df.columns:
            raise ValueError(f"'{self.config.target_col}' must exist in both train and validation datasets")

        preprocessor = TabularPreprocessor(config=self.config)
        processed_train, processed_val, selected_feature_names = preprocessor.fit_transform(train_df, val_df)
        TRAIN_LOGGER.info(
            "Выбраны группы признаков=%s | число признаков=%s",
            preprocessor.selected_groups,
            len(selected_feature_names),
        )
        self._plot_feature_distributions(
            train_df=processed_train,
            val_df=processed_val,
            feature_names=selected_feature_names,
            checkpoint_dir=checkpoint_dir,
        )

        x_train = processed_train[selected_feature_names].copy()
        y_train = processed_train[self.config.target_col].astype(int).copy()

        if self.config.under_sampling:
            from imblearn.under_sampling import RandomUnderSampler

            rus = RandomUnderSampler(random_state=self.config.random_seed)
            sampled_x, sampled_y = rus.fit_resample(x_train, y_train)
            x_train = pd.DataFrame(sampled_x, columns=selected_feature_names)
            y_train = pd.Series(sampled_y, name=self.config.target_col)
            TRAIN_LOGGER.info("Применен RandomUnderSampler. Распределение классов=%s", y_train.value_counts().to_dict())

        if self.config.oversampling:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=self.config.random_seed)

            x_train = x_train.astype(float).fillna(0.0)

            sampled_x, sampled_y = smote.fit_resample(x_train, y_train)
            x_train = pd.DataFrame(sampled_x, columns=selected_feature_names)
            y_train = pd.Series(sampled_y, name=self.config.target_col)
            TRAIN_LOGGER.info("Применен SMOTE. Распределение классов=%s", y_train.value_counts().to_dict())

        x_val = processed_val[selected_feature_names].copy()
        y_val = processed_val[self.config.target_col].astype(int).copy()

        train_counts = y_train.value_counts().to_dict()
        pos_count = int(train_counts.get(1, 0))
        neg_count = int(train_counts.get(0, 0))
        scale_pos_weight: float = float(neg_count / pos_count) if pos_count > 0 else 1.0
        TRAIN_LOGGER.info(
            "Данные подготовлены: X_train=%s, X_val=%s | распределение классов=%s | scale_pos_weight=%.4f",
            x_train.shape,
            x_val.shape,
            y_train.value_counts().to_dict(),
            scale_pos_weight,
        )

        best_model: Any | None = None
        best_variant: ModelVariantConfig | None = None
        best_scores: np.ndarray | None = None
        best_metrics: dict[str, float] = {}
        all_metrics: dict[str, dict[str, float]] = {}
        best_variant_optuna_params: dict[str, Any] = {}

        for variant in [variant for variant in self.config.model_variants if variant.enabled]:
            TRAIN_LOGGER.info("Обучение варианта '%s' (%s) с параметрами=%s", variant.name, variant.architecture, variant.params)
            try:
                model = self._build_estimator(variant=variant, scale_pos_weight=scale_pos_weight)
                model = self._fit_model(
                    model=model,
                    variant=variant,
                    x_train=x_train,
                    y_train=y_train,
                    x_val=x_val,
                    y_val=y_val,
                )

                val_scores = self._predict_scores(model=model, features_df=x_val)
                threshold, _ = self._best_threshold(y_val, val_scores)
                metrics = self._compute_validation_metrics(y_true=y_val, y_scores=val_scores, threshold=threshold)
                all_metrics[variant.name] = metrics
                TRAIN_LOGGER.info(
                    "Метрики варианта '%s': AP=%.6f F1=%.6f ROC_AUC=%.6f Precision=%.6f Recall=%.6f Accuracy=%.6f",
                    variant.name,
                    metrics["average_precision"],
                    metrics["f1"],
                    metrics.get("roc_auc", float("nan")),
                    metrics["precision"],
                    metrics["recall"],
                    metrics["accuracy"],
                )

                current_roc_auc: float = float(metrics.get("roc_auc", float("nan")))
                best_roc_auc: float = float(best_metrics.get("roc_auc", float("nan"))) if best_variant is not None else float("nan")
                is_better: bool = False
                if best_variant is None:
                    is_better = True
                elif np.isnan(best_roc_auc) and not np.isnan(current_roc_auc):
                    is_better = True
                elif not np.isnan(current_roc_auc) and current_roc_auc > best_roc_auc:
                    is_better = True
                elif np.isclose(current_roc_auc, best_roc_auc, rtol=0.0, atol=1e-8):
                    is_better = metrics["average_precision"] > best_metrics.get("average_precision", -1.0)

                if is_better:
                    best_model = model
                    best_variant = variant
                    best_scores = val_scores
                    best_metrics = metrics
            except Exception:
                TRAIN_LOGGER.exception("Вариант '%s' завершился с ошибкой и будет пропущен", variant.name)

        if best_model is None or best_variant is None or best_scores is None:
            raise ValueError("No model variants were successfully trained")

        if self.config.optuna.enabled:
            tuned_result = self._tune_best_variant_with_optuna(
                variant=best_variant,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                scale_pos_weight=scale_pos_weight,
            )
            if tuned_result is not None:
                tuned_model, tuned_params, tuned_threshold, tuned_metrics = tuned_result
                baseline_roc_auc: float = float(best_metrics.get("roc_auc", float("nan")))
                tuned_roc_auc: float = float(tuned_metrics.get("roc_auc", float("nan")))
                if (
                    np.isnan(baseline_roc_auc)
                    or (not np.isnan(tuned_roc_auc) and tuned_roc_auc >= baseline_roc_auc)
                ):
                    best_model = tuned_model
                    best_scores = self._predict_scores(model=tuned_model, features_df=x_val)
                    best_metrics = tuned_metrics
                    best_metrics["threshold"] = float(tuned_threshold)
                    best_variant_optuna_params = tuned_params
                    TRAIN_LOGGER.info(
                        "Применены Optuna-параметры для лучшей модели. ROC_AUC=%.6f",
                        tuned_roc_auc,
                    )

        best_threshold: float = float(best_metrics["threshold"])
        best_f1: float = float(best_metrics["f1"])
        ap_score: float = float(best_metrics["average_precision"])
        TRAIN_LOGGER.info(
            "Выбран лучший вариант: %s (%s) с ROC_AUC=%.6f, average_precision=%.6f",
            best_variant.name,
            best_variant.architecture,
            best_metrics.get("roc_auc", float("nan")),
            ap_score,
        )

        model_fpath = checkpoint_dir / "catboost_model.cbm"
        model_pickle_fpath = checkpoint_dir / "model.pkl"
        preprocessor_fpath = checkpoint_dir / "preprocessor.pkl"
        metadata_fpath = checkpoint_dir / "metadata.json"

        if best_variant.architecture == "catboost":
            best_model.save_model(str(model_fpath))
        with open(model_pickle_fpath, "wb") as model_file:
            pickle.dump(best_model, model_file)
        with open(preprocessor_fpath, "wb") as file:
            pickle.dump(preprocessor, file)

        metadata: dict[str, Any] = {
            "target_col": self.config.target_col,
            "selected_feature_names": selected_feature_names,
            "selected_groups": preprocessor.selected_groups,
            "best_threshold": best_threshold,
            "f1": best_f1,
            "average_precision": ap_score,
            "best_architecture": best_variant.architecture,
            "best_variant_name": best_variant.name,
            "best_variant_optuna_params": best_variant_optuna_params,
            "validation_metrics": best_metrics,
            "all_variants_metrics": all_metrics,
            "train_csv": str(self.config.train_csv),
            "val_csv": str(self.config.val_csv),
            "feature_flags": self.config.feature_flags.model_dump(),
        }
        metadata_fpath.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        self._update_latest_alias(output_root=checkpoint_dir.parent, checkpoint_dir=checkpoint_dir)

        TRAIN_LOGGER.info("Чекпоинт tabular-модели сохранен в %s", checkpoint_dir)
        return TrainResult(
            checkpoint_dir=str(checkpoint_dir),
            selected_feature_names=selected_feature_names,
            selected_groups=preprocessor.selected_groups,
            best_threshold=best_threshold,
            f1=best_f1,
            average_precision=ap_score,
            best_architecture=best_variant.architecture,
            validation_metrics=best_metrics,
        )


class TabularHallucinationPredictor:
    """Загружает сохраненный bundle и выполняет инференс."""

    def __init__(self, checkpoint_dir: PathLike) -> None:
        checkpoint_path: Path = Path(checkpoint_dir).expanduser()
        if checkpoint_path.is_absolute():
            self.checkpoint_dir = checkpoint_path
        else:
            self.checkpoint_dir = Path(get_public_checkpoints_dpath()) / checkpoint_path
        self.model: Any = cb.CatBoostClassifier()
        self.preprocessor: TabularPreprocessor | None = None
        self.metadata: dict[str, Any] = {}

    def load(self) -> None:
        model_fpath = self.checkpoint_dir / "catboost_model.cbm"
        model_pickle_fpath = self.checkpoint_dir / "model.pkl"
        preprocessor_fpath = self.checkpoint_dir / "preprocessor.pkl"
        metadata_fpath = self.checkpoint_dir / "metadata.json"

        if model_pickle_fpath.exists():
            with open(model_pickle_fpath, "rb") as model_file:
                self.model = pickle.load(model_file)
        else:
            self.model.load_model(str(model_fpath))
        with open(preprocessor_fpath, "rb") as file:
            self.preprocessor = pickle.load(file)
        self.metadata = json.loads(metadata_fpath.read_text(encoding="utf-8"))

    def predict_dataframe(self, dataframe: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        if self.preprocessor is None:
            self.load()
        assert self.preprocessor is not None

        selected_features: list[str] = list(self.metadata.get("selected_feature_names", []))
        if not selected_features:
            selected_features = self.preprocessor.selected_feature_names

        prepared = self.preprocessor.transform(dataframe)
        x_data = prepared[selected_features].copy()
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(x_data)[:, 1]
        elif hasattr(self.model, "decision_function"):
            decision = np.array(self.model.decision_function(x_data), dtype=float).reshape(-1)
            probs = 1.0 / (1.0 + np.exp(-decision))
        else:
            raise ValueError("Loaded model does not support scoring")

        effective_threshold: float = (
            float(threshold)
            if threshold is not None
            else float(self.metadata.get("best_threshold", 0.5))
        )
        result = dataframe.copy()
        result["hallucination_score"] = probs
        result["entailment_score"] = 1.0 - probs
        result["pred_is_hallucination"] = (probs >= effective_threshold).astype(int)

        INFER_LOGGER.info("Инференс завершен для %s строк с порогом=%.4f", len(result), effective_threshold)
        return result


__all__ = [
    "FeatureGroupFlags",
    "ModelVariantConfig",
    "TabularHallucinationPredictor",
    "TabularHallucinationTrainer",
    "TabularInferenceConfig",
    "TabularTrainConfig",
    "TrainResult",
]





