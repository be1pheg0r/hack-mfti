from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

import optuna
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OrdinalEncoder, StandardScaler
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from avito.features import (
    ShouldSplitFeatureConfig,
    TextEncoderLike,
    append_embedding_features,
    extract_should_split_features,
    resolve_keyphrases,
)
from common.files import read_yaml
from common.logger import AVITO_MICROCATS_LOGGER as logger
from common.logger import fit_model_with_progress, log_block_separator


TARGET_COLUMN = "targetDetectedMcIds"
SPLIT_COLUMN = "split"


class MicrocategoryTrainingConfig(BaseModel):
    """Гиперпараметры обучения выделения микрокатегорий."""

    model_config = ConfigDict(frozen=True)

    random_state: int = 42
    merge_train_test_for_fit: bool = True
    threshold_grid: list[float] = Field(default_factory=lambda: [0.5])
    log_reg_c: float = 1.0
    max_iter: int = 1500
    optuna_n_trials: int = 15
    optuna_timeout_sec: int | None = None
    categorical_features: list[str] = Field(default_factory=lambda: ["source_mc_id", "case_type"])
    backend_type: Literal["sklearn", "transformer"] = "sklearn"
    transformer_model_name: str = "jhu-clsp/mmBERT-base"
    transformer_max_length: int = 256
    transformer_batch_size: int = 8
    transformer_eval_batch_size: int = 16
    transformer_num_epochs: int = 2
    transformer_learning_rate: float = 2e-5
    transformer_weight_decay: float = 0.01
    transformer_warmup_ratio: float = 0.1
    transformer_gradient_accumulation_steps: int = 1
    transformer_device: str | None = None

    @field_validator("threshold_grid")
    @classmethod
    def validate_thresholds(cls, values: list[float]) -> list[float]:
        if not values:
            return [0.5]
        normalized: list[float] = []
        for value in values:
            if not 0.0 < value <= 1.0:
                raise ValueError("Пороговые значения должны быть в (0, 1].")
            normalized.append(float(value))
        return normalized

    @field_validator("log_reg_c")
    @classmethod
    def validate_log_reg_c(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("log_reg_c должен быть положительным")
        return float(value)

    @field_validator("max_iter")
    @classmethod
    def validate_max_iter(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_iter должен быть положительным")
        return int(value)

    @field_validator("optuna_n_trials")
    @classmethod
    def validate_optuna_trials(cls, value: int) -> int:
        if value < 0:
            raise ValueError("optuna_n_trials не может быть отрицательным")
        return int(value)


class TrainedMicrocategoryModel(BaseModel):
    """Контейнер с обученной моделью выделения микрокатегорий."""

    model_config = ConfigDict(arbitrary_types_allowed=True, protected_namespaces=())

    model_name: str
    pipeline: Pipeline | None = None
    threshold: float
    mlb_classes: list[int]
    metrics: dict[str, float]
    model_comparison_records: list[dict[str, float | str]] = Field(default_factory=list)
    tuned_params: dict[str, Any] = Field(default_factory=dict)
    backend_type: Literal["sklearn", "transformer"] = "sklearn"
    transformer_model_name: str | None = None
    transformer_model_dir: str | None = None
    transformer_model: Any | None = None
    transformer_tokenizer: Any | None = None


class TuneParamConfig(BaseModel):
    """Конфигурация одного тюнимого параметра для Optuna."""

    type: str
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
    """Конфиг одной кандидатной архитектуры микрокатегорий."""

    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    for_tune: dict[str, TuneParamConfig] = Field(default_factory=dict)


@dataclass(frozen=True)
class MicrocategoryModelConfigPaths:
    one_vs_rest_logreg: Path
    one_vs_rest_random_forest: Path
    one_vs_rest_xgboost: Path
    one_vs_rest_catboost: Path
    one_vs_rest_lightgbm: Path


def get_microcategory_model_config_paths() -> MicrocategoryModelConfigPaths:
    """Возвращает абсолютные пути к конфигам архитектур microcategories."""

    config_dir = Path(__file__).resolve().parent.parent / "configs" / "microcategories"
    return MicrocategoryModelConfigPaths(
        one_vs_rest_logreg=config_dir / "one_vs_rest_logreg.yaml",
        one_vs_rest_random_forest=config_dir / "one_vs_rest_random_forest.yaml",
        one_vs_rest_xgboost=config_dir / "one_vs_rest_xgboost.yaml",
        one_vs_rest_catboost=config_dir / "one_vs_rest_catboost.yaml",
        one_vs_rest_lightgbm=config_dir / "one_vs_rest_lightgbm.yaml",
    )


def _build_architecture_map(paths: MicrocategoryModelConfigPaths) -> dict[str, Path]:
    return {
        "one_vs_rest_logreg": paths.one_vs_rest_logreg,
        "one_vs_rest_random_forest": paths.one_vs_rest_random_forest,
        "one_vs_rest_xgboost": paths.one_vs_rest_xgboost,
        "one_vs_rest_catboost": paths.one_vs_rest_catboost,
        "one_vs_rest_lightgbm": paths.one_vs_rest_lightgbm,
    }


def _load_architecture_configs() -> dict[str, ModelArchitectureConfig]:
    paths = get_microcategory_model_config_paths()
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
        raise ValueError("Хотя бы одна архитектура должна быть включена в model-конфигах микрокатегорий")

    return configs


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


def _parse_mc_list(value: Any) -> list[int]:
    if isinstance(value, (list, tuple, set)):
        return [int(v) for v in value]
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, (list, tuple, set)):
                return [int(v) for v in loaded]
        except json.JSONDecodeError:
            pass
    return []


def _prepare_targets(df: pd.DataFrame) -> tuple[np.ndarray, MultiLabelBinarizer]:
    targets = df[TARGET_COLUMN].apply(_parse_mc_list)
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(targets)
    if y.ndim != 2 or y.shape[1] == 0:
        raise ValueError("Не удалось сформировать многометочные таргеты: пустой результат.")
    return y.astype(np.int8, copy=False), mlb


def _split_frame(
    X: pd.DataFrame,
    y: np.ndarray,
    split: pd.Series,
    *,
    merge_train_test_for_fit: bool,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    train_mask = split.eq("train")
    test_mask = split.eq("test")
    val_mask = split.eq("val")

    if not train_mask.any() or not val_mask.any():
        raise ValueError("Ожидаются непустые группы split: train/val (+ optional test)")

    fit_mask = (train_mask | test_mask) if merge_train_test_for_fit else train_mask
    return (
        X.loc[fit_mask],
        y[fit_mask],
        X.loc[val_mask],
        y[val_mask],
    )


def _build_preprocessor(feature_frame: pd.DataFrame, config: MicrocategoryTrainingConfig) -> ColumnTransformer:
    categorical_features = config.categorical_features
    numeric_features = [column for column in feature_frame.columns if column not in categorical_features]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value=0.0),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
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


def _predict_proba(pipeline: Pipeline, X_data: pd.DataFrame) -> np.ndarray:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X_data)
        return np.asarray(proba, dtype=np.float64)
    if hasattr(pipeline, "decision_function"):
        decision = np.asarray(pipeline.decision_function(X_data), dtype=np.float64)
        # Сигмоида для двоичных решений
        return 1.0 / (1.0 + np.exp(-decision))
    raise ValueError("Модель не поддерживает predict_proba или decision_function")


def _evaluate_thresholds(
    y_true: np.ndarray,
    proba: np.ndarray,
    thresholds: Iterable[float],
) -> tuple[float, dict[str, float]]:
    best_threshold: float | None = None
    best_metrics: dict[str, float] = {}
    best_f1: float = -1.0

    for threshold in thresholds:
        y_pred = (proba >= threshold).astype(np.int8)
        micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
        micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
        micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
        if micro_f1 > best_f1 or (np.isclose(micro_f1, best_f1) and (best_threshold is None or threshold < best_threshold)):
            best_f1 = micro_f1
            best_threshold = threshold
            best_metrics = {
                "micro_f1": float(micro_f1),
                "micro_precision": float(micro_precision),
                "micro_recall": float(micro_recall),
            }

    if best_threshold is None:
        raise RuntimeError("Не удалось подобрать порог для микрокатегорий")

    best_metrics["threshold"] = float(best_threshold)
    return best_threshold, best_metrics


def _resolve_transformer_thresholds(threshold_grid: Iterable[float]) -> list[float]:
    resolved = [float(value) for value in threshold_grid]
    if not resolved:
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    if len(resolved) == 1 and np.isclose(resolved[0], 0.5):
        # Для multi-label transformer один порог 0.5 часто дает пустые предсказания.
        return [0.1, 0.2, 0.3, 0.4, 0.5]
    return resolved


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _train_transformer_backend(
    df: pd.DataFrame,
    y: np.ndarray,
    split: pd.Series,
    cfg: MicrocategoryTrainingConfig,
) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Для transformer backend требуются torch и transformers. "
            "Установите зависимости из requirements-avito.txt"
        ) from exc

    class _TextMultilabelDataset(Dataset):
        def __init__(self, texts: list[str], labels: np.ndarray, tokenizer: Any, max_length: int) -> None:
            self.texts = texts
            self.labels = labels.astype(np.float32)
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            encoded = self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)
            return item

    set_seed(cfg.random_state)
    tokenizer = AutoTokenizer.from_pretrained(cfg.transformer_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.transformer_model_name,
        num_labels=int(y.shape[1]),
        problem_type="multi_label_classification",
    )

    device = cfg.transformer_device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_mask = split.eq("train")
    test_mask = split.eq("test")
    val_mask = split.eq("val")
    fit_mask = (train_mask | test_mask) if cfg.merge_train_test_for_fit else train_mask
    if not fit_mask.any() or not val_mask.any():
        raise ValueError("Ожидаются непустые группы split для transformer backend: fit/val")

    fit_texts = df.loc[fit_mask, "description"].astype(str).tolist()
    val_texts = df.loc[val_mask, "description"].astype(str).tolist()
    y_fit = y[fit_mask]
    y_val = y[val_mask]

    train_dataset = _TextMultilabelDataset(
        texts=fit_texts,
        labels=y_fit,
        tokenizer=tokenizer,
        max_length=cfg.transformer_max_length,
    )
    eval_dataset = _TextMultilabelDataset(
        texts=val_texts,
        labels=y_val,
        tokenizer=tokenizer,
        max_length=cfg.transformer_max_length,
    )

    positive_counts = np.asarray(y_fit.sum(axis=0), dtype=np.float32)
    negative_counts = np.asarray(y_fit.shape[0] - positive_counts, dtype=np.float32)
    pos_weight_np = (negative_counts + 1.0) / (positive_counts + 1.0)
    pos_weight_np = np.clip(pos_weight_np, 1.0, 50.0)
    pos_weight = torch.tensor(pos_weight_np, dtype=torch.float32)

    effective_batch = max(1, cfg.transformer_batch_size * cfg.transformer_gradient_accumulation_steps)
    steps_per_epoch = max(1, int(np.ceil(len(train_dataset) / effective_batch)))
    total_steps = max(1, int(steps_per_epoch * cfg.transformer_num_epochs))
    warmup_steps = max(0, int(total_steps * cfg.transformer_warmup_ratio))

    train_args = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="avito_microcats_mmbert_"),
        learning_rate=cfg.transformer_learning_rate,
        per_device_train_batch_size=cfg.transformer_batch_size,
        per_device_eval_batch_size=cfg.transformer_eval_batch_size,
        num_train_epochs=cfg.transformer_num_epochs,
        weight_decay=cfg.transformer_weight_decay,
        warmup_steps=warmup_steps,
        gradient_accumulation_steps=cfg.transformer_gradient_accumulation_steps,
        eval_strategy="no",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        report_to=[],
        seed=cfg.random_state,
        dataloader_pin_memory=torch.cuda.is_available(),
    )

    class _WeightedMultilabelTrainer(Trainer):
        def __init__(self, *args: Any, pos_weight_tensor: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._pos_weight_tensor = pos_weight_tensor

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            **kwargs: Any,
        ) -> Any:
            labels = inputs.get("labels")
            model_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**model_inputs)
            logits = outputs.logits
            if labels is None:
                loss = outputs.loss
            else:
                loss_fct = torch.nn.BCEWithLogitsLoss(
                    pos_weight=self._pos_weight_tensor.to(logits.device)
                )
                loss = loss_fct(logits, labels.to(logits.device).float())
            return (loss, outputs) if return_outputs else loss

    trainer = _WeightedMultilabelTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        pos_weight_tensor=pos_weight,
    )

    trainer.train()
    predictions = trainer.predict(eval_dataset).predictions
    val_proba = _sigmoid(np.asarray(predictions, dtype=np.float64))
    return model, tokenizer, val_proba, y_val


def _make_estimator(
    architecture_name: str,
    *,
    params: dict[str, Any],
    cfg: MicrocategoryTrainingConfig,
) -> OneVsRestClassifier:
    effective_params = dict(params)

    if architecture_name == "one_vs_rest_logreg":
        effective_params.setdefault("C", cfg.log_reg_c)
        effective_params.setdefault("class_weight", "balanced")
        effective_params.setdefault("max_iter", cfg.max_iter)
        effective_params.setdefault("random_state", cfg.random_state)
        effective_params.setdefault("solver", "liblinear")
        base_estimator = LogisticRegression(**effective_params)
        return OneVsRestClassifier(base_estimator)

    if architecture_name == "one_vs_rest_random_forest":
        effective_params.setdefault("n_estimators", 400)
        effective_params.setdefault("max_depth", None)
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("random_state", cfg.random_state)
        base_estimator = RandomForestClassifier(**effective_params)
        return OneVsRestClassifier(base_estimator)

    if architecture_name == "one_vs_rest_xgboost":
        effective_params.setdefault("n_estimators", 400)
        effective_params.setdefault("learning_rate", 0.05)
        effective_params.setdefault("max_depth", 6)
        effective_params.setdefault("subsample", 0.9)
        effective_params.setdefault("colsample_bytree", 0.9)
        effective_params.setdefault("eval_metric", "logloss")
        effective_params.setdefault("n_jobs", -1)
        effective_params.setdefault("random_state", cfg.random_state)
        effective_params.setdefault("tree_method", "hist")
        effective_params.setdefault("objective", "binary:logistic")
        base_estimator = XGBClassifier(**effective_params)
        return OneVsRestClassifier(base_estimator)

    if architecture_name == "one_vs_rest_catboost":
        effective_params.setdefault("iterations", 500)
        effective_params.setdefault("learning_rate", 0.1)
        effective_params.setdefault("depth", 6)
        effective_params.setdefault("loss_function", "Logloss")
        effective_params.setdefault("verbose", 100)
        effective_params.setdefault("random_seed", cfg.random_state)
        base_estimator = CatBoostClassifier(**effective_params)
        return OneVsRestClassifier(base_estimator)

    if architecture_name == "one_vs_rest_lightgbm":
        effective_params.setdefault("n_estimators", 500)
        effective_params.setdefault("learning_rate", 0.05)
        effective_params.setdefault("max_depth", -1)
        effective_params.setdefault("subsample", 0.9)
        effective_params.setdefault("colsample_bytree", 0.9)
        effective_params.setdefault("random_state", cfg.random_state)
        effective_params.setdefault("n_jobs", -1)
        base_estimator = LGBMClassifier(**effective_params)
        return OneVsRestClassifier(base_estimator)

    raise ValueError(f"Неизвестная архитектура: {architecture_name}")


def train_microcategory_model(
    df: pd.DataFrame,
    *,
    include_embeddings: bool = True,
    encoder: TextEncoderLike | None = None,
    feature_config: ShouldSplitFeatureConfig | None = None,
    training_config: MicrocategoryTrainingConfig | None = None,
) -> TrainedMicrocategoryModel:
    """Обучает мульти-лейбл модель выделения микрокатегорий.

    Args:
        df: Датасет с колонками description, targetDetectedMcIds и split (train/val/test).
        include_embeddings: Добавлять ли эмбеддинги FRIDA.
        encoder: Энкодер для генерации эмбеддингов при include_embeddings=True.
        feature_config: Конфиг извлечения признаков.
        training_config: Гиперпараметры обучения/порогов.

    Returns:
        TrainedMicrocategoryModel с пайплайном, лучшим порогом и метриками.
    """

    cfg = training_config or MicrocategoryTrainingConfig()
    feat_cfg = feature_config or ShouldSplitFeatureConfig()

    required_columns = {"description", "sourceMcId", "sourceMcTitle", TARGET_COLUMN, SPLIT_COLUMN}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"В DataFrame отсутствуют обязательные колонки: {sorted(missing)}")

    y, mlb = _prepare_targets(df)
    split = df[SPLIT_COLUMN].astype(str)

    if cfg.backend_type == "transformer":
        log_block_separator(logger)
        logger.info("Старт обучения transformer backend microcategories")
        logger.info(f"Модель: {cfg.transformer_model_name}")
        logger.info(f"Эпох: {cfg.transformer_num_epochs}")
        logger.info(f"Batch train/eval: {cfg.transformer_batch_size}/{cfg.transformer_eval_batch_size}")
        log_block_separator(logger)

        model, tokenizer, val_proba, y_val = _train_transformer_backend(
            df=df,
            y=y,
            split=split,
            cfg=cfg,
        )
        threshold_grid = _resolve_transformer_thresholds(cfg.threshold_grid)
        if threshold_grid != cfg.threshold_grid:
            logger.info(f"Transformer threshold grid переопределен: {threshold_grid}")
        best_threshold, best_metrics = _evaluate_thresholds(
            y_true=y_val,
            proba=val_proba,
            thresholds=threshold_grid,
        )

        comparison_records: list[dict[str, float | str]] = [
            {
                "model": "transformer_mmbert",
                "micro_f1": float(best_metrics["micro_f1"]),
                "micro_precision": float(best_metrics["micro_precision"]),
                "micro_recall": float(best_metrics["micro_recall"]),
                "threshold": float(best_threshold),
            }
        ]

        logger.info("Результаты transformer backend на валидации:")
        logger.info(f"  micro_f1        = {best_metrics['micro_f1']:.6f}")
        logger.info(f"  micro_precision = {best_metrics['micro_precision']:.6f}")
        logger.info(f"  micro_recall    = {best_metrics['micro_recall']:.6f}")
        logger.info(f"  threshold       = {best_threshold:.3f}")
        log_block_separator(logger)

        return TrainedMicrocategoryModel(
            model_name="transformer_mmbert",
            pipeline=None,
            threshold=best_threshold,
            mlb_classes=[int(cls) for cls in mlb.classes_.tolist()],
            metrics=best_metrics,
            model_comparison_records=comparison_records,
            tuned_params={},
            backend_type="transformer",
            transformer_model_name=cfg.transformer_model_name,
            transformer_model_dir=None,
            transformer_model=model,
            transformer_tokenizer=tokenizer,
        )

    keyphrases = resolve_keyphrases(df, feat_cfg)
    features = extract_should_split_features(df=df, config=feat_cfg, keyphrases=keyphrases)
    if include_embeddings:
        if encoder is None:
            raise ValueError("Для include_embeddings=True нужно передать encoder")
        features = append_embedding_features(
            features=features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
            keyphrases=keyphrases,
            config=feat_cfg,
        )

    X_fit, y_fit, X_val, y_val = _split_frame(
        X=features,
        y=y,
        split=split,
        merge_train_test_for_fit=cfg.merge_train_test_for_fit,
    )

    preprocessor = _build_preprocessor(features, cfg)
    architecture_configs = _load_architecture_configs()
    enabled_items = [(name, arch_cfg) for name, arch_cfg in architecture_configs.items() if arch_cfg.enabled]

    if not enabled_items:
        raise ValueError("Хотя бы одна архитектура должна быть включена для обучения микрокатегорий")

    log_block_separator(logger)
    logger.info("Старт сравнения архитектур microcategories")
    logger.info(f"Кандидатов: {len(enabled_items)}")
    logger.info(f"Порогов в сетке: {len(cfg.threshold_grid)}")
    logger.info(f"Признаков в матрице: {features.shape[1]}")
    log_block_separator(logger)

    model_rows: list[dict[str, float | str]] = []
    best_model_name: str | None = None
    best_pipeline: Pipeline | None = None
    best_threshold: float | None = None
    best_metrics: dict[str, float] | None = None
    best_base_params: dict[str, Any] | None = None
    best_micro_f1 = -1.0

    for architecture_index, (architecture_name, architecture_config) in enumerate(enabled_items, start=1):
        estimator = _make_estimator(
            architecture_name,
            params=architecture_config.params,
            cfg=cfg,
        )
        current_preprocessor = clone(preprocessor)
        pipeline = Pipeline(steps=[("preprocessor", current_preprocessor), ("model", estimator)])

        log_block_separator(logger)
        logger.info(f"[{architecture_index}/{len(enabled_items)}] Архитектура: {architecture_name}")
        logger.info(f"Базовые параметры: {architecture_config.params}")
        fit_model_with_progress(pipeline, X_fit, y_fit, X_val)

        val_proba = _predict_proba(pipeline, X_val)
        candidate_threshold, candidate_metrics = _evaluate_thresholds(
            y_true=y_val,
            proba=val_proba,
            thresholds=cfg.threshold_grid,
        )

        candidate_record: dict[str, float | str] = {"model": architecture_name, **candidate_metrics}
        model_rows.append(candidate_record)

        logger.info("Результаты на валидации:")
        logger.info(f"  micro_f1        = {candidate_metrics['micro_f1']:.6f}")
        logger.info(f"  micro_precision = {candidate_metrics['micro_precision']:.6f}")
        logger.info(f"  micro_recall    = {candidate_metrics['micro_recall']:.6f}")
        logger.info(f"  threshold       = {candidate_threshold:.3f}")

        if candidate_metrics["micro_f1"] > best_micro_f1 or (
            np.isclose(candidate_metrics["micro_f1"], best_micro_f1) and best_model_name is None
        ):
            best_micro_f1 = candidate_metrics["micro_f1"]
            best_model_name = architecture_name
            best_pipeline = pipeline
            best_threshold = candidate_threshold
            best_metrics = dict(candidate_metrics)
            best_base_params = dict(architecture_config.params)

    if best_model_name is None or best_pipeline is None or best_threshold is None or best_metrics is None or best_base_params is None:
        raise RuntimeError("Не удалось выбрать лучшую архитектуру микрокатегорий")

    tuned_params: dict[str, Any] = {}
    best_arch_config = architecture_configs[best_model_name]

    if best_arch_config.for_tune and cfg.optuna_n_trials > 0:
        log_block_separator(logger)
        logger.info("Запуск Optuna-тюнинга")
        logger.info(f"Архитектура: {best_model_name}")
        logger.info(f"n_trials: {cfg.optuna_n_trials}")
        logger.info(f"timeout: {cfg.optuna_timeout_sec}")
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
                cfg=cfg,
            )
            pipeline = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", estimator)])
            try:
                fit_model_with_progress(pipeline, X_fit, y_fit, X_val)
            except Exception as fit_error:  # noqa: BLE001
                logger.warning(f"Ошибка обучения во время trial: {fit_error}")
                return -1e9

            val_proba = _predict_proba(pipeline, X_val)
            _, candidate_metrics = _evaluate_thresholds(
                y_true=y_val,
                proba=val_proba,
                thresholds=cfg.threshold_grid,
            )
            return float(candidate_metrics.get("micro_f1", -1.0))

        def _trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
            logger.info(
                "Optuna trial завершен: "
                f"trial={trial.number}, value={trial.value:.6f}, best={study.best_value:.6f}"
            )
            logger.info(f"  params={trial.params}")

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize")
        study.optimize(
            objective,
            n_trials=cfg.optuna_n_trials,
            timeout=cfg.optuna_timeout_sec,
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
        logger.info(
            "Тюнинг пропущен: либо отсутствует for_tune, либо optuna_n_trials=0"
        )

    # Финальное обучение с учетом тюнинга
    final_params = {**best_base_params, **tuned_params}
    final_estimator = _make_estimator(
        best_model_name,
        params=final_params,
        cfg=cfg,
    )
    final_pipeline = Pipeline(steps=[("preprocessor", clone(preprocessor)), ("model", final_estimator)])
    fit_model_with_progress(final_pipeline, X_fit, y_fit, X_val)
    final_val_proba = _predict_proba(final_pipeline, X_val)
    final_threshold, final_metrics = _evaluate_thresholds(
        y_true=y_val,
        proba=final_val_proba,
        thresholds=cfg.threshold_grid,
    )

    comparison_records = sorted(
        model_rows,
        key=lambda row: float(row.get("micro_f1", -1)),
        reverse=True,
    )

    logger.info("Рейтинг архитектур по micro_f1:")
    for rank, row in enumerate(comparison_records, start=1):
        logger.info(
            f"  #{rank} {row['model']}: micro_f1={float(row['micro_f1']):.6f}, "
            f"precision={float(row['micro_precision']):.6f}, recall={float(row['micro_recall']):.6f}, "
            f"threshold={float(row['threshold']):.3f}"
        )
    log_block_separator(logger)

    return TrainedMicrocategoryModel(
        model_name=best_model_name,
        pipeline=final_pipeline,
        threshold=final_threshold,
        mlb_classes=[int(cls) for cls in mlb.classes_.tolist()],
        metrics=final_metrics,
        model_comparison_records=comparison_records,
        tuned_params=tuned_params,
        backend_type="sklearn",
        transformer_model_name=None,
        transformer_model_dir=None,
        transformer_model=None,
        transformer_tokenizer=None,
    )
