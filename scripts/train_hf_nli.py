from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import HfApi, hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sklearn.metrics import average_precision_score, f1_score, fbeta_score
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common.configs import load_config_from_namespace, load_pydantic_config
from common.logger import SBER_NLI_TRAIN_LOGGER as logger
from common.paths import PathLike, get_data_bench_dpath, get_data_raw_dpath, get_model_dpath
from src.sber.constants import (
    DEFAULT_SBER_HF_MODEL_REPO_ID,
    DEFAULT_SBER_HF_NLI_BASE_MODEL_REPO_ID,
    DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE,
    DEFAULT_SBER_HF_NLI_CONFIG_FPATH,
    DEFAULT_SBER_HF_NLI_TRAIN_CONFIG_FPATH,
)
from src.sber.utils.evaluate_utils import EvaluationSummary, evaluate_scoring_results
from src.sber.utils.text_features import FEATURE_NAMES, QAFeatureExtractor


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class TrainHFNLIConfig(BaseModel):
    """Конфигурация обучения и quality-gate публикации HF NLI модели."""

    model_config = ConfigDict(frozen=True)

    train_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / "features_with_judge_scores.csv")
    val_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "knowledge_bench_public.csv")
    base_model_repo_id: str = DEFAULT_SBER_HF_NLI_BASE_MODEL_REPO_ID
    target_repo_id: str = DEFAULT_SBER_HF_MODEL_REPO_ID
    output_dir: PathLike = Field(default_factory=lambda: Path(get_model_dpath()) / "hf_nli_train")
    best_weights_name: str = "best_hallucination_model.pt"
    max_length: int = 512
    batch_size: int = 8
    epochs: int = 5
    learning_rate: float = 1.5e-5
    seed: int = 42
    compute_dtype: str = DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE
    eval_autocast_dtype: str = "bfloat16"
    num_workers: int = 2
    pin_memory: bool = True
    positive_class_index: int = 1
    evaluate_batch_size: int = 8
    nli_config_path: PathLike = DEFAULT_SBER_HF_NLI_CONFIG_FPATH
    report_dir: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "reports" / "train_eval")
    hf_metrics_filename: str = "hf_eval_metrics.json"
    metric_primary: str = "weighted_average_precision"
    metric_secondary: str = "f1"
    min_relative_improvement: float = 0.0
    push_if_better: bool = True
    save_plots: bool = True
    hallucination_threshold: float = 0.5
    threshold_search_mode: str = "none"
    threshold_search_beta: float = 2.0
    threshold_trials: int = 50
    question_col: str | None = None
    warmup_ratio_per_epoch: float = 0.1
    feature_head_dropout: float = 0.1
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 2
    class_weights_before_balancing: bool = True
    use_class_weights: bool = True

    @field_validator(
        "base_model_repo_id",
        "target_repo_id",
        "best_weights_name",
        "hf_metrics_filename",
        "metric_primary",
        "metric_secondary",
    )
    @classmethod
    def validate_non_empty_str(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковый параметр не может быть пустым")
        return normalized

    @field_validator("max_length", "batch_size", "epochs", "evaluate_batch_size")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Параметр должен быть положительным")
        return value

    @field_validator("num_workers", "positive_class_index")
    @classmethod
    def validate_non_negative_int(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Параметр должен быть неотрицательным")
        return value

    @field_validator("learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("learning_rate должен быть больше нуля")
        return value

    @field_validator("compute_dtype", "eval_autocast_dtype")
    @classmethod
    def validate_dtype(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        allowed: set[str] = {"float32", "float16", "bfloat16"}
        if normalized not in allowed:
            raise ValueError("dtype должен быть одним из: float32, float16, bfloat16")
        return normalized

    @field_validator("min_relative_improvement")
    @classmethod
    def validate_min_relative_improvement(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("min_relative_improvement должен быть неотрицательным")
        return value

    @field_validator("hallucination_threshold")
    @classmethod
    def validate_hallucination_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("hallucination_threshold должен быть в диапазоне [0, 1]")
        return value

    @field_validator("threshold_search_mode")
    @classmethod
    def validate_threshold_search_mode(cls, value: str) -> str:
        normalized: str = value.strip().lower()
        allowed: set[str] = {"none", "grid", "optuna"}
        if normalized not in allowed:
            raise ValueError("threshold_search_mode должен быть одним из: none, grid, optuna")
        return normalized

    @field_validator("threshold_search_beta")
    @classmethod
    def validate_threshold_search_beta(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("threshold_search_beta должен быть больше нуля")
        return value

    @field_validator("threshold_trials")
    @classmethod
    def validate_threshold_trials(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("threshold_trials должен быть положительным")
        return value

    @field_validator("warmup_ratio_per_epoch", "feature_head_dropout")
    @classmethod
    def validate_ratio(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        return value

    @field_validator("early_stopping_patience")
    @classmethod
    def validate_early_stopping_patience(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("early_stopping_patience должен быть положительным")
        return value

    @classmethod
    def from_yaml(cls, fpath: PathLike) -> TrainHFNLIConfig:
        """Загружает train-конфиг из YAML-файла."""
        return load_pydantic_config(config_cls=cls, fpath=fpath, section_name="hf_nli_train")


class HallucinationDataset(Dataset[Any]):
    """Dataset с парами correct_answer/model_answer и бинарной меткой."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: Any,
        feature_extractor: QAFeatureExtractor,
        max_length: int,
        *,
        question_col: str,
        premise_col: str = "correct_answer",
        hypothesis_col: str = "model_answer",
        label_col: str = "is_hallucination",
    ) -> None:
        self.dataframe: pd.DataFrame = dataframe.reset_index(drop=True)
        self.tokenizer: Any = tokenizer
        self.feature_extractor: QAFeatureExtractor = feature_extractor
        self.max_length: int = max_length
        self.question_col: str = question_col
        self.premise_col: str = premise_col
        self.hypothesis_col: str = hypothesis_col
        self.label_col: str = label_col

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        encoded = self.tokenizer(
            str(row[self.premise_col]),
            str(row[self.hypothesis_col]),
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        label: torch.Tensor = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        feature_vector: list[float] = self.feature_extractor.extract_vector(
            question=str(row[self.question_col]),
            answer=str(row[self.hypothesis_col]),
        ).as_list()
        tabular_features: torch.Tensor = torch.tensor(feature_vector, dtype=torch.float32)
        return {key: value.squeeze(0) for key, value in encoded.items()}, label, tabular_features


@dataclass(slots=True)
class TrainArtifacts:
    """Локальные артефакты обучения."""

    model: Any
    tokenizer: Any
    device: torch.device
    output_dir: Path
    best_weights_fpath: Path
    feature_extractor: QAFeatureExtractor
    question_col: str


class NLIWithTabularHead(nn.Module):
    """NLI модель с конкатенацией pooled output и engineered tabular-фичей."""

    def __init__(self, backbone: Any, feature_dim: int, model_dtype: torch.dtype, dropout_prob: float) -> None:
        super().__init__()
        self.backbone: Any = backbone
        hidden_size: int = int(self.backbone.config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_prob)
        self.feature_head: nn.Linear = nn.Linear(hidden_size + feature_dim, 2)
        self.feature_head = self.feature_head.to(dtype=model_dtype)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.feature_head.weight)
            nn.init.zeros_(self.feature_head.bias)

    def forward(self, *, tabular_features: torch.Tensor, **encoded: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(**encoded, output_hidden_states=True, return_dict=True)
        pooled_output: torch.Tensor = outputs.hidden_states[-1][:, 0, :]
        fused: torch.Tensor = torch.cat([pooled_output.float(), tabular_features.float()], dim=-1)
        fused = fused.to(self.feature_head.weight.dtype)
        fused = self.dropout(fused)
        logits: torch.Tensor = self.feature_head(fused)
        return logits

    def push_to_hub(self, repo_id: str, *, token: str) -> None:
        self.backbone.push_to_hub(repo_id, token=token)
        api: HfApi = HfApi(token=token)
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as temp_file:
            temp_fpath: Path = Path(temp_file.name)
        torch.save(self.feature_head.state_dict(), temp_fpath)
        try:
            api.upload_file(
                path_or_fileobj=str(temp_fpath),
                path_in_repo="tabular_head_state.pt",
                repo_id=repo_id,
                repo_type="model",
            )
            api.upload_file(
                path_or_fileobj=json.dumps({"feature_names": FEATURE_NAMES}, ensure_ascii=False, indent=2).encode("utf-8"),
                path_in_repo="tabular_features_meta.json",
                repo_id=repo_id,
                repo_type="model",
            )
        finally:
            if temp_fpath.exists():
                temp_fpath.unlink()


def parse_args() -> tuple[TrainHFNLIConfig, str | None]:
    """Парсит CLI-аргументы и возвращает train-конфиг и HF token."""
    parser = argparse.ArgumentParser(description="Обучение HF NLI классификатора и conditional push в HF")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_SBER_HF_NLI_TRAIN_CONFIG_FPATH))
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--base-model-repo-id", type=str, default=None)
    parser.add_argument("--target-repo-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--compute-dtype", type=str, default=None)
    parser.add_argument("--eval-autocast-dtype", type=str, default=None)
    parser.add_argument("--evaluate-batch-size", type=int, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--push-if-better", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-plots", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hallucination-threshold", type=float, default=None)
    parser.add_argument("--threshold-search-mode", type=str, default=None)
    parser.add_argument("--threshold-search-beta", type=float, default=None)
    parser.add_argument("--threshold-trials", type=int, default=None)
    parser.add_argument("--question-col", type=str, default=None)
    parser.add_argument("--warmup-ratio-per-epoch", type=float, default=None)
    parser.add_argument("--feature-head-dropout", type=float, default=None)
    parser.add_argument("--early-stopping-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--class-weights-before-balancing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-class-weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--hf-token", type=str, default=None)

    namespace: argparse.Namespace = parser.parse_args()
    config: TrainHFNLIConfig = load_config_from_namespace(
        config_cls=TrainHFNLIConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="hf_nli_train",
    )
    return config, namespace.hf_token


def set_seed(seed: int) -> None:
    """Фиксирует random state для воспроизводимости."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_and_prepare_dataframe(csv_fpath: PathLike) -> pd.DataFrame:
    dataframe: pd.DataFrame = pd.read_csv(csv_fpath)
    required_columns: list[str] = ["correct_answer", "model_answer", "is_hallucination"]
    cleaned: pd.DataFrame = dataframe.dropna(subset=required_columns).reset_index(drop=True)
    cleaned["is_hallucination"] = cleaned["is_hallucination"].astype(int)
    return cleaned


def load_train_val_data(config: TrainHFNLIConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает train/val CSV и приводит колонки к нужным типам."""
    train_df: pd.DataFrame = _load_and_prepare_dataframe(config.train_csv)
    val_df: pd.DataFrame = _load_and_prepare_dataframe(config.val_csv)
    logger.info("Train shape=%s, Val shape=%s", train_df.shape, val_df.shape)
    return train_df, val_df


def resolve_question_column(dataframe: pd.DataFrame, configured_question_col: str | None = None) -> str:
    """Определяет колонку вопроса для табличных фичей."""
    if configured_question_col is not None and configured_question_col in dataframe.columns:
        return configured_question_col

    candidates: list[str] = ["query", "question", "prompt", "user_query"]
    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    logger.warning("Колонка вопроса не найдена, использую correct_answer как fallback")
    return "correct_answer"


def build_balanced_train_df(train_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Повторяет notebook-подход: downsample majority до minority."""
    positive: pd.DataFrame = train_df[train_df["is_hallucination"] == 1]
    negative: pd.DataFrame = train_df[train_df["is_hallucination"] == 0]
    if positive.empty or negative.empty:
        logger.warning("Train содержит один класс, балансировку пропускаю")
        return train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    minority_size: int = min(len(positive), len(negative))
    positive_down: pd.DataFrame = resample(positive, n_samples=minority_size, replace=False, random_state=seed)
    negative_down: pd.DataFrame = resample(negative, n_samples=minority_size, replace=False, random_state=seed)
    balanced: pd.DataFrame = pd.concat([positive_down, negative_down], axis=0)
    balanced = balanced.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    logger.info("Balanced train: %s", balanced["is_hallucination"].value_counts().to_dict())
    return balanced


def build_model_and_tokenizer(
    config: TrainHFNLIConfig,
) -> tuple[NLIWithTabularHead, Any, QAFeatureExtractor, torch.device, torch.dtype, torch.dtype]:
    """Создает NLI backbone и tabular head для конкатенации pooled output + features."""
    train_dtype: torch.dtype = _DTYPE_MAP[config.compute_dtype]
    eval_autocast_dtype: torch.dtype = _DTYPE_MAP[config.eval_autocast_dtype]
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_repo_id)
    backbone = AutoModelForSequenceClassification.from_pretrained(
        config.base_model_repo_id,
        num_labels=2,
        ignore_mismatched_sizes=True,
        dtype=train_dtype,
    ).to(device)

    feature_extractor: QAFeatureExtractor = QAFeatureExtractor(tokenizer=tokenizer)
    model: NLIWithTabularHead = NLIWithTabularHead(
        backbone=backbone,
        feature_dim=feature_extractor.feature_dim(),
        model_dtype=train_dtype,
        dropout_prob=config.feature_head_dropout,
    ).to(device)

    logger.info("Model init: base=%s, device=%s, dtype=%s", config.base_model_repo_id, device, train_dtype)
    logger.info("Tabular feature dim=%s, features=%s", feature_extractor.feature_dim(), FEATURE_NAMES)
    return model, tokenizer, feature_extractor, device, train_dtype, eval_autocast_dtype


def build_dataloader(
    dataframe: pd.DataFrame,
    *,
    tokenizer: Any,
    feature_extractor: QAFeatureExtractor,
    question_col: str,
    batch_size: int,
    max_length: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[Any]:
    """Создает DataLoader для HallucinationDataset."""
    dataset = HallucinationDataset(
        dataframe,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        max_length=max_length,
        question_col=question_col,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def compute_class_weights(train_df: pd.DataFrame) -> np.ndarray:
    """Считает веса классов как в notebook: inverse frequency + normalization."""
    class_counts: np.ndarray = train_df["is_hallucination"].value_counts().sort_index().values.astype(np.float64)
    weights: np.ndarray = 1.0 / class_counts
    return weights / weights.sum()


def calculate_warmup_steps(num_batches_per_epoch: int, warmup_ratio_per_epoch: float) -> int:
    """Считает warmup как долю шагов в пределах одной эпохи."""
    return int(num_batches_per_epoch * warmup_ratio_per_epoch)


def select_class_weight_dataframe(
    train_df_raw: pd.DataFrame,
    train_df_balanced: pd.DataFrame,
    *,
    class_weights_before_balancing: bool,
) -> pd.DataFrame:
    """Определяет, на каком train-срезе считать class weights."""
    return train_df_raw if class_weights_before_balancing else train_df_balanced


def build_classification_criterion(
    *,
    train_df_raw: pd.DataFrame,
    train_df_balanced: pd.DataFrame,
    config: TrainHFNLIConfig,
    device: torch.device,
    train_dtype: torch.dtype,
) -> nn.CrossEntropyLoss:
    """Создает criterion с опциональными class weights."""
    if not config.use_class_weights:
        return nn.CrossEntropyLoss()

    weights_df: pd.DataFrame = select_class_weight_dataframe(
        train_df_raw=train_df_raw,
        train_df_balanced=train_df_balanced,
        class_weights_before_balancing=config.class_weights_before_balancing,
    )
    class_weights: np.ndarray = compute_class_weights(weights_df)
    return nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=train_dtype))


def evaluate_model(
    model: Any,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
    positive_class_index: int,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Повторяет evaluate() из notebook: weighted AP + F1 по argmax."""
    model.eval()

    all_preds: list[int] = []
    all_probs: list[float] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch, labels, tabular_features in tqdm(loader, desc="Validation", leave=False):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            tabular_features = tabular_features.to(device)

            if device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(tabular_features=tabular_features, **batch)
            else:
                logits = model(tabular_features=tabular_features, **batch)

            probs = torch.softmax(logits.float(), dim=-1)[:, positive_class_index].cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(float(value) for value in probs.tolist())
            all_preds.extend(int(value) for value in preds.tolist())
            all_labels.extend(int(value) for value in labels.cpu().numpy().tolist())

    y_true: np.ndarray = np.array(all_labels)
    y_pred: np.ndarray = np.array(all_preds)
    y_probs: np.ndarray = np.array(all_probs)

    if np.any(~np.isfinite(y_probs)):
        y_probs = np.nan_to_num(y_probs, nan=0.5)

    pos_rate: float = float(y_true.mean()) if y_true.size else 0.0
    sample_weight: np.ndarray | None = None
    if 0.0 < pos_rate < 1.0:
        sample_weight = np.where(y_true == 1, (1.0 - pos_rate) / pos_rate, 1.0)

    weighted_ap: float = float(average_precision_score(y_true, y_probs, sample_weight=sample_weight))
    f1_value: float = float(f1_score(y_true, y_pred, pos_label=1))
    return weighted_ap, f1_value, y_true, y_pred, y_probs


def train_model(config: TrainHFNLIConfig, hf_token: str | None) -> TrainArtifacts:
    """Запускает цикл обучения и сохраняет лучший checkpoint по weighted AP."""
    _ = hf_token
    set_seed(config.seed)
    train_df_raw, val_df = load_train_val_data(config)
    train_df: pd.DataFrame = build_balanced_train_df(train_df_raw, seed=config.seed)
    question_col_train: str = resolve_question_column(train_df, configured_question_col=config.question_col)
    question_col_val: str = resolve_question_column(val_df, configured_question_col=config.question_col)
    if question_col_train != question_col_val:
        logger.warning("Разные question_col для train/val: %s vs %s", question_col_train, question_col_val)

    model, tokenizer, feature_extractor, device, train_dtype, eval_autocast_dtype = build_model_and_tokenizer(config)

    train_loader = build_dataloader(
        train_df,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        question_col=question_col_train,
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    val_loader = build_dataloader(
        val_df,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        question_col=question_col_val,
        batch_size=config.batch_size,
        max_length=config.max_length,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    warmup_steps: int = calculate_warmup_steps(
        num_batches_per_epoch=len(train_loader),
        warmup_ratio_per_epoch=config.warmup_ratio_per_epoch,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_loader) * config.epochs,
    )
    logger.info("Scheduler warmup steps=%s (ratio=%.3f per epoch)", warmup_steps, config.warmup_ratio_per_epoch)

    criterion = build_classification_criterion(
        train_df_raw=train_df_raw,
        train_df_balanced=train_df,
        config=config,
        device=device,
        train_dtype=train_dtype,
    )

    output_dir: Path = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_weights_fpath: Path = output_dir / config.best_weights_name

    best_ap: float = -1.0
    epochs_without_improvement: int = 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss: float = 0.0
        for batch, labels, tabular_features in tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels = labels.to(device)
            tabular_features = tabular_features.to(device)

            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda" and train_dtype in {torch.float16, torch.bfloat16}:
                with torch.autocast(device_type="cuda", dtype=train_dtype):
                    logits = model(tabular_features=tabular_features, **batch)
                    loss = criterion(logits, labels)
            else:
                logits = model(tabular_features=tabular_features, **batch)
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())

        weighted_ap, f1_value, _, _, _ = evaluate_model(
            model,
            val_loader,
            device=device,
            autocast_dtype=eval_autocast_dtype,
            positive_class_index=config.positive_class_index,
        )
        logger.info(
            "Epoch %s | loss=%.4f | weighted AP=%.6f | F1=%.6f",
            epoch,
            total_loss / max(len(train_loader), 1),
            weighted_ap,
            f1_value,
        )
        if weighted_ap > best_ap:
            best_ap = weighted_ap
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_weights_fpath)
            logger.info("Сохранил лучший checkpoint: %s", best_weights_fpath)
        else:
            epochs_without_improvement += 1
            if config.early_stopping_enabled and epochs_without_improvement >= config.early_stopping_patience:
                logger.info("Early stopping на epoch %s", epoch)
                break

    model.load_state_dict(torch.load(best_weights_fpath, map_location=device))
    model.eval()

    return TrainArtifacts(
        model=model,
        tokenizer=tokenizer,
        device=device,
        output_dir=output_dir,
        best_weights_fpath=best_weights_fpath,
        feature_extractor=feature_extractor,
        question_col=question_col_val,
    )


def score_with_trained_model(
    dataframe: pd.DataFrame,
    *,
    artifacts: TrainArtifacts,
    batch_size: int,
    max_length: int,
    positive_class_index: int,
    autocast_dtype: torch.dtype,
) -> tuple[pd.DataFrame, int]:
    """Скорит DataFrame напрямую обученной моделью в формате evaluate-пайплайна."""
    if "correct_answer" not in dataframe.columns or "model_answer" not in dataframe.columns:
        raise ValueError("Ожидаются колонки correct_answer и model_answer")

    model = artifacts.model
    tokenizer = artifacts.tokenizer
    device = artifacts.device
    feature_extractor: QAFeatureExtractor = artifacts.feature_extractor
    question_col: str = artifacts.question_col if artifacts.question_col in dataframe.columns else resolve_question_column(dataframe)

    hallucination_scores: list[float] = []
    entailment_scores: list[float] = []
    predicted_hallucinations: list[int] = []
    per_sample_times: list[float] = []

    total_batches: int = (len(dataframe) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Scoring trained model")

    for start in range(0, len(dataframe), batch_size):
        batch = dataframe.iloc[start : start + batch_size]
        questions: list[str] = batch[question_col].astype(str).tolist()
        premises: list[str] = batch["correct_answer"].astype(str).tolist()
        hypotheses: list[str] = batch["model_answer"].astype(str).tolist()

        encoded = tokenizer(
            premises,
            hypotheses,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        tabular_features: torch.Tensor = feature_extractor.extract_matrix(
            questions=questions,
            answers=hypotheses,
            device=device,
        )

        infer_start: float = time.perf_counter()
        with torch.inference_mode():
            if device.type == "cuda" and autocast_dtype in {torch.float16, torch.bfloat16}:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits: torch.Tensor = model(tabular_features=tabular_features, **encoded)
            else:
                logits = model(tabular_features=tabular_features, **encoded)
            probs: torch.Tensor = torch.softmax(logits.float(), dim=-1)
        infer_end: float = time.perf_counter()

        batch_hallucination: list[float] = [float(value) for value in probs[:, positive_class_index].cpu().tolist()]
        batch_entailment: list[float] = [1.0 - value for value in batch_hallucination]
        batch_preds: list[int] = [int(int(index) == positive_class_index) for index in logits.argmax(dim=-1).cpu().tolist()]

        batch_time_sec: float = infer_end - infer_start
        batch_sample_time_sec: float = batch_time_sec / max(len(batch), 1)

        hallucination_scores.extend(batch_hallucination)
        entailment_scores.extend(batch_entailment)
        predicted_hallucinations.extend(batch_preds)
        per_sample_times.extend([batch_sample_time_sec] * len(batch))
        progress_bar.update(1)

    progress_bar.close()
    scored: pd.DataFrame = dataframe.copy()
    scored["entailment_score"] = entailment_scores
    scored["hallucination_score"] = hallucination_scores
    scored["pred_is_hallucination"] = predicted_hallucinations
    scored["t_sample_sec"] = per_sample_times
    return scored, total_batches


def build_metrics_payload(summary: EvaluationSummary) -> dict[str, float | str | None]:
    """Готовит JSON-словарь с ключевыми метриками из evaluate-пайплайна."""
    return {
        "weighted_average_precision": summary.classification.weighted_average_precision,
        "average_precision": summary.classification.average_precision,
        "f1": summary.classification.f1,
        "accuracy": summary.classification.accuracy,
        "precision": summary.classification.precision,
        "recall": summary.classification.recall,
        "total_samples": summary.timing.total_samples,
        "throughput_samples_per_sec": summary.timing.throughput_samples_per_sec,
    }


def _find_best_threshold_grid(scores: Sequence[float], labels: Sequence[int], beta: float) -> float:
    y_true: list[int] = [int(value) for value in labels]
    y_score: list[float] = [float(value) for value in scores]
    best_threshold: float = 0.5
    best_metric: float = -1.0
    for step in range(1, 100):
        threshold: float = step / 100.0
        preds: list[int] = [int(score >= threshold) for score in y_score]
        metric_value: float = float(fbeta_score(y_true, preds, beta=beta, pos_label=1, zero_division=0))
        if metric_value > best_metric:
            best_metric = metric_value
            best_threshold = threshold
    return best_threshold


def _find_best_threshold_optuna(
    scores: Sequence[float],
    labels: Sequence[int],
    *,
    beta: float,
    trials: int,
) -> float:
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna не установлена, fallback на grid-search threshold")
        return _find_best_threshold_grid(scores=scores, labels=labels, beta=beta)

    y_true: list[int] = [int(value) for value in labels]
    y_score: list[float] = [float(value) for value in scores]

    def objective(trial: Any) -> float:
        threshold: float = float(trial.suggest_float("threshold", 0.01, 0.99))
        preds: list[int] = [int(score >= threshold) for score in y_score]
        return float(fbeta_score(y_true, preds, beta=beta, pos_label=1, zero_division=0))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return float(study.best_params["threshold"])


def resolve_hallucination_threshold(
    dataframe: pd.DataFrame,
    *,
    threshold_search_mode: str,
    threshold_search_beta: float,
    threshold_trials: int,
    fallback_threshold: float,
) -> float:
    if threshold_search_mode == "none" or "is_hallucination" not in dataframe.columns:
        return fallback_threshold

    scores: list[float] = [float(value) for value in dataframe["hallucination_score"].tolist()]
    labels: list[int] = [int(value) for value in dataframe["is_hallucination"].tolist()]
    if threshold_search_mode == "optuna":
        threshold: float = _find_best_threshold_optuna(
            scores=scores,
            labels=labels,
            beta=threshold_search_beta,
            trials=threshold_trials,
        )
    else:
        threshold = _find_best_threshold_grid(scores=scores, labels=labels, beta=threshold_search_beta)
    return threshold


def apply_hallucination_threshold(dataframe: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    result: pd.DataFrame = dataframe.copy()
    result["pred_is_hallucination"] = [int(float(score) >= threshold) for score in result["hallucination_score"].tolist()]
    return result


def load_remote_metrics(repo_id: str, metrics_filename: str, token: str | None) -> dict[str, Any] | None:
    """Читает baseline метрики из HF репозитория модели."""
    try:
        local_metrics_fpath: str = hf_hub_download(
            repo_id=repo_id,
            filename=metrics_filename,
            repo_type="model",
            token=token,
        )
    except Exception:
        return None

    try:
        return json.loads(Path(local_metrics_fpath).read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_metric(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if payload is None:
        return None
    value: Any = payload.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_candidate_better(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    *,
    primary_key: str,
    secondary_key: str,
    min_relative_improvement: float,
) -> bool:
    """Определяет, улучшилась ли новая модель по primary/secondary метрикам."""
    candidate_primary: float | None = _safe_metric(candidate, primary_key)
    if candidate_primary is None:
        return False

    baseline_primary: float | None = _safe_metric(baseline, primary_key)
    if baseline_primary is None:
        return True

    delta: float = candidate_primary - baseline_primary
    relative_gain: float = delta / max(abs(baseline_primary), 1e-12)
    if delta > 0.0 and relative_gain >= min_relative_improvement:
        return True
    if abs(delta) > 1e-12:
        return False

    candidate_secondary: float | None = _safe_metric(candidate, secondary_key)
    baseline_secondary: float | None = _safe_metric(baseline, secondary_key)
    if candidate_secondary is None or baseline_secondary is None:
        return False
    return candidate_secondary > baseline_secondary


def maybe_push_to_hf(
    *,
    artifacts: TrainArtifacts,
    config: TrainHFNLIConfig,
    hf_token: str | None,
    metrics_payload: dict[str, Any],
) -> bool:
    """Пушит модель и JSON метрик в HF при выполнении quality-gate."""
    baseline_metrics: dict[str, Any] | None = load_remote_metrics(
        repo_id=config.target_repo_id,
        metrics_filename=config.hf_metrics_filename,
        token=hf_token,
    )

    better: bool = is_candidate_better(
        candidate=metrics_payload,
        baseline=baseline_metrics,
        primary_key=config.metric_primary,
        secondary_key=config.metric_secondary,
        min_relative_improvement=config.min_relative_improvement,
    )
    if not better:
        logger.info("Quality-gate: новая модель не лучше baseline, push пропущен")
        return False

    if not config.push_if_better:
        logger.info("Quality-gate пройден, но push_if_better=false")
        return False

    if hf_token is None or not hf_token.strip():
        raise ValueError("Для push в Hugging Face нужен --hf-token")

    metrics_payload = dict(metrics_payload)
    metrics_payload["repo_id"] = config.target_repo_id

    artifacts.model.push_to_hub(config.target_repo_id, token=hf_token)
    artifacts.tokenizer.push_to_hub(config.target_repo_id, token=hf_token)

    api: HfApi = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=json.dumps(metrics_payload, ensure_ascii=False, indent=2).encode("utf-8"),
        path_in_repo=config.hf_metrics_filename,
        repo_id=config.target_repo_id,
        repo_type="model",
    )
    logger.info("Успешно запушил модель и метрики в %s", config.target_repo_id)
    return True


def run(config: TrainHFNLIConfig, hf_token: str | None) -> dict[str, Any]:
    """Тренирует модель, оценивает через evaluate-utils и при улучшении пушит в HF."""
    artifacts: TrainArtifacts = train_model(config=config, hf_token=hf_token)

    val_df: pd.DataFrame = _load_and_prepare_dataframe(config.val_csv)
    scored, total_batches = score_with_trained_model(
        val_df,
        artifacts=artifacts,
        batch_size=config.evaluate_batch_size,
        max_length=config.max_length,
        positive_class_index=config.positive_class_index,
        autocast_dtype=_DTYPE_MAP[config.eval_autocast_dtype],
    )

    selected_threshold: float = resolve_hallucination_threshold(
        scored,
        threshold_search_mode=config.threshold_search_mode,
        threshold_search_beta=config.threshold_search_beta,
        threshold_trials=config.threshold_trials,
        fallback_threshold=config.hallucination_threshold,
    )
    scored = apply_hallucination_threshold(scored, threshold=selected_threshold)
    logger.info("Использую threshold=%.4f (mode=%s)", selected_threshold, config.threshold_search_mode)

    report_dir: Path = Path(config.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    scored_csv: Path = report_dir / "candidate_scores.csv"
    scored.to_csv(scored_csv, index=False)

    summary: EvaluationSummary = evaluate_scoring_results(
        dataframe=scored,
        output_csv_fpath=scored_csv,
        report_dpath=report_dir,
        save_plots_enabled=config.save_plots,
        total_batches=total_batches,
    )

    metrics_payload: dict[str, Any] = build_metrics_payload(summary)
    metrics_payload["selected_threshold"] = selected_threshold
    metrics_payload["threshold_search_mode"] = config.threshold_search_mode
    local_metrics_fpath: Path = report_dir / "candidate_metrics.json"
    local_metrics_fpath.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    pushed: bool = maybe_push_to_hf(
        artifacts=artifacts,
        config=config,
        hf_token=hf_token,
        metrics_payload=metrics_payload,
    )

    result: dict[str, Any] = {
        "pushed": pushed,
        "report_fpath": str(summary.report_fpath) if summary.report_fpath is not None else None,
        "metrics_fpath": str(local_metrics_fpath),
        "scored_csv": str(scored_csv),
    }
    logger.info("Train pipeline finished: %s", result)
    return result


def main() -> None:
    """CLI entrypoint."""
    config, hf_token = parse_args()
    run(config=config, hf_token=hf_token)


if __name__ == "__main__":
    main()




