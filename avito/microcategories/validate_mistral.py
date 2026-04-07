from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer

from avito.config import AvitoCaseConfig
from avito.microcategories.inference import load_microcategory_artifact, predict_microcategories
from common.logger import (
    AVITO_MICROCATS_LOGGER,
    LOGGERS,
    LEVEL_MAP,
    ColoredFormatter,
    MISTRAL_LOGGER,
)
from common.mistral import MistralCallConfig
from common.paths import get_avito_checkpoints_dpath, get_avito_data_dpath


TARGET_COLUMN = "targetDetectedMcIds"
SPLIT_COLUMN = "split"


def _parse_target_ids(value: Any) -> list[int]:
    if isinstance(value, list):
        result: list[int] = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001
            return []
        if not isinstance(parsed, list):
            return []
        result = []
        for item in parsed:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result

    return []


def _ensure_stdout_handler(logger: logging.Logger, *, level_name: str, prefix: str) -> None:
    level = LEVEL_MAP[level_name]
    formatter = ColoredFormatter(LOGGERS.avito_microcategories.log_format, prefix=prefix)

    # Для standalone-CLI оставляем только один stdout-хендлер, чтобы не дублировать вывод.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)


def _build_arg_parser() -> argparse.ArgumentParser:
    case_config = AvitoCaseConfig.from_default_yaml()
    mistral_defaults = case_config.microcategories.mistral_inference

    parser = argparse.ArgumentParser(
        description="Отдельная Mistral-only валидация микрокатегорий без обучения и без sklearn-инференса.",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(get_avito_data_dpath()) / case_config.should_split.cli_defaults.dataset_filename),
        help="Путь к CSV-датасету.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(Path(get_avito_checkpoints_dpath()) / "avito_microcategories_model.joblib"),
        help="Путь к артефакту микрокатегорий.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
        help="Какой split оценивать.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Сколько строк взять из среза (<=0 означает весь срез).",
    )
    parser.add_argument(
        "--sample-strategy",
        type=str,
        default="head",
        choices=["head", "random"],
        help="Стратегия выбора подвыборки из split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed для случайной подвыборки (--sample-strategy random).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=mistral_defaults.models_list,
        help="Список Mistral моделей по приоритету.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=mistral_defaults.timeout,
        help="Таймаут одного вызова Mistral в секундах.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=mistral_defaults.max_attempts_per_call,
        help="Число попыток на один API вызов.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Уровень логов для avito/mistral логгеров.",
    )
    return parser


def _select_split(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name == "all":
        return df.copy()

    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"В датасете отсутствует колонка '{SPLIT_COLUMN}'")
    return df[df[SPLIT_COLUMN].astype(str).eq(split_name)].copy()


def _sample_frame(df: pd.DataFrame, *, sample_size: int, strategy: str, random_state: int) -> pd.DataFrame:
    if sample_size <= 0 or len(df) <= sample_size:
        return df
    if strategy == "random":
        return df.sample(n=sample_size, random_state=random_state)
    return df.head(sample_size)


def _build_true_labels(df: pd.DataFrame, *, allowed_ids: set[int]) -> list[list[int]]:
    true_labels: list[list[int]] = []

    for _, row in df.iterrows():
        source_raw = row.get("sourceMcId")
        source_mc_id: int | None
        try:
            source_mc_id = int(source_raw) if source_raw is not None else None
        except (TypeError, ValueError):
            source_mc_id = None

        labels: list[int] = []
        seen: set[int] = set()
        for mc_id in _parse_target_ids(row.get(TARGET_COLUMN)):
            if source_mc_id is not None and mc_id == source_mc_id:
                continue
            if mc_id not in allowed_ids or mc_id in seen:
                continue
            seen.add(mc_id)
            labels.append(mc_id)
        true_labels.append(labels)

    return true_labels


def main() -> None:
    args = _build_arg_parser().parse_args()

    _ensure_stdout_handler(
        AVITO_MICROCATS_LOGGER,
        level_name=args.log_level,
        prefix=LOGGERS.avito_microcategories.prefix,
    )
    _ensure_stdout_handler(
        MISTRAL_LOGGER,
        level_name=args.log_level,
        prefix=LOGGERS.mistral_call.prefix,
    )

    data_path = Path(args.data_path)
    artifact_path = Path(args.artifact_path)

    if not data_path.exists():
        raise FileNotFoundError(f"CSV-датасет не найден: {data_path}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Артефакт микрокатегорий не найден: {artifact_path}")

    AVITO_MICROCATS_LOGGER.info("Mistral-only validation started")
    AVITO_MICROCATS_LOGGER.info("data_path=%s", data_path)
    AVITO_MICROCATS_LOGGER.info("artifact_path=%s", artifact_path)

    df = pd.read_csv(data_path)
    required_columns = {"description", "sourceMcId", "sourceMcTitle", TARGET_COLUMN}
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {missing_columns}")

    split_df = _select_split(df, args.split)
    if split_df.empty:
        raise ValueError(f"Пустой срез split='{args.split}'")

    eval_df = _sample_frame(
        split_df,
        sample_size=args.sample_size,
        strategy=args.sample_strategy,
        random_state=args.random_state,
    ).reset_index(drop=True)
    AVITO_MICROCATS_LOGGER.info(
        "evaluation_rows=%d (from split_rows=%d, split=%s)",
        len(eval_df),
        len(split_df),
        args.split,
    )

    artifact = load_microcategory_artifact(artifact_path)
    mistral_config = MistralCallConfig(
        models_list=args.models,
        timeout=args.timeout,
        max_attempts_per_call=args.max_attempts,
    )

    prediction_result = predict_microcategories(
        eval_df,
        artifact=artifact,
        backend="mistral",
        mistral_config=mistral_config,
    )

    allowed_ids = set(artifact.mlb_classes)
    y_true_labels = _build_true_labels(eval_df, allowed_ids=allowed_ids)
    y_pred_labels = prediction_result.detected_mc_ids

    mlb = MultiLabelBinarizer(classes=artifact.mlb_classes)
    y_true = mlb.fit_transform(y_true_labels)
    y_pred = mlb.transform(y_pred_labels)

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average="micro", zero_division=0)

    metrics = {
        "backend": "mistral",
        "split": args.split,
        "sample_size": int(len(eval_df)),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "models": list(args.models),
        "timeout": int(args.timeout),
        "max_attempts": int(args.max_attempts),
    }

    AVITO_MICROCATS_LOGGER.info("Mistral-only validation finished")
    print("FINAL_METRICS=" + json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
