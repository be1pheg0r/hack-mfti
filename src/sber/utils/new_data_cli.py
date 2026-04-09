from __future__ import annotations

"""Builds a new train dataset by merging positive samples with freshly extracted negative samples."""

import argparse
import tempfile
from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.logger import SBER_DATASETS_LOGGER as logger
from common.paths import PathLike, get_data_raw_dpath, get_sber_configs_dpath
from src.sber.models.tabular_hallucination import FeatureSchema
from src.sber.datasets_utils import SberDatasetsConfig, retrieve_all_datasets, unpack_dataset
from src.sber.utils.extract_features_cli import ScriptConfig as ExtractScriptConfig
from src.sber.utils.extract_features_cli import run as extract_features_run


class NewDataConfig(BaseModel):
    """Конфигурация сборки train датасета.

    Attributes:
        source_csv: CSV с исходными фичами, откуда берутся только positive-сэмплы.
        output_csv: Путь до итогового train CSV.
        datasets_config_path: YAML-конфиг датасетов для negative-сэмплов.
        feature_model_name: Модель для извлечения фичей на negative-сэмплах.
        feature_config_path: YAML-конфиг feature extractor.
        batch_size: Batch size для извлечения фичей.
        seed: Seed для детерминированности.
        force_download: Принудительная загрузка датасетов.
    """

    model_config = ConfigDict(frozen=True)

    source_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / "merged_features_with_judge_scores.csv")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / "new_train_dataset.csv")
    datasets_config_path: PathLike = Field(default_factory=lambda: Path(get_sber_configs_dpath()) / "datasets_configs.yaml")
    feature_model_name: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
    feature_config_path: PathLike = Field(default_factory=lambda: Path(get_sber_configs_dpath()) / "hooks_config.yaml")
    batch_size: int = 2
    seed: int = 42
    force_download: bool = False

    @field_validator("batch_size")
    @classmethod
    def validate_positive_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size должен быть положительным")
        return value


class _PositiveLabelResolver:
    """Нормализует значения колонки target до bool для фильтрации positive-сэмплов."""

    TRUE_VALUES: set[str] = {"1", "true", "yes", "y", "да"}

    @classmethod
    def to_bool(cls, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            return int(value) == 1
        normalized: str = str(value).strip().lower()
        return normalized in cls.TRUE_VALUES


def parse_args() -> NewDataConfig:
    """Парсит CLI-аргументы new_data скрипта."""
    parser = argparse.ArgumentParser(description="Сборка нового train CSV из positive + свежих negative")
    parser.add_argument("--source-csv", type=str, default=str(Path(get_data_raw_dpath()) / "merged_features_with_judge_scores.csv"))
    parser.add_argument("--output-csv", type=str, default=str(Path(get_data_raw_dpath()) / "new_train_dataset.csv"))
    parser.add_argument(
        "--datasets-config-path",
        type=str,
        default=str(Path(get_sber_configs_dpath()) / "datasets_configs.yaml"),
    )
    parser.add_argument("--feature-model-name", type=str, default="ai-sage/GigaChat3-10B-A1.8B-bf16")
    parser.add_argument(
        "--feature-config-path",
        type=str,
        default=str(Path(get_sber_configs_dpath()) / "hooks_config.yaml"),
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force-download", action="store_true")
    namespace: argparse.Namespace = parser.parse_args()
    return NewDataConfig.model_validate(vars(namespace))


def _resolve_target_column(dataframe: pd.DataFrame) -> str:
    """Определяет колонку таргета с поддержкой опечатки hallutination."""
    if "is_hallucination" in dataframe.columns:
        return "is_hallucination"
    if "is_hallutination" in dataframe.columns:
        return "is_hallutination"
    raise ValueError("В source_csv должна быть колонка is_hallucination или is_hallutination")


def _resolve_prompt_column(dataframe: pd.DataFrame) -> str:
    """Определяет колонку prompt/query во входном DataFrame."""
    if "prompt" in dataframe.columns:
        return "prompt"
    if "query" in dataframe.columns:
        return "query"
    raise ValueError("Во входном DataFrame должна быть колонка prompt или query")


def _collect_train_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Возвращает список feature-колонок для train-датасета."""
    canonical_feature_names: set[str] = set(
        FeatureSchema.uncertainty_map
        + FeatureSchema.internal_scalars_map
        + FeatureSchema.probe_vec_map
        + FeatureSchema.attention_entropy_map
        + FeatureSchema.entropy_drops_map
        + FeatureSchema.moe_routing_map
    )
    collected: list[str] = []
    for column_name in dataframe.columns:
        if str(column_name).startswith("feature_") or str(column_name) in canonical_feature_names:
            collected.append(str(column_name))
    return collected


def _normalize_train_view(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только prompt/model_answer/target и фичи."""
    normalized: pd.DataFrame = dataframe.copy()
    prompt_column: str = _resolve_prompt_column(normalized)
    if prompt_column != "prompt":
        normalized["prompt"] = normalized[prompt_column]

    if "model_answer" not in normalized.columns:
        raise ValueError("Во входном DataFrame должна быть колонка model_answer")
    if "is_hallucination" not in normalized.columns:
        raise ValueError("Во входном DataFrame должна быть колонка is_hallucination")

    feature_columns: list[str] = _collect_train_feature_columns(normalized)
    ordered_columns: list[str] = ["prompt", "model_answer", "is_hallucination"] + feature_columns
    return normalized.loc[:, [column for column in ordered_columns if column in normalized.columns]].copy()


def _load_positive_samples(source_csv: PathLike) -> pd.DataFrame:
    """Читает source CSV и оставляет только positive-сэмплы."""
    dataframe: pd.DataFrame = pd.read_csv(source_csv)
    target_col: str = _resolve_target_column(dataframe)

    positive_mask: list[bool] = [_PositiveLabelResolver.to_bool(value) for value in dataframe[target_col].tolist()]
    positives: pd.DataFrame = dataframe.loc[positive_mask].copy()
    if positives.empty:
        raise ValueError("После фильтрации positive-сэмплов датасет пуст")

    if "is_hallucination" not in positives.columns:
        positives["is_hallucination"] = 1
    positives["is_hallucination"] = 1
    return positives


def _build_negative_base_dataframe(config: NewDataConfig) -> pd.DataFrame:
    """Собирает negative-сэмплы из датасетов конфига."""
    datasets_config: SberDatasetsConfig = SberDatasetsConfig.from_yaml(config.datasets_config_path)
    datasets_map: dict[str, Any] = retrieve_all_datasets(config=datasets_config, force_download=config.force_download)

    rows: list[dict[str, Any]] = []
    for dataset_name, dataset in datasets_map.items():
        queries, answers = unpack_dataset(dataset=dataset, name=dataset_name)
        for query, answer in zip(queries, answers):
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "prompt": str(query).strip(),
                    "model_answer": str(answer).strip(),
                    "is_hallucination": 0,
                }
            )

    negatives_raw: pd.DataFrame = pd.DataFrame(rows)
    if negatives_raw.empty:
        raise ValueError("Не удалось собрать negative-сэмплы из датасетов")

    negatives_raw = negatives_raw[negatives_raw["prompt"].astype("string").str.len() > 0].copy()
    negatives_raw = negatives_raw[negatives_raw["model_answer"].astype("string").str.len() > 0].copy()
    negatives_raw = negatives_raw.reset_index(drop=True)
    return negatives_raw


def _extract_negative_features(negatives_raw: pd.DataFrame, config: NewDataConfig) -> pd.DataFrame:
    """Прогоняет negative-сэмплы через feature extractor."""
    with tempfile.TemporaryDirectory(prefix="new_data_") as tmp_dir:
        tmp_dpath: Path = Path(tmp_dir)
        input_fpath: Path = tmp_dpath / "negative_input.csv"
        output_fpath: Path = tmp_dpath / "negative_features.csv"

        negatives_raw.to_csv(input_fpath, index=False)
        extract_config: ExtractScriptConfig = ExtractScriptConfig(
            seed=config.seed,
            model_name=config.feature_model_name,
            batch_size=config.batch_size,
            output_csv=output_fpath,
            force_download=False,
            input_csv_path=input_fpath,
            input_query_column="prompt",
            input_answer_column="model_answer",
            datasets_config_path=config.datasets_config_path,
            feature_config_path=config.feature_config_path,
        )
        extract_features_run(config=extract_config)
        negatives_features: pd.DataFrame = pd.read_csv(output_fpath)

    negatives_features["is_hallucination"] = 0
    negatives_features["hallucination_score"] = negatives_features.get("hallucination_score", 100.0)
    return negatives_features


def run(config: NewDataConfig) -> Path:
    """Собирает итоговый train датасет и сохраняет его на диск."""
    positives: pd.DataFrame = _load_positive_samples(source_csv=config.source_csv)
    negatives_raw: pd.DataFrame = _build_negative_base_dataframe(config=config)
    negatives_features: pd.DataFrame = _extract_negative_features(negatives_raw=negatives_raw, config=config)

    positives = _normalize_train_view(positives)
    negatives_features = _normalize_train_view(negatives_features)

    union_feature_columns: list[str] = sorted(
        set(_collect_train_feature_columns(positives)).union(set(_collect_train_feature_columns(negatives_features)))
    )
    base_columns: list[str] = ["prompt", "model_answer", "is_hallucination"]
    target_columns: list[str] = base_columns + union_feature_columns

    for column_name in target_columns:
        if column_name not in positives.columns:
            positives[column_name] = pd.NA
        if column_name not in negatives_features.columns:
            negatives_features[column_name] = pd.NA

    combined: pd.DataFrame = pd.concat(
        [positives.loc[:, target_columns], negatives_features.loc[:, target_columns]],
        ignore_index=True,
        sort=False,
    )
    dedup_column: str = "prompt"
    before_dedup: int = int(len(combined))
    combined = combined.drop_duplicates(subset=[dedup_column], keep="first").reset_index(drop=True)

    output_path: Path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    logger.info("Новый train датасет сохранен: %s", output_path)
    logger.info("Позитивных сэмплов: %s", len(positives))
    logger.info("Негативных сэмплов: %s", len(negatives_features))
    logger.info("Итог после drop_duplicates по %s: %s -> %s", dedup_column, before_dedup, len(combined))
    return output_path


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()

