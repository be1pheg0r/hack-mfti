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


def _load_positive_samples(source_csv: PathLike) -> pd.DataFrame:
    """Читает source CSV и оставляет только positive-сэмплы."""
    dataframe: pd.DataFrame = pd.read_csv(source_csv)
    target_col: str = _resolve_target_column(dataframe)
    prompt_col: str = _resolve_prompt_column(dataframe)

    if "model_answer" not in dataframe.columns:
        raise ValueError("В source_csv должна быть колонка model_answer")

    positive_mask: list[bool] = [_PositiveLabelResolver.to_bool(value) for value in dataframe[target_col].tolist()]
    positives: pd.DataFrame = dataframe.loc[positive_mask, [prompt_col, "model_answer"]].copy()
    if positives.empty:
        raise ValueError("После фильтрации positive-сэмплов датасет пуст")

    positives = positives.rename(columns={prompt_col: "prompt"})
    positives["is_hallucination"] = 1
    positives = positives[positives["prompt"].astype("string").str.len() > 0].copy()
    positives = positives[positives["model_answer"].astype("string").str.len() > 0].copy()
    positives = positives.reset_index(drop=True)
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


def _extract_features(rows: pd.DataFrame, config: NewDataConfig) -> pd.DataFrame:
    """Прогоняет строки через feature extractor и оставляет только base+feature_* колонки."""
    with tempfile.TemporaryDirectory(prefix="new_data_") as tmp_dir:
        tmp_dpath: Path = Path(tmp_dir)
        input_fpath: Path = tmp_dpath / "input.csv"
        output_fpath: Path = tmp_dpath / "features.csv"

        rows.to_csv(input_fpath, index=False)
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
        extracted: pd.DataFrame = pd.read_csv(output_fpath)

    if "prompt" not in extracted.columns and "query" in extracted.columns:
        extracted["prompt"] = extracted["query"]
    if "prompt" not in extracted.columns or "model_answer" not in extracted.columns:
        raise ValueError("После extract_features ожидаются колонки prompt/query и model_answer")
    if "is_hallucination" not in extracted.columns:
        extracted["is_hallucination"] = pd.NA

    feature_columns: list[str] = [column for column in extracted.columns if str(column).startswith("feature_")]
    return extracted.loc[:, ["prompt", "model_answer", "is_hallucination"] + feature_columns].copy()


def run(config: NewDataConfig) -> Path:
    """Собирает итоговый train датасет и сохраняет его на диск."""
    positives_raw: pd.DataFrame = _load_positive_samples(source_csv=config.source_csv)
    negatives_raw: pd.DataFrame = _build_negative_base_dataframe(config=config)
    positives_features: pd.DataFrame = _extract_features(rows=positives_raw, config=config)
    negatives_features: pd.DataFrame = _extract_features(rows=negatives_raw, config=config)

    positives_features["is_hallucination"] = 1
    negatives_features["is_hallucination"] = 0

    feature_columns: list[str] = sorted(
        set([column for column in positives_features.columns if str(column).startswith("feature_")]).union(
            set([column for column in negatives_features.columns if str(column).startswith("feature_")])
        )
    )
    target_columns: list[str] = ["prompt", "model_answer", "is_hallucination"] + feature_columns

    for column_name in target_columns:
        if column_name not in positives_features.columns:
            positives_features[column_name] = pd.NA
        if column_name not in negatives_features.columns:
            negatives_features[column_name] = pd.NA

    combined: pd.DataFrame = pd.concat(
        [positives_features.loc[:, target_columns], negatives_features.loc[:, target_columns]],
        ignore_index=True,
        sort=False,
    )
    combined = combined.reset_index(drop=True)

    output_path: Path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    logger.info("Новый train датасет сохранен: %s", output_path)
    logger.info("Позитивных сэмплов: %s", len(positives_features))
    logger.info("Негативных сэмплов: %s", len(negatives_features))
    logger.info("Итоговый размер train датасета: %s", len(combined))
    return output_path


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()

