from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm.auto import tqdm

from common.logger import SBER_DATASETS_LOGGER as logger
from common.mistral import MistralCallConfig, call_mistral
from common.paths import PathLike, get_data_raw_dpath, get_sber_configs_dpath
from src.sber.datasets_utils import SberDatasetsConfig, retrieve_all_datasets, unpack_dataset


VALID_JUDGE_LABELS: set[str] = {"галлюцинация", "не галлюцинация"}


class GenerateJudgeConfig(BaseModel):
    """Конфигурация генерации model_answer и judge-оценки через Mistral.

    Attributes:
        datasets_config_path: Путь к YAML конфигу датасетов.
        output_csv: Путь к итоговому CSV.
        generation_model: Название Mistral модели для генерации ответа.
        judge_model: Название Mistral модели для judge-оценки.
        generation_temperature: Температура генерации ответа.
        generation_top_p: Top-p генерации ответа.
        judge_temperature: Температура judge-вызова.
        judge_top_p: Top-p judge-вызова.
        max_samples: Опциональный лимит числа семплов.
        force_download: Принудительная перезагрузка датасетов.
        resume_from_output: Подхватывать существующий output_csv и пропускать готовые строки.
        save_every: Периодичность промежуточного сохранения в CSV.
    """

    model_config = ConfigDict(frozen=True)

    datasets_config_path: PathLike = Field(default_factory=lambda: Path(get_sber_configs_dpath()) / "datasets_configs.yaml")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / "mistral_judged_dataset.csv")
    generation_model: str = "mistral-small-latest"
    judge_model: str = "mistral-small-latest"
    generation_temperature: float = 0.3
    generation_top_p: float = 0.95
    judge_temperature: float = 0.1
    judge_top_p: float = 0.9
    max_samples: int | None = None
    force_download: bool = False
    resume_from_output: bool = True
    save_every: int = 20

    @field_validator("generation_model", "judge_model")
    @classmethod
    def validate_non_empty_model(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("model name не может быть пустым")
        return normalized

    @field_validator("generation_temperature", "generation_top_p", "judge_temperature", "judge_top_p")
    @classmethod
    def validate_generation_params(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("Параметр должен быть в диапазоне [0, 1]")
        return value

    @field_validator("max_samples")
    @classmethod
    def validate_max_samples(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("max_samples должен быть положительным")
        return value

    @field_validator("save_every")
    @classmethod
    def validate_save_every(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("save_every должен быть положительным")
        return value


def parse_args() -> GenerateJudgeConfig:
    """Парсит CLI-аргументы скрипта."""
    parser = argparse.ArgumentParser(description="Генерация answer + judge-оценка галлюцинаций через Mistral")
    parser.add_argument("--datasets-config-path", type=str, default=str(Path(get_sber_configs_dpath()) / "datasets_configs.yaml"))
    parser.add_argument("--output-csv", type=str, default=str(Path(get_data_raw_dpath()) / "mistral_judged_dataset.csv"))
    parser.add_argument("--generation-model", type=str, default="mistral-small-latest")
    parser.add_argument("--judge-model", type=str, default="mistral-small-latest")
    parser.add_argument("--generation-temperature", type=float, default=0.3)
    parser.add_argument("--generation-top-p", type=float, default=0.95)
    parser.add_argument("--judge-temperature", type=float, default=0.1)
    parser.add_argument("--judge-top-p", type=float, default=0.9)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-download", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-from-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-every", type=int, default=20)
    args = parser.parse_args()
    return GenerateJudgeConfig.model_validate(vars(args))


def build_base_dataset(config: GenerateJudgeConfig) -> pd.DataFrame:
    """Собирает базовый DataFrame с query/correct_answer из датасетов."""
    datasets_config = SberDatasetsConfig.from_yaml(config.datasets_config_path)
    datasets_map: dict[str, Any] = retrieve_all_datasets(config=datasets_config, force_download=config.force_download)

    rows: list[dict[str, Any]] = []
    for dataset_name, dataset in datasets_map.items():
        queries, answers = unpack_dataset(dataset=dataset, name=dataset_name)
        for query, answer in zip(queries, answers):
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "query": str(query),
                    "correct_answer": str(answer),
                }
            )

    dataframe = pd.DataFrame(rows).dropna(subset=["query", "correct_answer"]).reset_index(drop=True)
    if config.max_samples is not None:
        dataframe = dataframe.head(config.max_samples).copy()

    dataframe["query"] = dataframe["query"].astype("string").str.strip()
    dataframe["correct_answer"] = dataframe["correct_answer"].astype("string").str.strip()
    dataframe = dataframe.drop_duplicates(subset=["query"], keep="first").reset_index(drop=True)

    dataframe["model_answer"] = ""
    dataframe["judge_score"] = pd.NA
    dataframe["is_hallucination"] = pd.NA
    return dataframe


def build_generation_messages(query: str) -> list[dict[str, str]]:
    """Формирует сообщения для генерации ответа на вопрос."""
    return [
        {
            "role": "user",
            "content": query,
        },
    ]


def build_judge_messages(query: str, correct_answer: str, model_answer: str) -> list[dict[str, str]]:
    """Формирует сообщения judge-модели для оценки галлюцинации."""
    judge_prompt = (
        "Ты — судья, оценивающий точность ответа модели. "
        "Сравни ответ модели с эталонным ответом. "
        "Верни JSON строго формата {\"judge_score\": \"галлюцинация\"|\"не галлюцинация\"}.\n\n"
        f"Вопрос: {query}\n"
        f"Правильный ответ: {correct_answer}\n"
        f"Ответ модели: {model_answer}"
    )
    return [
        {
            "role": "system",
            "content": "Ты строгий факт-чекер и судья качества ответов. Отвечай только JSON-объектом.",
        },
        {
            "role": "user",
            "content": judge_prompt,
        },
    ]


def normalize_judge_score(raw_response: str) -> str:
    """Нормализует judge-ответ к одному из допустимых ярлыков."""
    text: str = str(raw_response).strip().lower()

    try:
        parsed: Any = json.loads(text)
        if isinstance(parsed, dict):
            judge_score: Any = parsed.get("judge_score")
            if isinstance(judge_score, str):
                normalized = judge_score.strip().lower()
                if normalized in VALID_JUDGE_LABELS:
                    return normalized
    except json.JSONDecodeError:
        pass

    if "не галлюцинация" in text:
        return "не галлюцинация"
    if "галлюцинация" in text:
        return "галлюцинация"
    return "неизвестно"


def merge_with_existing(base_df: pd.DataFrame, output_csv: PathLike, resume_enabled: bool) -> pd.DataFrame:
    """Подмешивает уже обработанные записи из output CSV при resume."""
    if not resume_enabled:
        return base_df

    output_path: Path = Path(output_csv)
    if not output_path.exists():
        return base_df

    existing_df: pd.DataFrame = pd.read_csv(output_path)
    required_cols: list[str] = ["query", "model_answer", "judge_score", "is_hallucination"]
    for col in required_cols:
        if col not in existing_df.columns:
            logger.warning("В output CSV нет колонки %s, resume пропускаю", col)
            return base_df

    existing_df["query"] = existing_df["query"].astype("string").str.strip()
    existing_df = existing_df.drop_duplicates(subset=["query"], keep="last")

    merge_cols: list[str] = ["query", "model_answer", "judge_score", "is_hallucination"]
    merged = base_df.merge(
        existing_df.loc[:, merge_cols],
        on="query",
        how="left",
        suffixes=("", "_existing"),
    )

    for target_col, existing_col in [
        ("model_answer", "model_answer_existing"),
        ("judge_score", "judge_score_existing"),
        ("is_hallucination", "is_hallucination_existing"),
    ]:
        merged[target_col] = merged[existing_col].combine_first(merged[target_col])
        merged = merged.drop(columns=[existing_col])

    return merged


def save_dataframe(df: pd.DataFrame, output_csv: PathLike) -> Path:
    """Сохраняет dataframe в CSV и возвращает путь."""
    output_path: Path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run(config: GenerateJudgeConfig) -> Path:
    """Запускает пайплайн генерации ответов и judge-оценки."""
    dataframe: pd.DataFrame = build_base_dataset(config)
    dataframe = merge_with_existing(base_df=dataframe, output_csv=config.output_csv, resume_enabled=config.resume_from_output)

    logger.info("Семплов к обработке: %s", len(dataframe))

    generation_call_config = MistralCallConfig(models_list=[config.generation_model])
    judge_call_config = MistralCallConfig(models_list=[config.judge_model])

    generated_counter: int = 0
    for index in tqdm(range(len(dataframe)), desc="Generating model answers"):
        current_answer: str = str(dataframe.at[index, "model_answer"])
        if current_answer.strip():
            continue

        query: str = str(dataframe.at[index, "query"])
        messages: list[dict[str, str]] = build_generation_messages(query=query)
        model_answer: str = call_mistral(
            generation_call_config,
            messages=messages,
            temperature=config.generation_temperature,
            top_p=config.generation_top_p,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            reasoning_effort="medium",
        )
        dataframe.at[index, "model_answer"] = str(model_answer).strip()
        generated_counter += 1

        if generated_counter % config.save_every == 0:
            save_dataframe(dataframe, config.output_csv)

    judge_counter: int = 0
    for index in tqdm(range(len(dataframe)), desc="Judging hallucinations"):
        current_judge: str = str(dataframe.at[index, "judge_score"])
        if current_judge in VALID_JUDGE_LABELS:
            continue

        query = str(dataframe.at[index, "query"])
        correct_answer = str(dataframe.at[index, "correct_answer"])
        model_answer = str(dataframe.at[index, "model_answer"])
        if not model_answer.strip():
            continue

        messages = build_judge_messages(
            query=query,
            correct_answer=correct_answer,
            model_answer=model_answer,
        )
        raw_judge_response: str = call_mistral(
            judge_call_config,
            messages=messages,
            temperature=config.judge_temperature,
            top_p=config.judge_top_p,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            reasoning_effort="high",
            response_format={"type": "json_object"},
        )
        normalized_judge: str = normalize_judge_score(raw_judge_response)
        dataframe.at[index, "judge_score"] = normalized_judge
        if normalized_judge in VALID_JUDGE_LABELS:
            dataframe.at[index, "is_hallucination"] = int(normalized_judge == "галлюцинация")

        judge_counter += 1
        if judge_counter % config.save_every == 0:
            save_dataframe(dataframe, config.output_csv)

    output_path: Path = save_dataframe(dataframe, config.output_csv)
    logger.info("Пайплайн завершен, результат: %s", output_path)
    logger.info("Сгенерировано новых ответов: %s", generated_counter)
    logger.info("Оценено judge-меткой: %s", judge_counter)
    return output_path


def main() -> None:
    """CLI entrypoint."""
    run(parse_args())


if __name__ == "__main__":
    main()



