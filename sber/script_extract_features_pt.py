from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import *
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rapidfuzz import fuzz
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike, get_sber_checkpoints_dpath, get_sber_gitignore_data_dpath
from sber.constants import (
    DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH,
    DEFAULT_HALLUCINATION_SCORE_THRESHOLD,
    DEFAULT_SBER_SCRIPT_BATCH_SIZE,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_MAX,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_MEAN,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_MIN,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_STD,
    DEFAULT_SBER_SCRIPT_MAX_NEW_TOKENS,
    DEFAULT_SBER_SCRIPT_MODEL_NAME,
    DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME,
    DEFAULT_SBER_SCRIPT_SEED,
)
from sber.datasets_utils import SberDatasetsConfig, retrieve_all_datasets, unpack_dataset
from sber.pt import FeatureExtractorConfig, FeatureGroups, LLMFeatureExtractor



class ScriptConfig(BaseModel):
    """Конфигурация запуска скрипта извлечения фичей.

    Attributes:
        n: Количество сэмплов для обработки после перемешивания.
        seed: Seed для воспроизводимого перемешивания.
        model_name: Hugging Face имя модели.
        batch_size: Размер батча для генерации.
        max_new_tokens: Максимум новых токенов при генерации.
        temperature_mean: Среднее температуры для сэмплирования из нормального распределения.
        temperature_std: Стандартное отклонение температуры.
        temperature_min: Нижняя граница температуры после clipping.
        temperature_max: Верхняя граница температуры после clipping.
        output_csv: Путь к итоговому CSV с фичами.
        force_download: Принудительная перезагрузка датасетов.
        datasets_config_path: Опциональный путь к YAML-конфигу датасетов.
        feature_config_path: Опциональный путь к YAML-конфигу фичей.
    """

    model_config = ConfigDict(frozen=True)

    n: int
    seed: int = DEFAULT_SBER_SCRIPT_SEED
    model_name: str = DEFAULT_SBER_SCRIPT_MODEL_NAME
    batch_size: int = DEFAULT_SBER_SCRIPT_BATCH_SIZE
    max_new_tokens: int = DEFAULT_SBER_SCRIPT_MAX_NEW_TOKENS
    temperature_mean: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_MEAN
    temperature_std: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_STD
    temperature_min: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_MIN
    temperature_max: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_MAX
    output_csv: PathLike = Field(default_factory=lambda: Path(get_sber_gitignore_data_dpath()) / DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME)
    force_download: bool = False
    datasets_config_path: PathLike | None = None
    feature_config_path: PathLike | None = DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH

    @field_validator("n", "batch_size", "max_new_tokens")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        """Проверяет, что числовые параметры положительные."""
        if value <= 0:
            raise ValueError("Параметр должен быть положительным")
        return value

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        """Проверяет, что имя модели не пустое."""
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("model_name не может быть пустым")
        return normalized

    @field_validator("temperature_mean", "temperature_min", "temperature_max")
    @classmethod
    def validate_positive_float(cls, value: float) -> float:
        """Проверяет, что параметры температуры больше нуля."""
        if value <= 0.0:
            raise ValueError("Параметры температуры должны быть больше нуля")
        return value

    @field_validator("temperature_std")
    @classmethod
    def validate_non_negative_std(cls, value: float) -> float:
        """Проверяет, что std температуры неотрицательна."""
        if value < 0.0:
            raise ValueError("temperature_std не может быть отрицательной")
        return value

    @model_validator(mode="after")
    def validate_output_suffix(self) -> ScriptConfig:
        """Проверяет расширение файла вывода."""
        output_path: Path = Path(self.output_csv)
        if output_path.suffix.lower() != ".csv":
            raise ValueError("output_csv должен указывать на .csv файл")
        if self.temperature_min > self.temperature_max:
            raise ValueError("temperature_min не может быть больше temperature_max")
        return self


def parse_args() -> ScriptConfig:
    """Парсит CLI-аргументы и возвращает валидированную конфигурацию."""
    parser = argparse.ArgumentParser(description="Извлечение фичей из LLM для Sber кейса")
    parser.add_argument("--n", type=int, required=True, help="Обязательное число обрабатываемых сэмплов")
    parser.add_argument("--seed", type=int, default=DEFAULT_SBER_SCRIPT_SEED, help="Seed для shuffle")
    parser.add_argument("--model-name", type=str, default=DEFAULT_SBER_SCRIPT_MODEL_NAME, help="Имя модели на Hugging Face")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SBER_SCRIPT_BATCH_SIZE, help="Размер батча для генерации")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_SBER_SCRIPT_MAX_NEW_TOKENS, help="Максимум новых токенов")
    parser.add_argument(
        "--temperature-mean",
        type=float,
        default=DEFAULT_SBER_SCRIPT_TEMPERATURE_MEAN,
        help="Средняя температура генерации (normal distribution)",
    )
    parser.add_argument(
        "--temperature-std",
        type=float,
        default=DEFAULT_SBER_SCRIPT_TEMPERATURE_STD,
        help="Std температуры генерации (normal distribution)",
    )
    parser.add_argument(
        "--temperature-min",
        type=float,
        default=DEFAULT_SBER_SCRIPT_TEMPERATURE_MIN,
        help="Минимальная температура после clipping",
    )
    parser.add_argument(
        "--temperature-max",
        type=float,
        default=DEFAULT_SBER_SCRIPT_TEMPERATURE_MAX,
        help="Максимальная температура после clipping",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path(get_sber_gitignore_data_dpath()) / DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME),
        help="Путь до итогового CSV",
    )
    parser.add_argument("--force-download", action="store_true", help="Принудительно перекачать датасеты")
    parser.add_argument("--datasets-config-path", type=str, default=None, help="Путь до YAML конфига datasets")
    parser.add_argument(
        "--feature-config-path",
        type=str,
        default=str(DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH),
        help="Путь до YAML конфига фичей",
    )
    namespace: argparse.Namespace = parser.parse_args()
    return ScriptConfig.model_validate(vars(namespace))


def _build_prompt(tokenizer: Any, query: str) -> str:
    """Формирует текст промпта с поддержкой chat-template."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=False,
        )
    return query


def _batched_pairs(queries: list[str], answers: list[str], batch_size: int) -> Iterator[tuple[list[str], list[str]]]:
    """Возвращает батчи пар вопрос-ответ."""
    for start in range(0, len(queries), batch_size):
        stop: int = min(start + batch_size, len(queries))
        yield queries[start:stop], answers[start:stop]


def _sample_queries_and_answers(
    queries: list[str],
    answers: list[str],
    n: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Перемешивает и выбирает n сэмплов."""
    if len(queries) != len(answers):
        raise ValueError("queries и answers должны быть одной длины")
    total_size: int = len(queries)
    if n > total_size:
        raise ValueError(f"Параметр n={n} больше размера датасета ({total_size})")

    paired: list[tuple[str, str]] = list(zip(queries, answers))
    rng: random.Random = random.Random()
    rng.shuffle(paired)
    sampled_pairs: list[tuple[str, str]] = paired[:n]
    sampled_queries: list[str] = [query for query, _ in sampled_pairs]
    sampled_answers: list[str] = [answer for _, answer in sampled_pairs]
    return sampled_queries, sampled_answers


def _sample_balanced_queries_and_answers(
    datasets_map: dict[str, Iterable[dict[str, Any]]],
    n: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Равномерно берет сэмплы из каждого датасета и перемешивает общий пул."""
    dataset_count: int = len(datasets_map)
    if dataset_count == 0:
        raise ValueError("datasets_map не должен быть пустым")
    if n % dataset_count != 0:
        raise ValueError(
            f"Параметр n={n} должен делиться на число датасетов ({dataset_count}) для равномерной выборки"
        )

    per_dataset: int = n // dataset_count
    rng: random.Random = random.Random(seed)
    sampled_pairs: list[tuple[str, str]] = []

    for dataset_name, dataset in datasets_map.items():
        queries, answers = unpack_dataset(dataset=dataset, name=dataset_name)
        if len(queries) != len(answers):
            raise ValueError(f"Датасет {dataset_name}: queries и answers должны быть одной длины")
        if len(queries) < per_dataset:
            raise ValueError(
                f"Датасет {dataset_name} содержит недостаточно сэмплов: {len(queries)} < {per_dataset}"
            )

        pairs: list[tuple[str, str]] = list(zip(queries, answers))
        rng.shuffle(pairs)
        sampled_pairs.extend(pairs[:per_dataset])

    rng.shuffle(sampled_pairs)
    sampled_queries: list[str] = [query for query, _ in sampled_pairs]
    sampled_answers: list[str] = [answer for _, answer in sampled_pairs]
    return sampled_queries, sampled_answers


def _flatten_feature_groups(features: FeatureGroups) -> list[float]:
    """Преобразует FeatureGroups в плоский список в фиксированном порядке."""
    ordered_groups: list[list[float]] = [
        features.uncertainty,
        features.internal_scalars,
        features.probe_vec,
        features.attention_entropy,
        features.entropy_drops,
        features.moe_routing,
    ]
    flattened: list[float] = []
    for group in ordered_groups:
        flattened.extend(float(value) for value in group)
    return flattened


def _sample_temperature(
    rng: random.Random,
    mean: float,
    std: float,
    min_value: float,
    max_value: float,
) -> float:
    """Сэмплирует температуру из нормального распределения и ограничивает диапазон."""
    sampled: float = mean if std == 0.0 else rng.normalvariate(mean, std)
    return max(min_value, min(max_value, sampled))


def _extract_logits(model_output: Any) -> torch.Tensor:
    """Извлекает логиты из выхода модели."""
    if hasattr(model_output, "logits"):
        return model_output.logits
    if isinstance(model_output, dict) and "logits" in model_output:
        logits_value: Any = model_output["logits"]
        if isinstance(logits_value, torch.Tensor):
            return logits_value
    raise ValueError("Не удалось извлечь logits из выхода модели")


def _prepare_tokenizer(tokenizer: Any) -> Any:
    """Настраивает токенизатор для батчевой генерации на decoder-only моделях."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_model_source(model_name: str) -> str:
    """Возвращает локальный путь модели при наличии, иначе исходное имя HF."""
    model_dir_name: str = model_name.rsplit("/", maxsplit=1)[-1]



    local_model_path: Path = Path(get_sber_checkpoints_dpath()) / model_dir_name
    logger.info(f"Пытаюсь найти модель {model_name} в локальных чекпоинтах по пути: {local_model_path}")
    if local_model_path.exists():
        logger.info(f"Загружаю модель из локального пути: {local_model_path}")
        return str(local_model_path)
    logger.warning(f"Локальный путь для модели {model_name} не найден, пробую загрузить из Hugging Face")
    return model_name


def _build_model_and_tokenizer(model_name: str) -> tuple[Any, Any, torch.device]:
    """Загружает модель и токенизатор с учетом multi-GPU конфигурации."""
    if torch.cuda.is_available():
        gpu_count: int = torch.cuda.device_count()
        torch.backends.cuda.matmul.allow_tf32 = True
        compute_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Доступно GPU: {gpu_count}, включаю device_map=auto и dtype={compute_dtype}")

        model = AutoModelForCausalLM.from_pretrained(
            _resolve_model_source(model_name),
            torch_dtype=compute_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(_resolve_model_source(model_name), trust_remote_code=True)
        model.eval()
        return model, _prepare_tokenizer(tokenizer), torch.device("cuda:0")

    logger.info("GPU не обнаружены, запускаю на CPU")
    model = AutoModelForCausalLM.from_pretrained(
        _resolve_model_source(model_name),
        torch_dtype=torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(_resolve_model_source(model_name), trust_remote_code=True)
    model.eval()
    return model, _prepare_tokenizer(tokenizer), torch.device("cpu")


def run(config: ScriptConfig) -> Path:
    """Запускает полный пайплайн извлечения и сохранения фичей.

    Args:
        config: Конфигурация запуска.

    Returns:
        Путь до сохраненного CSV файла.
    """
    datasets_config: SberDatasetsConfig
    if config.datasets_config_path is None:
        datasets_config = SberDatasetsConfig()
    else:
        datasets_config = SberDatasetsConfig.from_yaml(config.datasets_config_path)

    datasets_map: dict[str, Any] = retrieve_all_datasets(
        config=datasets_config,
        force_download=config.force_download,
    )
    sampled_queries, sampled_answers = _sample_balanced_queries_and_answers(
        datasets_map=datasets_map,
        n=config.n,
        seed=config.seed,
    )
    per_dataset: int = config.n // len(datasets_map)
    logger.info(
        f"После balanced sampling и shuffle выбрано {len(sampled_queries)} сэмплов "
        f"({per_dataset} из каждого датасета)"
    )

    model, tokenizer, input_device = _build_model_and_tokenizer(config.model_name)
    feature_config: FeatureExtractorConfig
    if config.feature_config_path is None:
        feature_config = FeatureExtractorConfig.from_default_yaml()
    else:
        feature_config = FeatureExtractorConfig.from_yaml(config.feature_config_path)
    extractor: LLMFeatureExtractor = LLMFeatureExtractor(model=model, config=feature_config)

    output_rows: list[dict[str, Any]] = []
    feature_columns: list[str] = []
    sample_counter: int = 0
    temperature_rng: random.Random = random.Random(config.seed)

    with torch.inference_mode():
        with extractor:
            total_batches: int = (len(sampled_queries) + config.batch_size - 1) // config.batch_size
            progress_bar = tqdm(total=total_batches, desc="Извлечение фичей")
            for batch_queries, batch_answers in _batched_pairs(
                queries=sampled_queries,
                answers=sampled_answers,
                batch_size=config.batch_size,
            ):
                batch_temperature: float = _sample_temperature(
                    rng=temperature_rng,
                    mean=config.temperature_mean,
                    std=config.temperature_std,
                    min_value=config.temperature_min,
                    max_value=config.temperature_max,
                )
                prompts: list[str] = [_build_prompt(tokenizer=tokenizer, query=query) for query in batch_queries]
                encoded: dict[str, torch.Tensor] = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                )
                input_ids: torch.Tensor = encoded["input_ids"].to(input_device)
                attention_mask: torch.Tensor = encoded["attention_mask"].to(input_device)

                generated: torch.Tensor = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=config.max_new_tokens,
                    do_sample=True,
                    temperature=batch_temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                padded_prompt_len: int = int(input_ids.shape[1])
                prompt_lengths: list[int] = [int(value) for value in attention_mask.sum(dim=1).tolist()]

                for index_in_batch in range(len(batch_queries)):
                    full_sequence: torch.Tensor = generated[index_in_batch]
                    prompt_len: int = prompt_lengths[index_in_batch]
                    left_padding_size: int = padded_prompt_len - prompt_len
                    if left_padding_size > 0:
                        full_sequence = full_sequence[left_padding_size:]

                    answer_start: int = prompt_len
                    answer_ids: torch.Tensor = full_sequence[answer_start:]
                    model_answer: str = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    full_ids: torch.Tensor = full_sequence.unsqueeze(0)

                    model_output: Any = extractor(full_ids)
                    logits: torch.Tensor = _extract_logits(model_output)
                    extracted_features: FeatureGroups = extractor.extract(
                        logits=logits,
                        input_ids=full_ids,
                        answer_start=answer_start,
                    )

                    feature_vector: list[float] = _flatten_feature_groups(extracted_features)
                    if not feature_columns:
                        feature_columns = [f"feature_{feature_index}" for feature_index in range(len(feature_vector))]

                    hallucination_score: float = float(
                        fuzz.partial_ratio(model_answer.lower(), batch_answers[index_in_batch].lower())
                    )
                    is_hallucination: int = int(hallucination_score < DEFAULT_HALLUCINATION_SCORE_THRESHOLD)

                    row: dict[str, Any] = {
                        "sample_id": sample_counter,
                        "query": batch_queries[index_in_batch],
                        "ground_truth": batch_answers[index_in_batch],
                        "model_answer": model_answer,
                        "temperature": batch_temperature,
                        "hallucination_score": hallucination_score,
                        "is_hallucination": is_hallucination,
                    }
                    row.update({feature_name: value for feature_name, value in zip(feature_columns, feature_vector)})
                    output_rows.append(row)
                    sample_counter += 1

                    # gc
                    del full_ids, logits, extracted_features, feature_vector

                progress_bar.update(1)
            progress_bar.close()

    dataframe: pd.DataFrame = pd.DataFrame(output_rows)
    output_path: Path = Path(config.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    hallucinations_count: int = int(dataframe["is_hallucination"].sum())
    hallucinations_ratio: float = (hallucinations_count / max(len(dataframe), 1)) * 100.0
    logger.info(f"CSV сохранен: {output_path}")
    logger.info(f"Обработано сэмплов: {len(dataframe)}")
    logger.info(f"Галлюцинаций: {hallucinations_count} ({hallucinations_ratio:.1f}%)")
    return output_path


def main() -> None:
    """Точка входа CLI-скрипта."""
    config: ScriptConfig = parse_args()
    run(config=config)


if __name__ == "__main__":
    main()
