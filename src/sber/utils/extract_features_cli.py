from __future__ import annotations

import argparse
import subprocess
import random
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import *

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from rapidfuzz import fuzz
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike, get_data_raw_dpath, get_model_dpath, get_sber_configs_dpath
from src.sber.constants import (
    DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH,
    DEFAULT_HALLUCINATION_SCORE_THRESHOLD,
    DEFAULT_SBER_SCRIPT_ENABLE_AUTO_SUBPROCESS,
    DEFAULT_SBER_SCRIPT_BATCH_SIZE,
    DEFAULT_SBER_SCRIPT_SUBPROCESS_MAX_WORKERS,
    DEFAULT_SBER_SCRIPT_SUBPROCESS_MIN_FREE_GPU_MEMORY_MIB,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_MAX,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_MEAN,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_MIN,
    DEFAULT_SBER_SCRIPT_TEMPERATURE_STD,
    DEFAULT_SBER_SCRIPT_MAX_NEW_TOKENS,
    DEFAULT_SBER_SCRIPT_MODEL_NAME,
    DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME,
    DEFAULT_SBER_SCRIPT_SEED,
)
from src.sber.datasets_utils import SberDatasetsConfig, retrieve_all_datasets, unpack_dataset
from src.sber.models.extract_features import FeatureExtractorConfig, FeatureGroups, LLMFeatureExtractor



class ScriptConfig(BaseModel):
    """Конфигурация запуска скрипта извлечения фичей.

    Attributes:
        n: Количество сэмплов для обработки после перемешивания.
            Если не задано, обрабатываются все сэмплы из всех датасетов.
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
        enable_auto_subprocess: Включает авто-запуск дополнительных subprocess при свободной VRAM.
        subprocess_max_workers: Максимальное число процессов для параллельной обработки.
        subprocess_min_free_gpu_memory_mib: Минимум свободной VRAM для добавления одного процесса.
        datasets_config_path: Опциональный путь к YAML-конфигу датасетов.
        feature_config_path: Опциональный путь к YAML-конфигу фичей.
    """

    model_config = ConfigDict(frozen=True)

    n: int | None = None
    seed: int = DEFAULT_SBER_SCRIPT_SEED
    model_name: str = DEFAULT_SBER_SCRIPT_MODEL_NAME
    batch_size: int = DEFAULT_SBER_SCRIPT_BATCH_SIZE
    max_new_tokens: int = DEFAULT_SBER_SCRIPT_MAX_NEW_TOKENS
    temperature_mean: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_MEAN
    temperature_std: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_STD
    temperature_min: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_MIN
    temperature_max: float = DEFAULT_SBER_SCRIPT_TEMPERATURE_MAX
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME)
    force_download: bool = False
    enable_auto_subprocess: bool = DEFAULT_SBER_SCRIPT_ENABLE_AUTO_SUBPROCESS
    subprocess_max_workers: int = DEFAULT_SBER_SCRIPT_SUBPROCESS_MAX_WORKERS
    subprocess_min_free_gpu_memory_mib: int = DEFAULT_SBER_SCRIPT_SUBPROCESS_MIN_FREE_GPU_MEMORY_MIB
    datasets_config_path: PathLike = Field(default_factory=lambda: Path(get_sber_configs_dpath()) / "datasets_configs.yaml")
    feature_config_path: PathLike | None = DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH

    @field_validator("batch_size", "max_new_tokens")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        """Проверяет, что числовые параметры положительные."""
        if value <= 0:
            raise ValueError("Параметр должен быть положительным")
        return value

    @field_validator("subprocess_max_workers", "subprocess_min_free_gpu_memory_mib")
    @classmethod
    def validate_subprocess_params(cls, value: int) -> int:
        """Проверяет параметры автозапуска subprocess."""
        if value <= 0:
            raise ValueError("Параметр subprocess должен быть положительным")
        return value

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int | None) -> int | None:
        """Проверяет, что n положительный, если параметр задан."""
        if value is not None and value <= 0:
            raise ValueError("Параметр n должен быть положительным")
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
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Число обрабатываемых сэмплов; если не указано, будут обработаны все сэмплы",
    )
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
        default=str(Path(get_data_raw_dpath()) / DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME),
        help="Путь до итогового CSV",
    )
    parser.add_argument("--force-download", action="store_true", help="Принудительно перекачать датасеты")
    parser.add_argument(
        "--enable-auto-subprocess",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_SBER_SCRIPT_ENABLE_AUTO_SUBPROCESS,
        help="Автоматически запускать дополнительные subprocess при достаточной свободной VRAM",
    )
    parser.add_argument(
        "--subprocess-max-workers",
        type=int,
        default=DEFAULT_SBER_SCRIPT_SUBPROCESS_MAX_WORKERS,
        help="Максимальное число процессов для параллельной обработки",
    )
    parser.add_argument(
        "--subprocess-min-free-gpu-memory-mib",
        type=int,
        default=DEFAULT_SBER_SCRIPT_SUBPROCESS_MIN_FREE_GPU_MEMORY_MIB,
        help="Минимум свободной VRAM (MiB) для запуска дополнительного процесса",
    )
    parser.add_argument(
        "--datasets-config-path",
        type=str,
        default=str(Path(get_sber_configs_dpath()) / "datasets_configs.yaml"),
        help="Путь до YAML конфига datasets",
    )
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
    rng: random.Random = random.Random(seed)
    rng.shuffle(paired)
    sampled_pairs: list[tuple[str, str]] = paired[:n]
    sampled_queries: list[str] = [query for query, _ in sampled_pairs]
    sampled_answers: list[str] = [answer for _, answer in sampled_pairs]
    return sampled_queries, sampled_answers


def _sample_balanced_queries_and_answers(
    datasets_map: dict[str, Iterable[dict[str, Any]]],
    n: int | None,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Выбирает сэмплы из датасетов и перемешивает общий пул.

    Если `n` указан, делает balanced sampling: одинаковое число из каждого датасета.
    Если `n` не указан, берет все сэмплы из всех датасетов.
    """
    dataset_count: int = len(datasets_map)
    if dataset_count == 0:
        raise ValueError("datasets_map не должен быть пустым")
    if n is not None and n % dataset_count != 0:
        raise ValueError(
            f"Параметр n={n} должен делиться на число датасетов ({dataset_count}) для равномерной выборки"
        )

    rng: random.Random = random.Random(seed)
    sampled_pairs: list[tuple[str, str]] = []
    per_dataset: int | None = None if n is None else n // dataset_count

    for dataset_name, dataset in datasets_map.items():
        queries, answers = unpack_dataset(dataset=dataset, name=dataset_name)
        if len(queries) != len(answers):
            raise ValueError(f"Датасет {dataset_name}: queries и answers должны быть одной длины")
        if per_dataset is not None and len(queries) < per_dataset:
            raise ValueError(
                f"Датасет {dataset_name} содержит недостаточно сэмплов: {len(queries)} < {per_dataset}"
            )

        pairs: list[tuple[str, str]] = list(zip(queries, answers))
        rng.shuffle(pairs)
        if per_dataset is None:
            sampled_pairs.extend(pairs)
        else:
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


def _build_feature_column_names(features: FeatureGroups, probe_layers: Sequence[int]) -> list[str]:
    """Возвращает семантические имена колонок фичей в порядке flatten()."""
    uncertainty_names: list[str] = [
        "feature_uncertainty_token_logprob_mean",
        "feature_uncertainty_token_logprob_min",
        "feature_uncertainty_token_logprob_max",
        "feature_uncertainty_token_logprob_std",
        "feature_uncertainty_entropy_mean",
        "feature_uncertainty_entropy_min",
        "feature_uncertainty_entropy_max",
        "feature_uncertainty_entropy_std",
        "feature_uncertainty_answer_token_count",
        "feature_uncertainty_first_token_logprob",
        "feature_uncertainty_top1_prob_mean",
        "feature_uncertainty_top5_prob_mean",
    ]
    if len(features.uncertainty) != len(uncertainty_names):
        uncertainty_names = [f"feature_uncertainty_{index}" for index in range(len(features.uncertainty))]

    internal_names: list[str] = []
    for layer_idx in probe_layers:
        internal_names.extend(
            [
                f"feature_internal_pre_answer_norm_layer_{layer_idx}",
                f"feature_internal_answer_norm_mean_layer_{layer_idx}",
                f"feature_internal_logit_entropy_mean_layer_{layer_idx}",
            ]
        )
    if len(features.internal_scalars) != len(internal_names):
        internal_names = [f"feature_internal_{index}" for index in range(len(features.internal_scalars))]

    probe_vec_names: list[str] = [f"feature_probe_vec_{index}" for index in range(len(features.probe_vec))]

    attention_names: list[str] = []
    for layer_idx in probe_layers:
        attention_names.extend(
            [
                f"feature_attention_entropy_mean_layer_{layer_idx}",
                f"feature_attention_entropy_max_layer_{layer_idx}",
                f"feature_attention_entropy_std_layer_{layer_idx}",
            ]
        )
    if len(features.attention_entropy) != len(attention_names):
        attention_names = [f"feature_attention_entropy_{index}" for index in range(len(features.attention_entropy))]

    entropy_drop_names: list[str] = []
    for current_layer, next_layer in zip(probe_layers, probe_layers[1:]):
        entropy_drop_names.append(f"feature_entropy_drop_layer_{current_layer}_to_{next_layer}")
    if len(features.entropy_drops) != len(entropy_drop_names):
        entropy_drop_names = [f"feature_entropy_drop_{index}" for index in range(len(features.entropy_drops))]

    moe_names_default: list[str] = [
        "feature_moe_top_prob_mean",
        "feature_moe_top_prob_std",
        "feature_moe_top_prob_dispersion_mean",
        "feature_moe_top_prob_dispersion_std",
        "feature_moe_entropy_mean",
        "feature_moe_entropy_std",
        "feature_moe_entropy_dispersion_mean",
        "feature_moe_entropy_dispersion_std",
        "feature_moe_active_ratio_mean",
        "feature_moe_active_ratio_std",
    ]
    moe_names: list[str]
    if len(features.moe_routing) == len(moe_names_default):
        moe_names = moe_names_default
    else:
        moe_names = [f"feature_moe_{index}" for index in range(len(features.moe_routing))]

    return uncertainty_names + internal_names + probe_vec_names + attention_names + entropy_drop_names + moe_names


def _feature_row_from_groups(features: FeatureGroups, *, probe_layers: Sequence[int]) -> tuple[list[str], list[float]]:
    """Формирует согласованные имена и значения фичей для строки CSV."""
    feature_values: list[float] = _flatten_feature_groups(features)
    feature_names: list[str] = _build_feature_column_names(features=features, probe_layers=probe_layers)
    if len(feature_names) != len(feature_values):
        raise ValueError("Количество имен фичей не совпадает с количеством значений")
    return feature_names, feature_values


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


def _query_free_gpu_memory_mib() -> int | None:
    """Возвращает свободную VRAM в MiB по `nvidia-smi` для GPU 0."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as error:
        logger.warning(f"Не удалось получить свободную VRAM через nvidia-smi: {error}")
        return None

    lines: list[str] = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    first_value: str = lines[0].split()[0]
    try:
        return int(first_value)
    except ValueError:
        return None


def _compute_subprocess_worker_count(
    *,
    total_samples: int,
    auto_enabled: bool,
    cuda_available: bool,
    max_workers: int,
    min_free_gpu_memory_mib: int,
    free_gpu_memory_mib: int | None,
) -> int:
    """Определяет число процессов обработки с учетом доступной VRAM."""
    if total_samples <= 1:
        return 1
    if not auto_enabled or not cuda_available:
        return 1
    if free_gpu_memory_mib is None or free_gpu_memory_mib < min_free_gpu_memory_mib:
        return 1

    memory_based_workers: int = 1 + (free_gpu_memory_mib // min_free_gpu_memory_mib)
    return max(1, min(total_samples, max_workers, memory_based_workers))


def _resolve_subprocess_worker_count(config: ScriptConfig, total_samples: int) -> int:
    """Обертка для расчета числа процессов с чтением текущей VRAM."""
    free_gpu_memory_mib: int | None = _query_free_gpu_memory_mib()
    worker_count: int = _compute_subprocess_worker_count(
        total_samples=total_samples,
        auto_enabled=config.enable_auto_subprocess,
        cuda_available=torch.cuda.is_available(),
        max_workers=config.subprocess_max_workers,
        min_free_gpu_memory_mib=config.subprocess_min_free_gpu_memory_mib,
        free_gpu_memory_mib=free_gpu_memory_mib,
    )
    if free_gpu_memory_mib is not None:
        logger.info(f"Свободная VRAM: {free_gpu_memory_mib} MiB, процессов для экстракции: {worker_count}")
    else:
        logger.info(f"Свободная VRAM не определена, процессов для экстракции: {worker_count}")
    return worker_count


def _build_shard_ranges(total: int, workers: int) -> list[tuple[int, int]]:
    """Делит `total` сэмплов на близкие по размеру непрерывные диапазоны."""
    if total <= 0:
        return []

    base_size: int = total // workers
    remainder: int = total % workers
    ranges: list[tuple[int, int]] = []
    start: int = 0
    for worker_idx in range(workers):
        shard_size: int = base_size + (1 if worker_idx < remainder else 0)
        if shard_size <= 0:
            continue
        stop: int = start + shard_size
        ranges.append((start, stop))
        start = stop
    return ranges


def _prepare_tokenizer(tokenizer: Any) -> Any:
    """Настраивает токенизатор для батчевой генерации на decoder-only моделях."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _resolve_model_source(model_name: str) -> str:
    """Возвращает локальный путь модели при наличии, иначе исходное имя HF."""
    model_dir_name: str = model_name.rsplit("/", maxsplit=1)[-1]



    local_model_path: Path = Path(get_model_dpath()) / model_dir_name
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
            attn_implementation="eager"
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


def _extract_rows_for_samples(
    *,
    config: ScriptConfig,
    sampled_queries: list[str],
    sampled_answers: list[str],
    sample_id_offset: int,
) -> list[dict[str, Any]]:
    """Выполняет экстракцию фичей для заданного слайса сэмплов."""
    model, tokenizer, input_device = _build_model_and_tokenizer(config.model_name)
    feature_config: FeatureExtractorConfig
    if config.feature_config_path is None:
        feature_config = FeatureExtractorConfig.from_default_yaml()
    else:
        feature_config = FeatureExtractorConfig.from_yaml(config.feature_config_path)
    extractor: LLMFeatureExtractor = LLMFeatureExtractor(model=model, config=feature_config)

    output_rows: list[dict[str, Any]] = []
    feature_columns: list[str] = []
    sample_counter: int = sample_id_offset
    temperature_rng: random.Random = random.Random(config.seed + sample_id_offset)

    with torch.inference_mode():
        with extractor:  # type: ignore[arg-type]
            total_batches: int = (len(sampled_queries) + config.batch_size - 1) // config.batch_size
            progress_bar = tqdm(total=total_batches, desc="Извлечение фичей", disable=sample_id_offset != 0)
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

                    extracted_feature_columns, feature_vector = _feature_row_from_groups(
                        extracted_features,
                        probe_layers=feature_config.probe_layers,
                    )
                    if not feature_columns:
                        feature_columns = extracted_feature_columns

                    hallucination_score: float = float(
                        fuzz.partial_ratio(model_answer.lower(), batch_answers[index_in_batch].lower())
                    )
                    is_hallucination: int = int(hallucination_score < DEFAULT_HALLUCINATION_SCORE_THRESHOLD)

                    row: dict[str, Any] = {
                        "sample_id": sample_counter,
                        "query": batch_queries[index_in_batch],
                        "correct_answer": batch_answers[index_in_batch],
                        "model_answer": model_answer,
                        "temperature": batch_temperature,
                        "hallucination_score": hallucination_score,
                        "is_hallucination": is_hallucination,
                    }
                    row.update({feature_name: value for feature_name, value in zip(feature_columns, feature_vector)})
                    output_rows.append(row)
                    sample_counter += 1

                    del full_ids, logits, extracted_features, feature_vector

                progress_bar.update(1)
            progress_bar.close()

    return output_rows


def _extract_rows_worker(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Воркeр multiprocessing для экстракции фичей на шардe."""
    config: ScriptConfig = ScriptConfig.model_validate(payload["config"])
    sampled_queries: list[str] = list(payload["queries"])
    sampled_answers: list[str] = list(payload["answers"])
    sample_id_offset: int = int(payload["sample_id_offset"])
    return _extract_rows_for_samples(
        config=config,
        sampled_queries=sampled_queries,
        sampled_answers=sampled_answers,
        sample_id_offset=sample_id_offset,
    )


def _extract_rows_in_parallel(
    *,
    config: ScriptConfig,
    sampled_queries: list[str],
    sampled_answers: list[str],
    worker_count: int,
) -> list[dict[str, Any]]:
    """Запускает multiprocessing-экстракцию по шардированным сэмплам."""
    shard_ranges: list[tuple[int, int]] = _build_shard_ranges(total=len(sampled_queries), workers=worker_count)
    payloads: list[dict[str, Any]] = []
    for start, stop in shard_ranges:
        payloads.append(
            {
                "config": config.model_dump(),
                "queries": sampled_queries[start:stop],
                "answers": sampled_answers[start:stop],
                "sample_id_offset": start,
            }
        )

    output_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=get_context("spawn")) as executor:
        futures = [executor.submit(_extract_rows_worker, payload) for payload in payloads]
        for future in futures:
            output_rows.extend(future.result())

    output_rows.sort(key=lambda row: int(row["sample_id"]))
    return output_rows


def run(config: ScriptConfig) -> Path:
    """Запускает полный пайплайн извлечения и сохранения фичей.

    Args:
        config: Конфигурация запуска.

    Returns:
        Путь до сохраненного CSV файла.
    """
    datasets_config: SberDatasetsConfig = SberDatasetsConfig.from_yaml(config.datasets_config_path)

    datasets_map: dict[str, Any] = retrieve_all_datasets(
        config=datasets_config,
        force_download=config.force_download,
    )
    sampled_queries, sampled_answers = _sample_balanced_queries_and_answers(
        datasets_map=datasets_map,
        n=config.n,
        seed=config.seed,
    )
    if config.n is None:
        logger.info(
            f"Параметр n не задан: после shuffle выбраны все сэмплы из датасетов (всего {len(sampled_queries)})"
        )
    else:
        per_dataset: int = config.n // len(datasets_map)
        logger.info(
            f"После balanced sampling и shuffle выбрано {len(sampled_queries)} сэмплов "
            f"({per_dataset} из каждого датасета)"
        )

    worker_count: int = _resolve_subprocess_worker_count(config=config, total_samples=len(sampled_queries))
    output_rows: list[dict[str, Any]]
    if worker_count <= 1:
        output_rows = _extract_rows_for_samples(
            config=config,
            sampled_queries=sampled_queries,
            sampled_answers=sampled_answers,
            sample_id_offset=0,
        )
    else:
        logger.info(f"Запускаю параллельную экстракцию в {worker_count} процессах")
        try:
            output_rows = _extract_rows_in_parallel(
                config=config,
                sampled_queries=sampled_queries,
                sampled_answers=sampled_answers,
                worker_count=worker_count,
            )
        except Exception as error:
            logger.warning(f"Параллельный режим завершился с ошибкой ({error}), fallback на single-process")
            output_rows = _extract_rows_for_samples(
                config=config,
                sampled_queries=sampled_queries,
                sampled_answers=sampled_answers,
                sample_id_offset=0,
            )

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

