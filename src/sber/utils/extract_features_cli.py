from __future__ import annotations

import argparse
import json
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
from common.paths import PathLike, get_data_raw_dpath, get_model_dpath, get_sber_configs_dpath
from src.sber.constants import (
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
        input_csv_path: Опциональный путь к входному CSV с колонкой `query` или `prompt`.
        input_query_column: Явное имя колонки с текстом запроса во входном CSV.
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
    input_csv_path: PathLike | None = None
    input_query_column: str | None = None
    datasets_config_path: PathLike = Field(default_factory=lambda: Path(get_sber_configs_dpath()) / "datasets_configs.yaml")
    feature_config_path: PathLike | None = DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH

    @field_validator("batch_size", "max_new_tokens")
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        """Проверяет, что числовые параметры положительные."""
        if value <= 0:
            raise ValueError("Параметр должен быть положительным")
        return value

    @field_validator("n")
    @classmethod
    def validate_n(cls, value: int | None) -> int | None:
        """Проверяет, что n положительный, если параметр задан."""
        if value is not None and value <= 0:
            raise ValueError("Параметр n должен быть положительным")
        return value

    @field_validator("input_query_column")
    @classmethod
    def validate_input_query_column(cls, value: str | None) -> str | None:
        """Проверяет, что имя query-колонки не пустое, если задано."""
        if value is None:
            return None
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("input_query_column не может быть пустым")
        return normalized

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
        "--input-csv-path",
        type=str,
        default=None,
        help="Путь к входному CSV. Если задан, датасеты из YAML не загружаются",
    )
    parser.add_argument(
        "--input-query-column",
        type=str,
        default=None,
        help="Явное имя колонки с запросом во входном CSV (иначе auto: query -> prompt)",
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




def _resolve_query_column_name(columns: Sequence[str], preferred_column: str | None = None) -> str:
    """Определяет колонку запроса во входном CSV."""
    available: set[str] = {str(column) for column in columns}
    if preferred_column is not None:
        if preferred_column in available:
            return preferred_column
        raise ValueError(f"Колонка {preferred_column} не найдена во входном CSV")

    for candidate in ("query", "prompt"):
        if candidate in available:
            return candidate
    raise ValueError("Во входном CSV должна быть колонка query или prompt")


def _sample_dataframe_rows(dataframe: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    """Возвращает детерминированный сэмпл строк DataFrame."""
    total_size: int = int(len(dataframe))
    if n is None:
        return dataframe.reset_index(drop=True)
    if n > total_size:
        raise ValueError(f"Параметр n={n} больше размера входного CSV ({total_size})")

    indices: list[int] = list(range(total_size))
    rng: random.Random = random.Random(seed)
    rng.shuffle(indices)
    sampled_indices: list[int] = indices[:n]
    return dataframe.iloc[sampled_indices].reset_index(drop=True)


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


def _extract_expected_float_field_name(error_text: str) -> str | None:
    """Извлекает имя поля из ошибки вида `Field 'x' expected float, got int`."""
    marker: str = "Field '"
    start: int = error_text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end: int = error_text.find("'", start)
    if end < 0:
        return None
    return error_text[start:end]


def _get_local_config_json_fpath(model_source: str) -> Path | None:
    """Возвращает локальный путь к config.json для модели, если он доступен."""
    source_path: Path = Path(model_source)
    if source_path.exists() and source_path.is_dir():
        config_fpath: Path = source_path / "config.json"
        if config_fpath.exists():
            return config_fpath

    # Для удаленного HF repo_id используем локальный файл из кэша huggingface_hub.
    try:
        from huggingface_hub import hf_hub_download

        cached_config_fpath: str = hf_hub_download(
            repo_id=model_source,
            filename="config.json",
            repo_type="model",
        )
        cached_path: Path = Path(cached_config_fpath)
        if cached_path.exists():
            return cached_path
    except Exception as error:
        logger.warning(f"Не удалось получить config.json из HF Hub для {model_source}: {error}")

    return None


def _coerce_field_to_float_if_needed(payload: Any, field_name: str) -> bool:
    """Рекурсивно приводит найденные int-значения заданного поля к float."""
    changed: bool = False
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key == field_name and isinstance(value, int) and not isinstance(value, bool):
                payload[key] = float(value)
                changed = True
            else:
                changed = _coerce_field_to_float_if_needed(value, field_name) or changed
    elif isinstance(payload, list):
        for item in payload:
            changed = _coerce_field_to_float_if_needed(item, field_name) or changed
    return changed


def _try_patch_model_config_for_float_field(model_source: str, error: Exception) -> bool:
    """Пытается исправить config.json при ошибке float/int и возвращает факт правки."""
    error_text: str = str(error)
    lowered: str = error_text.lower()
    if "expected float" not in lowered or "got int" not in lowered:
        return False

    field_name: str | None = _extract_expected_float_field_name(error_text)
    if field_name is None:
        return False

    config_fpath: Path | None = _get_local_config_json_fpath(model_source)
    if config_fpath is None:
        logger.warning(
            f"Не удалось автоисправить конфиг модели {model_source}: локальный config.json не найден для поля {field_name}"
        )
        return False

    with open(config_fpath, "r", encoding="utf-8") as file:
        config_payload: Any = json.load(file)

    if not _coerce_field_to_float_if_needed(config_payload, field_name=field_name):
        logger.warning(f"Поле {field_name} не найдено в {config_fpath}, автоисправление не применено")
        return False

    with open(config_fpath, "w", encoding="utf-8") as file:
        json.dump(config_payload, file, ensure_ascii=False, indent=2)
        file.write("\n")

    logger.warning(f"Автоисправление config.json: поле {field_name} приведено к float в {config_fpath}")
    return True


def _load_model_with_config_autofix(
    model_source: str,
    *,
    torch_dtype: torch.dtype,
    device_map: str | None,
    attn_implementation: str,
) -> Any:
    """Загружает модель и при необходимости делает один retry после автофикса config.json."""
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "attn_implementation": attn_implementation,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    try:
        return AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)
    except Exception as error:
        if not _try_patch_model_config_for_float_field(model_source=model_source, error=error):
            raise

    # После правки config.json пробуем загрузить еще раз.
    return AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)


def _build_model_and_tokenizer(model_name: str) -> tuple[Any, Any, torch.device]:
    """Загружает модель и токенизатор с учетом multi-GPU конфигурации."""
    if torch.cuda.is_available():
        gpu_count: int = torch.cuda.device_count()
        torch.backends.cuda.matmul.allow_tf32 = True
        compute_dtype: torch.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        logger.info(f"Доступно GPU: {gpu_count}, включаю device_map=auto и dtype={compute_dtype}")

        model_source: str = _resolve_model_source(model_name)
        model = _load_model_with_config_autofix(
            model_source=model_source,
            torch_dtype=compute_dtype,
            device_map="auto",
            attn_implementation="eager",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
        model.eval()
        return model, _prepare_tokenizer(tokenizer), torch.device("cuda:0")

    logger.info("GPU не обнаружены, запускаю на CPU")
    model_source = _resolve_model_source(model_name)
    model = _load_model_with_config_autofix(
        model_source=model_source,
        torch_dtype=torch.float32,
        device_map=None,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    model.eval()
    return model, _prepare_tokenizer(tokenizer), torch.device("cpu")


def _extract_rows_for_samples(
    *,
    config: ScriptConfig,
    sampled_queries: list[str],
    sampled_answers: list[str] | None,
    base_rows: list[dict[str, Any]] | None,
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
    sample_index_in_slice: int = 0
    temperature_rng: random.Random = random.Random(config.seed + sample_id_offset)

    with torch.inference_mode():
        with extractor:  # type: ignore[arg-type]
            total_batches: int = (len(sampled_queries) + config.batch_size - 1) // config.batch_size
            progress_bar = tqdm(total=total_batches, desc="Извлечение фичей", disable=sample_id_offset != 0)
            for batch_queries, batch_answers in _batched_pairs(
                queries=sampled_queries,
                answers=sampled_answers if sampled_answers is not None else [""] * len(sampled_queries),
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

                    row: dict[str, Any] = {}
                    if base_rows is not None and sample_index_in_slice < len(base_rows):
                        row = dict(base_rows[sample_index_in_slice])

                    row.update(
                        {
                            "sample_id": sample_counter,
                            "query": batch_queries[index_in_batch],
                            "model_answer": model_answer,
                            "temperature": batch_temperature,
                        }
                    )

                    if sampled_answers is not None and sample_index_in_slice < len(sampled_answers):
                        reference_answer: str = sampled_answers[sample_index_in_slice]
                        row["correct_answer"] = reference_answer
                        hallucination_score: float = float(fuzz.partial_ratio(model_answer.lower(), reference_answer.lower()))
                        row["hallucination_score"] = hallucination_score
                        row["is_hallucination"] = int(hallucination_score < DEFAULT_HALLUCINATION_SCORE_THRESHOLD)
                    else:
                        row["hallucination_score"] = None
                        row["is_hallucination"] = None

                    row.update({feature_name: value for feature_name, value in zip(feature_columns, feature_vector)})
                    output_rows.append(row)
                    sample_index_in_slice += 1
                    sample_counter += 1

                    del full_ids, logits, extracted_features, feature_vector

                progress_bar.update(1)
            progress_bar.close()

    return output_rows


def run(config: ScriptConfig) -> Path:
    """Запускает полный пайплайн извлечения и сохранения фичей.

    Args:
        config: Конфигурация запуска.

    Returns:
        Путь до сохраненного CSV файла.
    """
    sampled_queries: list[str]
    sampled_answers: list[str] | None
    base_rows: list[dict[str, Any]] | None

    if config.input_csv_path is not None:
        input_path: Path = Path(config.input_csv_path)
        input_dataframe: pd.DataFrame = pd.read_csv(input_path)
        sampled_dataframe: pd.DataFrame = _sample_dataframe_rows(input_dataframe, n=config.n, seed=config.seed)
        query_column: str = _resolve_query_column_name(
            columns=[str(column) for column in sampled_dataframe.columns],
            preferred_column=config.input_query_column,
        )
        normalized_queries: list[str] = sampled_dataframe[query_column].astype("string").fillna("").str.strip().tolist()
        if any(not query for query in normalized_queries):
            raise ValueError(f"Во входном CSV найдены пустые значения в колонке {query_column}")

        sampled_queries = [str(query) for query in normalized_queries]
        sampled_answers = None
        if "correct_answer" in sampled_dataframe.columns:
            answers_series = sampled_dataframe["correct_answer"].astype("string").fillna("").str.strip()
            if answers_series.str.len().gt(0).all():
                sampled_answers = [str(answer) for answer in answers_series.tolist()]

        base_rows = sampled_dataframe.to_dict(orient="records")
        logger.info(
            f"Использую входной CSV {input_path}. Выбрано {len(sampled_queries)} строк, query-колонка: {query_column}"
        )
    else:
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
        base_rows = [
            {"query": query, "correct_answer": answer}
            for query, answer in zip(sampled_queries, sampled_answers)
        ]

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

    output_rows: list[dict[str, Any]] = _extract_rows_for_samples(
        config=config,
        sampled_queries=sampled_queries,
        sampled_answers=sampled_answers,
        base_rows=base_rows,
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

