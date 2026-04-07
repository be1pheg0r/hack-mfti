from __future__ import annotations

import time
from pathlib import Path
from typing import *

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike
from src.sber.constants import DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH, DEFAULT_SBER_SCRIPT_MODEL_NAME
from src.sber.models.extract_features import FeatureExtractorConfig, LLMFeatureExtractor
from src.sber.models.tabular_hallucination import FeatureSchema, TabularHallucinationPredictor


class TabularPipelineServerConfig(BaseModel):
    """Конфигурация HTTP-сервера пайплайна feature extractor -> tabular classifier.

    Attributes:
        host: Хост сервера.
        port: Порт сервера.
        serve_mode: Режим запуска (`tabular_pipeline` или `dummy`).
        feature_model_name: Имя/путь модели для извлечения внутренних фичей.
        checkpoint_dir: Директория tabular-чекпоинта.
        feature_config_path: Путь до YAML-конфига feature extractor.
        feature_extractor_mode: Режим экстрактора фичей (`real` или `dummy`).
    """

    model_config = ConfigDict(frozen=True)

    host: str = "0.0.0.0"
    port: int = 8020
    serve_mode: str = "tabular_pipeline"
    feature_model_name: str = DEFAULT_SBER_SCRIPT_MODEL_NAME
    checkpoint_dir: PathLike = "sber_tabular/latest"
    feature_config_path: PathLike = DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH
    feature_extractor_mode: str = "real"

    @field_validator("host", "feature_model_name", "serve_mode", "feature_extractor_mode")
    @classmethod
    def validate_non_empty_str(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("Строковый параметр не может быть пустым")
        return normalized

    @field_validator("port")
    @classmethod
    def validate_positive_port(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("port должен быть положительным")
        return value

    @field_validator("serve_mode")
    @classmethod
    def validate_serve_mode(cls, value: str) -> str:
        allowed: set[str] = {"tabular_pipeline", "dummy"}
        if value not in allowed:
            raise ValueError(f"serve_mode должен быть одним из: {sorted(allowed)}")
        return value

    @field_validator("feature_extractor_mode")
    @classmethod
    def validate_feature_extractor_mode(cls, value: str) -> str:
        allowed: set[str] = {"real", "dummy"}
        if value not in allowed:
            raise ValueError(f"feature_extractor_mode должен быть одним из: {sorted(allowed)}")
        return value


class TabularPipelineRequest(BaseModel):
    """Входной payload для tabular pipeline endpoint.

    Attributes:
        queries: Список пользовательских запросов.
        model_answers: Список ответов модели для валидации.
        threshold: Опциональный override порога классификации.
        precomputed_features: Опциональные предвычисленные признаки для bypass LLM.
    """

    model_config = ConfigDict(frozen=True)

    queries: list[str]
    model_answers: list[str]
    threshold: float | None = None
    precomputed_features: list[dict[str, Any]] | None = None

    @field_validator("queries", "model_answers")
    @classmethod
    def validate_non_empty_list(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("Список не должен быть пустым")
        return value

    @field_validator("threshold")
    @classmethod
    def validate_threshold(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold должен быть в диапазоне [0, 1]")
        return value

    @field_validator("precomputed_features")
    @classmethod
    def validate_precomputed_features(cls, value: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if value is None:
            return value
        if not value:
            raise ValueError("precomputed_features не должен быть пустым")
        for row in value:
            if not isinstance(row, dict):
                raise ValueError("Каждая строка precomputed_features должна быть объектом")
        return value

    @model_validator(mode="after")
    def validate_equal_lengths(self) -> TabularPipelineRequest:
        if len(self.queries) != len(self.model_answers):
            raise ValueError("queries и model_answers должны быть одинаковой длины")
        if self.precomputed_features is not None and len(self.precomputed_features) != len(self.queries):
            raise ValueError("precomputed_features должен быть той же длины, что и queries")
        return self


class BaseFeatureExtractorBackend:
    """Базовый интерфейс backend-компоненты экстрактора фичей."""

    def build_features(self, queries: list[str], model_answers: list[str]) -> pd.DataFrame:
        raise NotImplementedError


class DummyFeatureExtractorBackend(BaseFeatureExtractorBackend):
    """Dummy-экстрактор фичей для быстрых тестов без LLM."""

    def _row(self, query: str, model_answer: str) -> dict[str, Any]:
        row: dict[str, Any] = {
            "query": query,
            "model_answer": model_answer,
        }
        for feature_name in (
            FeatureSchema.uncertainty_map
            + FeatureSchema.internal_scalars_map
            + FeatureSchema.probe_vec_map
            + FeatureSchema.attention_entropy_map
            + FeatureSchema.entropy_drops_map
            + FeatureSchema.moe_routing_map
        ):
            row[feature_name] = 0.0
        row["n_answer_tokens"] = float(max(len(model_answer.split()), 1))
        return row

    def build_features(self, queries: list[str], model_answers: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = [
            self._row(query=query, model_answer=answer)
            for query, answer in zip(queries, model_answers)
        ]
        return pd.DataFrame(rows)


class LLMFeatureExtractorBackend(BaseFeatureExtractorBackend):
    """Компонента экстрактора фичей на базе LLM hooks."""

    def __init__(self, config: TabularPipelineServerConfig) -> None:
        self.config: TabularPipelineServerConfig = config
        self._device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._feature_model: Any | None = None
        self._tokenizer: Any | None = None
        self._extractor: LLMFeatureExtractor | None = None
        self._extractor_config: FeatureExtractorConfig | None = None
        self._load_feature_extractor()

    def _resolve_feature_source(self) -> str:
        configured_path: Path = Path(self.config.feature_model_name).expanduser()
        if configured_path.exists():
            return str(configured_path)
        return self.config.feature_model_name

    def _build_prompt(self, query: str) -> str:
        tokenizer: Any = self._tokenizer
        assert tokenizer is not None
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": query}],
                add_generation_prompt=True,
                tokenize=False,
            )
        return query

    def _load_feature_extractor(self) -> None:
        source: str = self._resolve_feature_source()
        logger.info("Загружаю модель для feature extractor: %s", source)

        self._feature_model = AutoModelForCausalLM.from_pretrained(source, trust_remote_code=True)
        self._tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
        self._feature_model = self._feature_model.to(self._device)
        self._feature_model.eval()

        feature_config = FeatureExtractorConfig.from_yaml(self.config.feature_config_path)
        self._extractor_config = feature_config
        self._extractor = LLMFeatureExtractor(model=self._feature_model, config=feature_config)

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if hasattr(model_output, "logits"):
            logits: Any = model_output.logits
            if isinstance(logits, torch.Tensor):
                return logits
        if isinstance(model_output, dict) and "logits" in model_output:
            logits_value: Any = model_output["logits"]
            if isinstance(logits_value, torch.Tensor):
                return logits_value
        raise ValueError("Не удалось извлечь logits из выхода модели")

    def _align_group(self, values: list[float], target_size: int) -> list[float]:
        if len(values) == target_size:
            return values
        if len(values) > target_size:
            return values[:target_size]
        return values + [0.0] * (target_size - len(values))

    def _feature_row_from_pair(self, query: str, model_answer: str) -> dict[str, Any]:
        tokenizer: Any = self._tokenizer
        assert tokenizer is not None
        assert self._extractor is not None

        prompt_text: str = self._build_prompt(query)
        prompt_ids: torch.Tensor = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self._device)
        answer_ids: torch.Tensor = tokenizer(model_answer, return_tensors="pt", add_special_tokens=False)["input_ids"].to(
            self._device
        )
        if int(answer_ids.shape[1]) == 0:
            answer_ids = tokenizer(" ", return_tensors="pt", add_special_tokens=False)["input_ids"].to(self._device)

        full_ids: torch.Tensor = torch.cat([prompt_ids, answer_ids], dim=1)
        answer_start: int = int(prompt_ids.shape[1])

        with torch.inference_mode():
            with self._extractor:
                model_output: Any = self._extractor(full_ids)
                logits: torch.Tensor = self._extract_logits(model_output)
                groups = self._extractor.extract(logits=logits, input_ids=full_ids, answer_start=answer_start)

        feature_row: dict[str, Any] = {
            "query": query,
            "model_answer": model_answer,
        }

        uncertainty_values = self._align_group(groups.uncertainty, len(FeatureSchema.uncertainty_map))
        internal_values = self._align_group(groups.internal_scalars, len(FeatureSchema.internal_scalars_map))
        probe_values = self._align_group(groups.probe_vec, len(FeatureSchema.probe_vec_map))
        attention_values = self._align_group(groups.attention_entropy, len(FeatureSchema.attention_entropy_map))
        entropy_drop_values = self._align_group(groups.entropy_drops, len(FeatureSchema.entropy_drops_map))
        moe_values = self._align_group(groups.moe_routing, len(FeatureSchema.moe_routing_map))

        feature_row.update(dict(zip(FeatureSchema.uncertainty_map, uncertainty_values)))
        feature_row.update(dict(zip(FeatureSchema.internal_scalars_map, internal_values)))
        feature_row.update(dict(zip(FeatureSchema.probe_vec_map, probe_values)))
        feature_row.update(dict(zip(FeatureSchema.attention_entropy_map, attention_values)))
        feature_row.update(dict(zip(FeatureSchema.entropy_drops_map, entropy_drop_values)))
        feature_row.update(dict(zip(FeatureSchema.moe_routing_map, moe_values)))
        return feature_row

    def build_features(self, queries: list[str], model_answers: list[str]) -> pd.DataFrame:
        rows: list[dict[str, Any]] = [
            self._feature_row_from_pair(query=query, model_answer=answer)
            for query, answer in zip(queries, model_answers)
        ]
        return pd.DataFrame(rows)


class TabularClassifierBackend:
    """Компонента классификатора, использующая сохраненный tabular checkpoint."""

    def __init__(self, checkpoint_dir: PathLike) -> None:
        self.predictor: TabularHallucinationPredictor = TabularHallucinationPredictor(checkpoint_dir)
        self.predictor.load()

    def classify(self, features_df: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        return self.predictor.predict_dataframe(features_df, threshold=threshold)


class TabularPipelineService:
    """Сервис пайплайна feature extractor -> tabular classifier."""

    def __init__(self, config: TabularPipelineServerConfig) -> None:
        self.config: TabularPipelineServerConfig = config
        self.classifier_backend: TabularClassifierBackend | None = TabularClassifierBackend(config.checkpoint_dir)
        self.feature_extractor_backend: BaseFeatureExtractorBackend = DummyFeatureExtractorBackend()

        # Dummy-режим не должен инициализировать тяжелые зависимости (LLM/checkpoint).
        if config.serve_mode == "dummy":
            return

        if config.feature_extractor_mode == "real":
            self.feature_extractor_backend = LLMFeatureExtractorBackend(config=config)

    def _ensure_classifier_backend(self) -> TabularClassifierBackend:
        if self.classifier_backend is None:
            self.classifier_backend = TabularClassifierBackend(self.config.checkpoint_dir)
        return self.classifier_backend

    def _dummy_predict(self, request: TabularPipelineRequest) -> dict[str, Any]:
        if request.precomputed_features is not None:
            total_start: float = time.perf_counter()
            classifier: TabularClassifierBackend = self._ensure_classifier_backend()
            features_df: pd.DataFrame = pd.DataFrame(request.precomputed_features)
            classify_start: float = time.perf_counter()
            scored_df: pd.DataFrame = classifier.classify(features_df, threshold=request.threshold)
            classification_elapsed: float = time.perf_counter() - classify_start

            total_elapsed: float = time.perf_counter() - total_start
            per_sample_total: float = total_elapsed / max(len(scored_df), 1)
            per_sample_classification: float = classification_elapsed / max(len(scored_df), 1)
            return {
                "pred_is_hallucination": [int(value) for value in scored_df["pred_is_hallucination"].tolist()],
                "hallucination_score": [float(value) for value in scored_df["hallucination_score"].tolist()],
                "entailment_score": [float(value) for value in scored_df["entailment_score"].tolist()],
                "t_feature_extraction_sec": [0.0 for _ in range(len(scored_df))],
                "t_classification_sec": [per_sample_classification for _ in range(len(scored_df))],
                "t_total_sec": [per_sample_total for _ in range(len(scored_df))],
                "t_sample_sec": [per_sample_total for _ in range(len(scored_df))],
            }

        start: float = time.perf_counter()
        scores: list[float] = []
        preds: list[int] = []
        entails: list[float] = []
        for query, answer in zip(request.queries, request.model_answers):
            score: float = float(1.0 if not answer.strip() or query.strip() == answer.strip() else 0.2)
            pred: int = int(score >= float(request.threshold if request.threshold is not None else 0.5))
            scores.append(score)
            preds.append(pred)
            entails.append(1.0 - score)

        elapsed: float = time.perf_counter() - start
        per_sample: float = elapsed / max(len(request.queries), 1)
        return {
            "pred_is_hallucination": preds,
            "hallucination_score": scores,
            "entailment_score": entails,
            "t_feature_extraction_sec": [0.0 for _ in preds],
            "t_classification_sec": [0.0 for _ in preds],
            "t_total_sec": [per_sample for _ in preds],
            "t_sample_sec": [per_sample for _ in preds],
        }

    def predict(self, request: TabularPipelineRequest) -> dict[str, Any]:
        if self.config.serve_mode == "dummy":
            return self._dummy_predict(request)

        if self.classifier_backend is None:
            raise RuntimeError("classifier backend не инициализирован")

        total_start: float = time.perf_counter()
        feature_start: float = time.perf_counter()
        features_df: pd.DataFrame = self.feature_extractor_backend.build_features(
            queries=request.queries,
            model_answers=request.model_answers,
        )
        feature_elapsed: float = time.perf_counter() - feature_start

        classify_start: float = time.perf_counter()
        scored_df: pd.DataFrame = self.classifier_backend.classify(features_df, threshold=request.threshold)
        classification_elapsed: float = time.perf_counter() - classify_start

        total_elapsed: float = time.perf_counter() - total_start
        per_sample_total: float = total_elapsed / max(len(scored_df), 1)
        per_sample_feature: float = feature_elapsed / max(len(scored_df), 1)
        per_sample_classification: float = classification_elapsed / max(len(scored_df), 1)
        return {
            "pred_is_hallucination": [int(value) for value in scored_df["pred_is_hallucination"].tolist()],
            "hallucination_score": [float(value) for value in scored_df["hallucination_score"].tolist()],
            "entailment_score": [float(value) for value in scored_df["entailment_score"].tolist()],
            "t_feature_extraction_sec": [per_sample_feature for _ in range(len(scored_df))],
            "t_classification_sec": [per_sample_classification for _ in range(len(scored_df))],
            "t_total_sec": [per_sample_total for _ in range(len(scored_df))],
            "t_sample_sec": [per_sample_total for _ in range(len(scored_df))],
        }


def parse_tabular_pipeline_request(payload: Mapping[str, Any]) -> TabularPipelineRequest:
    """Нормализует JSON payload и валидирует вход endpoint-а."""
    if "queries" in payload and "model_answers" in payload:
        normalized: dict[str, Any] = dict(payload)
    else:
        features_raw: Any = payload.get("features")
        normalized_features: list[dict[str, Any]] | None = None
        if isinstance(features_raw, Mapping):
            normalized_features = [dict(features_raw)]
        elif isinstance(features_raw, list):
            rows: list[dict[str, Any]] = []
            for item in features_raw:
                if not isinstance(item, Mapping):
                    raise ValueError("Каждый элемент features должен быть объектом")
                rows.append(dict(item))
            normalized_features = rows

        normalized = {
            "queries": [str(payload.get("query", ""))],
            "model_answers": [str(payload.get("model_answer", ""))],
            "threshold": payload.get("threshold"),
            "precomputed_features": normalized_features,
        }
    return TabularPipelineRequest.model_validate(normalized)


__all__ = [
    "BaseFeatureExtractorBackend",
    "DummyFeatureExtractorBackend",
    "LLMFeatureExtractorBackend",
    "TabularClassifierBackend",
    "TabularPipelineRequest",
    "TabularPipelineServerConfig",
    "TabularPipelineService",
    "parse_tabular_pipeline_request",
]



