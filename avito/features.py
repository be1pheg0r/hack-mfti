from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator
from rapidfuzz import fuzz, process

from avito.constants import COMPLEX_MARKERS, SPLIT_MARKERS, TURNKEY_SOURCE_TITLE


_BULLET_PATTERN = re.compile(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+\S", flags=re.MULTILINE)
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")


class TextEncoderLike(Protocol):
    """Структурный протокол для текстовых энкодеров в генерации признаков."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Преобразует входные тексты в плотные вектора."""
        ...


class ShouldSplitFeatureConfig(BaseModel):
    """Конфигурация извлечения признаков для классификации shouldSplit."""

    model_config = ConfigDict(frozen=True)

    split_markers: tuple[str, ...] = SPLIT_MARKERS
    complex_markers: tuple[str, ...] = COMPLEX_MARKERS
    turnkey_source_title: str = TURNKEY_SOURCE_TITLE
    include_extra_text_features: bool = True
    include_case_type_feature: bool = True
    case_type_unknown_label: str = "unknown"
    correlation_threshold: float | None = None

    # Расширенные признаки по ключевым фразам микрокатегорий
    include_keyphrase_similarity: bool = True
    include_rapidfuzz_score: bool = True
    include_entailment_similarity: bool = True
    include_split_marker_window: bool = True
    keyphrase_columns: tuple[str, ...] = ("sourceMcTitle",)
    extra_keyphrases: tuple[str, ...] = ()
    split_marker_window: int = 2
    entailment_template: str = "селлер предлагает {phrase} отдельно"

    @field_validator("split_marker_window")
    @classmethod
    def validate_window(cls, value: int) -> int:
        if value < 0:
            raise ValueError("split_marker_window должен быть неотрицательным")
        return int(value)


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _count_markers(text: str, markers: Sequence[str]) -> int:
    lowered = text.lower()
    return int(sum(lowered.count(marker.lower()) for marker in markers))


def _has_bullets(text: str) -> int:
    return int(bool(_BULLET_PATTERN.search(text)))


def _sentence_count(text: str) -> int:
    chunks = [chunk for chunk in _SENTENCE_SPLIT_PATTERN.split(text) if chunk.strip()]
    return max(len(chunks), 1) if text.strip() else 0


def _avg_word_length(text: str) -> float:
    words = [w for w in re.findall(r"\w+", text, flags=re.UNICODE) if w]
    if not words:
        return 0.0
    return float(sum(len(w) for w in words) / len(words))


def _punctuation_ratio(text: str) -> float:
    if not text:
        return 0.0
    punct_count = len(re.findall(r"[.,!?;:]", text))
    return float(punct_count / max(len(text), 1))


def resolve_keyphrases(df: pd.DataFrame, config: ShouldSplitFeatureConfig | None = None) -> list[str]:
    """Формирует список ключевых фраз для микрокатегорий из датафрейма и конфига."""

    cfg = config or ShouldSplitFeatureConfig()
    phrases: list[str] = []

    for column in cfg.keyphrase_columns:
        if column in df.columns:
            phrases.extend(df[column].dropna().astype(str).tolist())

    phrases.extend(cfg.extra_keyphrases)

    normalized: list[str] = []
    seen: set[str] = set()
    for phrase in phrases:
        cleaned = _safe_text(phrase).strip()
        lowered = cleaned.lower()
        if cleaned and lowered not in seen:
            seen.add(lowered)
            normalized.append(cleaned)

    return normalized


def _split_into_sentences(text: str) -> list[str]:
    return [chunk.strip() for chunk in _SENTENCE_SPLIT_PATTERN.split(text) if chunk.strip()]


def _split_marker_near_keyphrase(
    text: str,
    keyphrases: Sequence[str],
    split_markers: Sequence[str],
    window: int,
) -> int:
    if not text or not keyphrases:
        return 0

    sentences = _split_into_sentences(text)
    if not sentences:
        return 0

    lowered_keyphrases = [phrase.lower() for phrase in keyphrases]
    lowered_markers = [marker.lower() for marker in split_markers]

    for idx, sentence in enumerate(sentences):
        lowered_sentence = sentence.lower()
        if not any(phrase in lowered_sentence for phrase in lowered_keyphrases):
            continue

        start = max(0, idx - window)
        end = min(len(sentences), idx + window + 1)
        window_sentences = sentences[start:end]
        for window_sentence in window_sentences:
            lowered_window = window_sentence.lower()
            if any(marker in lowered_window for marker in lowered_markers):
                return 1
    return 0


def _max_rapidfuzz_score(text: str, keyphrases: Sequence[str]) -> float:
    if not text or not keyphrases:
        return 0.0

    # Используем token_set_ratio для устойчивости к порядку слов
    result = process.extractOne(text, keyphrases, scorer=fuzz.token_set_ratio, score_cutoff=0)
    if result is None:
        return 0.0
    # extractOne возвращает (match, score, index)
    return float(result[1])


def extract_should_split_features(
    df: pd.DataFrame,
    config: ShouldSplitFeatureConfig | None = None,
    keyphrases: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Извлекает текстовые и категориальные признаки для shouldSplit."""
    cfg = config or ShouldSplitFeatureConfig()

    required_columns = {"description", "sourceMcId", "sourceMcTitle"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"В DataFrame отсутствуют обязательные колонки: {sorted(missing_columns)}")

    resolved_keyphrases = list(keyphrases) if keyphrases is not None else resolve_keyphrases(df, cfg)

    descriptions = df["description"].map(_safe_text)
    source_titles = df["sourceMcTitle"].map(_safe_text)
    source_ids = df["sourceMcId"].map(_safe_text)
    case_types = (
        df["caseType"].map(_safe_text)
        if "caseType" in df.columns
        else pd.Series(cfg.case_type_unknown_label, index=df.index)
    )

    features = pd.DataFrame(index=df.index)
    features["description_word_count"] = descriptions.map(lambda text: len(text.split())).astype(np.float32)
    features["description_char_count"] = descriptions.map(len).astype(np.float32)
    features["split_marker_count"] = descriptions.map(lambda text: _count_markers(text, cfg.split_markers)).astype(np.float32)
    features["complex_marker_count"] = descriptions.map(lambda text: _count_markers(text, cfg.complex_markers)).astype(np.float32)
    features["marker_ratio"] = (features["split_marker_count"] / (features["complex_marker_count"] + 1.0)).astype(np.float32)
    features["has_bullets"] = descriptions.map(_has_bullets).astype(np.int8)

    features["source_mc_id"] = source_ids
    features["is_turnkey"] = source_titles.map(lambda title: int(title.strip() == cfg.turnkey_source_title)).astype(np.int8)

    if cfg.include_case_type_feature:
        features["case_type"] = case_types.astype(str)

    if cfg.include_extra_text_features:
        features["sentence_count"] = descriptions.map(_sentence_count).astype(np.float32)
        features["avg_word_len"] = descriptions.map(_avg_word_length).astype(np.float32)
        features["punctuation_ratio"] = descriptions.map(_punctuation_ratio).astype(np.float32)

    if cfg.include_rapidfuzz_score:
        features["max_keyphrase_rapidfuzz"] = descriptions.map(
            lambda text: _max_rapidfuzz_score(text, resolved_keyphrases)
        ).astype(np.float32)

    if cfg.include_split_marker_window:
        features["split_marker_near_keyphrase"] = descriptions.map(
            lambda text: _split_marker_near_keyphrase(
                text,
                resolved_keyphrases,
                cfg.split_markers,
                cfg.split_marker_window,
            )
        ).astype(np.int8)

    return features


def _compute_max_cosine_similarity(
    embeddings: np.ndarray,
    keyphrase_embeddings: np.ndarray,
) -> np.ndarray:
    if embeddings.size == 0 or keyphrase_embeddings.size == 0:
        return np.zeros((embeddings.shape[0],), dtype=np.float32)

    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    key_norm = keyphrase_embeddings / np.linalg.norm(keyphrase_embeddings, axis=1, keepdims=True).clip(min=1e-12)
    similarity = emb_norm @ key_norm.T
    return similarity.max(axis=1)


def append_embedding_features(
    features: pd.DataFrame,
    descriptions: Sequence[Any],
    encoder: TextEncoderLike,
    prefix: str = "embedding",
    *,
    keyphrases: Sequence[str] | None = None,
    config: ShouldSplitFeatureConfig | None = None,
) -> pd.DataFrame:
    """Добавляет эмбеддинговые признаки в DataFrame фичей и производные фичи по ключевым фразам."""

    cfg = config or ShouldSplitFeatureConfig()
    resolved_keyphrases = list(keyphrases) if keyphrases is not None else []

    prepared_descriptions = [(_safe_text(text).strip() or "<EMPTY>") for text in descriptions]
    raw_embeddings = encoder.encode(prepared_descriptions)
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Ожидаются эмбеддинги размерности N x D.")
    if embeddings.shape[0] != len(features):
        raise ValueError("Число эмбеддингов не совпадает с числом строк признаков.")

    embedding_columns = [f"{prefix}_{idx:03d}" for idx in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns, index=features.index)

    augmented = pd.concat([features, embedding_df], axis=1)

    if cfg.include_keyphrase_similarity and resolved_keyphrases:
        kp_embeddings = np.asarray(encoder.encode(resolved_keyphrases), dtype=np.float32)
        max_cosine = _compute_max_cosine_similarity(embeddings, kp_embeddings)
        augmented["max_keyphrase_cosine_similarity"] = max_cosine.astype(np.float32)

    if cfg.include_entailment_similarity and resolved_keyphrases:
        entailment_phrases = [cfg.entailment_template.format(phrase=phrase) for phrase in resolved_keyphrases]
        entailment_embeddings = np.asarray(encoder.encode(entailment_phrases), dtype=np.float32)
        max_entailment = _compute_max_cosine_similarity(embeddings, entailment_embeddings)
        augmented["max_entailment_similarity"] = max_entailment.astype(np.float32)

    return augmented


def build_training_matrix(
    df: pd.DataFrame,
    *,
    encoder: TextEncoderLike | None = None,
    include_embeddings: bool = False,
    config: ShouldSplitFeatureConfig | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Строит матрицу признаков X, таргет y и колонку split."""
    required_columns = {"shouldSplit", "split"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"В DataFrame отсутствуют обязательные колонки: {sorted(missing_columns)}")

    cfg = config or ShouldSplitFeatureConfig()
    keyphrases = resolve_keyphrases(df, cfg)

    features = extract_should_split_features(df=df, config=cfg, keyphrases=keyphrases)
    if include_embeddings:
        if encoder is None:
            raise ValueError("Для include_embeddings=True нужно передать encoder.")
        features = append_embedding_features(
            features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
            keyphrases=keyphrases,
            config=cfg,
        )

    if cfg.correlation_threshold is not None:
        features = _drop_highly_correlated_features(features, threshold=cfg.correlation_threshold)

    y = df["shouldSplit"].astype(bool)
    split = df["split"].astype(str)
    return features, y, split


def _drop_highly_correlated_features(features: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Удаляет числовые признаки с высокой взаимной корреляцией."""
    if features.empty:
        return features

    numeric_cols = features.select_dtypes(include=[np.number, "bool"]).columns
    if len(numeric_cols) < 2:
        return features

    corr_matrix = features[numeric_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop: set[str] = set()
    for column in upper.columns:
        high_corr = upper[column] > threshold
        if high_corr.any():
            to_drop.add(column)

    if not to_drop:
        return features

    return features.drop(columns=list(to_drop), errors="ignore")
