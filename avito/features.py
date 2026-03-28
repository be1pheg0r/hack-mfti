from __future__ import annotations

import re
from typing import Any, Protocol, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from avito.constants import COMPLEX_MARKERS, SPLIT_MARKERS, TURNKEY_SOURCE_TITLE


_BULLET_PATTERN = re.compile(r"(^|\n)\s*(?:[-*•]|\d+[.)])\s+\S", flags=re.MULTILINE)


class TextEncoderLike(Protocol):
    """Structural protocol for text encoders used in feature generation."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode input texts into dense vectors."""
        ...


class ShouldSplitFeatureConfig(BaseModel):
    """Конфигурация извлечения признаков для классификации shouldSplit."""

    model_config = ConfigDict(frozen=True)

    split_markers: tuple[str, ...] = SPLIT_MARKERS
    complex_markers: tuple[str, ...] = COMPLEX_MARKERS
    turnkey_source_title: str = TURNKEY_SOURCE_TITLE
    include_extra_text_features: bool = True


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
    chunks = [chunk for chunk in re.split(r"[.!?]+", text) if chunk.strip()]
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


def extract_should_split_features(
    df: pd.DataFrame,
    config: ShouldSplitFeatureConfig | None = None,
) -> pd.DataFrame:
    """Извлекает текстовые и категориальные признаки для shouldSplit."""
    cfg = config or ShouldSplitFeatureConfig()

    required_columns = {"description", "sourceMcId", "sourceMcTitle"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(f"В DataFrame отсутствуют обязательные колонки: {sorted(missing_columns)}")

    descriptions = df["description"].map(_safe_text)
    source_titles = df["sourceMcTitle"].map(_safe_text)
    source_ids = df["sourceMcId"].map(_safe_text)

    features = pd.DataFrame(index=df.index)
    features["description_word_count"] = descriptions.map(lambda text: len(text.split())).astype(np.float32)
    features["description_char_count"] = descriptions.map(len).astype(np.float32)
    features["split_marker_count"] = descriptions.map(lambda text: _count_markers(text, cfg.split_markers)).astype(np.float32)
    features["complex_marker_count"] = descriptions.map(lambda text: _count_markers(text, cfg.complex_markers)).astype(np.float32)
    features["marker_ratio"] = (features["split_marker_count"] / (features["complex_marker_count"] + 1.0)).astype(np.float32)
    features["has_bullets"] = descriptions.map(_has_bullets).astype(np.int8)

    features["source_mc_id"] = source_ids
    features["is_turnkey"] = source_titles.map(lambda title: int(title.strip() == cfg.turnkey_source_title)).astype(np.int8)

    if cfg.include_extra_text_features:
        features["sentence_count"] = descriptions.map(_sentence_count).astype(np.float32)
        features["avg_word_len"] = descriptions.map(_avg_word_length).astype(np.float32)
        features["punctuation_ratio"] = descriptions.map(_punctuation_ratio).astype(np.float32)

    return features


def append_embedding_features(
    features: pd.DataFrame,
    descriptions: Sequence[Any],
    encoder: TextEncoderLike,
    prefix: str = "embedding",
) -> pd.DataFrame:
    """Добавляет эмбеддинговые признаки в DataFrame фичей."""
    prepared_descriptions = [(_safe_text(text).strip() or "<EMPTY>") for text in descriptions]
    raw_embeddings = encoder.encode(prepared_descriptions)
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)

    if embeddings.ndim != 2:
        raise ValueError("Ожидаются эмбеддинги размерности N x D.")
    if embeddings.shape[0] != len(features):
        raise ValueError("Число эмбеддингов не совпадает с числом строк признаков.")

    embedding_columns = [f"{prefix}_{idx:03d}" for idx in range(embeddings.shape[1])]
    embedding_df = pd.DataFrame(embeddings, columns=embedding_columns, index=features.index)
    return pd.concat([features, embedding_df], axis=1)


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

    features = extract_should_split_features(df=df, config=config)
    if include_embeddings:
        if encoder is None:
            raise ValueError("Для include_embeddings=True нужно передать encoder.")
        features = append_embedding_features(
            features,
            descriptions=df["description"].tolist(),
            encoder=encoder,
        )

    y = df["shouldSplit"].astype(bool)
    split = df["split"].astype(str)
    return features, y, split