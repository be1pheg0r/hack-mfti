from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import *

import pandas as pd

from avito.should_split.core.config import EncoderConfig, VectorDbConfig
from avito.should_split.domain.models import RetrievedExample


class ExampleRetriever(Protocol):
    """Контракт retrieval слоя."""

    def search(self, description: str, limit: int) -> list[RetrievedExample]:
        """Возвращает top-N релевантных примеров."""


class InMemoryRetriever:
    """Простой ретривер по строковому сходству для локальной отладки."""

    def __init__(self, examples_df: pd.DataFrame) -> None:
        self._examples_df = examples_df.copy()

    def search(self, description: str, limit: int) -> list[RetrievedExample]:
        scored_rows: list[tuple[float, pd.Series]] = []
        for _, row in self._examples_df.iterrows():
            text = str(row["description"])
            score = SequenceMatcher(None, description, text).ratio()
            scored_rows.append((score, row))

        scored_rows.sort(key=lambda item: item[0], reverse=True)
        top_rows = scored_rows[:limit]
        return [
            RetrievedExample(
                text=str(row["description"]),
                shouldSplit=bool(row["shouldSplit"]),
                categories=[str(category) for category in row["targetSplitMcTitles"]],
                score=float(score),
            )
            for score, row in top_rows
        ]


class ChromaRetriever:
    """Обертка над Chroma retriever с lazy import."""

    def __init__(
        self,
        examples_df: pd.DataFrame,
        vector_db_config: VectorDbConfig,
        encoder_config: EncoderConfig,
    ) -> None:
        try:
            from langchain_huggingface.embeddings import HuggingFaceEmbeddings
            from langchain_community.vectorstores import Chroma
        except ImportError as error:
            raise ImportError(
                "Для backend=chroma установите langchain-community и chromadb"
            ) from error

        self._examples_df = examples_df.copy()
        embedding_model = HuggingFaceEmbeddings(model_name=encoder_config.model_name)
        texts = self._examples_df["description"].astype(str).tolist()

        persist_dir = Path(vector_db_config.persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._vectorstore = Chroma(
            embedding_function=embedding_model,
            collection_name=vector_db_config.collection_name,
            persist_directory=str(persist_dir),
        )

        try:
            collection_count = int(self._vectorstore._collection.count())
        except Exception:
            collection_count = 0

        if collection_count == 0 and texts:
            self._vectorstore.add_texts(texts=texts)
            if vector_db_config.persist:
                self._vectorstore.persist()
        self._search_k = vector_db_config.search_k

    def search(self, description: str, limit: int) -> list[RetrievedExample]:
        k = min(max(limit, 1), self._search_k)
        docs = self._vectorstore.similarity_search(description, k=k)

        examples: list[RetrievedExample] = []
        for doc in docs:
            text = str(doc.page_content)
            rows = self._examples_df.loc[self._examples_df["description"] == text]
            if rows.empty:
                continue
            row = rows.iloc[0]
            examples.append(
                RetrievedExample(
                    text=text,
                    shouldSplit=bool(row["shouldSplit"]),
                    categories=[str(category) for category in row["targetSplitMcTitles"]],
                    score=0.0,
                )
            )
        return examples


def build_retriever(
    examples_df: pd.DataFrame,
    vector_db_config: VectorDbConfig,
    encoder_config: EncoderConfig,
) -> ExampleRetriever:
    """Фабрика retrieval слоя по конфигу backend."""
    if vector_db_config.backend == "chroma":
        return ChromaRetriever(
            examples_df=examples_df,
            vector_db_config=vector_db_config,
            encoder_config=encoder_config,
        )
    return InMemoryRetriever(examples_df=examples_df)

