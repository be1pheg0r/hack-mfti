from __future__ import annotations

import re
from dataclasses import dataclass
from typing import *

import numpy as np
import torch
from rapidfuzz import fuzz


FEATURE_NAMES: list[str] = [
    "answer_token_len",
    "question_token_len",
    "answer_to_question_len_ratio",
    "question_number_count",
    "answer_number_count",
    "question_numbers_missing_in_answer",
    "answer_entity_count",
    "question_entity_count",
    "question_answer_entity_overlap_fuzzy",
]


@dataclass(slots=True)
class QAFeatureVector:
    """Структурированный вектор engineered-фичей для пары question/answer."""

    answer_token_len: float
    question_token_len: float
    answer_to_question_len_ratio: float
    question_number_count: float
    answer_number_count: float
    question_numbers_missing_in_answer: float
    answer_entity_count: float
    question_entity_count: float
    question_answer_entity_overlap_fuzzy: float

    def as_list(self) -> list[float]:
        return [
            self.answer_token_len,
            self.question_token_len,
            self.answer_to_question_len_ratio,
            self.question_number_count,
            self.answer_number_count,
            self.question_numbers_missing_in_answer,
            self.answer_entity_count,
            self.question_entity_count,
            self.question_answer_entity_overlap_fuzzy,
        ]


class QAFeatureExtractor:
    """Единый интерфейс вычисления фичей для question/answer."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer: Any = tokenizer

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES

    def extract_vector(self, *, question: str, answer: str) -> QAFeatureVector:
        question_text: str = str(question)
        answer_text: str = str(answer)

        question_token_len: int = self._token_len(question_text)
        answer_token_len: int = self._token_len(answer_text)
        len_ratio: float = float(answer_token_len / max(question_token_len, 1))

        question_numbers: set[str] = self._extract_numbers(question_text)
        answer_numbers: set[str] = self._extract_numbers(answer_text)
        question_number_count: int = len(question_numbers)
        answer_number_count: int = len(answer_numbers)
        numbers_missing: int = int(bool(question_numbers) and not question_numbers.issubset(answer_numbers))

        question_entities: list[str] = self._extract_entities(question_text)
        answer_entities: list[str] = self._extract_entities(answer_text)
        entity_overlap: float = self._fuzzy_entity_overlap(question_entities, answer_entities)

        return QAFeatureVector(
            answer_token_len=float(answer_token_len),
            question_token_len=float(question_token_len),
            answer_to_question_len_ratio=len_ratio,
            question_number_count=float(question_number_count),
            answer_number_count=float(answer_number_count),
            question_numbers_missing_in_answer=float(numbers_missing),
            answer_entity_count=float(len(answer_entities)),
            question_entity_count=float(len(question_entities)),
            question_answer_entity_overlap_fuzzy=entity_overlap,
        )

    def extract_matrix(self, *, questions: Sequence[str], answers: Sequence[str], device: torch.device) -> torch.Tensor:
        if len(questions) != len(answers):
            raise ValueError("questions и answers должны быть одинаковой длины")

        features: list[list[float]] = [
            self.extract_vector(question=question, answer=answer).as_list()
            for question, answer in zip(questions, answers)
        ]
        matrix = np.array(features, dtype=np.float32)
        return torch.tensor(matrix, dtype=torch.float32, device=device)

    def feature_dim(self) -> int:
        return len(FEATURE_NAMES)

    def _token_len(self, text: str) -> int:
        encoded: Any = self.tokenizer(text, add_special_tokens=False, truncation=False)
        input_ids: list[int] = encoded.get("input_ids", []) if isinstance(encoded, dict) else []
        return int(len(input_ids))

    def _extract_numbers(self, text: str) -> set[str]:
        return set(re.findall(r"\d+(?:[.,]\d+)?", text))

    def _extract_entities(self, text: str) -> list[str]:
        title_case_entities: set[str] = set(
            re.findall(r"\b[A-ZА-ЯЁ][a-zа-яё]+(?:[-\s][A-ZА-ЯЁ][a-zа-яё]+)*\b", text)
        )
        acronyms: set[str] = set(re.findall(r"\b[A-ZА-ЯЁ]{2,}\b", text))
        entities: set[str] = {value.strip() for value in title_case_entities.union(acronyms) if value.strip()}
        return sorted(value.lower() for value in entities)

    def _fuzzy_entity_overlap(self, question_entities: Sequence[str], answer_entities: Sequence[str]) -> float:
        if not question_entities:
            return 1.0
        if not answer_entities:
            return 0.0

        overlap_scores: list[float] = []
        for question_entity in question_entities:
            best_match: float = max(float(fuzz.ratio(question_entity, answer_entity)) for answer_entity in answer_entities)
            overlap_scores.append(float(best_match) / 100.0)
        return float(sum(overlap_scores) / max(len(overlap_scores), 1))


__all__ = ["FEATURE_NAMES", "QAFeatureExtractor", "QAFeatureVector"]


