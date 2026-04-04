from __future__ import annotations

import torch

from src.sber.utils.text_features import FEATURE_NAMES, QAFeatureExtractor


class DummyTokenizer:
    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False) -> dict[str, list[int]]:
        _ = add_special_tokens
        _ = truncation
        token_count: int = len([chunk for chunk in text.split() if chunk.strip()])
        return {"input_ids": list(range(token_count))}


def test_feature_extractor_builds_expected_vector_shape() -> None:
    extractor = QAFeatureExtractor(tokenizer=DummyTokenizer())

    vector = extractor.extract_vector(
        question="Сколько планет в Солнечной системе в 2024 году?",
        answer="В Солнечной системе 8 планет.",
    )

    values = vector.as_list()
    assert len(values) == len(FEATURE_NAMES)
    assert values[0] > 0
    assert values[1] > 0
    assert values[3] >= 1
    assert values[4] >= 1


def test_feature_extractor_matrix_and_missing_numbers_flag() -> None:
    extractor = QAFeatureExtractor(tokenizer=DummyTokenizer())

    matrix: torch.Tensor = extractor.extract_matrix(
        questions=["Назови 3 закона Ньютона"],
        answers=["Законы Ньютона перечислены"],
        device=torch.device("cpu"),
    )

    assert matrix.shape == (1, len(FEATURE_NAMES))
    missing_numbers_flag_index: int = FEATURE_NAMES.index("question_numbers_missing_in_answer")
    assert float(matrix[0, missing_numbers_flag_index].item()) == 1.0
