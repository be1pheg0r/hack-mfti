from __future__ import annotations

"""Scores `knowledge_bench_private.csv` with the HF hallucination classifier."""

import argparse
import sys
from pathlib import Path
from typing import *

import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from tqdm.auto import tqdm

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from common.logger import SBER_HOOKS_LOGGER as logger
from common.paths import PathLike, get_data_bench_dpath
from src.sber.constants import DEFAULT_SBER_HF_MODEL_REPO_ID
from src.sber.models.model_loader import SberHFModelLoader


class ScoreConfig(BaseModel):
    """Конфигурация скрипта скоринга benchmark-датасета.

    Attributes:
        input_csv: Входной CSV с колонками `ground_truth` и `model_answer`.
        output_csv: Выходной CSV со score-колонками.
        repo_id: Hugging Face repo ID модели-классификатора.
        batch_size: Размер батча для инференса.
    """

    model_config = ConfigDict(frozen=True)

    input_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "knowledge_bench_private.csv")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "knowledge_bench_private_scores.csv")
    repo_id: str = DEFAULT_SBER_HF_MODEL_REPO_ID
    batch_size: int = 8

    @field_validator("repo_id")
    @classmethod
    def validate_repo_id(cls, value: str) -> str:
        normalized: str = value.strip()
        if not normalized:
            raise ValueError("repo_id не может быть пустым")
        return normalized

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("batch_size должен быть положительным")
        return value


def parse_args() -> ScoreConfig:
    """Парсит CLI-аргументы."""
    parser = argparse.ArgumentParser(description="Скоринг knowledge_bench_private.csv через HF классификатор")
    parser.add_argument("--input-csv", type=str, default=str(Path(get_data_bench_dpath()) / "knowledge_bench_private.csv"))
    parser.add_argument("--output-csv", type=str, default=str(Path(get_data_bench_dpath()) / "knowledge_bench_private_scores.csv"))
    parser.add_argument("--repo-id", type=str, default=DEFAULT_SBER_HF_MODEL_REPO_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    namespace: argparse.Namespace = parser.parse_args()
    return ScoreConfig.model_validate(vars(namespace))


def _resolve_entailment_index(model: Any) -> int:
    """Пытается определить индекс класса entailment."""
    id2label: Any = getattr(getattr(model, "config", None), "id2label", None)
    if isinstance(id2label, dict):
        for index, label in id2label.items():
            if str(label).lower() == "entailment":
                return int(index)
    return 0


def _score_batch(model: Any, tokenizer: Any, premise_batch: list[str], hypothesis_batch: list[str], device: torch.device) -> torch.Tensor:
    """Возвращает probability entailment для батча пар текстов."""
    inputs = tokenizer(
        premise_batch,
        hypothesis_batch,
        truncation=True,
        padding=True,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        logits = model(**inputs).logits
    entailment_idx: int = _resolve_entailment_index(model)
    probs = torch.softmax(logits.float(), dim=-1)[:, entailment_idx]
    return probs.detach().cpu()


def score_dataframe(df: pd.DataFrame, repo_id: str, batch_size: int) -> pd.DataFrame:
    """Скорит DataFrame с колонками `ground_truth` и `model_answer`."""
    loader = SberHFModelLoader(repo_id=repo_id)
    bundle = loader.load()
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = bundle.device

    if "ground_truth" not in df.columns or "model_answer" not in df.columns:
        raise ValueError("Ожидаются колонки ground_truth и model_answer")

    entailment_scores: list[float] = []
    total_batches: int = (len(df) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Scoring benchmark")

    for start in range(0, len(df), batch_size):
        batch = df.iloc[start : start + batch_size]
        scores = _score_batch(
            model=model,
            tokenizer=tokenizer,
            premise_batch=batch["ground_truth"].astype(str).tolist(),
            hypothesis_batch=batch["model_answer"].astype(str).tolist(),
            device=device,
        )
        entailment_scores.extend(float(score) for score in scores.tolist())
        progress_bar.update(1)

    progress_bar.close()
    result = df.copy()
    result["entailment_score"] = entailment_scores
    result["hallucination_score"] = [1.0 - score for score in entailment_scores]
    return result


def run(config: ScoreConfig) -> Path:
    """Скорит benchmark и сохраняет CSV."""
    input_path = Path(config.input_csv)
    output_path = Path(config.output_csv)
    logger.info("Читаю benchmark: %s", input_path)
    dataframe = pd.read_csv(input_path)
    scored = score_dataframe(df=dataframe, repo_id=config.repo_id, batch_size=config.batch_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    logger.info("Сохранил score-файл: %s", output_path)
    return output_path


def main() -> None:
    """Точка входа CLI."""
    run(parse_args())


if __name__ == "__main__":
    main()

