from __future__ import annotations

import argparse
from typing import *

from common.configs import load_config_from_namespace
from common.logger import SBER_TRAIN_LOGGER as logger
from src.sber.constants import DEFAULT_SBER_TABULAR_TRAIN_CONFIG_FPATH
from src.sber.models.tabular_hallucination import TabularHallucinationTrainer, TabularTrainConfig


def parse_args() -> TabularTrainConfig:
    """Парсит CLI-аргументы и возвращает train-конфиг tabular-модели."""
    parser = argparse.ArgumentParser(description="Train tabular hallucination model")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_SBER_TABULAR_TRAIN_CONFIG_FPATH))
    parser.add_argument("--train-csv", type=str, default=None)
    parser.add_argument("--val-csv", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--pca-n-components", type=int, default=None)
    parser.add_argument("--tfidf-max-features", type=int, default=None)
    parser.add_argument("--tfidf-n-components", type=int, default=None)
    parser.add_argument("--scaling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--under-sampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--oversampling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--plot-feature-distributions", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--feature-plot-batch-size", type=int, default=None)
    parser.add_argument("--feature-plot-dir", type=str, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--depth", type=int, default=None)

    parser.add_argument("--uncertainty", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--internal-scalars", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--probe-vec", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--attention-entropy", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--entropy-drops", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--moe-routing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tfidf", action=argparse.BooleanOptionalAction, default=None)

    namespace: argparse.Namespace = parser.parse_args()
    base_config: TabularTrainConfig = load_config_from_namespace(
        config_cls=TabularTrainConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="tabular_hallucination_train",
    )

    flags_payload: dict[str, Any] = base_config.feature_flags.model_dump()
    for source_key, target_key in {
        "uncertainty": "uncertainty",
        "internal_scalars": "internal_scalars",
        "probe_vec": "probe_vec",
        "attention_entropy": "attention_entropy",
        "entropy_drops": "entropy_drops",
        "moe_routing": "moe_routing",
        "text_features": "text_features",
        "tfidf": "tfidf",
    }.items():
        override_value = getattr(namespace, source_key, None)
        if override_value is not None:
            flags_payload[target_key] = bool(override_value)

    payload: dict[str, Any] = base_config.model_dump()
    payload["feature_flags"] = flags_payload
    return TabularTrainConfig.model_validate(payload)


def run(config: TabularTrainConfig) -> dict[str, Any]:
    """Запускает обучение tabular-модели."""
    trainer = TabularHallucinationTrainer(config=config)
    train_result = trainer.train()
    result: dict[str, Any] = train_result.model_dump()
    logger.info("Обучение tabular-модели завершено: %s", result)
    return result


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    run(config=config)


if __name__ == "__main__":
    main()

