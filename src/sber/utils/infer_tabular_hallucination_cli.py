from __future__ import annotations

import argparse
from pathlib import Path
from typing import *

import pandas as pd

from common.configs import load_config_from_namespace
from common.logger import SBER_NLI_INFERENCE_LOGGER as logger
from src.sber.constants import DEFAULT_SBER_TABULAR_INFER_CONFIG_FPATH
from src.sber.models.tabular_hallucination import (
    TabularHallucinationPredictor,
    TabularInferenceConfig,
)


def parse_args() -> TabularInferenceConfig:
    """Парсит CLI-аргументы инференса tabular-модели."""
    parser = argparse.ArgumentParser(description="Inference for tabular hallucination model")
    parser.add_argument("--config-path", type=str, default=str(DEFAULT_SBER_TABULAR_INFER_CONFIG_FPATH))
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--input-csv", type=str, default=None)
    parser.add_argument("--output-csv", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)

    namespace: argparse.Namespace = parser.parse_args()
    return load_config_from_namespace(
        config_cls=TabularInferenceConfig,
        namespace=namespace,
        fpath=namespace.config_path,
        section_name="tabular_hallucination_infer",
    )


def run(config: TabularInferenceConfig) -> Path:
    """Запускает инференс и сохраняет CSV с предсказаниями."""
    predictor = TabularHallucinationPredictor(checkpoint_dir=config.checkpoint_dir)
    input_df = pd.read_csv(Path(config.input_csv).expanduser())
    scored = predictor.predict_dataframe(dataframe=input_df, threshold=config.threshold)

    output_path = Path(config.output_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    logger.info("Результаты инференса tabular-модели сохранены в %s", output_path)
    return output_path


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    run(config=config)


if __name__ == "__main__":
    main()


