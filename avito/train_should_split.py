from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from avito.config import AvitoCaseConfig
from avito.classifier import train_should_split_models
from avito.embeddings import EncoderConfig, SentenceTransformerEncoder
from avito.features import ShouldSplitFeatureConfig
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.paths import get_avito_data_dpath, get_checkpoints_dpath


def _build_arg_parser() -> argparse.ArgumentParser:
    case_config = AvitoCaseConfig.from_default_yaml()

    parser = argparse.ArgumentParser(description="Train shouldSplit classifier for Avito.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(get_avito_data_dpath()) / case_config.should_split.cli_defaults.dataset_filename),
        help="Path to the dataset CSV.",
    )
    parser.add_argument(
        "--with-embeddings",
        action="store_true",
        help="Use FRIDA embeddings in the feature matrix.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(Path(get_checkpoints_dpath()) / case_config.should_split.cli_defaults.artifact_filename),
        help="Where to save the trained artifact.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path(get_checkpoints_dpath()) / case_config.should_split.cli_defaults.report_filename),
        help="Where to save metrics report JSON.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    case_config = AvitoCaseConfig.from_default_yaml()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {data_path}")

    logger.info(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    encoder = None
    if args.with_embeddings:
        logger.info("Initializing FRIDA encoder for embedding features")
        encoder = SentenceTransformerEncoder(EncoderConfig.from_default_yaml())

    feature_config = ShouldSplitFeatureConfig(
        include_extra_text_features=case_config.should_split.include_extra_text_features,
    )

    result = train_should_split_models(
        df=df,
        include_embeddings=args.with_embeddings,
        encoder=encoder,
        feature_config=feature_config,
        training_config=case_config.should_split.training,
    )

    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_payload = {
        "best_model_name": result.model_name,
        "pipeline": result.pipeline,
        "gt_should_split_ratio": result.gt_should_split_ratio,
        "model_should_split_ratio": result.model_should_split_ratio,
        "ratio_delta": result.ratio_delta,
        "ratio_abs_delta": result.ratio_abs_delta,
        "tuned_params": result.tuned_params,
        "model_comparison": result.model_comparison_records,
        "with_embeddings": bool(args.with_embeddings),
        "feature_config": feature_config.model_dump(mode="json"),
    }
    joblib.dump(artifact_payload, artifact_path)
    logger.info(f"Saved model artifact: {artifact_path}")

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "best_model_name": result.model_name,
        "gt_should_split_ratio": result.gt_should_split_ratio,
        "model_should_split_ratio": result.model_should_split_ratio,
        "ratio_delta": result.ratio_delta,
        "ratio_abs_delta": result.ratio_abs_delta,
        "tuned_params": result.tuned_params,
        "model_comparison": result.model_comparison_records,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved report JSON: {report_path}")

    logger.info(f"Best model: {result.model_name}")
    logger.info(f"GT shouldSplit ratio: {result.gt_should_split_ratio:.4f}")
    logger.info(f"Model shouldSplit ratio: {result.model_should_split_ratio:.4f}")
    logger.info(f"Ratio abs delta: {result.ratio_abs_delta:.6f}")


if __name__ == "__main__":
    main()