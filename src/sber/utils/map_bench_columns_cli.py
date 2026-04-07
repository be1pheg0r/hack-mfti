from __future__ import annotations

import argparse
from pathlib import Path
from typing import *

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.logger import SBER_DATASETS_LOGGER as logger
from common.paths import PathLike, get_data_bench_dpath, get_data_raw_dpath

DEFAULT_PROBE_LAYERS: list[int] = [0, 4, 8, 12, 16, 20, 24, 25]
DEFAULT_PROBE_VECTOR_SIZE: int = 1536


class BenchMappingConfig(BaseModel):
    """Config for bench->train column mapping.

    Attributes:
        train_csv: Path to train CSV with reference feature names.
        val_csv: Path to validation CSV that contains feature_* names.
        output_csv: Path to save mapped validation CSV.
        strict: If True, fail when mapped val features do not match train features.
        inplace: If True, overwrite val_csv file.
    """

    model_config = ConfigDict(frozen=True)

    train_csv: PathLike = Field(default_factory=lambda: Path(get_data_raw_dpath()) / "merged_features_with_judge_scores.csv")
    val_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged.csv")
    output_csv: PathLike = Field(default_factory=lambda: Path(get_data_bench_dpath()) / "bench_processed_judged_mapped.csv")
    strict: bool = True
    inplace: bool = False

    @field_validator("train_csv", "val_csv", "output_csv")
    @classmethod
    def _validate_csv_suffix(cls, value: PathLike) -> PathLike:
        path = Path(value)
        if path.suffix.lower() != ".csv":
            raise ValueError("Expected .csv path")
        return value


def parse_args() -> BenchMappingConfig:
    """Parse CLI arguments for bench column mapping."""
    parser = argparse.ArgumentParser(description="Map bench feature_* columns to train-compatible names")
    parser.add_argument(
        "--train-csv",
        type=str,
        default=str(Path(get_data_raw_dpath()) / "merged_features_with_judge_scores.csv"),
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        default=str(Path(get_data_bench_dpath()) / "bench_processed_judged.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(Path(get_data_bench_dpath()) / "bench_processed_judged_mapped.csv"),
    )
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inplace", action=argparse.BooleanOptionalAction, default=False)
    namespace = parser.parse_args()
    return BenchMappingConfig.model_validate(vars(namespace))


def build_val_feature_to_train_mapping(
    *,
    probe_layers: Sequence[int] = DEFAULT_PROBE_LAYERS,
    probe_vector_size: int = DEFAULT_PROBE_VECTOR_SIZE,
) -> dict[str, str]:
    """Build explicit mapping from bench feature_* names to train names.

    Args:
        probe_layers: Ordered list of model layers used in features.
        probe_vector_size: Number of probe vector dimensions.

    Returns:
        Dictionary for pandas.DataFrame.rename.
    """
    mapping: dict[str, str] = {
        "feature_uncertainty_token_logprob_mean": "mean_log_prob",
        "feature_uncertainty_token_logprob_min": "min_log_prob",
        "feature_uncertainty_token_logprob_max": "max_log_prob",
        "feature_uncertainty_token_logprob_std": "std_log_prob",
        "feature_uncertainty_entropy_mean": "mean_entropy",
        "feature_uncertainty_entropy_min": "min_entropy",
        "feature_uncertainty_entropy_max": "max_entropy",
        "feature_uncertainty_entropy_std": "std_entropy",
        "feature_uncertainty_answer_token_count": "n_answer_tokens",
        "feature_uncertainty_first_token_logprob": "first_token_log_prob",
        "feature_uncertainty_top1_prob_mean": "mean_top1_prob",
        "feature_uncertainty_top5_prob_mean": "mean_top5_prob_sum",
        "feature_moe_top_prob_mean": "moe_mean_top_prob_mean",
        "feature_moe_top_prob_std": "moe_mean_top_prob_std",
        "feature_moe_top_prob_dispersion_mean": "moe_std_top_prob_mean",
        "feature_moe_top_prob_dispersion_std": "moe_std_top_prob_std",
        "feature_moe_entropy_mean": "moe_mean_entropy_mean",
        "feature_moe_entropy_std": "moe_mean_entropy_std",
        "feature_moe_entropy_dispersion_mean": "moe_std_entropy_mean",
        "feature_moe_entropy_dispersion_std": "moe_std_entropy_std",
        "feature_moe_active_ratio_mean": "moe_active_ratio_mean",
        "feature_moe_active_ratio_std": "moe_active_ratio_std",
    }

    for layer in probe_layers:
        mapping.update(
            {
                f"feature_internal_pre_answer_norm_layer_{layer}": f"layer_{layer}_prompt_last_token_norm",
                f"feature_internal_answer_norm_mean_layer_{layer}": f"layer_{layer}_answer_hidden_norm_mean",
                f"feature_internal_logit_entropy_mean_layer_{layer}": f"layer_{layer}_logit_lens_entropy",
                f"feature_attention_entropy_mean_layer_{layer}": f"attn_layer_{layer}_entropy_mean",
                f"feature_attention_entropy_max_layer_{layer}": f"attn_layer_{layer}_entropy_max",
                f"feature_attention_entropy_std_layer_{layer}": f"attn_layer_{layer}_entropy_std",
            }
        )

    for index in range(probe_vector_size):
        mapping[f"feature_probe_vec_{index}"] = f"probe_vec_{index}"

    for index in range(len(probe_layers) - 1):
        layer_from = probe_layers[index]
        layer_to = probe_layers[index + 1]
        mapping[f"feature_entropy_drop_layer_{layer_from}_to_{layer_to}"] = f"entropy_drop_{layer_from}_to_{layer_to}"

    return mapping


def _resolve_feature_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return model feature columns (exclude service/metadata columns)."""
    service_columns: set[str] = {
        "sample_id",
        "query",
        "correct_answer",
        "model_answer",
        "comment",
        "temperature",
        "hallucination_score",
        "is_hallucination",
        "judge_prompt",
        "judge_score",
    }
    return [column for column in dataframe.columns if column not in service_columns]


def _validate_mapping_sources(val_df: pd.DataFrame, mapping: Mapping[str, str]) -> None:
    """Validate that all source mapping columns exist in validation dataframe."""
    missing_sources = sorted(set(mapping.keys()) - set(val_df.columns))
    if missing_sources:
        raise ValueError(f"Validation CSV is missing source feature columns: {missing_sources[:10]}")


def _validate_train_val_features(*, train_df: pd.DataFrame, val_df: pd.DataFrame, strict: bool) -> None:
    """Validate that mapped validation features match train features."""
    train_features = _resolve_feature_columns(train_df)
    val_features = _resolve_feature_columns(val_df)

    missing_in_val = sorted(set(train_features) - set(val_features))
    extra_in_val = sorted(set(val_features) - set(train_features))

    if strict and (missing_in_val or extra_in_val):
        raise ValueError(
            "Mapped validation features do not match train features. "
            f"missing={missing_in_val[:10]}, extra={extra_in_val[:10]}"
        )

    logger.info("Train feature columns: %s", len(train_features))
    logger.info("Val feature columns: %s", len(val_features))
    if missing_in_val:
        logger.warning("Missing features in val (first 10): %s", missing_in_val[:10])
    if extra_in_val:
        logger.warning("Extra features in val (first 10): %s", extra_in_val[:10])


def run(config: BenchMappingConfig) -> Path:
    """Run bench column mapping pipeline.

    Args:
        config: CLI configuration.

    Returns:
        Output CSV path.
    """
    train_path = Path(config.train_csv)
    val_path = Path(config.val_csv)
    output_path = val_path if config.inplace else Path(config.output_csv)

    logger.info("Reading train CSV: %s", train_path)
    train_df = pd.read_csv(train_path)

    logger.info("Reading val CSV: %s", val_path)
    val_df = pd.read_csv(val_path)

    mapping = build_val_feature_to_train_mapping()
    logger.info("Manual mapping size: %s", len(mapping))

    _validate_mapping_sources(val_df=val_df, mapping=mapping)
    mapped_val_df = val_df.rename(columns=mapping)

    _validate_train_val_features(train_df=train_df, val_df=mapped_val_df, strict=config.strict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_val_df.to_csv(output_path, index=False)
    logger.info("Mapped validation CSV saved to: %s", output_path)
    return output_path


def main() -> None:
    """CLI entrypoint."""
    config = parse_args()
    run(config=config)


if __name__ == "__main__":
    main()

