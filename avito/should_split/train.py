from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import joblib

from avito.config import AvitoCaseConfig
from avito.should_split.classifier import train_should_split_models
from avito.embeddings import EncoderConfig, SentenceTransformerEncoder
from avito.features import ShouldSplitFeatureConfig
from common.checkpoints import resolve_checkpoint_path, save_checkpoint
from common.logger import AVITO_SHOULD_SPLIT_LOGGER as logger
from common.paths import get_avito_checkpoints_dpath, get_avito_data_dpath


def _load_existing_best_metric(artifact_path: Path, report_path: Path) -> float | None:
    """Возвращает ratio_abs_delta из уже сохраненного артефакта/репорта, если он есть."""
    for path, loader in (
        (artifact_path, "joblib"),
        (report_path, "json"),
    ):
        if not path.exists():
            continue
        try:
            if loader == "joblib":
                payload = joblib.load(path)
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
            metric = payload.get("ratio_abs_delta")
            if isinstance(metric, (int, float)):
                return float(metric)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось прочитать {path}: {exc}")
    return None


def _build_arg_parser() -> argparse.ArgumentParser:
    case_config = AvitoCaseConfig.from_default_yaml()

    parser = argparse.ArgumentParser(description="Обучение классификатора shouldSplit для кейса Авито.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(get_avito_data_dpath()) / case_config.should_split.cli_defaults.dataset_filename),
        help="Путь к CSV-датасету.",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(Path(get_avito_checkpoints_dpath()) / case_config.should_split.cli_defaults.artifact_filename),
        help="Путь для сохранения артефакта обученной модели.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path(get_avito_checkpoints_dpath()) / case_config.should_split.cli_defaults.report_filename),
        help="Путь для сохранения JSON-отчета.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    case_config = AvitoCaseConfig.from_default_yaml()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV-датасет не найден: {data_path}")

    logger.info(f"Загрузка данных из: {data_path}")
    df = pd.read_csv(data_path)

    logger.info("Инициализация FRIDA-энкодера для эмбеддинговых признаков")
    encoder = SentenceTransformerEncoder(EncoderConfig.from_default_yaml())

    feature_config = ShouldSplitFeatureConfig(
        include_extra_text_features=case_config.should_split.include_extra_text_features,
    )

    result = train_should_split_models(
        df=df,
        include_embeddings=True,
        encoder=encoder,
        feature_config=feature_config,
        training_config=case_config.should_split.training,
    )

    artifact_path = resolve_checkpoint_path(filename=Path(args.artifact_path))
    report_path = resolve_checkpoint_path(filename=Path(args.report_path))

    existing_metric = _load_existing_best_metric(artifact_path, report_path)
    if existing_metric is not None and existing_metric <= result.ratio_abs_delta:
        logger.info(
            "Пропускаю сохранение: существующая ratio_abs_delta=%.6f лучше или равна новой %.6f",
            existing_metric,
            result.ratio_abs_delta,
        )
        logger.info("Лучшая модель остается: %s", artifact_path)
        return

    artifact_payload = {
        "best_model_name": result.model_name,
        "pipeline": result.pipeline,
        "gt_should_split_ratio": result.gt_should_split_ratio,
        "model_should_split_ratio": result.model_should_split_ratio,
        "ratio_delta": result.ratio_delta,
        "ratio_abs_delta": result.ratio_abs_delta,
        "tuned_params": result.tuned_params,
        "model_comparison": result.model_comparison_records,
        "with_embeddings": True,
        "feature_config": feature_config.model_dump(mode="json"),
    }
    artifact_path = save_checkpoint(artifact_payload, filename=artifact_path)
    logger.info(f"Артефакт модели сохранен: {artifact_path}")

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
    logger.info(f"JSON-отчет сохранен: {report_path}")

    logger.info(f"Лучшая модель: {result.model_name}")
    logger.info(f"GT shouldSplit ratio: {result.gt_should_split_ratio:.4f}")
    logger.info(f"Model shouldSplit ratio: {result.model_should_split_ratio:.4f}")
    logger.info(f"Абсолютная разница ratio: {result.ratio_abs_delta:.6f}")


if __name__ == "__main__":
    main()