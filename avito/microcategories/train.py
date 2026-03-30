from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from avito.config import AvitoCaseConfig
from avito.embeddings import EncoderConfig, SentenceTransformerEncoder
from avito.features import ShouldSplitFeatureConfig
from avito.microcategories.classifier import MicrocategoryTrainingConfig, train_microcategory_model
from common.checkpoints import resolve_checkpoint_path, save_checkpoint
from common.logger import AVITO_MICROCATS_LOGGER as logger
from common.paths import get_avito_checkpoints_dpath, get_avito_data_dpath


DEFAULT_ARTIFACT_FILENAME = "avito_microcategories_model.joblib"
DEFAULT_REPORT_FILENAME = "avito_microcategories_report.json"


def _build_arg_parser() -> argparse.ArgumentParser:
    case_config = AvitoCaseConfig.from_default_yaml()

    parser = argparse.ArgumentParser(description="Обучение модели выделения микрокатегорий.")
    parser.add_argument(
        "--data-path",
        type=str,
        default=str(Path(get_avito_data_dpath()) / case_config.should_split.cli_defaults.dataset_filename),
        help="Путь к CSV-датасету.",
    )
    parser.add_argument(
        "--extra-data-path",
        type=str,
        default=None,
        help="Опциональный дополнительный CSV-датасет (например, rnc_dataset_augmented_pilot).",
    )
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=str(Path(get_avito_checkpoints_dpath()) / DEFAULT_ARTIFACT_FILENAME),
        help="Путь для сохранения артефакта модели.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=str(Path(get_avito_checkpoints_dpath()) / DEFAULT_REPORT_FILENAME),
        help="Путь для сохранения JSON-отчета.",
    )
    return parser


def _load_existing_best_metric(artifact_path: Path, report_path: Path) -> float | None:
    """Возвращает micro_f1 из существующего артефакта/репорта, если он лучше."""
    for path, loader in ((artifact_path, "joblib"), (report_path, "json")):
        if not path.exists():
            continue
        try:
            payload = joblib.load(path) if loader == "joblib" else json.loads(path.read_text(encoding="utf-8"))
            metrics = payload.get("metrics") if isinstance(payload, dict) else None
            if metrics and isinstance(metrics, dict):
                micro_f1 = metrics.get("micro_f1")
                if isinstance(micro_f1, (int, float)):
                    return float(micro_f1)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Не удалось прочитать {path}: {exc}")
    return None


def main() -> None:
    args = _build_arg_parser().parse_args()
    case_config = AvitoCaseConfig.from_default_yaml()

    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"CSV-датасет не найден: {data_path}")

    logger.info(f"Загрузка данных из: {data_path}")
    df = pd.read_csv(data_path)

    if args.extra_data_path:
        extra_path = Path(args.extra_data_path)
        if not extra_path.exists():
            raise FileNotFoundError(f"Дополнительный датасет не найден: {extra_path}")
        logger.info(f"Загрузка дополнительного датасета: {extra_path}")
        extra_df = pd.read_csv(extra_path)
        df = pd.concat([df, extra_df], ignore_index=True)
        logger.info(f"Общий размер датасета после объединения: {len(df)}")

    logger.info("Инициализация FRIDA-энкодера для эмбеддинговых признаков")
    encoder = SentenceTransformerEncoder(EncoderConfig.from_default_yaml())

    feature_config = ShouldSplitFeatureConfig(
        include_extra_text_features=case_config.should_split.include_extra_text_features,
    )
    training_config = MicrocategoryTrainingConfig()

    result = train_microcategory_model(
        df=df,
        include_embeddings=True,
        encoder=encoder,
        feature_config=feature_config,
        training_config=training_config,
    )

    artifact_path = resolve_checkpoint_path(filename=Path(args.artifact_path))
    report_path = resolve_checkpoint_path(filename=Path(args.report_path))

    existing_metric = _load_existing_best_metric(artifact_path, report_path)
    if existing_metric is not None and existing_metric >= result.metrics.get("micro_f1", -1):
        logger.info(
            "Пропускаю сохранение: существующий micro_f1=%.6f не хуже нового %.6f",
            existing_metric,
            result.metrics.get("micro_f1", -1),
        )
        logger.info("Лучшая модель остается: %s", artifact_path)
        return

    artifact_payload = {
        "model_name": result.model_name,
        "pipeline": result.pipeline,
        "threshold": result.threshold,
        "mlb_classes": result.mlb_classes,
        "metrics": result.metrics,
        "model_comparison_records": result.model_comparison_records,
        "with_embeddings": True,
        "feature_config": feature_config.model_dump(mode="json"),
        "training_config": training_config.model_dump(mode="json"),
    }
    artifact_path = save_checkpoint(artifact_payload, filename=artifact_path)
    logger.info(f"Артефакт модели сохранен: {artifact_path}")

    report_payload = {
        "model_name": result.model_name,
        "threshold": result.threshold,
        "metrics": result.metrics,
        "mlb_classes": result.mlb_classes,
        "model_comparison_records": result.model_comparison_records,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON-отчет сохранен: {report_path}")


if __name__ == "__main__":
    main()
