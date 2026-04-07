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
    parser.add_argument(
        "--hub-model-name",
        type=str,
        default=None,
        help="Опциональное имя модели HF Hub для эмбеддингов (переопределяет YAML).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Опциональный batch_size для энкодера эмбеддингов.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Опциональный prompt для encode (переопределяет YAML).",
    )
    parser.add_argument(
        "--normalize-embeddings",
        type=str,
        choices=["true", "false"],
        default=None,
        help="Нормализовать эмбеддинги (true/false), если нужно переопределить YAML.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sklearn", "transformer"],
        default="sklearn",
        help="Backend обучения: sklearn (по умолчанию) или transformer (full fine-tuning mmBERT).",
    )
    parser.add_argument(
        "--transformer-model-name",
        type=str,
        default="jhu-clsp/mmBERT-base",
        help="HF модель для transformer backend.",
    )
    parser.add_argument("--transformer-num-epochs", type=int, default=2, help="Количество эпох для transformer backend.")
    parser.add_argument("--transformer-batch-size", type=int, default=8, help="Batch size train для transformer backend.")
    parser.add_argument(
        "--transformer-eval-batch-size",
        type=int,
        default=16,
        help="Batch size eval для transformer backend.",
    )
    parser.add_argument("--transformer-max-length", type=int, default=256, help="Max token length для transformer backend.")
    parser.add_argument(
        "--transformer-learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate для transformer backend.",
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


def _normalize_cli_output_path(raw_path: str) -> Path:
    """Нормализует путь вывода из CLI.

    Если передан путь вида `avito/checkpoints/...`, считаем его относительным к корню
    репозитория и превращаем в абсолютный, чтобы избежать дублирования checkpoints/checkpoints.
    """

    path = Path(raw_path)
    if path.is_absolute():
        return path
    normalized_parts = tuple(part.lower() for part in path.parts[:2])
    if normalized_parts == ("avito", "checkpoints"):
        return Path.cwd() / path
    return path


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

    encoder: SentenceTransformerEncoder | None = None
    if args.backend == "sklearn":
        encoder_kwargs: dict[str, object] = {
            "config_path": None,
        }
        base_encoder_config = EncoderConfig.from_default_yaml().model_dump(mode="python")
        encoder_kwargs.update(base_encoder_config)
        if args.hub_model_name is not None:
            encoder_kwargs["hub_model_name"] = args.hub_model_name
        if args.batch_size is not None:
            encoder_kwargs["batch_size"] = args.batch_size
        if args.prompt is not None:
            encoder_kwargs["prompt"] = args.prompt
        if args.normalize_embeddings is not None:
            encoder_kwargs["normalize_embeddings"] = args.normalize_embeddings.lower() == "true"

        encoder_config = EncoderConfig.model_validate(encoder_kwargs)
        logger.info(
            "Инициализация энкодера эмбеддингов: hub_model_name=%s, local_model_path=%s, batch_size=%s",
            encoder_config.hub_model_name,
            encoder_config.local_model_path,
            encoder_config.batch_size,
        )
        encoder = SentenceTransformerEncoder(encoder_config)

    feature_config = ShouldSplitFeatureConfig(
        include_extra_text_features=case_config.should_split.include_extra_text_features,
    )
    training_config = MicrocategoryTrainingConfig(
        backend_type=args.backend,
        transformer_model_name=args.transformer_model_name,
        transformer_num_epochs=args.transformer_num_epochs,
        transformer_batch_size=args.transformer_batch_size,
        transformer_eval_batch_size=args.transformer_eval_batch_size,
        transformer_max_length=args.transformer_max_length,
        transformer_learning_rate=args.transformer_learning_rate,
    )

    result = train_microcategory_model(
        df=df,
        include_embeddings=args.backend == "sklearn",
        encoder=encoder,
        feature_config=feature_config,
        training_config=training_config,
    )

    artifact_path = resolve_checkpoint_path(filename=_normalize_cli_output_path(args.artifact_path))
    report_path = resolve_checkpoint_path(filename=_normalize_cli_output_path(args.report_path))

    existing_metric = _load_existing_best_metric(artifact_path, report_path)
    if existing_metric is not None and existing_metric >= result.metrics.get("micro_f1", -1):
        logger.info(
            "Пропускаю сохранение: существующий micro_f1=%.6f не хуже нового %.6f",
            existing_metric,
            result.metrics.get("micro_f1", -1),
        )
        logger.info("Лучшая модель остается: %s", artifact_path)
        return

    transformer_model_dir: Path | None = None
    if result.backend_type == "transformer":
        if result.transformer_model is None or result.transformer_tokenizer is None:
            raise RuntimeError("Transformer backend вернул пустую модель/tokenizer")
        transformer_model_dir = artifact_path.parent / f"{artifact_path.stem}_transformer"
        transformer_model_dir.mkdir(parents=True, exist_ok=True)
        result.transformer_tokenizer.save_pretrained(str(transformer_model_dir))
        result.transformer_model.save_pretrained(str(transformer_model_dir))

    artifact_payload = {
        "model_name": result.model_name,
        "pipeline": result.pipeline,
        "threshold": result.threshold,
        "mlb_classes": result.mlb_classes,
        "metrics": result.metrics,
        "model_comparison_records": result.model_comparison_records,
        "tuned_params": result.tuned_params,
        "with_embeddings": args.backend == "sklearn",
        "backend_type": result.backend_type,
        "transformer_model_name": result.transformer_model_name,
        "transformer_model_dir": str(transformer_model_dir) if transformer_model_dir is not None else None,
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
        "tuned_params": result.tuned_params,
        "backend_type": result.backend_type,
        "transformer_model_name": result.transformer_model_name,
        "transformer_model_dir": str(transformer_model_dir) if transformer_model_dir is not None else None,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON-отчет сохранен: {report_path}")


if __name__ == "__main__":
    main()
