from __future__ import annotations

from common.paths import PathLike, get_configs_dpath

DEFAULT_PROBE_LAYERS: list[int] = [0, 4, 8, 12, 16, 20, 24, 28, 31]
DEFAULT_LOGIT_EPSILON: float = 1e-10
DEFAULT_ATTENTION_EPSILON: float = 1e-10
DEFAULT_ENABLE_ATTENTION_ENTROPY: bool = True
DEFAULT_ENABLE_MOE_ROUTING: bool = True
DEFAULT_OUTPUT_ATTENTIONS: bool = True
DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH: PathLike = get_configs_dpath() / "sber" / "hooks_config.yaml"

DEFAULT_RUBQ_DATASET_SLUG: str = "valentinbiryukov/rubq-20"
DEFAULT_RUBQ_ARCHIVE_NAME: str = "rubq-20.zip"
DEFAULT_RUBQ_JSON_NAME: str = "RuBQ_2.0_dev.json"
DEFAULT_TAPE_DATASET_REPO: str = "RussianNLP/tape"
DEFAULT_TAPE_CACHE_SUBDIR: str = "tape"
DEFAULT_DATASET_NAMES: list[str] = ["rubq-20", "tape-chegeka.raw", "tape-multiq.raw"]
DEFAULT_KAGGLE_API_URL_TEMPLATE: str = "https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"

DEFAULT_SBER_SCRIPT_MODEL_NAME: str = "ai-sage/GigaChat3-10B-A1.8B-bf16"
DEFAULT_SBER_SCRIPT_OUTPUT_FILENAME: str = "model_features.csv"
DEFAULT_SBER_SCRIPT_BATCH_SIZE: int = 2
DEFAULT_SBER_SCRIPT_MAX_NEW_TOKENS: int = 64
DEFAULT_SBER_SCRIPT_SEED: int = 42
DEFAULT_SBER_SCRIPT_TEMPERATURE_MEAN: float = 1.0
DEFAULT_SBER_SCRIPT_TEMPERATURE_STD: float = 0.2
DEFAULT_SBER_SCRIPT_TEMPERATURE_MIN: float = 0.3
DEFAULT_SBER_SCRIPT_TEMPERATURE_MAX: float = 1.7
DEFAULT_HALLUCINATION_SCORE_THRESHOLD: int = 70

DEFAULT_SBER_HF_MODEL_REPO_ID: str = "be1pheg0r/hack-mfti-sbercase"
DEFAULT_SBER_HF_MODEL_DIRNAME: str = DEFAULT_SBER_HF_MODEL_REPO_ID.rsplit("/", maxsplit=1)[-1]

