from __future__ import annotations

from common.paths import PathLike, get_servers_configs_dpath, get_sber_configs_dpath

DEFAULT_PROBE_LAYERS: list[int] = [0, 4, 8, 12, 16, 20, 24, 25]
DEFAULT_LOGIT_EPSILON: float = 1e-10
DEFAULT_ATTENTION_EPSILON: float = 1e-10
DEFAULT_ENABLE_ATTENTION_ENTROPY: bool = True
DEFAULT_ENABLE_MOE_ROUTING: bool = True
DEFAULT_OUTPUT_ATTENTIONS: bool = True
DEFAULT_FEATURE_EXTRACTION_CONFIGS_FPATH: PathLike = get_sber_configs_dpath() / "hooks_config.yaml"

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
DEFAULT_SBER_HF_NLI_CONFIG_FPATH: PathLike = get_sber_configs_dpath() / "hf_nli_clf.yaml"
DEFAULT_SBER_HF_NLI_MAX_LENGTH: int = 512
DEFAULT_SBER_HF_NLI_BATCH_SIZE: int = 8
DEFAULT_SBER_HF_NLI_HALLUCINATION_THRESHOLD: float = 0.5
DEFAULT_SBER_HF_NLI_POSITIVE_CLASS_INDEX: int = 1
DEFAULT_SBER_HF_NLI_COMPUTE_DTYPE: str = "float32"
DEFAULT_SBER_HF_NLI_BASE_MODEL_REPO_ID: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
DEFAULT_SBER_HF_NLI_TRAIN_CONFIG_FPATH: PathLike = get_sber_configs_dpath() / "hf_nli_train.yaml"
DEFAULT_SBER_TABULAR_TRAIN_CONFIG_FPATH: PathLike = get_sber_configs_dpath() / "tabular_hallucination_train.yaml"
DEFAULT_SBER_TABULAR_INFER_CONFIG_FPATH: PathLike = get_sber_configs_dpath() / "tabular_hallucination_infer.yaml"

DEFAULT_VLLM_MODEL_NAME: str = DEFAULT_SBER_SCRIPT_MODEL_NAME
DEFAULT_VLLM_SERVE_MODE: str = "vllm"
DEFAULT_VLLM_DUMMY_MODEL_NAME: str = "dummy-vllm"
DEFAULT_VLLM_DUMMY_RESPONSE_TEXT: str = "Dummy vLLM server is working."
DEFAULT_VLLM_HOST: str = "0.0.0.0"
DEFAULT_VLLM_PORT: int = 8000
DEFAULT_VLLM_DTYPE: str = "bfloat16"
DEFAULT_VLLM_TENSOR_PARALLEL_SIZE: int = 1
DEFAULT_VLLM_GPU_MEMORY_UTILIZATION: float = 0.9
DEFAULT_VLLM_MAX_MODEL_LEN: int | None = None
DEFAULT_VLLM_TRUST_REMOTE_CODE: bool = True
DEFAULT_VLLM_DOWNLOAD_DIR: PathLike | None = None
DEFAULT_VLLM_SERVE_CONFIG_FPATH: PathLike = get_servers_configs_dpath() / "vllm_server.yaml"

DEFAULT_HF_NLI_SERVER_SERVE_MODE: str = "hf_nli"
DEFAULT_HF_NLI_SERVER_HOST: str = "0.0.0.0"
DEFAULT_HF_NLI_SERVER_PORT: int = 8010
DEFAULT_HF_NLI_SERVER_CONFIG_FPATH: PathLike = get_servers_configs_dpath() / "hf_nli_server.yaml"

DEFAULT_TABULAR_PIPELINE_SERVER_HOST: str = "0.0.0.0"
DEFAULT_TABULAR_PIPELINE_SERVER_PORT: int = 8020
DEFAULT_TABULAR_PIPELINE_SERVER_MODE: str = "tabular_pipeline"
DEFAULT_TABULAR_PIPELINE_SERVER_CONFIG_FPATH: PathLike = get_servers_configs_dpath() / "tabular_pipeline_server.yaml"

DEFAULT_GRADIO_DEMO_HOST: str = "0.0.0.0"
DEFAULT_GRADIO_DEMO_PORT: int = 7860
DEFAULT_GRADIO_DEMO_SHARE: bool = False
DEFAULT_GRADIO_DEMO_TITLE: str = "Sber FeatureExtractor + Tabular Classifier Demo"
DEFAULT_GRADIO_DEMO_PIPELINE_BASE_URL: str = "http://127.0.0.1:8020"
DEFAULT_GRADIO_DEMO_TIMEOUT_SEC: float = 60.0
DEFAULT_GRADIO_DEMO_CLASSIFICATION_THRESHOLD: float = 0.5
DEFAULT_GRADIO_DEMO_CONFIG_FPATH: PathLike = get_servers_configs_dpath() / "gradio_demo.yaml"

