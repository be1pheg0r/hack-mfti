#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.txt -r requirements-dev.txt
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
python - <<'PY'
from src.sber.models.model_loader import SberHFModelLoader

loader = SberHFModelLoader()
print(loader.resolve_source())
PY



