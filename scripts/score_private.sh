#!/usr/bin/env bash
set -euo pipefail

python scripts/evaluate.py \
  --input-csv data/bench/knowledge_bench_private.csv \
  --output-csv data/bench/knowledge_bench_private_scores.csv

