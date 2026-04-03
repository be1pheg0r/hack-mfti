#!/usr/bin/env bash
set -euo pipefail

python scripts/score_private.py \
  --input-csv data/bench/knowledge_bench_private.csv \
  --output-csv data/bench/knowledge_bench_private_scores.csv

