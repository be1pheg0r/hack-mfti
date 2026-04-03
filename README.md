# hack-mfti

Репозиторий с решением кейса Sber для IT Purple Hack 2026.

## Структура

- `configs/` — конфиги проекта, включая `configs/sber/hooks_config.yaml`.
- `data/bench/` — benchmark-файлы, включая `knowledge_bench_private.csv` и `knowledge_bench_private_scores.csv`.
- `data/raw/` — сырые датасеты и производные таблицы.
- `model/` — локальный cache моделей или пустой каталог, если модель загружается с Hugging Face.
- `src/sber/` — канонический Python-код кейса.
- `scripts/` — входные скрипты для извлечения фичей, установки и скоринга.
- `notebooks/sber/` — ноутбуки и исследовательские артефакты.
- `tests/` — pytest-тесты.

## Быстрый старт

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Основные команды

```bash
bash scripts/install.sh
bash scripts/score_private.sh
```

Если нужен feature extraction pipeline:

```bash
python scripts/extract_features.py --n 16
```

## Примечание по данным

Датасеты не анонимизированы: в репозитории оставлены текстовые исходники и код препроцессинга, чтобы можно было восстановить признаки end-to-end и проверить решение без подмешивания приватного теста.

