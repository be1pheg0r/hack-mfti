# IT Purple Hack 2026 - Avito case solution | RnD Свердловской синагоги

## Обзор

Для кейса Avito реализован пайплайн с LLM + RAG, где в качестве LLM используются модели Mistral (API).

- Основной pipeline: [`avito/should_split/core/pipeline.py`](avito/should_split/core/pipeline.py)
- Конфиг pipeline: [`avito/configs/should_split_graph.yaml`](avito/configs/should_split_graph.yaml)
- Скрипт оценки/генерации предсказаний: [
  `avito/should_split/interfaces/jobs/evaluate.py`](avito/should_split/interfaces/jobs/evaluate.py)
- API сервер: [`avito/should_split/api/server.py`](avito/should_split/api/server.py)
- Gradio UI: [`avito/should_split/interfaces/ui/gradio_app.py`](avito/should_split/interfaces/ui/gradio_app.py)

## Схема пайплайна

```
START
  │
  ▼
[clean]
  │
  ▼
[pre_filter]
  │
  ├─ pre_filter_verdict=False ──────────────────────────────► [llm_exit] ──► END
  │                                                            (shouldSplit=False)
  │
  └─ pre_filter_verdict=True
       │
       ▼
    [rag_split]  (LLM → извлекает rag_split_verdict + multiCats)
       │
       ├─ multiCats=False ───────────────────────────────────► [llm_exit] ──► END
       │                                                        (shouldSplit=False)
       │
       ├─ multiCats=True, rag_split_verdict=False ───────────► [llm_exit] ──► END
       │                                                        (shouldSplit=False)
       │
       └─ multiCats=True, rag_split_verdict=True
            │
            ▼
         [categorize]  (LLM → извлекает categorized_mc_ids)
            │
            ├─ enable_drafts=False ──────────────────────────► [llm_exit] ──► END
            │                                                   (shouldSplit=True)
            │
            └─ enable_drafts=True
                 │
                 ▼
              [draft_generate]
                 │
                 ▼
              [drafts_exit] ──────────────────────────────────► END
                                                                (shouldSplit=True)
```
## Решение

[solution.csv](solution.csv) в корне проекта.

## Установка

- Python `>=3.10,<3.13`

Находясь в корне репозитория:

```bash
pip install -e .
```

### Optional extras

В `pyproject.toml` определены дополнительные группы зависимостей:

- `ui` — зависимости для интерфейса Gradio
- `dev` — зависимости для тестирования
- `notebooks` — зависимости для ноутбуков и экспериментов

Примеры установки:

```bash
pip install -e .[ui]
pip install -e .[dev]
pip install -e .[notebooks]
pip install -e .[ui,dev]
pip install -e .[ui,dev,notebooks]
```

## Запуск

`solution.csv` генерируется через скрипт [`avito/should_split/interfaces/jobs/evaluate.py`](avito/should_split/interfaces/jobs/evaluate.py):

```bash
python -m avito.should_split.interfaces.jobs.evaluate --dataset <dataset_fpath>
```

Примечание: для [`evaluate.py`](avito/should_split/interfaces/jobs/evaluate.py) предполагается,
что [`avito/should_split/api/server.py`](avito/should_split/api/server.py) уже запущен, так как инференс идет через API.

## API ключи

API ключи для теста лежат в <project_root>/.credentials/demo_keys
