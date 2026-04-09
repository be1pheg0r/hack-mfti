# Скрипты и входные аргументы

Документ описывает CLI-аргументы для скриптов из [`scripts/`](../scripts).

Общие правила:
- Аргументы можно посмотреть локально через `--help`.
- Если у скрипта есть `--config-path`, значения из CLI имеют приоритет над YAML.
- Пути в дефолтах указаны относительно корня репозитория.

## [`scripts/evaluate.py`](../scripts/evaluate.py)
Обертка над [`src/sber/utils/evaluate_cli.py`](../src/sber/utils/evaluate_cli.py).

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--input-csv` | `str` | `data/bench/bench_processed_judged_mapped.csv` | Входной CSV для скоринга. |
| `--output-csv` | `str` | `data/bench/bench_processed_judged_tabular_scores.csv` | CSV с предсказаниями и score-колонками. |
| `--checkpoint-dir` | `str` | `sber_tabular/latest` | Директория чекпоинта tabular модели. |
| `--threshold` | `float` | `None` | Override порога классификации. |
| `--report-dir` | `str` | `None` | Каталог для отчета (если не задан, рядом с output). |
| `--save-plots` / `--no-save-plots` | `bool` | `True` | Сохранять графики в отчете. |

## [`scripts/full_evaluate.py`](../scripts/full_evaluate.py)
Обертка над [`src/sber/utils/full_evaluate_cli.py`](../src/sber/utils/full_evaluate_cli.py).

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--input-csv` | `str` | `data/bench/bench_processed_judged.csv` | Входной CSV с `query`/`prompt` и `model_answer`. |
| `--output-csv` | `str` | `data/bench/bench_processed_judged_full_eval.csv` | Итоговый CSV с `predict_proba` и колонками пайплайна. |
| `--checkpoint-dir` | `str` | `sber_tabular/latest` | Директория tabular-чекпоинта. |
| `--feature-model-name` | `str` | `ai-sage/GigaChat3-10B-A1.8B-bf16` | LLM для feature extraction. |
| `--feature-config-path` | `str` | [`configs/sber/hooks_config.yaml`](../configs/sber/hooks_config.yaml) | YAML-конфиг извлечения фичей. |
| `--input-query-column` | `str` | `None` | Явное имя query-колонки (иначе auto: `query` -> `prompt`). |
| `--threshold` | `float` | `None` | Override порога бинарной метки. |

## [`scripts/extract_features.py`](../scripts/extract_features.py)
Обертка над [`src/sber/utils/extract_features_cli.py`](../src/sber/utils/extract_features_cli.py).

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--n` | `int` | `None` | Число обрабатываемых сэмплов (иначе все). |
| `--seed` | `int` | `42` | Seed для shuffle и sampling. |
| `--model-name` | `str` | `ai-sage/GigaChat3-10B-A1.8B-bf16` | HF-модель для извлечения фичей. |
| `--batch-size` | `int` | `2` | Размер батча обработки пар `query + answer`. |
| `--output-csv` | `str` | `data/raw/model_features.csv` | Путь до итогового CSV. |
| `--force-download` | `flag` | `False` | Принудительно перекачать датасеты. |
| `--input-csv-path` | `str` | `None` | Входной CSV; если задан, YAML-датасеты не используются. |
| `--input-query-column` | `str` | `None` | Явная колонка с запросом (`query`/`prompt`). |
| `--input-answer-column` | `str` | `None` | Явная колонка с ответом (`model_answer`/`answer`/`generated_answer`/`correct_answer`). |
| `--datasets-config-path` | `str` | [`configs/sber/datasets_configs.yaml`](../configs/sber/datasets_configs.yaml) | YAML-конфиг датасетов. |
| `--feature-config-path` | `str` | [`configs/sber/hooks_config.yaml`](../configs/sber/hooks_config.yaml) | YAML-конфиг извлечения фичей. |

Примечание: `extract_features` теперь использует teacher-forcing режим — для каждой пары `query + model_answer` выполняется единый `forward`, затем фичи считаются по answer-span (`answer_start..end`) без вызова `generate()`.

## [`scripts/generate_mistral_judged_dataset.py`](../scripts/generate_mistral_judged_dataset.py)
Обертка над [`src/sber/utils/generate_mistral_judged_dataset_cli.py`](../src/sber/utils/generate_mistral_judged_dataset_cli.py).

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--datasets-config-path` | `str` | [`configs/sber/datasets_configs.yaml`](../configs/sber/datasets_configs.yaml) | YAML-конфиг датасетов. |
| `--output-csv` | `str` | `data/raw/mistral_judged_dataset.csv` | Итоговый CSV. |
| `--generation-model` | `str` | `mistral-small-latest` | Модель Mistral для генерации ответов. |
| `--judge-model` | `str` | `mistral-small-latest` | Модель Mistral для judge-оценки. |
| `--generation-temperature` | `float` | `0.3` | Температура генерации ответа. |
| `--generation-top-p` | `float` | `0.95` | Top-p генерации ответа. |
| `--judge-temperature` | `float` | `0.1` | Температура judge-вызова. |
| `--judge-top-p` | `float` | `0.9` | Top-p judge-вызова. |
| `--max-samples` | `int` | `None` | Ограничение на число обрабатываемых сэмплов. |
| `--force-download` / `--no-force-download` | `bool` | `False` | Принудительная перезагрузка датасетов. |
| `--resume-from-output` / `--no-resume-from-output` | `bool` | `True` | Продолжить обработку из существующего output CSV. |
| `--save-every` | `int` | `20` | Периодичность промежуточных сохранений. |

## [`scripts/map_bench_columns.py`](../scripts/map_bench_columns.py)
Обертка над [`src/sber/utils/map_bench_columns_cli.py`](../src/sber/utils/map_bench_columns_cli.py).

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--train-csv` | `str` | `data/raw/merged_features_with_judge_scores.csv` | CSV train с эталонными именами фичей. |
| `--val-csv` | `str` | `data/bench/bench_processed_judged.csv` | CSV bench до маппинга колонок. |
| `--output-csv` | `str` | `data/bench/bench_processed_judged_mapped.csv` | CSV после маппинга. |
| `--strict` / `--no-strict` | `bool` | `True` | Падать при несовпадении train/val фичей. |
| `--inplace` / `--no-inplace` | `bool` | `False` | Перезаписывать `val_csv` вместо `output_csv`. |

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--config-path` | `str` | [`configs/sber/tabular_hallucination_train.yaml`](../configs/sber/tabular_hallucination_train.yaml) | YAML train-конфиг. |
| `--train-csv` | `str` | `None` | Override train CSV. |
| `--val-csv` | `str` | `None` | Override validation CSV. |
| `--run-name` | `str` | `None` | Имя запуска (иначе авто). |
| `--output-root` | `str` | `None` | Корневая директория чекпоинтов. |
| `--pca-n-components` | `int` | `None` | Override PCA-компонент для probe vec. |
| `--tfidf-max-features` | `int` | `None` | Override словаря TF-IDF. |
| `--tfidf-n-components` | `int` | `None` | Override PCA-компонент для TF-IDF. |
| `--scaling` / `--no-scaling` | `bool` | `None` | Override включения scaling. |
| `--under-sampling` / `--no-under-sampling` | `bool` | `None` | Override undersampling. |
| `--oversampling` / `--no-oversampling` | `bool` | `None` | Override oversampling. |
| `--plot-feature-distributions` / `--no-plot-feature-distributions` | `bool` | `None` | Override построения графиков фичей. |
| `--feature-plot-batch-size` | `int` | `None` | Override размера батча графиков. |
| `--feature-plot-dir` | `str` | `None` | Override каталога графиков. |
| `--random-seed` | `int` | `None` | Override seed обучения. |
| `--iterations` | `int` | `None` | Override итераций модели. |
| `--learning-rate` | `float` | `None` | Override learning rate. |
| `--depth` | `int` | `None` | Override depth. |
| `--uncertainty` / `--no-uncertainty` | `bool` | `None` | Override feature flag `uncertainty`. |
| `--internal-scalars` / `--no-internal-scalars` | `bool` | `None` | Override feature flag `internal_scalars`. |
| `--probe-vec` / `--no-probe-vec` | `bool` | `None` | Override feature flag `probe_vec`. |
| `--attention-entropy` / `--no-attention-entropy` | `bool` | `None` | Override feature flag `attention_entropy`. |
| `--entropy-drops` / `--no-entropy-drops` | `bool` | `None` | Override feature flag `entropy_drops`. |
| `--moe-routing` / `--no-moe-routing` | `bool` | `None` | Override feature flag `moe_routing`. |
| `--text-features` / `--no-text-features` | `bool` | `None` | Override feature flag `text_features`. |
| `--tfidf` / `--no-tfidf` | `bool` | `None` | Override feature flag `tfidf`. |

## [`scripts/tabular_pipeline_server.py`](../scripts/tabular_pipeline_server.py)
Обертка над [`src/servers/tabular_pipeline_server.py`](../src/servers/tabular_pipeline_server.py).

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--config-path` | `str` | [`configs/servers/tabular_pipeline_server.yaml`](../configs/servers/tabular_pipeline_server.yaml) | YAML-конфиг сервера. |
| `--host` | `str` | `None` | Override хоста HTTP-сервера. |
| `--port` | `int` | `None` | Override порта HTTP-сервера. |
| `--serve-mode` | `str` | `None` | Override режима (`tabular_pipeline`/`dummy`). |
| `--feature-model-name` | `str` | `None` | Override имени/пути LLM для feature extractor. |
| `--feature-extractor-mode` | `str` | `None` | Override режима экстрактора (`real`/`dummy`). |
| `--checkpoint-dir` | `str` | `None` | Override директории tabular checkpoint. |
| `--feature-config-path` | `str` | `None` | Override YAML-конфига feature extractor. |

## [`scripts/init_servers.py`](../scripts/init_servers.py)
Оркестратор запуска tabular pipeline и Gradio.

### Аргументы
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--tabular-config-path` | `str` | [`configs/servers/tabular_pipeline_server.yaml`](../configs/servers/tabular_pipeline_server.yaml) | YAML-конфиг tabular сервера. |
| `--gradio-config-path` | `str` | [`configs/servers/gradio_demo.yaml`](../configs/servers/gradio_demo.yaml) | YAML-конфиг Gradio. |
| `--without-gradio` | `flag` | `False` | Не запускать Gradio UI. |
| `--dummy` | `flag` | `False` | Запустить оба сервера в dummy-режиме. |
| `--dry-run` | `flag` | `False` | Только вывести сводку и команды запуска. |

## Дополнительно: CLI Gradio (запускается из `init_servers`)
[`scripts/init_servers.py`](../scripts/init_servers.py) стартует Gradio через `python -m src.servers.gradio_demo` (модуль: [`src/servers/gradio_demo.py`](../src/servers/gradio_demo.py)).

### Аргументы `src.servers.gradio_demo`
| Аргумент | Тип | По умолчанию | Описание |
|---|---|---|---|
| `--config-path` | `str` | [`configs/servers/gradio_demo.yaml`](../configs/servers/gradio_demo.yaml) | YAML-конфиг Gradio. |
| `--host` | `str` | `None` | Override host Gradio UI. |
| `--port` | `int` | `None` | Override port Gradio UI. |
| `--title` | `str` | `None` | Override заголовка UI. |
| `--pipeline-base-url` | `str` | `None` | Override URL tabular pipeline backend. |
| `--timeout-sec` | `float` | `None` | Override таймаута HTTP-запросов. |
| `--classification-threshold` | `float` | `None` | Override порога label по score. |
| `--dummy-mode` / `--no-dummy-mode` | `bool` | `None` | Включить/выключить dummy UX. |
| `--dummy-examples-csv` | `str` | `None` | Override CSV с примерами для dummy-режима. |
| `--share` / `--no-share` | `bool` | `None` | Публичная ссылка Gradio. |

## Быстрые примеры
```bash
python scripts/evaluate.py --help
python scripts/full_evaluate.py --help
python scripts/extract_features.py --help
python scripts/generate_mistral_judged_dataset.py --help
python scripts/map_bench_columns.py --help
python scripts/new_data.py --help
python scripts/train_tabular_hallucination.py --help
python scripts/tabular_pipeline_server.py --help
python scripts/init_servers.py --help
```



