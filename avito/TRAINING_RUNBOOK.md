# Руководство По Запуску Обучения Avito

В этом файле собраны пошаговые инструкции по запуску обучения для кейса `avito`.
Покрываются оба скрипта:
- `avito/microcategories/train.py`
- `avito/should_split/train.py`

## 1. Предварительные Требования

1. Python 3.10+
2. Активированное виртуальное окружение
3. Установленные зависимости

Пример:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-avito.txt
```

## 2. Входные Данные

### 2.1 Датасет Для microcategories

Обязательные колонки:
- `description`
- `sourceMcId`
- `sourceMcTitle`
- `targetDetectedMcIds`
- `split` (`train`/`val`/`test`)

Базовый датасет по умолчанию:
- `avito/data/rnc_dataset.csv`

Опциональный дополнительный датасет:
- `avito/data/rnc_dataset_augmented_pilot.csv`
- Подключается через `--extra-data-path`
- Объединение выполняется через `pd.concat(..., ignore_index=True)`

### 2.2 Датасет Для should_split

Проверка обязательных колонок выполняется внутри пайплайна `should_split`.
Базовый датасет по умолчанию:
- `avito/data/rnc_dataset.csv`

## 3. CLI Обучения microcategories

Скрипт:
- `python -m avito.microcategories.train`

### 3.1 Аргументы

| Аргумент | Тип | Значение По Умолчанию | Назначение |
|---|---|---|---|
| `--data-path` | str | `avito/data/rnc_dataset.csv` | Путь к базовому CSV-датасету |
| `--extra-data-path` | str | `None` | Путь к дополнительному CSV-датасету |
| `--artifact-path` | str | `avito/checkpoints/avito_microcategories_model.joblib` | Путь сохранения артефакта модели |
| `--report-path` | str | `avito/checkpoints/avito_microcategories_report.json` | Путь сохранения JSON-отчета |
| `--hub-model-name` | str | `None` | Переопределение HF-модели эмбеддингов для backend `sklearn` |
| `--batch-size` | int | `None` | Переопределение batch size для энкодера эмбеддингов |
| `--prompt` | str | `None` | Переопределение prompt для encode |
| `--normalize-embeddings` | str | `None` (`true`/`false`) | Переопределение нормализации эмбеддингов |
| `--backend` | str | `sklearn` | Режим обучения: `sklearn` или `transformer` |
| `--transformer-model-name` | str | `jhu-clsp/mmBERT-base` | HF-модель для backend `transformer` |
| `--transformer-num-epochs` | int | `2` | Количество эпох обучения transformer backend |
| `--transformer-batch-size` | int | `8` | Batch size обучения transformer backend |
| `--transformer-eval-batch-size` | int | `16` | Batch size валидации transformer backend |
| `--transformer-max-length` | int | `256` | Максимальная длина токенизированной последовательности |
| `--transformer-learning-rate` | float | `2e-5` | Learning rate для transformer backend |

### 3.2 Примеры Запуска

#### A) Базовый запуск backend `sklearn`

```bash
python -m avito.microcategories.train
```

#### B) backend `sklearn` с extra data

```bash
python -m avito.microcategories.train \
  --backend sklearn \
  --extra-data-path avito/data/rnc_dataset_augmented_pilot.csv \
  --artifact-path avito_microcategories_model_sklearn_extra.joblib \
  --report-path avito_microcategories_report_sklearn_extra.json
```

#### C) backend `transformer` (mmBERT) с extra data

```bash
python -m avito.microcategories.train \
  --backend transformer \
  --transformer-model-name jhu-clsp/mmBERT-base \
  --extra-data-path avito/data/rnc_dataset_augmented_pilot.csv \
  --artifact-path avito_microcategories_model_mmbert_transformer.joblib \
  --report-path avito_microcategories_report_mmbert_transformer.json
```

#### D) Быстрый smoke-запуск backend `transformer`

```bash
python -m avito.microcategories.train \
  --data-path avito/data/rnc_dataset_transformer_smoke.csv \
  --backend transformer \
  --transformer-model-name jhu-clsp/mmBERT-base \
  --transformer-num-epochs 1 \
  --transformer-batch-size 2 \
  --transformer-eval-batch-size 2 \
  --artifact-path avito_microcategories_model_mmbert_transformer_smoke.joblib \
  --report-path avito_microcategories_report_mmbert_transformer_smoke.json
```

### 3.3 Выходные Артефакты

Для backend `sklearn`:
- Joblib-артефакт (pipeline + metadata)
- JSON-отчет с метриками

Для backend `transformer`:
- Joblib-артефакт (metadata)
- JSON-отчет с метриками
- Директория transformer-весов с HF-файлами:
  - `config.json`
  - `model.safetensors`
  - файлы токенайзера

### 3.4 Логика Сохранения Лучшей Модели

Целевая метрика: `micro_f1`.
Если в существующем артефакте/отчете `micro_f1 >= new_micro_f1`, сохранение новой модели пропускается.

## 4. CLI Обучения should_split

Скрипт:
- `python -m avito.should_split.train`

### 4.1 Аргументы

| Аргумент | Тип | Значение По Умолчанию | Назначение |
|---|---|---|---|
| `--data-path` | str | `avito/data/rnc_dataset.csv` | Путь к базовому CSV-датасету |
| `--artifact-path` | str | `avito/checkpoints/avito_should_split_model.joblib` | Путь сохранения артефакта модели |
| `--report-path` | str | `avito/checkpoints/avito_should_split_report.json` | Путь сохранения JSON-отчета |

### 4.2 Примеры Запуска

#### A) Базовый запуск

```bash
python -m avito.should_split.train
```

#### B) Кастомные пути выхода

```bash
python -m avito.should_split.train \
  --artifact-path avito_should_split_model_custom.joblib \
  --report-path avito_should_split_report_custom.json
```

### 4.3 Выходы И Критерий Выбора

Выходы:
- Joblib-артефакт
- JSON-отчет

Критерий выбора лучшей модели:
- `ratio_abs_delta`
- Меньше — лучше
- Сохранение пропускается, если `existing ratio_abs_delta <= new ratio_abs_delta`

## 5. Справка По Метрикам

### microcategories
- `micro_f1` (основная)
- `micro_precision`
- `micro_recall`
- `threshold`

### should_split
- `gt_should_split_ratio`
- `model_should_split_ratio`
- `ratio_delta`
- `ratio_abs_delta` (основная)

## 6. Troubleshooting

### 6.1 Сообщения HF: UNEXPECTED / MISSING

Пример:
- `decoder.* = UNEXPECTED`
- `classifier.* = MISSING`

Это нормальное поведение при загрузке базовой модели и инициализации новой task-specific classification head.

### 6.2 В backend `transformer` низкий micro_f1 или около нуля

Рекомендации:
1. Увеличить эпохи (`--transformer-num-epochs 2` или `3`)
2. Включить extra data
3. Оставить подбор порога включенным
4. На CPU использовать меньший batch size при нестабильности

### 6.3 Время обучения

Ориентиры на CPU:
- backend `sklearn` обычно заметно быстрее
- backend `transformer` на полном датасете может занимать часы

### 6.4 Предупреждение HF Hub про unauthenticated requests

Если видите предупреждение про unauthenticated requests, задайте `HF_TOKEN` для ускорения скачивания и повышения лимитов.

## 7. Воспроизводимое Side-by-Side Сравнение

Запустите оба backend на одних и тех же данных:

```bash
python -m avito.microcategories.train \
  --backend sklearn \
  --extra-data-path avito/data/rnc_dataset_augmented_pilot.csv \
  --artifact-path avito_microcategories_model_sklearn_extra.joblib \
  --report-path avito_microcategories_report_sklearn_extra.json

python -m avito.microcategories.train \
  --backend transformer \
  --transformer-model-name jhu-clsp/mmBERT-base \
  --extra-data-path avito/data/rnc_dataset_augmented_pilot.csv \
  --artifact-path avito_microcategories_model_mmbert_transformer.joblib \
  --report-path avito_microcategories_report_mmbert_transformer.json
```

Сравнивайте:
- `metrics.micro_f1`
- `metrics.micro_precision`
- `metrics.micro_recall`
- `threshold`

## 8. Быстрый Чеклист После Запуска

1. Проверить, что файл артефакта создан
2. Проверить, что JSON-отчет создан
3. Проверить улучшение основной метрики (`micro_f1` или `ratio_abs_delta`)
4. Проверить в логах факт объединения данных и выбранный backend
5. Проверить корректность путей сохранения
