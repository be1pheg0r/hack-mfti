# Changelog Используемых В Методологии Подходов

30.03.2026 - 05.04.2026

## Abstract

Бейзлайн: multi-label классификация микрокатегорий через `OneVsRest` на инженерных признаках + эмбеддингах.

Основные гипотезы в текущем цикле:
1. Можно улучшить результат заменой эмбеддинг-модели, сохранив классический head (`logreg`/`xgboost`/`rf`).
2. Можно улучшить результат, если дообучать `jhu-clsp/mmBERT-base` end-to-end как multi-label классификатор.

---

## 1. Бейзлайн: OVR + инженерные признаки + эмбеддинги

### Что было реализовано
- Вынесены и унифицированы фичи в общем пайплайне.
- Добавлено сравнение нескольких архитектур для microcategories по аналогии с should_split:
  - `one_vs_rest_logreg`
  - `one_vs_rest_xgboost`
  - `one_vs_rest_random_forest`
- Добавлены YAML-конфиги архитектур и подбор гиперпараметров.
- Включен Optuna-тюнинг по умолчанию.

### Результат
- Лучшей моделью в классическом стеке остался `one_vs_rest_logreg`.
- Метрика на отчете бейзлайна:
  - `micro_f1 = 0.8480`
  - `micro_precision = 0.8002`
  - `micro_recall = 0.9019`

Источник: `avito/checkpoints/avito_microcategories_report.json`.

---

## 2. Гипотеза 1: смена эмбеддинг-бэкбона при неизменном классификаторе

### Идея
Проверить, улучшится ли качество при замене модели эмбеддингов, но без смены классификационного стека.

### Экспериментальные варианты
1. `jhu-clsp/mmBERT-base` как эмбеддер (backend `sklearn` остается прежним).
2. `deepvk/RuModernBERT-base` как эмбеддер (backend `sklearn` остается прежним).

### Результаты
- `mmBERT` как эмбеддер:
  - `micro_f1 = 0.8437`
  - `micro_precision = 0.8004`
  - `micro_recall = 0.8918`
  - Источник: `avito/checkpoints/avito_microcategories_report_mmbert.json`
- `RuModernBERT` как эмбеддер:
  - `micro_f1 = 0.8457`
  - `micro_precision = 0.8142`
  - `micro_recall = 0.8798`
  - Источник: `avito/checkpoints/avito_microcategories_report_rumodernbert.json`

### Вывод
Смена только эмбеддинг-модели в этом цикле не дала прироста относительно бейзлайна (`0.8480`).

---

## 3. Гипотеза 2: end-to-end transformer backend (mmBERT)

### Что было сделано
Реализован отдельный backend `transformer` для `avito/microcategories/train.py`, где `jhu-clsp/mmBERT-base` дообучается напрямую как multi-label классификатор.

Добавлено:
- Прямое обучение через `transformers`/`torch`.
- Сохранение `save_pretrained` директории рядом с joblib-артефактом.
- Поддержка инференса из сохраненной transformer-директории.
- Подбор порога на валидации тем же механизмом метрик.

### Промежуточная проблема
На первых прогонах наблюдался `micro_f1 = 0.0000` в smoke-сценариях.

### Исправления
1. Добавлен `BCEWithLogitsLoss` с `pos_weight` (class imbalance handling).
2. Добавлена адаптивная сетка порогов для transformer-ветки при дефолтном одиночном пороге (`[0.1, 0.2, 0.3, 0.4, 0.5]`).

### Результаты
- Smoke после фиксов:
  - `micro_f1 = 0.5714`
  - `threshold = 0.2`
  - Источник: `avito/checkpoints/avito_microcategories_report_mmbert_transformer_smoke.json`
- Полный запуск с extra data (`rnc_dataset + rnc_dataset_augmented_pilot`):
  - `micro_f1 = 0.9263`
  - `micro_precision = 0.9208`
  - `micro_recall = 0.9319`
  - `threshold = 0.5`
  - Источник: `avito/checkpoints/avito_microcategories_report_mmbert_transformer.json`

### Вывод
В этом экспериментальном цикле end-to-end mmBERT показал значительный прирост по `micro_f1` относительно классического бейзлайна.

---

## 4. Конфигурации И Артефакты

### Ключевые точки в коде
- `avito/microcategories/classifier.py`:
  - мультиархитектурный sklearn-пайплайн
  - Optuna-тюнинг
  - transformer backend и обучение
- `avito/microcategories/train.py`:
  - расширенный CLI (backend, transformer-параметры, extra data)
  - сохранение joblib + JSON + transformer directory
- `avito/microcategories/inference.py`:
  - загрузка артефактов для `sklearn` и `transformer`

### Основные отчеты
- `avito/checkpoints/avito_microcategories_report.json`
- `avito/checkpoints/avito_microcategories_report_mmbert.json`
- `avito/checkpoints/avito_microcategories_report_rumodernbert.json`
- `avito/checkpoints/avito_microcategories_report_mmbert_transformer.json`
- `avito/checkpoints/avito_microcategories_report_mmbert_transformer_smoke.json`

---

## 5. Риски И Примечания

1. Время обучения transformer backend на CPU существенно выше sklearn-режима.
2. Сообщения вида `UNEXPECTED/MISSING` при первой загрузке base HF-модели для новой головы классификации являются ожидаемыми.
3. Высокий валид. результат (`0.9263`) требует дополнительной проверки на независимом holdout / blind-сценариях.
4. Для честного сравнения режимов нужно фиксировать:
   - одинаковый датасет и split
   - одинаковые условия запуска
   - отдельные пути артефактов/отчетов

---

## 6. Следующие Шаги

1. Сравнить `sklearn` vs `transformer` на нескольких фиксированных random seeds.
2. Добавить отдельный эксперимент с `deepvk/RuModernBERT-base` как end-to-end backend.
3. Проверить обобщающую способность на отдельном закрытом тесте/holdout.
4. При необходимости добавить early stopping и планомерный подбор `max_length`, `epochs`, `lr`.
