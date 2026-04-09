# IT Purple Hack 2026 | Сбер | RnD Свердловской синагоги

---

## Краткое описание подхода

Для определения галлюцинаций в ответах модели `ai-sage/GigaChat3.1-10B-A1.8B-bf16` при pass forward извлекаются
следующие признаки:

1. **Uncertainty features** (12 признаков) — статистики по логитам токенов ответа: среднее/мин/макс/std
   log-вероятностей, среднее/мин/макс/std энтропии распределения, длина ответа в токенах, log-prob первого токена,
   средние top-1 и top-5 вероятности.

2. **Internal scalars** (3 × N_probe_layers признаков) — поэлементные скаляры из probe-слоёв: L2-норма hidden state
   последнего токена промпта, средняя L2-норма hidden states токенов ответа, средняя logit-lens энтропия по токенам
   ответа.

3. **Probe vector** (hidden_size признаков) — усреднённый по всем probe-слоям вектор mean-pooling hidden states токенов
   ответа. Используется как dense embedding для downstream-классификатора.

4. **Attention entropy** (3 × N_probe_layers признаков, опционально) — статистики энтропии attention weights по токенам
   ответа в каждом probe-слое: среднее, максимум, std.

5. **Entropy drops** (N_probe_layers − 1 признаков) — разности logit-lens энтропий между соседними probe-слоями.

6. **MoE routing features** (10 признаков, опционально) — агрегированные routing-статистики по токенам ответа: mean/std
   максимальной routing-вероятности, mean/std std routing-вероятностей, mean/std routing-энтропии, mean/std std
   routing-энтропии, mean/std доли уникальных активных экспертов.

На *некоторых* из этих групп признаков обучались бустинги (CatBoost, LightGBM, XGBoost) и логистическая регрессия.
Последняя на момент 07.04 используется для инференса.

---

## Решение

Сабмит в корне файл [`submit.csv`](submit.csv).

## Установка и запуск

* Python: 3.10>=, 3.13<

### Установка

Установка зависимостей. В корне репозитория:

```bash
pip install -e .
```

### Запуск

1. Валидация last state модели на публичном бенче (скрипт: [`scripts/evaluate.py`](scripts/evaluate.py)):

```bash
python scripts/evaluate.py
```

Прим. - это для проверки на csv уже с фичами с паблик бенча.

2. Запуск демо сервера на Gradio (скрипт: [`scripts/init_servers.py`](scripts/init_servers.py)):

```bash
python scripts/init_servers.py
```

3. Обучение (скрипт: [`scripts/train_tabular_hallucination.py`](scripts/train_tabular_hallucination.py)):

```bash
python scripts/train_tabular_hallucination.py 
```

4. Predict (solution inference) (скрипт: [`scripts/predict_csv.py`](scripts/predict_csv.py)):

```bash
python scripts/predict_csv.py --input_csv <path_to_csv> -output_csv <path_to_output_csv>
```

Этим скриптом собирались метрики для сабмита.

Сигнатуры и параметры отдельных скриптов можно посмотреть через `--help` или в [`docs/scripts.md`](docs/scripts.md).

## Возможные ошибки

Корректность установки проверялась на сервере с A100, python 3.12, однако могут быть следующие ошибки:

1. NotImplementedError (via transformers) при загрузке токенизатора. Помогает обновление библиотеки.
2. "Невозможно подключиться с HuggingFace", надо в env подгрузить свой токен HF, иногда при закачке весов платформа его
   требует, иногда нет.
3. Share ссылка Gradio может не работать. Я решал прокидыванием порта через powershell.