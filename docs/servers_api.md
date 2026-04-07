# API серверов

Актуальная серверная схема проекта:
- [`scripts/tabular_pipeline_server.py`](../scripts/tabular_pipeline_server.py) -> HTTP API классификации.
- [`src/servers/tabular_pipeline_server.py`](../src/servers/tabular_pipeline_server.py) -> реализация сервера.
- [`configs/servers/tabular_pipeline_server.yaml`](../configs/servers/tabular_pipeline_server.yaml) -> конфиг tabular backend.
- [`src/servers/gradio_demo.py`](../src/servers/gradio_demo.py) -> Gradio UI клиент к tabular backend.
- [`configs/servers/gradio_demo.yaml`](../configs/servers/gradio_demo.yaml) -> конфиг Gradio.
- [`scripts/init_servers.py`](../scripts/init_servers.py) -> оркестратор запуска backend + UI.

Подробные CLI-аргументы: [`docs/scripts.md`](scripts.md).

## 1) Tabular Pipeline HTTP сервер

### Запуск

```bash
python scripts/tabular_pipeline_server.py --config-path configs/servers/tabular_pipeline_server.yaml
```

Через оркестратор:

```bash
python scripts/init_servers.py
python scripts/init_servers.py --dummy
```

### Endpoints

- `GET /health`
  - Ответ: `{"status": "ok"}`

- `POST /v1/tabular/predict`
  - Назначение: оценка галлюцинации для одной или нескольких пар `query/model_answer`.

### Варианты входного payload

1) **Single-pair** формат:

```json
{
  "query": "Кто написал Войну и мир?",
  "model_answer": "Лев Толстой",
  "threshold": 0.5
}
```

2) **Batch** формат:

```json
{
  "queries": ["q1", "q2"],
  "model_answers": ["a1", "a2"],
  "threshold": 0.5
}
```

3) **Single-pair + precomputed features** (для bypass LLM и работы по train-фичам):

```json
{
  "query": "q",
  "model_answer": "a",
  "threshold": 0.5,
  "features": {
    "mean_log_prob": -0.2,
    "n_answer_tokens": 12.0
  }
}
```

### Формат ответа

```json
{
  "pred_is_hallucination": [0],
  "hallucination_score": [0.12],
  "entailment_score": [0.88],
  "t_feature_extraction_sec": [0.034],
  "t_classification_sec": [0.004],
  "t_total_sec": [0.038],
  "t_sample_sec": [0.038]
}
```

Примечания:
- `pred_is_hallucination`: `1` = галлюцинация, `0` = не галлюцинация.
- `t_sample_sec` сохранен для обратной совместимости и равен `t_total_sec`.

## 2) Gradio Demo

### Запуск

```bash
python -m src.servers.gradio_demo --config-path configs/servers/gradio_demo.yaml
```

Или через оркестратор:

```bash
python scripts/init_servers.py
```

### Что делает demo

- отправляет запросы в tabular backend (`POST /v1/tabular/predict`);
- показывает:
  - лейбл классификации;
  - score;
  - время этапов пайплайна (`feature_extraction`, `classification`, `total`);
- в `dummy mode` берет случайный train-сэмпл из CSV и отправляет precomputed features.

### Важные поля Gradio-конфига

См. [`configs/servers/gradio_demo.yaml`](../configs/servers/gradio_demo.yaml):
- `pipeline_base_url` — URL tabular backend;
- `classification_threshold` — порог интерпретации score в лейбл;
- `dummy_mode` — режим smoke-теста без ручного ввода;
- `dummy_examples_csv` — CSV с train-примерами для dummy режима.

## 3) Ошибки API

- `404` — неизвестный endpoint.
- `400` — невалидный payload (неверные типы, пустые списки, несоответствие длин и т.д.).

