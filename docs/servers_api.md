# API серверов

Документ описывает HTTP API для серверов `vLLM` и `HF NLI`, которые используются в проекте.

## 1) vLLM сервер (для end-to-end пайплайна)

CLI запуск:

```bash
python -m src.servers.vllm_server --config-path configs/servers/vllm_server.yaml
python -m src.servers.vllm_server --config-path configs/servers/vllm_server_dummy.yaml --serve-mode dummy
```

### Endpoints

- `GET /health` и `GET /v1/health`
  - Ответ: `{"status": "ok", "mode": "vllm|dummy"}`

- `GET /v1/models`
  - OpenAI-compatible список моделей.

- `POST /v1/chat/completions`
  - OpenAI-compatible chat completions.

- `POST /v1/completions`
  - OpenAI-compatible completions.

### Пример запроса (dummy)

```bash
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dummy-vllm",
    "messages": [
      {"role": "user", "content": "Привет"}
    ]
  }'
```

## 2) HF NLI (Бертовый классификатор) сервер

CLI запуск:

```bash
python -m src.servers.hf_nli_server --config-path configs/servers/hf_nli_server.yaml
python -m src.servers.hf_nli_server --config-path configs/servers/hf_nli_server_dummy.yaml --serve-mode dummy
```

### Endpoints

- `GET /health` и `GET /v1/health`
  - Ответ: `{"status": "ok", "mode": "hf_nli|dummy", "model": "..."}`

- `POST /v1/hf-nli/predict`
  - Назначение: классификация пар `correct_answer/model_answer` на галлюцинацию.
  - Вход:
    - `premises: list[str]` (или alias `correct_answers`)
    - `hypotheses: list[str]` (или alias `model_answers`)
    - `batch_size: int` (опционально)
  - Выход:
    - `model: str`
    - `pred_is_hallucination: list[int]`
    - `hallucination_score: list[float]`
    - `entailment_score: list[float]`

### Пример запроса

```bash
curl -X POST "http://127.0.0.1:8010/v1/hf-nli/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "premises": ["Париж - столица Франции"],
    "hypotheses": ["Париж - столица Франции"],
    "batch_size": 8
  }'
```

### Пример ответа

```json
{
  "model": "be1pheg0r/hack-mfti-sbercase",
  "pred_is_hallucination": [0],
  "hallucination_score": [0.12],
  "entailment_score": [0.88]
}
```

## Ошибки API

- `404` для неизвестного endpoint.
- `400` для невалидного payload (разная длина списков, неверные типы, пустые поля и т.д.).

