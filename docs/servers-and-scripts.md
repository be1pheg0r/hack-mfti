# Серверы и скрипты (Avito Should Split)

Этот документ описывает точки запуска для `should_split` в кейсе Avito: API-сервер, UI и batch-скрипты.

## Где смотреть код

- FastAPI сервер: [`avito/should_split/api/server.py`](../avito/should_split/api/server.py)
- API-контракты: [`avito/should_split/api/schemas.py`](../avito/should_split/api/schemas.py)
- Преобразование ответа API: [`avito/should_split/api/service.py`](../avito/should_split/api/service.py)
- Gradio UI: [`avito/should_split/interfaces/ui/gradio_app.py`](../avito/should_split/interfaces/ui/gradio_app.py)
- CLI запуск пайплайна: [`avito/should_split/interfaces/cli/run_pipeline.py`](../avito/should_split/interfaces/cli/run_pipeline.py)
- Evaluate по разметке: [`avito/should_split/interfaces/jobs/evaluate.py`](../avito/should_split/interfaces/jobs/evaluate.py)
- Переразметка Mistral (без RAG): [`avito/should_split/interfaces/jobs/relabel_mistral_no_rag.py`](../avito/should_split/interfaces/jobs/relabel_mistral_no_rag.py)
- Конфиг графа: [`avito/should_split/core/config.py`](../avito/should_split/core/config.py)
- Константы: [`avito/constants.py`](../avito/constants.py)

---

## 1) FastAPI сервер

Реализация: [`avito/should_split/api/server.py`](../avito/should_split/api/server.py)

### Эндпоинты

- `GET /health` -> `{"status": "ok"}`
- `POST /infer` -> ответ по модели [`AvitoApiResponse`](../avito/should_split/api/schemas.py)

### Запуск

```powershell
Set-Location "C:\Users\User\Desktop\dirs\Dev\hack-mfti"
python -m avito.should_split.api.server --host 127.0.0.1 --port 8080
```

### Минимальный пример запроса

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/infer" -Method Post -ContentType "application/json" -Body '{"itemId":1,"mcId":101,"mcTitle":"Ремонт под ключ","description":"Делаем ремонт и отдельно электрику"}'
```

---

## 2) Gradio UI

Реализация: [`avito/should_split/interfaces/ui/gradio_app.py`](../avito/should_split/interfaces/ui/gradio_app.py)

UI отправляет запросы в API-клиент [`PipelineApiClient`](../avito/should_split/api/client.py), поэтому API должен быть доступен.

### Запуск

```powershell
Set-Location "C:\Users\User\Desktop\dirs\Dev\hack-mfti"
python -m avito.should_split.interfaces.ui.gradio_app --host 127.0.0.1 --port 7860 --api-url "http://127.0.0.1:8080"
```

### Полезные флаги

- `--host` — адрес Gradio UI
- `--port` — порт Gradio UI
- `--api-url` — базовый URL API
- `--share` — публичная ссылка Gradio

---

## 3) CLI: одиночный прогон пайплайна

Реализация: [`avito/should_split/interfaces/cli/run_pipeline.py`](../avito/should_split/interfaces/cli/run_pipeline.py)

### Запуск

```powershell
Set-Location "C:\Users\User\Desktop\dirs\Dev\hack-mfti"
python -m avito.should_split.interfaces.cli.run_pipeline "Делаем ремонт под ключ и отдельно сантехнику"
```

### Что печатает

- `shouldSplit`
- `stage_durations_sec`
- `metadata`

---

## 4) Evaluate по разметке

Реализация: [`avito/should_split/interfaces/jobs/evaluate.py`](../avito/should_split/interfaces/jobs/evaluate.py)

### Что считает

- `precision_micro` — доля корректно предложенных доп. категорий среди всех предложенных
- `recall_micro` — доля корректно найденных доп. категорий среди всех эталонных
- `f1_micro` — основная агрегированная метрика по доп. категориям
- `should_split_accuracy` — точность флага `shouldSplit`

### Важные особенности

- На время прохода отключаются логи Mistral и пайплайна.
- Включен progress bar (`tqdm`).
- В runtime-конфиге принудительно отключается draft-ветка (`enable_drafts=False`).
- Скрипт сохраняет копию входного датасета с добавленным полем `prediction` в API-формате:
  - `detectedMcIds`
  - `shouldSplit`
  - `drafts`
- Есть отдельный режим `--official-test`: чтение датасета формата `request/response`
  и инференс через HTTP-клиент [`PipelineApiClient`](../avito/should_split/api/client.py)
  к уже запущенному серверу (скрипт сервер не поднимает).
- Входной датасет поддерживается в форматах `.json` и `.csv`.

### Запуск

```powershell
Set-Location "C:\Users\User\Desktop\dirs\Dev\hack-mfti"
python -m avito.should_split.interfaces.jobs.evaluate --dataset "avito/data/rnc_dataset_markup.json" --output "avito/data/gitignore/rnc_dataset_markup_with_predictions.json"
```

### Запуск (official test)

```powershell
Set-Location "C:\Users\User\Desktop\dirs\Dev\hack-mfti"
python -m avito.should_split.interfaces.jobs.evaluate --official-test --dataset "avito/data/official_test.json" --api-url "http://127.0.0.1:8080" --output "avito/data/gitignore/official_test_with_predictions.json"
```

`official_test` датасет ожидается в JSON-массиве, где каждая запись содержит:
- `request` — JSON-объект или JSON-строка объекта запроса API
- `response` — JSON-объект или JSON-строка эталонного ответа API

Для `.csv` используются те же поля (`request`, `response`) в колонках, значения — JSON-строки.
Если `response` пустой, скрипт не считает такую строку в метриках, но записывает в выходной файл
ответ системы в поле `response` (и дублирует его в `prediction`).

### Флаги

- `--dataset` — входной датасет `.json` или `.csv` (по умолчанию `avito/data/rnc_dataset_markup.json`)
- `--config` — опциональный YAML-конфиг графа
- `--limit` — ограничение числа записей
- `--output` — путь для JSON с добавленным `prediction`
- `--official-test` — режим official-датасета (`request/response`) через уже запущенный API
- `--api-url` — базовый URL API-сервера для `--official-test`

---

## 5) Переразметка Mistral без RAG

Реализация: [`avito/should_split/interfaces/jobs/relabel_mistral_no_rag.py`](../avito/should_split/interfaces/jobs/relabel_mistral_no_rag.py)

Скрипт генерирует переразмеченный датасет в формате `OutputSample` (с полями `targetDetectedMcIds`, `targetSplitMcIds`, `shouldSplit`).

### Запуск

```powershell
Set-Location "C:\Users\User\Desktop\dirs\Dev\hack-mfti"
python -m avito.should_split.interfaces.jobs.relabel_mistral_no_rag --input "avito/data/rnc_dataset_auto_annoted.json" --output "avito/data/gitignore/relabel_output.json" --model "mistral-medium-latest" --save-every 50
```

### Флаги

- `--input` — входной датасет
- `--output` — выходной JSON
- `--model` — имя модели Mistral
- `--save-every` — частота checkpoint-сохранений

---

## 6) Данные, конфиги и тесты

- Данные Avito: [`avito/data/`](../avito/data/)
- YAML-конфиги: [`avito/configs/`](../avito/configs/)
- Тесты API: [`avito/tests/test_api_server_fastapi.py`](../avito/tests/test_api_server_fastapi.py)
- Тесты evaluate: [`avito/tests/test_evaluate_markup.py`](../avito/tests/test_evaluate_markup.py)
- Тесты пайплайна: [`avito/tests/test_should_split_pipeline.py`](../avito/tests/test_should_split_pipeline.py)

Для ретроспективы по подходам см. [`docs/story.md`](./story.md).

