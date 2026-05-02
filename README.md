# LiteRT AI — LLM Inference Server

REST API сервер для запуска LLM моделей в формате `.litertlm` через [LiteRT LM](https://ai.google.dev/edge/litert/models/overview).

## Быстрый старт

### Через Docker Compose (рекомендуется)

```bash
cp .env.example .env
# Отредактируйте .env — укажите MODEL_NAME и HUGGING_FACE_HUB_TOKEN
docker compose up
```

### Через Docker напрямую

```bash
docker build -t litert-ai .

docker run -d \
  -e MODEL_NAME=litert-community/gemma-4-E4B-it-litert-lm \
  -e PORT=8000 \
  -e HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxx \
  -e LITERT_BACKEND=cpu \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  litert-ai
```

## Переменные окружения

| Переменная | Обязательная | По умолчанию | Описание |
|---|---|---|---|
| `MODEL_NAME` | Да | — | ID модели на Hugging Face (должна содержать `.litertlm` файл) |
| `PORT` | Нет | `8000` | Порт сервера |
| `HUGGING_FACE_HUB_TOKEN` | Нет | — | HF токен для приватных/gated моделей |
| `MODELS_DIR` | Нет | `/app/models` | Папка для кэша моделей |
| `LITERT_BACKEND` | Нет | `cpu` | Вычислительный бэкенд: `cpu` или `gpu` |

> При первом запуске модель автоматически скачивается с Hugging Face. Сервер отвечает на запросы сразу, пока модель загружается (статус `loading`).

## Консольный вывод при старте

При запуске сервер печатает прогресс загрузки модели. **Первый запуск** (модель скачивается):

```
[startup] Searching for model in /app/models...
[startup] Downloading litert-community/gemma-4-E4B-it-litert-lm from Hugging Face...
gemma-4-E4B-it.litertlm:  34%|████      | 1.2G/3.5G [00:42<01:18, 29.1MB/s]
[startup] Loading model into engine...
[startup] Model ready
```

**Повторный запуск** (модель уже в кэше):

```
[startup] Searching for model in /app/models...
[startup] Found model: gemma-4-E4B-it.litertlm
[startup] Loading model into engine...
[startup] Model ready
```

В случае ошибки:

```
[startup] ERROR: No .litertlm file found in repo <model_name>
```

## API

Совместим с [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md). Клиенты OpenWebUI, Continue.dev и другие работают без изменений.

### `GET /` — проверка работоспособности

```bash
curl http://localhost:8000/
```

```
Ollama is running
```

### `GET /api/tags` — список моделей

```bash
curl http://localhost:8000/api/tags
```

```json
{
  "models": [
    {
      "name": "gemma-4-E4B-it",
      "model": "gemma-4-E4B-it",
      "modified_at": "2026-04-30T12:00:00Z",
      "size": 4294967296,
      "digest": "",
      "details": {
        "format": "litertlm",
        "family": "gemma",
        "parameter_size": "",
        "quantization_level": ""
      }
    }
  ]
}
```

Возвращает `"models": []`, пока модель ещё загружается.

### `POST /api/show` — информация о модели

```bash
curl -X POST http://localhost:8000/api/show \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it"}'
```

```json
{
  "modelfile": "",
  "parameters": "",
  "template": "",
  "details": {"format": "litertlm", "family": "gemma", "parameter_size": "", "quantization_level": ""}
}
```

### `POST /api/generate` — генерация текста

**Без стриминга:**

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it", "prompt": "Расскажи историю", "stream": false}'
```

```json
{"model": "gemma-4-E4B-it", "created_at": "2026-04-30T12:00:00Z", "response": "...", "done": true}
```

**Со стримингом (NDJSON):**

```bash
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it", "prompt": "Расскажи историю", "stream": true}'
```

```
{"model":"gemma-4-E4B-it","created_at":"...","response":"Жил","done":false}
{"model":"gemma-4-E4B-it","created_at":"...","response":"-был","done":false}
{"model":"gemma-4-E4B-it","created_at":"...","response":"","done":true}
```

**Параметры:**

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model` | string | — | Имя модели (принимается, игнорируется — движок один) |
| `prompt` | string | **обязательный** | Текст запроса |
| `stream` | bool | `false` | NDJSON-стриминг |
| `options.num_predict` | int | `512` | Принимается для совместимости, не применяется |

### `POST /api/chat` — чат с историей сообщений

**Без стриминга:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it", "messages": [{"role": "user", "content": "Привет!"}]}'
```

```json
{
  "model": "gemma-4-E4B-it",
  "created_at": "2026-04-30T12:00:00Z",
  "message": {"role": "assistant", "content": "Привет! Чем могу помочь?"},
  "done": true
}
```

**Со стримингом:**

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma-4-E4B-it", "messages": [{"role": "user", "content": "Привет!"}], "stream": true}'
```

```
{"model":"gemma-4-E4B-it","created_at":"...","message":{"role":"assistant","content":"Привет"},"done":false}
{"model":"gemma-4-E4B-it","created_at":"...","message":{"role":"assistant","content":""},"done":true}
```

**Параметры:**

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `model` | string | — | Имя модели |
| `messages` | array | **обязательный** | `[{"role": "user"/"system"/"assistant", "content": "..."}]` |
| `stream` | bool | `false` | NDJSON-стриминг |

> Все сообщения конкатенируются в один prompt. Клиент управляет историей самостоятельно (stateless).

## Поддерживаемые модели

Любая модель на Hugging Face с файлом `.litertlm`. Примеры:

- `litert-community/gemma-4-E2B-it-litert-lm`
- `litert-community/gemma-4-E4B-it-litert-lm`

## Локальная разработка

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

export MODEL_NAME=litert-community/gemma-4-E4B-it-litert-lm
export HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxx

export LITERT_BACKEND=cpu

uvicorn app.main:app --reload
```

### Тесты

```bash
pytest tests/
```
