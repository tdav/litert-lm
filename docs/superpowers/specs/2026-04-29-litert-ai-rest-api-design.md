# LiteRT-AI REST API — Design Spec

**Date:** 2026-04-29

## Overview

Python REST web application, которое предоставляет HTTP API для запуска LLM-моделей через библиотеку LiteRT-LM. Запускается в Docker, модель загружается с Hugging Face при первом старте и кешируется в volume.

## Architecture

```
Docker Container
├── FastAPI app (uvicorn)
│   ├── POST /generate      — генерация текста
│   ├── GET  /health        — healthcheck
│   └── GET  /info          — информация о модели
├── Startup logic
│   ├── Проверить /app/models/*.litertlm
│   ├── Если не найден — скачать через huggingface_hub
│   └── Загрузить litert_lm.Engine(model_path)
└── Model volume: /app/models/
```

## Environment Variables

| Переменная | Обязательна | Пример | Описание |
|---|---|---|---|
| `MODEL_NAME` | да | `litert-community/gemma-4-E2B-it-litert-lm` | HF репозиторий |
| `PORT` | нет (default: 8000) | `8000` | Порт сервера |
| `HUGGING_FACE_HUB_TOKEN` | нет* | `hf_...` | Токен HF (нужен для приватных моделей) |

\* Публичные модели не требуют токена.

## File Structure

```
litert-ai/
├── app/
│   └── main.py
├── Dockerfile
├── requirements.txt
└── docker-compose.yml
```

## API Specification

### POST /generate

**Request body:**
```json
{
  "prompt": "What is the capital of France?",
  "stream": false,
  "max_tokens": 512
}
```

- `prompt` (string, required) — входной текст
- `stream` (bool, default: false) — включить SSE streaming
- `max_tokens` (int, default: 512) — максимальная длина ответа

**Response (stream=false):**
```json
{
  "response": "Paris is the capital of France."
}
```

**Response (stream=true):**
Server-Sent Events, каждое событие содержит фрагмент текста:
```
data: {"chunk": "Paris"}
data: {"chunk": " is"}
data: {"chunk": " the capital"}
data: [DONE]
```

### GET /health

```json
{"status": "ok"}
```

Возвращает `503` если модель ещё не загружена.

### GET /info

```json
{
  "model_name": "litert-community/gemma-4-E2B-it-litert-lm",
  "model_file": "gemma-4-E2B-it.litertlm",
  "status": "ready"
}
```

## Startup Flow

1. Читаем `MODEL_NAME` из env
2. Ищем `*.litertlm` файл в `/app/models/`
3. Если не найден — скачиваем через `huggingface_hub.hf_hub_download()`
4. Загружаем `litert_lm.Engine(model_path)` как глобальный синглтон
5. Запускаем uvicorn

## Dockerfile

- Base image: `python:3.12-slim`
- WORKDIR: `/app`
- Копируем `requirements.txt` и `app/`
- `CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]`

## requirements.txt

```
fastapi
uvicorn[standard]
huggingface_hub
litert-lm-api-nightly
```

## Docker Run Example

```bash
# С загрузкой модели (первый запуск)
docker run -d \
  -e MODEL_NAME=litert-community/gemma-4-E2B-it-litert-lm \
  -e PORT=8000 \
  -e HUGGING_FACE_HUB_TOKEN=hf_xxx \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  litert-ai

# Пример вызова API
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "stream": false}'

# Пример streaming
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"prompt": "Tell me a story", "stream": true}'
```

## Error Handling

- `503` — модель не готова (загружается или ошибка загрузки)
- `422` — неверные параметры запроса (FastAPI валидация автоматически)
- `500` — ошибка генерации

## Out of Scope

- Аутентификация API
- Многопользовательские сессии / история диалогов
- GPU поддержка (CPU only)
- Батчинг запросов
