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

## API

### `GET /info` — статус сервера

```bash
curl http://localhost:8000/info
```

```json
{
  "model_name": "litert-community/gemma-4-E4B-it-litert-lm",
  "model_file": "gemma-4-E4B-it-litert-lm.litertlm",
  "status": "ready",
  "error": null
}
```

Возможные значения `status`: `loading`, `ready`, `error`, `stopped`.

### `GET /health` — health check

```bash
curl http://localhost:8000/health
```

- `200 OK` — модель готова
- `503 Service Unavailable` — модель загружается
- `500 Internal Server Error` — ошибка загрузки модели

### `POST /generate` — генерация текста

**Без стриминга:**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Расскажи историю", "stream": false, "max_tokens": 512}'
```

```json
{
  "response": {
    "role": "assistant",
    "content": [{"type": "text", "text": "..."}]
  }
}
```

**Со стримингом (SSE):**

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"prompt": "Расскажи историю", "stream": true}'
```

```
data: {"chunk": {"role": "assistant", "content": [{"type": "text", "text": "Жил"}]}}
data: {"chunk": {"role": "assistant", "content": [{"type": "text", "text": "-был"}]}}
...
data: [DONE]
```

**Параметры запроса:**

| Поле | Тип | По умолчанию | Описание |
|---|---|---|---|
| `prompt` | string | — | Текст запроса |
| `stream` | bool | `false` | Включить SSE стриминг |
| `max_tokens` | int | `512` | Максимальное количество токенов |

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
