# Ollama-compatible API — Design Spec

**Date:** 2026-04-30  
**Scope:** Полная замена текущих эндпоинтов на Ollama-совместимый интерфейс

---

## Цель

Сделать `litert-ai` дроп-ин заменой Ollama. Клиенты (OpenWebUI, Continue.dev и другие), настроенные на Ollama, должны работать без изменений.

---

## Эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `GET` | `/` | Возвращает `"Ollama is running"` (plain text) |
| `GET` | `/api/tags` | Список доступных моделей |
| `POST` | `/api/show` | Детали конкретной модели |
| `POST` | `/api/generate` | Генерация текста по prompt |
| `POST` | `/api/chat` | Генерация текста по массиву messages |

Старые эндпоинты (`/health`, `/info`, `/generate`) удаляются полностью.

---

## Форматы

### `GET /`
```
200 OK
Content-Type: text/plain

Ollama is running
```

### `GET /api/tags`
```json
{
  "models": [
    {
      "name": "gemma-4-E4B-it",
      "model": "gemma-4-E4B-it",
      "modified_at": "<ISO8601 mtime файла модели>",
      "size": <размер файла модели в байтах>,
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
Модель берётся из `_model_name` / `_model_file`. `modified_at` и `size` — из `os.path.getmtime` / `os.path.getsize` пути к файлу модели. Если модель ещё не загружена — возвращается пустой список `"models": []`.

### `POST /api/show`
**Запрос:**
```json
{"model": "gemma-4-E4B-it"}
```
**Ответ:**
```json
{
  "modelfile": "",
  "parameters": "",
  "template": "",
  "details": {
    "format": "litertlm",
    "family": "gemma",
    "parameter_size": "",
    "quantization_level": ""
  }
}
```
Если `_model_status != "ready"` — `404 Not Found`.

### `POST /api/generate`
**Запрос:**
```json
{
  "model": "gemma-4-E4B-it",
  "prompt": "Why is the sky blue?",
  "stream": true,
  "options": {"num_predict": 512}
}
```
Поле `model` принимается, но игнорируется (движок один). `options.num_predict` маппится на `max_tokens` (по умолчанию 512).

**Ответ без стриминга:**
```json
{
  "model": "gemma-4-E4B-it",
  "created_at": "2026-04-30T12:00:00Z",
  "response": "The sky is blue because...",
  "done": true
}
```

**Ответ со стримингом (NDJSON, `stream: true`):**
```
{"model":"gemma-4-E4B-it","created_at":"...","response":"The","done":false}
{"model":"gemma-4-E4B-it","created_at":"...","response":" sky","done":false}
{"model":"gemma-4-E4B-it","created_at":"...","response":"","done":true}
```
Каждая строка — отдельный JSON-объект, разделитель `\n`. Media type: `application/x-ndjson`.

### `POST /api/chat`
**Запрос:**
```json
{
  "model": "gemma-4-E4B-it",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Why is the sky blue?"}
  ],
  "stream": true
}
```

**Обработка истории:** все сообщения конкатенируются в один prompt:
```
system: You are a helpful assistant.
user: Why is the sky blue?
```
Результат передаётся движку как единый текст. Клиент управляет историей сам (stateless).

**Ответ без стриминга:**
```json
{
  "model": "gemma-4-E4B-it",
  "created_at": "2026-04-30T12:00:00Z",
  "message": {"role": "assistant", "content": "The sky is blue because..."},
  "done": true
}
```

**Ответ со стримингом:**
```
{"model":"gemma-4-E4B-it","created_at":"...","message":{"role":"assistant","content":"The"},"done":false}
{"model":"gemma-4-E4B-it","created_at":"...","message":{"role":"assistant","content":""},"done":true}
```

---

## Обработка ошибок

| Ситуация | Код |
|----------|-----|
| Модель загружается | `503 Service Unavailable` |
| Ошибка загрузки модели | `500 Internal Server Error` |
| Невалидный запрос | `422 Unprocessable Entity` (FastAPI) |
| Модель не найдена (`/api/show`) | `404 Not Found` |

---

## Тесты

Файл `tests/test_main.py` переписывается полностью под новые эндпоинты:

- `GET /` → `200`, body `"Ollama is running"`
- `GET /api/tags` → список с одной моделью
- `POST /api/show` → details модели
- `POST /api/generate` без стриминга → `{"response": "...", "done": true, ...}`
- `POST /api/generate` со стримингом → NDJSON строки
- `POST /api/chat` без стриминга → `{"message": {"role": "assistant", "content": "..."}, ...}`
- `POST /api/chat` со стримингом → NDJSON строки с `message.content`
- `POST /api/generate` без `prompt` → `422`
- `POST /api/chat` без `messages` → `422`

---

## Изменения в файлах

| Файл | Действие |
|------|----------|
| `app/main.py` | Полная замена эндпоинтов и моделей запросов/ответов |
| `tests/test_main.py` | Полная замена тестов |
| `CLAUDE.md` | Обновить описание эндпоинтов |
