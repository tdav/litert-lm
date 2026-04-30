# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Локальная разработка
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

export MODEL_NAME=litert-community/gemma-4-E4B-it-litert-lm
export LITERT_BACKEND=cpu
uvicorn app.main:app --reload

# Тесты
pytest tests/

# Запустить один тест
pytest tests/test_main.py::test_root_returns_ollama_running

# Docker
cp .env.example .env  # заполнить MODEL_NAME и HUGGING_FACE_HUB_TOKEN
docker compose up
```

## Архитектура

Единственный модуль `app/main.py` — FastAPI-приложение с тремя эндпоинтами:

- `GET /` — `"Ollama is running"` (plain text)
- `GET /api/tags` — список моделей; пустой список пока статус не `ready`
- `POST /api/show` — детали модели; 404 если модель не готова
- `POST /api/generate` — генерация по prompt; стриминг NDJSON (`stream: true`)
- `POST /api/chat` — генерация по `messages[]`; история конкатенируется в один prompt

**Жизненный цикл модели:** при старте приложения `lifespan` запускает `_load_model_task` как фоновую asyncio-задачу. Сервер принимает запросы сразу, пока статус `_model_status` остаётся `"loading"`. Загрузка: поиск `.litertlm` в `MODELS_DIR` → если нет, скачивание с Hugging Face через `hf_hub_download`.

**Движок:** `litert_lm.Engine` используется как контекстный менеджер (`__enter__`/`__exit__` вызываются вручную, т.к. lifespan не поддерживает `async with` для синхронных CM). Каждый запрос создаёт отдельный `create_conversation()`.

**Стриминг:** `_generate_stream` и `_chat_stream` — синхронные генераторы, обёрнутые в `StreamingResponse` с `media_type="application/x-ndjson"`. Каждая строка — самостоятельный JSON-объект. `send_message_async` возвращает синхронный итератор (не asyncio).

**Тесты:** `conftest.py` заглушает `litert_lm` через `sys.modules` до импорта приложения. Фикстура `client` патчит `_find_or_download_model` и `litert_lm.Engine`, поэтому тесты не требуют реальной модели.
