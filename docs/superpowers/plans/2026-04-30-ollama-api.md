# Ollama-compatible API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Заменить текущие эндпоинты (`/health`, `/info`, `/generate`) на Ollama-совместимые (`/`, `/api/tags`, `/api/show`, `/api/generate`, `/api/chat`).

**Architecture:** Единственный модуль `app/main.py` переписывается полностью. Стриминг меняется с SSE на NDJSON. Pydantic-модели запросов и ответов заменяются на Ollama-совместимые. Тесты перезаписываются под новые эндпоинты.

**Tech Stack:** Python, FastAPI, Pydantic v2, pytest, litert_lm, huggingface_hub

---

## Файлы

| Файл | Действие |
|------|----------|
| `app/main.py` | Полная замена |
| `tests/test_main.py` | Полная замена |
| `README.md` | Обновить раздел API |
| `CLAUDE.md` | Обновить список эндпоинтов |

---

## Task 1: Написать новые тесты (TDD — красная фаза)

**Files:**
- Modify: `tests/test_main.py`

- [ ] **Step 1: Заменить tests/test_main.py на новые тесты**

```python
# tests/test_main.py
import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "test/model")

    mock_eng = MagicMock()
    mock_eng.__enter__ = MagicMock(return_value=mock_eng)
    mock_eng.__exit__ = MagicMock(return_value=False)
    mock_conv = MagicMock()
    mock_conv.__enter__ = MagicMock(return_value=mock_conv)
    mock_conv.__exit__ = MagicMock(return_value=False)
    mock_eng.create_conversation.return_value = mock_conv
    mock_conv.send_message.return_value = "Paris is the capital of France."
    mock_conv.send_message_async.return_value = iter(["Paris", " is", " the capital."])

    with patch("app.main._find_or_download_model", return_value="/tmp/test.litertlm"), \
         patch("app.main.litert_lm.Engine", return_value=mock_eng):
        from app.main import app
        with TestClient(app) as c:
            yield c, mock_conv


def test_root_returns_ollama_running(client):
    c, _ = client
    response = c.get("/")
    assert response.status_code == 200
    assert response.text == "Ollama is running"


def test_tags_returns_model_list(client):
    c, _ = client
    with patch("os.path.getmtime", return_value=1746000000.0), \
         patch("os.path.getsize", return_value=123456):
        response = c.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert len(data["models"]) == 1
    model = data["models"][0]
    assert model["name"] == "test"
    assert model["size"] == 123456
    assert model["details"]["format"] == "litertlm"


def test_show_returns_model_details(client):
    c, _ = client
    response = c.post("/api/show", json={"model": "test"})
    assert response.status_code == 200
    data = response.json()
    assert data["details"]["format"] == "litertlm"
    assert "modelfile" in data


def test_generate_returns_response(client):
    c, mock_conv = client
    response = c.post("/api/generate", json={"model": "test", "prompt": "What is the capital of France?"})
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Paris is the capital of France."
    assert data["done"] is True
    assert "model" in data
    assert "created_at" in data
    mock_conv.send_message.assert_called_once_with("What is the capital of France?")


def test_generate_requires_prompt(client):
    c, _ = client
    response = c.post("/api/generate", json={"model": "test"})
    assert response.status_code == 422


def test_generate_streaming_returns_ndjson(client):
    c, mock_conv = client
    mock_conv.send_message_async.return_value = iter(["Paris", " is", " the capital."])

    response = c.post("/api/generate", json={"model": "test", "prompt": "Tell me about Paris", "stream": True})
    assert response.status_code == 200
    assert "json" in response.headers["content-type"]

    lines = [line for line in response.text.splitlines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[0]["response"] == "Paris"
    assert chunks[0]["done"] is False
    assert chunks[1]["response"] == " is"
    assert chunks[2]["response"] == " the capital."
    assert chunks[-1]["done"] is True
    assert chunks[-1]["response"] == ""


def test_chat_returns_response(client):
    c, mock_conv = client
    response = c.post("/api/chat", json={
        "model": "test",
        "messages": [{"role": "user", "content": "What is the capital of France?"}],
    })
    assert response.status_code == 200
    data = response.json()
    assert data["message"]["role"] == "assistant"
    assert data["message"]["content"] == "Paris is the capital of France."
    assert data["done"] is True
    mock_conv.send_message.assert_called_once_with("user: What is the capital of France?")


def test_chat_requires_messages(client):
    c, _ = client
    response = c.post("/api/chat", json={"model": "test"})
    assert response.status_code == 422


def test_chat_streaming_returns_ndjson(client):
    c, mock_conv = client
    mock_conv.send_message_async.return_value = iter(["Paris", " is", " the capital."])

    response = c.post("/api/chat", json={
        "model": "test",
        "messages": [{"role": "user", "content": "Tell me about Paris"}],
        "stream": True,
    })
    assert response.status_code == 200

    lines = [line for line in response.text.splitlines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[0]["message"]["content"] == "Paris"
    assert chunks[0]["done"] is False
    assert chunks[-1]["done"] is True
    assert chunks[-1]["message"]["content"] == ""


def test_find_model_uses_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "test/repo")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    model_file = tmp_path / "test-model.litertlm"
    model_file.write_text("fake")

    from app.main import _find_or_download_model
    result = _find_or_download_model()
    assert result == str(model_file)


def test_find_model_downloads_when_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "test/repo")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))

    with patch("app.main.huggingface_hub.list_repo_files",
               return_value=["model.litertlm", "config.json"]), \
         patch("app.main.huggingface_hub.hf_hub_download",
               return_value=str(tmp_path / "model.litertlm")) as mock_dl:
        from app.main import _find_or_download_model
        result = _find_or_download_model()

    mock_dl.assert_called_once_with(
        repo_id="test/repo",
        filename="model.litertlm",
        local_dir=str(tmp_path),
        token=None,
    )
    assert result == str(tmp_path / "model.litertlm")
```

- [ ] **Step 2: Запустить тесты — убедиться, что они красные**

```bash
cd /home/davron/litert-ai && source .venv/bin/activate && pytest tests/ -v 2>&1 | tail -30
```

Ожидание: большинство тестов FAILED (404/422 на старых эндпоинтах, новых ещё нет).

---

## Task 2: Переписать app/main.py под Ollama API

**Files:**
- Modify: `app/main.py`

- [ ] **Step 1: Заменить app/main.py полностью**

```python
# app/main.py
import asyncio
import json
import os
import glob
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import litert_lm
import huggingface_hub

_engine = None
_engine_cm = None
_model_name: str = ""
_model_file: str = ""
_model_path: str = ""
_model_status: str = "loading"
_load_error: str = ""


def _find_or_download_model() -> str:
    models_dir = os.environ.get("MODELS_DIR", "/app/models")
    model_name = os.environ["MODEL_NAME"]
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    os.makedirs(models_dir, exist_ok=True)
    existing = glob.glob(f"{models_dir}/*.litertlm")
    if existing:
        return existing[0]

    files = list(huggingface_hub.list_repo_files(repo_id=model_name, token=token))
    litertlm_files = [f for f in files if f.endswith(".litertlm")]
    if not litertlm_files:
        raise RuntimeError(f"No .litertlm file found in repo {model_name}")

    return huggingface_hub.hf_hub_download(
        repo_id=model_name,
        filename=litertlm_files[0],
        local_dir=models_dir,
        token=token,
    )


async def _load_model_task():
    global _engine, _engine_cm, _model_file, _model_path, _model_status, _load_error
    try:
        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(None, _find_or_download_model)
        _model_path = path
        _model_file = os.path.basename(path)
        _backend_env = os.environ.get("LITERT_BACKEND", "cpu").lower()
        _backend = litert_lm.Backend.GPU if _backend_env == "gpu" else litert_lm.Backend.CPU
        _engine_cm = litert_lm.Engine(path, backend=_backend)
        _engine = _engine_cm.__enter__()
        _model_status = "ready"
    except Exception as exc:
        _load_error = str(exc)
        _model_status = "error"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_name, _model_status
    _model_name = os.environ.get("MODEL_NAME", "")
    _model_status = "loading"
    task = asyncio.create_task(_load_model_task())
    yield
    task.cancel()
    if _engine_cm is not None:
        _engine_cm.__exit__(None, None, None)
    _model_status = "stopped"


class GenerateOptions(BaseModel):
    num_predict: int = 512


class GenerateRequest(BaseModel):
    model: str = ""
    prompt: str
    stream: bool = False
    options: Optional[GenerateOptions] = None


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = ""
    messages: List[Message]
    stream: bool = False


class ShowRequest(BaseModel):
    model: str


app = FastAPI(lifespan=lifespan)


def _check_ready():
    if _model_status == "error":
        raise HTTPException(status_code=500, detail=_load_error)
    if _model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _model_details() -> dict:
    return {
        "format": "litertlm",
        "family": "gemma",
        "parameter_size": "",
        "quantization_level": "",
    }


@app.get("/")
def root():
    return PlainTextResponse("Ollama is running")


@app.get("/api/tags")
def tags():
    if _model_status != "ready":
        return {"models": []}
    name = os.path.splitext(_model_file)[0]
    mtime = datetime.fromtimestamp(
        os.path.getmtime(_model_path), tz=timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    size = os.path.getsize(_model_path)
    return {
        "models": [
            {
                "name": name,
                "model": name,
                "modified_at": mtime,
                "size": size,
                "digest": "",
                "details": _model_details(),
            }
        ]
    }


@app.post("/api/show")
def show(request: ShowRequest):
    if _model_status != "ready":
        raise HTTPException(status_code=404, detail="Model not found")
    return {
        "modelfile": "",
        "parameters": "",
        "template": "",
        "details": _model_details(),
    }


def _generate_stream(prompt: str):
    with _engine.create_conversation() as conv:
        for chunk in conv.send_message_async(prompt):
            yield json.dumps({
                "model": _model_name,
                "created_at": _now_iso(),
                "response": chunk,
                "done": False,
            }) + "\n"
    yield json.dumps({
        "model": _model_name,
        "created_at": _now_iso(),
        "response": "",
        "done": True,
    }) + "\n"


def _chat_stream(prompt: str):
    with _engine.create_conversation() as conv:
        for chunk in conv.send_message_async(prompt):
            yield json.dumps({
                "model": _model_name,
                "created_at": _now_iso(),
                "message": {"role": "assistant", "content": chunk},
                "done": False,
            }) + "\n"
    yield json.dumps({
        "model": _model_name,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": ""},
        "done": True,
    }) + "\n"


@app.post("/api/generate")
def generate(request: GenerateRequest):
    _check_ready()
    if request.stream:
        return StreamingResponse(
            _generate_stream(request.prompt),
            media_type="application/x-ndjson",
        )
    with _engine.create_conversation() as conv:
        response = conv.send_message(request.prompt)
    return {
        "model": _model_name,
        "created_at": _now_iso(),
        "response": response,
        "done": True,
    }


@app.post("/api/chat")
def chat(request: ChatRequest):
    _check_ready()
    prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
    if request.stream:
        return StreamingResponse(
            _chat_stream(prompt),
            media_type="application/x-ndjson",
        )
    with _engine.create_conversation() as conv:
        response = conv.send_message(prompt)
    return {
        "model": _model_name,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": response},
        "done": True,
    }
```

- [ ] **Step 2: Запустить тесты — убедиться, что все зелёные**

```bash
cd /home/davron/litert-ai && source .venv/bin/activate && pytest tests/ -v
```

Ожидание: все тесты PASSED.

- [ ] **Step 3: Закоммитить**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: replace endpoints with Ollama-compatible API"
```

---

## Task 3: Обновить README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Заменить раздел API в README.md**

Найти раздел `## API` и заменить его целиком на:

```markdown
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
| `options.num_predict` | int | `512` | Максимум токенов |

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
```

- [ ] **Step 2: Проверить, что README корректно отображается**

```bash
cat /home/davron/litert-ai/README.md | grep -A 3 "## API"
```

Ожидание: видна новая секция, нет старых `/health`, `/info`, `/generate`.

- [ ] **Step 3: Закоммитить**

```bash
git add README.md
git commit -m "docs: update README with Ollama-compatible API"
```

---

## Task 4: Обновить CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Обновить раздел Архитектура и пример команды теста**

Найти строку:
```
pytest tests/test_main.py::test_health_returns_ok
```
Заменить на:
```
pytest tests/test_main.py::test_root_returns_ollama_running
```

Найти блок `## Архитектура` и заменить список эндпоинтов:
```
- `GET /health` — 200 если модель готова, 503 при загрузке, 500 при ошибке
- `GET /info` — текущий статус и имя файла модели
- `POST /generate` — генерация текста; поддерживает обычный ответ и SSE-стриминг (`stream: true`)
```
на:
```
- `GET /` — `"Ollama is running"` (plain text)
- `GET /api/tags` — список моделей; пустой список пока статус не `ready`
- `POST /api/show` — детали модели; 404 если модель не готова
- `POST /api/generate` — генерация по prompt; стриминг NDJSON (`stream: true`)
- `POST /api/chat` — генерация по `messages[]`; история конкатенируется в один prompt
```

Найти строку про стриминг:
```
**Стриминг:** `_stream_generator` — синхронный генератор, обёрнутый в `StreamingResponse`; `send_message_async` возвращает синхронный итератор (не asyncio).
```
Заменить на:
```
**Стриминг:** `_generate_stream` и `_chat_stream` — синхронные генераторы, обёрнутые в `StreamingResponse` с `media_type="application/x-ndjson"`. Каждая строка — самостоятельный JSON-объект. `send_message_async` возвращает синхронный итератор (не asyncio).
```

- [ ] **Step 2: Закоммитить**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for Ollama-compatible endpoints"
```
