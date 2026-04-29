# LiteRT-AI REST API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Python FastAPI приложение для запуска LLM через LiteRT-LM с поддержкой sync/streaming генерации, упакованное в Docker.

**Architecture:** Lifespan FastAPI ищет или скачивает `.litertlm` файл с HF при старте, инициализирует `litert_lm.Engine` как глобальный синглтон. Каждый `/generate` запрос создаёт новый `Conversation`.

**Tech Stack:** Python 3.12, FastAPI, uvicorn, litert-lm-api-nightly, huggingface_hub, pytest, httpx, Docker

---

## File Map

| Файл | Описание |
|---|---|
| `app/__init__.py` | Пустой, делает app пакетом |
| `app/main.py` | FastAPI app, все endpoints, startup logic |
| `tests/__init__.py` | Пустой |
| `tests/conftest.py` | Мок litert_lm для тестов |
| `tests/test_main.py` | Все тесты |
| `requirements.txt` | Production зависимости |
| `requirements-dev.txt` | Test зависимости |
| `Dockerfile` | Docker образ |
| `docker-compose.yml` | Запуск через compose |
| `.env.example` | Пример переменных окружения |

---

### Task 1: requirements files

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

- [ ] **Step 1: Создать requirements.txt**

```
fastapi
uvicorn[standard]
huggingface_hub
litert-lm-api-nightly
```

- [ ] **Step 2: Создать requirements-dev.txt**

```
pytest
httpx
```

- [ ] **Step 3: Инициализировать git и сделать commit**

```bash
cd /home/davron/litert-ai
git init
git add requirements.txt requirements-dev.txt
git commit -m "chore: add requirements files"
```

---

### Task 2: Test infrastructure + /health endpoint

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Создать пустые __init__.py**

`app/__init__.py` — пустой файл.

`tests/__init__.py` — пустой файл.

- [ ] **Step 2: Создать tests/conftest.py (мок litert_lm)**

`litert_lm` — нативная библиотека, может отсутствовать в окружении тестов. Мокируем через `sys.modules` до любых импортов.

```python
# tests/conftest.py
import sys
from unittest.mock import MagicMock

sys.modules["litert_lm"] = MagicMock()
```

- [ ] **Step 3: Написать failing тест для /health**

```python
# tests/test_main.py
import os
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


def test_health_returns_ok(client):
    c, _ = client
    response = c.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_503_when_not_ready():
    import app.main as m
    original = m._model_status
    m._model_status = "loading"
    try:
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            m.health()
        assert exc.value.status_code == 503
    finally:
        m._model_status = original
```

- [ ] **Step 4: Запустить тесты — убедиться что падают**

```bash
cd /home/davron/litert-ai
pip install -r requirements-dev.txt
python -m pytest tests/test_main.py -v
```
Ожидаем: `ERROR` — `app/main.py` не существует.

- [ ] **Step 5: Создать app/main.py**

```python
# app/main.py
import json
import os
import glob
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import litert_lm
import huggingface_hub

_engine = None
_model_name: str = ""
_model_file: str = ""
_model_status: str = "loading"


def _find_or_download_model() -> str:
    models_dir = os.environ.get("MODELS_DIR", "/app/models")
    model_name = os.environ["MODEL_NAME"]
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _model_name, _model_file, _model_status
    _model_name = os.environ.get("MODEL_NAME", "")
    model_path = _find_or_download_model()
    _model_file = os.path.basename(model_path)
    with litert_lm.Engine(model_path) as eng:
        _engine = eng
        _model_status = "ready"
        yield
    _engine = None
    _model_status = "stopped"


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    if _model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ok"}
```

- [ ] **Step 6: Запустить тесты — убедиться что проходят**

```bash
python -m pytest tests/test_main.py::test_health_returns_ok tests/test_main.py::test_health_returns_503_when_not_ready -v
```
Ожидаем: оба `PASSED`.

- [ ] **Step 7: Commit**

```bash
git add app/__init__.py app/main.py tests/__init__.py tests/conftest.py tests/test_main.py
git commit -m "feat: add FastAPI skeleton with /health endpoint"
```

---

### Task 3: /info endpoint

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Написать failing тест**

Добавить в конец `tests/test_main.py`:

```python
def test_info_returns_model_info(client):
    c, _ = client
    response = c.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test/model"
    assert data["model_file"] == "test.litertlm"
    assert data["status"] == "ready"
```

- [ ] **Step 2: Запустить тест — убедиться что падает**

```bash
python -m pytest tests/test_main.py::test_info_returns_model_info -v
```
Ожидаем: `FAILED` — 404 Not Found.

- [ ] **Step 3: Добавить /info в app/main.py после /health**

```python
@app.get("/info")
def info():
    return {
        "model_name": _model_name,
        "model_file": _model_file,
        "status": _model_status,
    }
```

- [ ] **Step 4: Запустить тест — убедиться что проходит**

```bash
python -m pytest tests/test_main.py::test_info_returns_model_info -v
```
Ожидаем: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add /info endpoint"
```

---

### Task 4: Model loading тесты

**Files:**
- Modify: `tests/test_main.py`

- [ ] **Step 1: Написать тест — используется существующий файл**

Добавить в конец `tests/test_main.py`:

```python
def test_find_model_uses_existing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "test/repo")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    model_file = tmp_path / "test-model.litertlm"
    model_file.write_text("fake")

    from app.main import _find_or_download_model
    result = _find_or_download_model()
    assert result == str(model_file)
```

- [ ] **Step 2: Запустить тест — убедиться что проходит (уже реализовано)**

```bash
python -m pytest tests/test_main.py::test_find_model_uses_existing_file -v
```
Ожидаем: `PASSED`.

- [ ] **Step 3: Написать тест — скачивание с HF когда файла нет**

Добавить в конец `tests/test_main.py`:

```python
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

- [ ] **Step 4: Запустить тест — убедиться что проходит**

```bash
python -m pytest tests/test_main.py::test_find_model_downloads_when_missing -v
```
Ожидаем: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_main.py
git commit -m "test: add model loading unit tests"
```

---

### Task 5: /generate non-streaming

**Files:**
- Modify: `app/main.py`
- Modify: `tests/test_main.py`

- [ ] **Step 1: Написать failing тесты**

Добавить в конец `tests/test_main.py`:

```python
def test_generate_returns_response(client):
    c, mock_conv = client
    response = c.post("/generate", json={"prompt": "What is the capital of France?"})
    assert response.status_code == 200
    assert response.json() == {"response": "Paris is the capital of France."}
    mock_conv.send_message.assert_called_once_with("What is the capital of France?")


def test_generate_requires_prompt(client):
    c, _ = client
    response = c.post("/generate", json={})
    assert response.status_code == 422
```

- [ ] **Step 2: Запустить тесты — убедиться что падают**

```bash
python -m pytest tests/test_main.py::test_generate_returns_response tests/test_main.py::test_generate_requires_prompt -v
```
Ожидаем: `FAILED` — 404 Not Found.

- [ ] **Step 3: Добавить GenerateRequest и /generate в app/main.py**

Добавить перед `@asynccontextmanager` (после импортов и глобалей):

```python
class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = False
    max_tokens: int = 512
```

Добавить в конец файла (после `/info`):

```python
def _stream_generator(prompt: str):
    with _engine.create_conversation() as conv:
        for chunk in conv.send_message_async(prompt):
            yield f'data: {json.dumps({"chunk": chunk})}\n\n'
    yield "data: [DONE]\n\n"


@app.post("/generate")
def generate(request: GenerateRequest):
    if _model_status != "ready":
        raise HTTPException(status_code=503, detail="Model not ready")

    if request.stream:
        return StreamingResponse(
            _stream_generator(request.prompt),
            media_type="text/event-stream",
        )

    with _engine.create_conversation() as conv:
        response = conv.send_message(request.prompt)
    return {"response": response}
```

- [ ] **Step 4: Запустить тесты — убедиться что проходят**

```bash
python -m pytest tests/test_main.py::test_generate_returns_response tests/test_main.py::test_generate_requires_prompt -v
```
Ожидаем: оба `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/test_main.py
git commit -m "feat: add /generate endpoint (non-streaming)"
```

---

### Task 6: /generate streaming

**Files:**
- Modify: `tests/test_main.py`

(`_stream_generator` уже добавлен в Task 5)

- [ ] **Step 1: Написать failing тест для SSE streaming**

Добавить в конец `tests/test_main.py`:

```python
def test_generate_streaming_returns_sse(client):
    c, mock_conv = client
    mock_conv.send_message_async.return_value = iter(["Paris", " is", " the capital."])

    response = c.post("/generate", json={"prompt": "Tell me about Paris", "stream": True})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = [line for line in response.text.splitlines() if line.startswith("data:")]
    assert lines[0] == 'data: {"chunk": "Paris"}'
    assert lines[1] == 'data: {"chunk": " is"}'
    assert lines[2] == 'data: {"chunk": " the capital."}'
    assert lines[3] == "data: [DONE]"
```

- [ ] **Step 2: Запустить тест — убедиться что проходит**

```bash
python -m pytest tests/test_main.py::test_generate_streaming_returns_sse -v
```
Ожидаем: `PASSED`.

- [ ] **Step 3: Запустить все тесты**

```bash
python -m pytest tests/ -v
```
Ожидаем: все `PASSED`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_main.py
git commit -m "test: add streaming SSE test for /generate"
```

---

### Task 7: Dockerfile

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Создать Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
```

- [ ] **Step 2: Собрать образ**

```bash
docker build -t litert-ai .
```
Ожидаем: `Successfully built` без ошибок.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "feat: add Dockerfile"
```

---

### Task 8: docker-compose.yml + .env.example

**Files:**
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Создать .env.example**

```
MODEL_NAME=litert-community/gemma-4-E2B-it-litert-lm
PORT=8000
HUGGING_FACE_HUB_TOKEN=hf_your_token_here
```

- [ ] **Step 2: Создать docker-compose.yml**

```yaml
services:
  litert-ai:
    build: .
    ports:
      - "${PORT:-8000}:${PORT:-8000}"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_NAME=${MODEL_NAME}
      - PORT=${PORT:-8000}
      - HUGGING_FACE_HUB_TOKEN=${HUGGING_FACE_HUB_TOKEN:-}
    restart: unless-stopped
```

- [ ] **Step 3: Проверить валидность compose файла**

```bash
docker compose config
```
Ожидаем: валидный YAML без ошибок.

- [ ] **Step 4: Проверить полный запуск (опционально, требует HF токен)**

```bash
# Скопировать .env.example в .env и заполнить токен
cp .env.example .env
# Затем:
docker compose up
```

- [ ] **Step 5: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: add docker-compose.yml and .env.example"
```

---

## Примеры вызова API

```bash
# Health check
curl http://localhost:8000/health

# Информация о модели
curl http://localhost:8000/info

# Генерация (sync)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "stream": false}'

# Генерация (streaming SSE)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"prompt": "Tell me a story", "stream": true}'
```

## Docker run (без compose)

```bash
docker run -d \
  -e MODEL_NAME=litert-community/gemma-4-E2B-it-litert-lm \
  -e PORT=8000 \
  -e HUGGING_FACE_HUB_TOKEN=hf_xxx \
  -v $(pwd)/models:/app/models \
  -p 8000:8000 \
  litert-ai
```
