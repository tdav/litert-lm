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
    assert "application/x-ndjson" in response.headers["content-type"]

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
    assert "application/x-ndjson" in response.headers["content-type"]

    lines = [line for line in response.text.splitlines() if line.strip()]
    chunks = [json.loads(line) for line in lines]
    assert chunks[0]["message"]["role"] == "assistant"
    assert chunks[0]["message"]["content"] == "Paris"
    assert chunks[0]["done"] is False
    assert chunks[-1]["done"] is True
    assert chunks[-1]["message"]["content"] == ""


def test_find_model_uses_existing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_NAME", "test/repo")
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    model_file = tmp_path / "test-model.litertlm"
    model_file.write_text("fake")

    from app.main import _find_or_download_model
    result = _find_or_download_model()
    assert result == str(model_file)

    out = capsys.readouterr().out
    assert f"[startup] Searching for model in {str(tmp_path)}" in out
    assert "[startup] Found model: test-model.litertlm" in out


def test_find_model_downloads_when_missing(tmp_path, monkeypatch, capsys):
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

    out = capsys.readouterr().out
    assert f"[startup] Searching for model in {str(tmp_path)}" in out
    assert "[startup] Downloading test/repo from HuggingFace..." in out


def test_generate_returns_503_when_not_ready():
    import app.main as m
    from unittest.mock import patch, MagicMock
    # Patch _find_or_download_model so lifespan's background task never
    # completes and flips _model_status away from "loading".
    with patch("app.main._find_or_download_model", side_effect=Exception("blocked")), \
         patch("app.main.litert_lm.Engine", return_value=MagicMock()):
        m._model_status = "loading"
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            # Force status to loading right before the request so even a
            # racing background task cannot flip it.
            m._model_status = "loading"
            response = c.post("/api/generate", json={"prompt": "hi"})
    assert response.status_code == 503


def test_show_returns_404_when_not_ready():
    import app.main as m
    from unittest.mock import patch, MagicMock
    with patch("app.main._find_or_download_model", side_effect=Exception("blocked")), \
         patch("app.main.litert_lm.Engine", return_value=MagicMock()):
        m._model_status = "loading"
        from fastapi.testclient import TestClient
        from app.main import app
        with TestClient(app, raise_server_exceptions=False) as c:
            m._model_status = "loading"
            response = c.post("/api/show", json={"model": "test"})
    assert response.status_code == 404
