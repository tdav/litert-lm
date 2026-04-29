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


def test_info_returns_model_info(client):
    c, _ = client
    response = c.get("/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "test/model"
    assert data["model_file"] == "test.litertlm"
    assert data["status"] == "ready"


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
