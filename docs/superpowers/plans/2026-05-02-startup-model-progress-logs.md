# Startup Model Progress Logs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `print`-based console logs to `app/main.py` so operators can see model search, download progress, and engine load status at startup.

**Architecture:** Add 6 `print(…, flush=True)` calls at key points in `_find_or_download_model()` and `_load_model_task()`. The `hf_hub_download` function already uses tqdm internally for the download progress bar — no additional configuration needed.

**Tech Stack:** Python `print()` with `flush=True`, existing `huggingface_hub` tqdm integration, `pytest` + `capsys` for output capture.

---

### Task 1: Write failing tests for logs in `_find_or_download_model`

**Files:**
- Modify: `tests/test_main.py`

- [ ] **Step 1: Add capsys to the two existing unit tests for `_find_or_download_model`**

Replace the two existing tests at the bottom of `tests/test_main.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_main.py::test_find_model_uses_existing_file tests/test_main.py::test_find_model_downloads_when_missing -v
```

Expected: both FAIL — no output captured because print statements don't exist yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_main.py
git commit -m "test: add capsys assertions for startup log messages in _find_or_download_model"
```

---

### Task 2: Implement logs in `_find_or_download_model`

**Files:**
- Modify: `app/main.py:24-44`

- [ ] **Step 1: Replace `_find_or_download_model` with the logged version**

Replace the entire function body in `app/main.py`:

```python
def _find_or_download_model() -> str:
    models_dir = os.environ.get("MODELS_DIR", "/app/models")
    model_name = os.environ["MODEL_NAME"]
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    os.makedirs(models_dir, exist_ok=True)
    print(f"[startup] Searching for model in {models_dir}...", flush=True)
    existing = glob.glob(f"{models_dir}/*.litertlm")
    if existing:
        print(f"[startup] Found model: {os.path.basename(existing[0])}", flush=True)
        return existing[0]

    print(f"[startup] Downloading {model_name} from HuggingFace...", flush=True)
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
pytest tests/test_main.py::test_find_model_uses_existing_file tests/test_main.py::test_find_model_downloads_when_missing -v
```

Expected: both PASS.

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: add startup log messages to _find_or_download_model"
```

---

### Task 3: Write failing tests for logs in `_load_model_task`

**Files:**
- Modify: `tests/test_main.py`

- [ ] **Step 1: Add two tests for `_load_model_task` at the end of `tests/test_main.py`**

```python
def test_load_model_task_logs_ready(monkeypatch, capsys):
    import asyncio
    import app.main as m

    monkeypatch.setenv("MODEL_NAME", "test/repo")
    monkeypatch.setenv("LITERT_BACKEND", "cpu")

    mock_eng = MagicMock()
    mock_eng.__enter__ = MagicMock(return_value=mock_eng)
    mock_eng.__exit__ = MagicMock(return_value=False)

    with patch("app.main._find_or_download_model", return_value="/tmp/test.litertlm"), \
         patch("app.main.litert_lm.Engine", return_value=mock_eng):
        asyncio.run(m._load_model_task())

    out = capsys.readouterr().out
    assert "[startup] Loading model into engine..." in out
    assert "[startup] Model ready" in out


def test_load_model_task_logs_error(monkeypatch, capsys):
    import asyncio
    import app.main as m

    monkeypatch.setenv("MODEL_NAME", "test/repo")

    with patch("app.main._find_or_download_model", side_effect=RuntimeError("disk full")):
        asyncio.run(m._load_model_task())

    out = capsys.readouterr().out
    assert "[startup] ERROR: disk full" in out
    assert m._model_status == "error"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_main.py::test_load_model_task_logs_ready tests/test_main.py::test_load_model_task_logs_error -v
```

Expected: both FAIL — no output captured yet.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_main.py
git commit -m "test: add capsys assertions for startup log messages in _load_model_task"
```

---

### Task 4: Implement logs in `_load_model_task`

**Files:**
- Modify: `app/main.py:47-61`

- [ ] **Step 1: Replace `_load_model_task` with the logged version**

Replace the entire function body in `app/main.py`:

```python
async def _load_model_task():
    global _engine, _engine_cm, _model_file, _model_path, _model_status, _load_error
    try:
        loop = asyncio.get_event_loop()
        path = await loop.run_in_executor(None, _find_or_download_model)
        _model_path = path
        _model_file = os.path.basename(path)
        _backend_env = os.environ.get("LITERT_BACKEND", "cpu").lower()
        _backend = litert_lm.Backend.GPU if _backend_env == "gpu" else litert_lm.Backend.CPU
        print("[startup] Loading model into engine...", flush=True)
        _engine_cm = litert_lm.Engine(path, backend=_backend)
        _engine = _engine_cm.__enter__()
        _model_status = "ready"
        print("[startup] Model ready", flush=True)
    except Exception as exc:
        _load_error = str(exc)
        _model_status = "error"
        print(f"[startup] ERROR: {exc}", flush=True)
```

- [ ] **Step 2: Run the new tests to verify they pass**

```bash
pytest tests/test_main.py::test_load_model_task_logs_ready tests/test_main.py::test_load_model_task_logs_error -v
```

Expected: both PASS.

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 4: Commit**

```bash
git add app/main.py
git commit -m "feat: add startup log messages to _load_model_task"
```
