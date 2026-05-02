# app/main.py
import asyncio
import json
import os
import glob
import threading
import urllib.request
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
_MAX_CONCURRENT: int = int(os.environ.get("MAX_CONCURRENT_REQUESTS", "4"))
_engine_semaphore: asyncio.Semaphore | None = None


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

    print(f"[startup] Downloading {model_name} from Hugging Face...", flush=True)
    files = list(huggingface_hub.list_repo_files(repo_id=model_name, token=token))
    litertlm_files = [f for f in files if f.endswith(".litertlm")]
    if not litertlm_files:
        raise RuntimeError(f"No .litertlm file found in repo {model_name}")

    filename = litertlm_files[0]
    url = huggingface_hub.hf_hub_url(repo_id=model_name, filename=filename)
    dest = os.path.join(models_dir, os.path.basename(filename))
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length") or 0)
        downloaded = 0
        last_pct = -1
        with open(dest, "wb") as f:
            while True:
                chunk = r.read(8 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100 // total
                    if pct >= last_pct + 10:
                        last_pct = pct
                        print(
                            f"[startup] Downloading... {pct}%"
                            f" ({downloaded / 1048576:.0f}/{total / 1048576:.0f} MB)",
                            flush=True,
                        )
    print(f"[startup] Download complete: {os.path.basename(dest)}", flush=True)
    return dest


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model_name, _model_status, _engine_semaphore
    _model_name = os.environ.get("MODEL_NAME", "")
    _model_status = "loading"
    _engine_semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
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
    try:
        mtime = datetime.fromtimestamp(
            os.path.getmtime(_model_path), tz=timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        size = os.path.getsize(_model_path)
    except OSError:
        mtime = ""
        size = 0
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


async def _stream_with_semaphore(sync_gen_fn, *args):
    """Acquire engine semaphore for the full duration of a streaming response."""
    async with _engine_semaphore:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=8)
        stop_event = threading.Event()

        def worker():
            try:
                for chunk in sync_gen_fn(*args):
                    if stop_event.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result(timeout=10.0)
            except Exception as exc:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(exc), loop).result(timeout=1.0)
                except Exception:
                    pass
            finally:
                try:
                    asyncio.run_coroutine_threadsafe(queue.put(None), loop).result(timeout=1.0)
                except Exception:
                    pass

        loop.run_in_executor(None, worker)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        except GeneratorExit:
            stop_event.set()


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
async def generate(request: GenerateRequest):
    _check_ready()
    # options.num_predict is accepted for API compatibility but not forwarded;
    # litert_lm.Engine does not currently expose a max-tokens parameter
    if request.stream:
        return StreamingResponse(
            _stream_with_semaphore(_generate_stream, request.prompt),
            media_type="application/x-ndjson",
        )
    loop = asyncio.get_running_loop()
    async with _engine_semaphore:
        def _call():
            with _engine.create_conversation() as conv:
                return conv.send_message(request.prompt)
        response = await loop.run_in_executor(None, _call)
    return {
        "model": _model_name,
        "created_at": _now_iso(),
        "response": response,
        "done": True,
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    _check_ready()
    prompt = "\n".join(f"{m.role}: {m.content}" for m in request.messages)
    if request.stream:
        return StreamingResponse(
            _stream_with_semaphore(_chat_stream, prompt),
            media_type="application/x-ndjson",
        )
    loop = asyncio.get_running_loop()
    async with _engine_semaphore:
        def _call():
            with _engine.create_conversation() as conv:
                return conv.send_message(prompt)
        response = await loop.run_in_executor(None, _call)
    return {
        "model": _model_name,
        "created_at": _now_iso(),
        "message": {"role": "assistant", "content": response},
        "done": True,
    }
