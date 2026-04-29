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


class GenerateRequest(BaseModel):
    prompt: str
    stream: bool = False
    max_tokens: int = 512


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


@app.get("/info")
def info():
    return {
        "model_name": _model_name,
        "model_file": _model_file,
        "status": _model_status,
    }


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
