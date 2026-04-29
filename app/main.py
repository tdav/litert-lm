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
