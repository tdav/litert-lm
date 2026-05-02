# Design: Startup Model Progress Logs

**Date:** 2026-05-02  
**Status:** Approved

## Problem

When the application starts, model loading (and potential download from HuggingFace) happens silently in the background. There is no console output indicating what is happening or how far the download has progressed.

## Goal

Add `print`-based progress logs to `app/main.py` so that the operator can see in the console:
- When the app is searching for a local model file
- Whether the model was found locally or needs to be downloaded
- Download progress (percentage, speed, ETA) via tqdm
- When the model is being loaded into the engine
- When the model is ready (or failed)

## Approach

Use `print` statements (not the `logging` module) at key points in `_find_or_download_model()` and `_load_model_task()`. The `hf_hub_download` function from `huggingface_hub` already uses `tqdm` internally — no extra configuration needed for the download progress bar.

For Docker environments, `PYTHONUNBUFFERED=1` should be set (already standard in Python Docker images) to prevent output buffering.

## Changes

Only `app/main.py` is modified. No new dependencies.

### `_find_or_download_model()`

| Point | Message |
|-------|---------|
| Start | `[startup] Searching for model in <MODELS_DIR>...` |
| Found locally | `[startup] Found model: <filename>` |
| Not found, downloading | `[startup] Downloading <model_name> from HuggingFace...` |

tqdm progress bar appears automatically after the download message.

### `_load_model_task()`

| Point | Message |
|-------|---------|
| Before engine init | `[startup] Loading model into engine...` |
| After success | `[startup] Model ready` |
| On exception | `[startup] ERROR: <exception message>` |

## Expected Console Output

```
[startup] Searching for model in /app/models...
[startup] Downloading litert-community/gemma-4-E4B-it-litert-lm from HuggingFace...
gemma-4-E4B-it.litertlm:  34%|████      | 1.2G/3.5G [00:42<01:18, 29.1MB/s]
[startup] Loading model into engine...
[startup] Model ready
```

Or if model already exists locally:

```
[startup] Searching for model in /app/models...
[startup] Found model: gemma-4-E4B-it.litertlm
[startup] Loading model into engine...
[startup] Model ready
```

## Out of Scope

- Switching to the `logging` module
- Blocking server startup until model is ready
- Progress reporting via HTTP endpoint
