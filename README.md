## Drone PI0.5 Inference API

Backend ini menjalankan inference model OpenPI `pi0_drone_lite` untuk drone, dengan deployment utama ke Modal.
Model checkpoint diambil dari Hugging Face, disimpan ke Modal Volume, lalu di-load sekali saat container startup.

## Checkpoint layout

```text
metadata.pt
model.safetensors
norm_stats.json
assets/rafli/drone_roblox/norm_stats.json
```

The API reads those files from `./model` by default, or from `MODEL_DIR` if you set that env var.
If only the flat files exist, startup will automatically materialize the OpenPI asset layout under `assets/rafli/drone_roblox/`.

## Routes

```text
GET  /health
POST /infer
```

`POST /infer` uses `multipart/form-data`.

Required fields:

```text
image
task
```

Optional state fields, default `0.0`:

```text
vx
vy
vz
yaw
```

## Local run

Use this only if you want to run inference locally and already have the full OpenPI runtime installed on your machine.

```bash
uv run fastapi dev main.py
```

Local health check:

```bash
curl http://127.0.0.1:8000/health
```

Local inference test:

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -F "image=@frame.jpg" \
  -F "task=move forward"
```

With explicit state:

```bash
curl -X POST http://127.0.0.1:8000/infer \
  -F "image=@frame.jpg" \
  -F "task=move forward" \
  -F "vx=0.0" \
  -F "vy=0.0" \
  -F "vz=0.0" \
  -F "yaw=0.0"
```

## Modal deployment flow

### 1. Create the Volume

```bash
modal volume create drone-pi05-models
```

### 2. Download the model from Hugging Face into the Volume

This runs on Modal, not on your laptop. It downloads:

```text
metadata.pt
model.safetensors
norm_stats.json
```

from:

```text
izunx/drone-roblok-2500
```

Command:

```bash
modal run modal_app.py --download
```

### 3. Deploy the inference API

```bash
modal deploy modal_app.py
```

After deploy, Modal will print a public app URL.

## Test the deployed API

Replace `<base-url>` with the URL shown by `modal deploy`.

Health check:

```bash
curl https://<base-url>/health
```

Inference test:

```bash
curl -X POST https://<base-url>/infer \
  -F "image=@frame.jpg" \
  -F "task=move forward"
```

Inference with explicit state:

```bash
curl -X POST https://<base-url>/infer \
  -F "image=@frame.jpg" \
  -F "task=move forward" \
  -F "vx=0.0" \
  -F "vy=0.0" \
  -F "vz=0.0" \
  -F "yaw=0.0"
```

## Expected health output

You want to see values like:

```json
{
  "ready": true,
  "checkpoint_found": true,
  "metadata_found": true,
  "norm_stats_found": true,
  "asset_norm_stats_found": true,
  "asset_id": "rafli/drone_roblox"
}
```

If `ready` is `false`, check the `error` field in the response.

## How it works on Modal

- Modal downloads the checkpoint directly from Hugging Face into the Volume
- the download job uses a small image that only installs `huggingface_hub`
- the inference image uses the OpenPI `.venv` from `uv sync --frozen --no-dev` and applies the required PyTorch `transformers` patch
- the runtime now prepares the OpenPI checkpoint layout, installs the `transformers` patch, preloads the PaliGemma tokenizer, and uses a LoRA-compatible PyTorch checkpoint loader that matches the successful notebook flow
- model files live in a Modal `Volume`
- the volume is mounted at `/models`
- the app reads the checkpoint from `/models/drone-roblok-2500`
- the model is loaded once at container startup via `@modal.enter()`
- `/infer` returns the raw action output from `policy.infer(...)`

## Notes

- You do not need to install PyTorch locally if you only deploy to Modal
- Local inference only works if your local machine also has a compatible OpenPI runtime
- For normal usage, the simplest path is: `modal run ... --download`, then `modal deploy ...`, then call the Modal URL
