## Drone PI0.5 Inference (Modal Only)

Inference backend disederhanakan total ke Modal, dengan core inference OpenPI tetap mengikuti alur notebook (`pi0_drone_lite`, patch runtime, patch LoRA loader non-strict, patch transformers).

Perbedaan utama hanya I/O:

- Input via `multipart/form-data` (`image_input`, `task_input`, `state_input`)
- Output via JSON (`success`, `first_action`, `trajectory`, `inference_time_ms`, `error`)

## Endpoints

```text
GET  /health
POST /infer
```

## Request/Response

### POST /infer (`multipart/form-data`)

Fields:

- `image_input`: image file (required)
- `task_input`: instruction text (required)
- `state_input`: string list optional, contoh `"[0.0, 0.0, 0.0, 0.0]"`

Notes:

- Jika `task_input` kosong -> error `Task required`.
- Jika image tidak ada -> error `Image required`.
- Jika `state_input` invalid atau shape bukan `(4,)` -> fallback ke `[0, 0, 0, 0]`.

### Response

```json
{
  "success": true,
  "first_action": [0.1123, -0.0312, 0.0054, 0.1901],
  "trajectory": [[0.1123, -0.0312, 0.0054, 0.1901]],
  "inference_time_ms": 241.6,
  "error": null
}
```

## Deploy

1. Buat volume model:

```bash
modal volume create drone-pi05-models
```

2. Download checkpoint ke volume:

```bash
modal run modal_app.py --download
```

3. Deploy API:

```bash
modal deploy modal_app.py
```

## Quick Test

```bash
curl https://<base-url>/health
```

```bash
curl -X POST https://<base-url>/infer \
  -F "image_input=@frame.jpg" \
  -F "task_input=look at the red cube" \
  -F "state_input=[0.0, 0.0, 0.0, 0.0]"
```

## Core Runtime Notes

- Model dimuat satu kali per container melalui `@modal.enter()`.
- Checkpoint source: `izunx/drone-roblok-2500`.
- Asset layout OpenPI (`assets/rafli/drone_roblox`) dibuat otomatis saat startup.
- Runtime patch yang dipakai:
  - `datetime.UTC` compatibility patch
  - `transformers_replace` patch dari OpenPI
  - LoRA non-strict loader patch di `openpi/models/model.py`
