# VLA Drone Inference API

Backend inference API for a Vision-Language-Action (VLA) drone. Accepts a camera frame and a language instruction, returns an action vector `[vx, vy, vz, yaw]`.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
uv sync
```

## Running

```bash
uv run fastapi dev main.py
```

API is available at `http://localhost:8000`.
Interactive docs at `http://localhost:8000/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Run inference |

### POST `/predict`

Accepts `multipart/form-data`:

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | Camera frame (any format, resized to 224x224 internally) |
| `language_instruction` | string | Mission instruction, e.g. `"Selesaikan lintasan"` |

**Response:**

```json
{
  "action": {
    "vx": 1.3437,
    "vy": -0.0537,
    "vz": 0.1321,
    "yaw": -0.0514
  },
  "timestamp": "2026-04-09T14:56:26.351045+00:00"
}
```

**Example cURL:**

```bash
curl -X POST http://localhost:8000/predict \
  -F "image=@frame.jpg" \
  -F "language_instruction=Selesaikan lintasan"
```

## Testing

```bash
uv run python test_predict.py
```

## Tunnel (Cloudflare)

```bash
cloudflared tunnel run --url http://localhost:8000 dronevla-backend
```