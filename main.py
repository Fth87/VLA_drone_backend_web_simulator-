import io
import random
from datetime import datetime, timezone
from typing import Annotated

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

app = FastAPI(title="VLA Drone Inference API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])



class Action(BaseModel):
    vx: float
    vy: float
    vz: float
    yaw: float


class InferenceResponse(BaseModel):
    action: Action
    timestamp: str


async def preprocess_image(file: UploadFile) -> np.ndarray:
    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize((224, 224))
        return np.array(img, dtype=np.float32) / 255.0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def run_inference(image: np.ndarray, instruction: str) -> Action:
    rng = random.Random(hash(instruction) % 2**32)
    return Action(
        vx=round(rng.uniform(0.0, 1.5), 4),
        vy=round(rng.uniform(-0.3, 0.3), 4),
        vz=round(rng.uniform(-0.2, 0.2), 4),
        yaw=round(rng.uniform(-0.1, 0.1), 4),
    )


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": False}


@app.post("/predict", response_model=InferenceResponse)
async def predict(
    image: Annotated[UploadFile, File()],
    language_instruction: Annotated[str, Form()],
):
    img_arr = await preprocess_image(image)
    action = run_inference(img_arr, language_instruction)
    return InferenceResponse(
        action=action,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
