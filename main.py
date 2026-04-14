from __future__ import annotations

import ast
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from pydantic import BaseModel

from openpi_runtime import create_policy, notebook_prepare_inference_runtime, run_policy_inference


DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))


class InferenceResponse(BaseModel):
    success: bool
    first_action: list[float] | None = None
    trajectory: list[list[float]] | None = None
    inference_time_ms: float | None = None
    error: str | None = None


def _parse_state_input(state_input: str) -> list[float]:
    if not state_input or not state_input.strip():
        return [0.0, 0.0, 0.0, 0.0]

    try:
        parsed = ast.literal_eval(state_input)
        arr = np.asarray(parsed, dtype=np.float32)
        if arr.shape != (4,):
            return [0.0, 0.0, 0.0, 0.0]
        return arr.tolist()
    except Exception:
        return [0.0, 0.0, 0.0, 0.0]


def create_app(model_dir: Path = DEFAULT_MODEL_DIR) -> FastAPI:
    state: dict[str, object] = {
        "prepared": None,
        "policy": None,
        "error": None,
        "model_dir": model_dir,
    }

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            state["prepared"] = notebook_prepare_inference_runtime(model_dir)
            _, state["policy"] = create_policy(model_dir)
            state["error"] = None
        except Exception as exc:  # pragma: no cover
            state["error"] = str(exc)
        yield

    app = FastAPI(title="Pi0 Drone VLA Local API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {
            "ready": state["policy"] is not None,
            "model_dir": str(state["model_dir"]),
            "error": state["error"],
        }

    @app.post("/infer", response_model=InferenceResponse)
    async def infer(
        image_input: UploadFile = File(...),
        task_input: str = Form(""),
        state_input: str = Form(""),
    ) -> InferenceResponse:
        if state["policy"] is None:
            return InferenceResponse(success=False, error=f"Model not ready: {state['error']}")

        try:
            task = task_input.strip()
            if not task:
                return InferenceResponse(success=False, error="Task required")

            image_bytes = await image_input.read()
            if not image_bytes:
                return InferenceResponse(success=False, error="Image required")

            start = time.time()
            actions = run_policy_inference(
                policy=state["policy"],
                image_bytes=image_bytes,
                task=task,
                state=_parse_state_input(state_input),
            )
            return InferenceResponse(
                success=True,
                first_action=actions[0].round(4).tolist(),
                trajectory=actions.round(4).tolist(),
                inference_time_ms=round((time.time() - start) * 1000, 2),
            )
        except Exception as exc:
            return InferenceResponse(success=False, error=str(exc))

    return app


app = create_app()
