from __future__ import annotations

import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel
from PIL import Image
from openpi_runtime import (
    DEFAULT_ASSET_ID,
    PreparedCheckpoint,
    create_policy,
    notebook_prepare_inference_runtime,
)


logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))


class InferenceResponse(BaseModel):
    actions: list[list[float]]
    first_action: list[float]
    action_horizon: int
    action_dim: int


class HealthResponse(BaseModel):
    ready: bool
    model_dir: str
    checkpoint_found: bool
    metadata_found: bool
    norm_stats_found: bool
    asset_norm_stats_found: bool
    asset_id: str
    error: str | None = None


class OpenPiService:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.policy: Any | None = None
        self.prepared_checkpoint: PreparedCheckpoint | None = None
        self.error: str | None = None

    @property
    def checkpoint_path(self) -> Path:
        if self.model_dir.is_file():
            return self.model_dir
        for candidate in (self.model_dir / "model.safetensors", self.model_dir / "checkpoint" / "model.safetensors"):
            if candidate.exists():
                return candidate
        return self.model_dir / "model.safetensors"

    @property
    def norm_stats_path(self) -> Path:
        if self.model_dir.is_file():
            return self.model_dir.parent / "norm_stats.json"
        return self.model_dir / "norm_stats.json"

    @property
    def metadata_path(self) -> Path:
        if self.model_dir.is_file():
            return self.model_dir.parent / "metadata.pt"
        return self.model_dir / "metadata.pt"

    @property
    def asset_norm_stats_path(self) -> Path:
        if self.prepared_checkpoint is not None:
            return self.prepared_checkpoint.asset_norm_stats_path
        base_dir = self.model_dir.parent if self.model_dir.is_file() else self.model_dir
        return base_dir / "assets" / DEFAULT_ASSET_ID / "norm_stats.json"

    def health(self) -> HealthResponse:
        return HealthResponse(
            ready=self.policy is not None,
            model_dir=str(self.model_dir),
            checkpoint_found=self.checkpoint_path.exists(),
            metadata_found=self.metadata_path.exists(),
            norm_stats_found=self.norm_stats_path.exists(),
            asset_norm_stats_found=self.asset_norm_stats_path.exists(),
            asset_id=DEFAULT_ASSET_ID,
            error=self.error,
        )

    def load(self) -> None:
        try:
            self.prepared_checkpoint = notebook_prepare_inference_runtime(self.model_dir)
            self.prepared_checkpoint, self.policy = create_policy(self.model_dir)
            self.error = None
            logger.info("OpenPI policy loaded from %s", self.prepared_checkpoint.checkpoint_dir)
        except Exception as exc:  # pragma: no cover - depends on checkpoint/runtime
            self.policy = None
            self.error = f"Failed to load policy: {exc}"
            logger.exception("Failed to load OpenPI policy")

    def infer(
        self,
        *,
        image_bytes: bytes,
        task: str,
        state: list[float],
    ) -> InferenceResponse:
        if self.policy is None:
            raise RuntimeError(self.error or "Model is not loaded.")

        image = self._decode_image(image_bytes)
        observation = {"image": image, "state": np.asarray(state, dtype=np.float32), "task": task}

        import torch

        with torch.no_grad():
            result = self.policy.infer(observation)

        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise RuntimeError(f"Unexpected action tensor shape: {actions.shape}")

        return InferenceResponse(
            actions=actions.tolist(),
            first_action=actions[0].tolist(),
            action_horizon=int(actions.shape[0]),
            action_dim=int(actions.shape[1]),
        )

    @staticmethod
    def _decode_image(image_bytes: bytes) -> np.ndarray:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224))
        except Exception as exc:
            raise ValueError(f"Invalid image upload: {exc}") from exc
        return np.asarray(image, dtype=np.uint8)


def create_app(
    service: OpenPiService | None = None,
    *,
    load_on_startup: bool = True,
) -> FastAPI:
    service = service or OpenPiService(DEFAULT_MODEL_DIR)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if load_on_startup and service.policy is None:
            service.load()
        yield

    app = FastAPI(
        title="Drone PI0.5 Inference API",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/", response_model=HealthResponse)
    def root() -> HealthResponse:
        return service.health()

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return service.health()

    @app.post("/infer", response_model=InferenceResponse)
    async def infer(
        image: UploadFile = File(..., description="Image file to run through the policy."),
        task: str = Form(""),
        prompt: str = Form(""),
        vx: float = Form(0.0),
        vy: float = Form(0.0),
        vz: float = Form(0.0),
        yaw: float = Form(0.0),
    ) -> InferenceResponse:
        health_state = service.health()
        if not health_state.ready:
            raise HTTPException(status_code=503, detail=health_state.model_dump())

        try:
            user_instruction = (prompt or task).strip()
            if not user_instruction:
                raise HTTPException(status_code=400, detail="Either `task` or `prompt` must be provided.")

            return service.infer(
                image_bytes=await image.read(),
                task=user_instruction,
                state=[vx, vy, vz, yaw],
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
