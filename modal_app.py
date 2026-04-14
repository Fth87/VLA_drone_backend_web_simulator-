from __future__ import annotations

import os
import time
from pathlib import Path

import modal


APP_NAME = "drone-pi05-inference"
VOLUME_NAME = "drone-pi05-models"
MOUNT_PATH = "/models"
MODEL_DIR = f"{MOUNT_PATH}/drone-roblok-2500"
HF_REPO_ID = "izunx/drone-roblok-2500"
OPENPI_LOCAL_DIR = "model/openpi"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub[hf_transfer]>=0.30.0")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "MODEL_DIR": MODEL_DIR,
        }
    )
)

inference_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("uv>=0.8.0")
    .add_local_dir(OPENPI_LOCAL_DIR, remote_path="/root/openpi", copy=True)
    .add_local_file("openpi_runtime.py", "/root/openpi_runtime.py", copy=True)
    .run_commands(
        "cd /root/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev",
        "uv pip install --python /root/openpi/.venv/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision torchaudio",
        "uv pip install --python /root/openpi/.venv/bin/python peft safetensors transformers==4.53.2 'fastapi[standard]>=0.135.3' numpy==1.26.4 'pillow>=11.0.0' pytest",
    )
    .env(
        {
            "MODEL_DIR": MODEL_DIR,
            "PATH": "/root/openpi/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONPATH": "/root:/root/openpi/src:/root/openpi/packages/openpi-client/src",
            "HF_HUB_OFFLINE": "1",
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
)


@app.function(
    image=download_image,
    timeout=60 * 60,
    volumes={MOUNT_PATH: volume},
)
def download_model() -> dict[str, object]:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        allow_patterns=["metadata.pt", "model.safetensors", "norm_stats.json"],
    )
    volume.commit()
    return {
        "repo_id": HF_REPO_ID,
        "model_dir": MODEL_DIR,
        "files": ["metadata.pt", "model.safetensors", "norm_stats.json"],
    }


@app.local_entrypoint()
def main(download: bool = False) -> None:
    if download:
        print(download_model.remote())
    else:
        print("Run `modal run modal_app.py --download` to sync the model into the Modal Volume.")


def _parse_state_input(state_input: str) -> list[float]:
    import numpy as np

    if state_input and state_input.strip():
        try:
            state = np.array(state_input, dtype=np.float32)
            if state.shape != (4,):
                state = np.zeros(4, dtype=np.float32)
        except Exception:
            state = np.zeros(4, dtype=np.float32)
    else:
        state = np.zeros(4, dtype=np.float32)
    return state.tolist()


@app.cls(
    image=inference_image,
    gpu="A10G",
    scaledown_window=150,
    timeout=400,
    volumes={MOUNT_PATH: volume},
)
class InferenceAPI:
    @modal.enter()
    def load_model(self) -> None:
        from openpi_runtime import create_policy, notebook_prepare_inference_runtime

        checkpoint_dir = Path(os.environ.get("MODEL_DIR", MODEL_DIR))
        self.prepared = notebook_prepare_inference_runtime(checkpoint_dir)
        _, self.policy = create_policy(checkpoint_dir)

    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Form, UploadFile
        from openpi_runtime import run_policy_inference
        from pydantic import BaseModel

        class InferenceResponse(BaseModel):
            success: bool
            first_action: list[float] | None = None
            trajectory: list[list[float]] | None = None
            inference_time_ms: float | None = None
            error: str | None = None

        api = FastAPI(title="Pi0 Drone VLA Modal API", version="1.0.0")

        @api.get("/health")
        def health() -> dict[str, object]:
            return {
                "ready": self.policy is not None,
                "model_dir": str(self.prepared.checkpoint_dir),
                "asset_id": self.prepared.asset_id,
            }

        @api.post("/infer", response_model=InferenceResponse)
        async def infer(
            image_input: UploadFile | None = File(None),
            task_input: str = Form(""),
            state_input: str = Form(""),
        ) -> InferenceResponse:
            try:
                if image_input is None:
                    return InferenceResponse(success=False, error="Image required")

                task = task_input.strip()
                if not task:
                    return InferenceResponse(success=False, error="Task required")

                image_bytes = await image_input.read()
                if not image_bytes:
                    return InferenceResponse(success=False, error="Image required")

                start = time.time()
                actions = run_policy_inference(
                    policy=self.policy,
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

        return api
