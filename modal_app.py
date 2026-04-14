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
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.30.0",
    )
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
    .env(
        {
            "PYTHONPATH": "/root:/root/openpi/src:/root/openpi/packages/openpi-client/src",
        }
    )
    .add_local_dir(OPENPI_LOCAL_DIR, remote_path="/root/openpi", copy=True)
    .add_local_file("main.py", "/root/main.py", copy=True)
    .add_local_file("openpi_runtime.py", "/root/openpi_runtime.py", copy=True)
    .run_commands(
        "cd /root/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-dev",
        "uv pip install --python /root/openpi/.venv/bin/python --index-url https://download.pytorch.org/whl/cu121 torch==2.4.0 torchvision torchaudio",
        "uv pip install --python /root/openpi/.venv/bin/python pytest peft safetensors transformers==4.53.2 'fastapi[standard]>=0.135.3' numpy==1.26.4 'pillow>=11.0.0'",
        "/root/openpi/.venv/bin/python -c \"from openpi.training import config as c; print('OpenPI configs available:', len(getattr(c, '_CONFIGS_DICT', {}))); print('pi0_drone_lite available:', 'pi0_drone_lite' in getattr(c, '_CONFIGS_DICT', {}))\"",
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
def download_model():
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
def main(download: bool = False):
    if download:
        print(download_model.remote())
    else:
        print("Run `modal run modal_app.py --download` to sync the model into the Modal Volume.")


@app.cls(
    image=inference_image,
    gpu="A10G",
    scaledown_window=150,
    timeout=400,
    volumes={MOUNT_PATH: volume},
)
class InferenceAPI:
    @modal.enter()
    def load_model(self):
        from main import OpenPiService
        from pathlib import Path

        self.service = OpenPiService(model_dir=Path(MODEL_DIR))
        self.service.load()

    @modal.asgi_app()
    def fastapi_app(self):
        from main import create_app

        return create_app(service=self.service, load_on_startup=False)
