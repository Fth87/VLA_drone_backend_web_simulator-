from __future__ import annotations

import dataclasses
import importlib
import logging
import os
import shutil
import sys
import urllib.request
from pathlib import Path


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
OPENPI_ROOT_CANDIDATES = (
    Path("/root/openpi"),
    REPO_ROOT / "model" / "openpi",
)
DEFAULT_ASSET_ID = "rafli/drone_roblox"
PALIGEMMA_TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"


def _resolve_openpi_root() -> Path:
    for candidate in OPENPI_ROOT_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / "model" / "openpi").resolve()


OPENPI_ROOT = _resolve_openpi_root()
OPENPI_SRC = OPENPI_ROOT / "src"
OPENPI_CLIENT_SRC = OPENPI_ROOT / "packages" / "openpi-client" / "src"
OPENPI_DOWNLOAD_PY = OPENPI_SRC / "openpi" / "shared" / "download.py"
OPENPI_MODEL_PY = OPENPI_SRC / "openpi" / "models" / "model.py"
TRANSFORMERS_PATCH_ROOT = OPENPI_SRC / "openpi" / "models_pytorch" / "transformers_replace"


@dataclasses.dataclass(frozen=True)
class PreparedCheckpoint:
    checkpoint_dir: Path
    weight_path: Path
    metadata_path: Path
    root_norm_stats_path: Path
    asset_id: str = DEFAULT_ASSET_ID

    @property
    def asset_dir(self) -> Path:
        return self.checkpoint_dir / "assets" / self.asset_id

    @property
    def asset_norm_stats_path(self) -> Path:
        return self.asset_dir / "norm_stats.json"

    @property
    def asset_metadata_path(self) -> Path:
        return self.asset_dir / "metadata.pt"


def notebook_prepare_inference_runtime(model_dir: Path) -> PreparedCheckpoint:
    """Run the same inference preparation stages used in `model/rapli inference.ipynb`."""
    configure_environment()
    extend_python_path()
    prepared = prepare_checkpoint_assets(model_dir)
    patch_openpi_download_datetime()
    patch_transformers_install()
    ensure_paligemma_tokenizer()
    return prepared


def configure_environment() -> None:
    os.environ["PYTHONPATH"] = f"{OPENPI_SRC}:{OPENPI_CLIENT_SRC}"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ.setdefault("OPENPI_DATA_HOME", str(Path("~/.cache/openpi").expanduser()))


def extend_python_path() -> None:
    for path in (OPENPI_SRC, OPENPI_CLIENT_SRC):
        path_str = str(path)
        if path.exists() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def prepare_checkpoint_assets(model_dir: Path, *, asset_id: str = DEFAULT_ASSET_ID) -> PreparedCheckpoint:
    checkpoint_dir = model_dir if model_dir.is_dir() else model_dir.parent
    prepared = PreparedCheckpoint(
        checkpoint_dir=checkpoint_dir.resolve(),
        weight_path=(checkpoint_dir / "model.safetensors").resolve(),
        metadata_path=(checkpoint_dir / "metadata.pt").resolve(),
        root_norm_stats_path=(checkpoint_dir / "norm_stats.json").resolve(),
        asset_id=asset_id,
    )

    if not prepared.weight_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found at `{prepared.weight_path}`.")

    prepared.asset_dir.mkdir(parents=True, exist_ok=True)
    _replace_with_link_or_copy(prepared.root_norm_stats_path, prepared.asset_norm_stats_path)
    _replace_with_link_or_copy(prepared.metadata_path, prepared.asset_metadata_path)
    return prepared


def _replace_with_link_or_copy(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if dst.exists() or dst.is_symlink():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        logger.info("Linked asset %s -> %s", src, dst)
    except OSError:
        shutil.copy2(src, dst)
        logger.info("Copied asset %s -> %s", src, dst)


def patch_openpi_download_datetime() -> None:
    if not OPENPI_DOWNLOAD_PY.exists():
        return
    content = OPENPI_DOWNLOAD_PY.read_text()
    updated = content.replace("datetime.UTC", "datetime.timezone.utc")
    if updated != content:
        OPENPI_DOWNLOAD_PY.write_text(updated)
        logger.info("Patched OpenPI download.py datetime.UTC compatibility")


def patch_transformers_install() -> None:
    if not TRANSFORMERS_PATCH_ROOT.exists():
        raise FileNotFoundError(f"OpenPI transformers patch directory not found at `{TRANSFORMERS_PATCH_ROOT}`.")

    transformers = importlib.import_module("transformers")
    transformers_root = Path(transformers.__file__).resolve().parent
    for item in TRANSFORMERS_PATCH_ROOT.iterdir():
        target = transformers_root / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def ensure_paligemma_tokenizer() -> Path:
    target_path = Path("/root/.cache/openpi/big_vision/paligemma_tokenizer.model")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        return target_path
    urllib.request.urlretrieve(PALIGEMMA_TOKENIZER_URL, target_path)
    return target_path


def create_policy(model_dir: Path, *, config_name: str = "pi0_drone_lite", pytorch_device: str | None = None):
    prepared = prepare_checkpoint_assets(model_dir)

    import torch
    import torch._dynamo
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    torch._dynamo.config.suppress_errors = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_config = _config.get_config(config_name)
    train_config = dataclasses.replace(
        train_config,
        model=dataclasses.replace(train_config.model, pytorch_compile_mode=None),
    )

    device = pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu")
    policy = _policy_config.create_trained_policy(
        train_config,
        prepared.checkpoint_dir,
        pytorch_device=device,
    )
    return prepared, policy
