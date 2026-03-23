"""Load a trained CFM model from a run directory or checkpoint path."""

import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf, DictConfig

# Ensure runner/src/ is importable so Hydra can resolve targets like
# "backbones.mlp_simple.MLP" regardless of where the caller's cwd is.
_SRC_DIR = str(Path(__file__).resolve().parent.parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


def load_model(
    path: str | Path,
    device: str | torch.device | None = None,
) -> tuple[torch.nn.Module, DictConfig]:
    """Load a trained model from a run directory or a checkpoint file.

    Parameters
    ----------
    path : str or Path
        Either a training run directory (contains ``.hydra/`` and ``checkpoints/``)
        or a direct path to a ``.ckpt`` file.
    device : str, torch.device, or None
        Device to move the model to.  If *None*, uses CUDA when available.

    Returns
    -------
    model : torch.nn.Module
        The model in eval mode on the requested device.
    cfg : DictConfig
        The full Hydra config from the training run.
    """
    path = Path(path).resolve()

    if path.suffix == ".ckpt":
        ckpt_path = path
        run_dir = path.parent.parent  # checkpoints/ -> run_dir/
    else:
        run_dir = path
        ckpt_path = _find_best_checkpoint(run_dir / "checkpoints")

    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    net = hydra.utils.instantiate(cfg.model.net)
    optimizer_partial = hydra.utils.instantiate(cfg.model.optimizer)
    model_cls = hydra.utils.get_class(cfg.model._target_)

    model = model_cls.load_from_checkpoint(
        ckpt_path,
        cfg=cfg.model,
        net=net,
        optimizer=optimizer_partial,
    )
    model.eval()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, cfg


def _find_best_checkpoint(ckpt_dir: Path) -> Path:
    """Return the best checkpoint in *ckpt_dir*, falling back to ``last.ckpt``."""
    ckpts = sorted(p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt")
    if ckpts:
        return ckpts[-1]
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
