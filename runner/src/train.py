import sys
from pathlib import Path

# Ensure runner/src is on the Python path so Hydra can resolve targets
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import torch
import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything

from utils import get_pylogger

log = get_pylogger(__name__)

# Opt into Tensor-Core (TF32) float32 matmuls on capable GPUs (e.g. A40).
# Only affects float32 ops; the float64 SPD computations are unchanged. Mainly
# silences Lightning's "set_float32_matmul_precision" notice.
torch.set_float32_matmul_precision("high")


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    # Instantiate backbone
    log.info(f"Instantiating net <{cfg.model.net._target_}>")
    net = hydra.utils.instantiate(cfg.model.net)

    # Resolve the optimizer partial so the model can call it directly
    optimizer_partial = hydra.utils.instantiate(cfg.model.optimizer)

    # Instantiate model — constructor takes (cfg, net, optimizer),
    # so we resolve the class and pass cfg.model as a plain config object
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model_cls = hydra.utils.get_class(cfg.model._target_)
    model = model_cls(cfg.model, net, optimizer_partial)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Instantiate callbacks (can be null in debug mode)
    callbacks = (
        [hydra.utils.instantiate(cb) for cb in cfg.callbacks.values()]
        if cfg.get("callbacks")
        else []
    )

    # Instantiate loggers (can be null in debug mode)
    loggers = (
        [hydra.utils.instantiate(lg) for lg in cfg.logger.values()]
        if cfg.get("logger")
        else []
    )

    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=loggers,
    )

    if cfg.get("train"):
        log.info("Starting training!")
        t0 = time.time()
        trainer.fit(model=model, datamodule=datamodule)
        elapsed = time.time() - t0
        log.info(f"Training finished in {elapsed:.1f}s")
        with open("training_time.log", "w") as f:
            f.write(f"{elapsed:.1f}s\n")

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")


if __name__ == "__main__":
    main()
