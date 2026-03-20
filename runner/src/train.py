import sys
from pathlib import Path

# Ensure runner/src is on the Python path so Hydra can resolve targets
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hydra
from omegaconf import DictConfig
from lightning.pytorch import Trainer, seed_everything

from utils import get_pylogger

log = get_pylogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    # Instantiate backbone
    log.info(f"Instantiating net <{cfg.net._target_}>")
    net = hydra.utils.instantiate(cfg.net)

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

    # Instantiate loggers (config may define multiple under named keys)
    loggers = [hydra.utils.instantiate(lg) for lg in cfg.logger.values()]

    # Instantiate trainer
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=loggers,
    )

    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
