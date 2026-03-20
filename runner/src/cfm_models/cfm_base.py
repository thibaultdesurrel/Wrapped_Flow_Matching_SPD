"""Base class for all Conditional Flow Matching models on SPD matrices.

Provides shared infrastructure: LR scheduling and the common training loop skeleton.
Subclasses implement sample_xt(), ut_cond(), and step() for their specific
parameterization of the conditional probability path
p_t(·|X₁) = WG(X̄_t, μ_t, σ_t² I_D).

"""

import torch
import torch.nn.functional as F

import math
from lightning.pytorch import LightningModule
from src.utils.sampling_wrapped_gaussian import sample_wrapped_gaussian

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
}


class CFMBase(LightningModule):
    def __init__(self, cfg, net):
        super().__init__()
        self.cfg = cfg
        self.net = net
        self.dtype: torch.dtype = DTYPE_MAP[self.cfg.dtype]
        self.num_classes: int = self.cfg.num_classes
        self.dim: int = self.cfg.dim

        self.sigma_init: float = 1.0 / math.sqrt(self.dim * (self.dim + 1) / 2)
        self.criterion = torch.nn.MSELoss()

    def preprocess_batch(self, batch):

        # Get the x1 and labels from the batch
        x1 = batch.x1.to(self.dtype)
        labels = batch.labels

        # Convert the labels to one-hot encoding
        labels = F.one_hot(labels, num_classes=self.num_classes).to(self.dtype)

        batch_size = x1.shape[0]
        dim = x1.shape[1]
        assert dim == self.dim, f"Expected input dimension {self.dim}, but got {dim}"

        # Sample the random time steps for each sample in the batch
        t = torch.rand(batch_size, dtype=self.dtype, device=x1.device)

        # Sample x0
        dim_TS = self.dim * (self.dim + 1) // 2
        p0 = torch.eye(dim, dtype=self.dtype, device=x1.device)
        mu0 = torch.zeros(dim_TS, dtype=self.dtype, device=x1.device)
        sigma0 = (self.sigma_init**2) * torch.eye(
            dim_TS, dtype=self.dtype, device=x1.device
        )
        x0 = sample_wrapped_gaussian(batch_size, p0, mu0, sigma0)

        return x0, x1, t, labels

    def configure_optimizers(self):
        LR = self.cfg.lr
        FACTOR_LR = self.cfg.factor_lr
        WARMUP_EPOCHS = self.cfg.warmup_epochs
        EPOCHS = self.cfg.epochs

        optimizer = self.cfg.optimizer(self.parameters(), lr=LR, weight_decay=0)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=FACTOR_LR,
                    end_factor=1.0,
                    total_iters=WARMUP_EPOCHS,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=LR * FACTOR_LR
                ),
            ],
            milestones=[WARMUP_EPOCHS],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    def step(self, batch):
        raise NotImplementedError("The step method must be implemented by subclasses.")

    def training_step(self, batch):
        loss, _, _ = self.step(batch)
        self.log_step("train", loss, batch_size=batch.x1.shape[0])
        return loss

    def validation_step(self, batch):
        loss, _, _ = self.step(batch)
        self.log_step("val", loss, batch_size=batch.x1.shape[0], on_step=False)

    def test_step(self, batch):
        loss, _, _ = self.step(batch)
        self.log_step("test", loss, batch_size=batch.x1.shape[0], on_step=False)

    def log_step(self, prefix, loss, batch_size, on_step=True):
        """Log the training/val/test loss metric."""
        self.log(
            f"{prefix}/loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

    def forward(self, x, t, y):
        return self.net(x, t, y)
