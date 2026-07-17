"""Exponential moving average (EMA) of model weights, as a Lightning callback.

The shadow weights are updated after every optimizer step:

    s <- decay * s + (1 - decay) * p

and are swapped into the module for every validation/test run. They are left
in place through the end of validation, so that:

  * ``val/loss`` (monitored by ModelCheckpoint / EarlyStopping) scores the
    EMA model, making checkpoint selection much less noisy,
  * checkpoints saved by ModelCheckpoint (which runs after this callback on
    validation end) contain the EMA weights — the evaluation scripts that
    ``load_from_checkpoint`` therefore evaluate the EMA model with no change,
  * the raw (non-averaged) weights are swapped back in before the next
    training epoch, so training itself is unaffected.

Note on the horizon: the synthetic runs take ~3 optimizer steps per epoch
(8000 train samples, batch 2048, drop_last), so decay=0.999 averages over
~1000 steps ≈ 330 epochs.
"""

import torch
from lightning.pytorch import Callback


class EMA(Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = float(decay)
        self.shadow = None  # list of EMA tensors, aligned with pl_module.parameters()
        self._in_model = False  # True while the EMA weights are inside the module

    def on_fit_start(self, trainer, pl_module):
        if self.shadow is None:
            self.shadow = [p.detach().clone() for p in pl_module.parameters()]

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._in_model:  # safety: never average while swapped in
            return
        with torch.no_grad():
            for s, p in zip(self.shadow, pl_module.parameters()):
                s.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def _swap(self, pl_module):
        """Exchange the module parameters with the shadow tensors in place."""
        with torch.no_grad():
            for s, p in zip(self.shadow, pl_module.parameters()):
                tmp = p.detach().clone()
                p.copy_(s.to(p.device))
                s.copy_(tmp)

    def _swap_in(self, pl_module):
        if self.shadow is not None and not self._in_model:
            self._swap(pl_module)
            self._in_model = True

    def _swap_out(self, pl_module):
        if self.shadow is not None and self._in_model:
            self._swap(pl_module)
            self._in_model = False

    # Evaluate with the EMA weights. No swap-out on validation end: the weights
    # stay in place so ModelCheckpoint (which runs last) saves the EMA model.
    def on_validation_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_test_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_predict_start(self, trainer, pl_module):
        self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    def on_predict_end(self, trainer, pl_module):
        self._swap_out(pl_module)

    # Restore the raw weights before continuing to train.
    def on_train_epoch_start(self, trainer, pl_module):
        self._swap_out(pl_module)

    # After fit, leave the EMA weights in the module (used when no checkpoint
    # is available and the in-memory model is evaluated directly).
    def on_train_end(self, trainer, pl_module):
        self._swap_in(pl_module)

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow,
            "in_model": self._in_model,
        }

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]
        self._in_model = state_dict["in_model"]
