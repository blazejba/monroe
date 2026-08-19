"""Exponential Moving Average (EMA) of model parameters."""

from copy import deepcopy

import torch
import torch.nn as nn


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    After each optimizer step, call `update(model)` to blend the latest
    weights into the shadow copy: ema_param = decay * ema_param + (1 - decay) * param.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = deepcopy(model)
        self.model.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for ema_p, model_p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(model_p.data, 1.0 - self.decay)
        for ema_b, model_b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(model_b)

    def state_dict(self) -> dict:
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        self.model.load_state_dict(state_dict)
