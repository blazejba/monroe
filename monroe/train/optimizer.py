import math
from typing import Tuple

from torch.nn import Module
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR, _LRScheduler


def optimizer_factory(
    model: Module,
    optim_params: dict,
    scheduler_params: dict,
) -> Tuple[Optimizer, _LRScheduler]:
    assert "lr" in optim_params, "Learning rate must be specified"
    assert "weight_decay" in optim_params, "Weight decay must be specified"
    assert "warmup_steps" in scheduler_params, "Warmup steps must be specified"
    assert "total_steps" in scheduler_params, "Total steps must be specified"

    optimizer = AdamW(
        model.parameters(),
        lr=optim_params["lr"],
        weight_decay=optim_params["weight_decay"],
        fused=True,
        eps=1e-10,
    )

    total_steps = scheduler_params["total_steps"]
    warmup_steps = scheduler_params["warmup_steps"]
    cosine_alpha = scheduler_params.get("cosine_alpha", 1.0)
    decay_steps = total_steps - warmup_steps

    def cosine_lambda(t):
        if decay_steps <= 0:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * (t / decay_steps) ** cosine_alpha))

    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1, total_iters=warmup_steps)
    main_scheduler = LambdaLR(optimizer, lr_lambda=cosine_lambda)
    scheduler = SequentialLR(optimizer, [warmup_scheduler, main_scheduler], milestones=[warmup_steps])
    return optimizer, scheduler
