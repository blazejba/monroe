import torch
import torch.distributed as dist

from monroe.mtl.abstract_weighting import AbsWeighting


class UW(AbsWeighting):
    r"""Uncertainty Weights (UW).

    This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics (CVPR 2018)
    <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_
    and implemented by us.

    """

    def __init__(self, *args, **kwargs):
        super(UW, self).__init__(*args, **kwargs)

    def init_param(self):
        # Use a plain tensor (not nn.Parameter) to avoid DDP "marked ready twice" errors.
        # DDP expects parameters to be used in forward(), but loss_scale is used in backward().
        # We manually sync gradients via all_reduce in backward().
        loss_scale = torch.tensor([-0.5] * self.task_num, device=self.device, requires_grad=True)
        self.loss_scale = loss_scale
    
    def parameters(self, recurse=True):
        # Include loss_scale in parameters so optimizer picks it up
        yield from super().parameters(recurse)
        if hasattr(self, 'loss_scale') and self.loss_scale.requires_grad:
            yield self.loss_scale

    def serialize_weighting_state(self) -> dict:
        return {"loss_scale": self.loss_scale.detach().cpu()}

    def deserialize_weighting_state(self, state: dict) -> None:
        loss_scale_tensor = state["loss_scale"].to(self.device)
        with torch.no_grad():
            self.loss_scale.copy_(loss_scale_tensor)

    def backward(self, losses, **kwargs):
        precision = 1.0 / (2 * self.loss_scale.exp())
        self._accumulate_weights((precision / precision.sum()).detach())
        loss = (losses * precision + self.loss_scale / 2).sum()
        extra_losses = kwargs.get("extra_losses")
        if extra_losses is not None:
            loss = loss + extra_losses.sum()
        loss.backward()
        
        # Manually sync loss_scale gradients across DDP replicas
        if dist.is_available() and dist.is_initialized():
            if self.loss_scale.grad is not None:
                dist.all_reduce(self.loss_scale.grad, op=dist.ReduceOp.AVG)
        
        return loss
