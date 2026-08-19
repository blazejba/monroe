import torch
import torch.nn.functional as F

from monroe.mtl.abstract_weighting import AbsWeighting


class RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW).

    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus
    Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_
    and implemented by us.

    """

    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(losses.numel()), dim=-1).to(self.device)
        self._accumulate_weights(batch_weight)
        loss = torch.mul(losses, batch_weight).sum()
        extra_losses = kwargs.get("extra_losses")
        if extra_losses is not None:
            loss = loss + extra_losses.sum()
        loss.backward()
        return loss
