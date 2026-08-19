from monroe.mtl.abstract_weighting import AbsWeighting


class EW(AbsWeighting):
    r"""Equal Weighting (EW).

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """

    def backward(self, losses, **kwargs):
        self._accumulate_weights(losses.new_full((losses.numel(),), 1.0 / losses.numel()))
        loss = losses.sum()
        extra_losses = kwargs.get("extra_losses")
        if extra_losses is not None:
            loss = loss + extra_losses.sum()
        loss.backward()
        return loss
