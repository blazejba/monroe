import torch
import torch.nn.functional as F

from monroe.mtl.abstract_weighting import AbsWeighting


class DWA(AbsWeighting):
    r"""Dynamic Weight Average (DWA).

    This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019)
    <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_
    and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.

    Args:
        T (float, default=2.0): The softmax temperature.

    """

    def backward(self, losses, **kwargs):
        T = kwargs["DWA_T"]
        train_loss_buffer = self.train_loss_buffer
        task_num = losses.numel()
        if self.shard > 1:
            w_i = torch.Tensor(
                train_loss_buffer[:, self.shard - 1] / train_loss_buffer[:, self.shard - 2]
            ).to(self.device)
            batch_weight = task_num * F.softmax(w_i / T, dim=-1)
        else:
            batch_weight = torch.ones_like(losses).to(self.device)
        self._accumulate_weights(batch_weight / batch_weight.sum())
        loss = torch.mul(losses, batch_weight).sum()
        extra_losses = kwargs.get("extra_losses")
        if extra_losses is not None:
            loss = loss + extra_losses.sum()
        loss.backward()
        return loss
