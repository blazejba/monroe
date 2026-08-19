import torch

from monroe.mtl.abstract_weighting import AbsWeighting


class STCH(AbsWeighting):
    r"""STCH.

    This method is proposed in `Smooth Tchebycheff Scalarization for Multi-Objective
    Optimization (ICML 2024) <https://openreview.net/forum?id=m4dO5L6eCp>`_
    and implemented by modifying from the
    `official PyTorch implementation <https://github.com/Xi-L/STCH/tree/main/STCH_MTL>`_.

    """

    def __init__(self, *args, **kwargs):
        super(STCH, self).__init__(*args, **kwargs)

    def init_param(self):
        self.step = 0
        self.shard = 0
        self.nadir_vector = None
        self._nadir_pending = False
        self._prev_shard = -1

        self.average_loss = 0.0
        self.average_loss_count = 0

    def serialize_weighting_state(self) -> dict:
        nadir_vector = self.nadir_vector.detach().cpu() if self.nadir_vector is not None else None
        if isinstance(self.average_loss, torch.Tensor):
            average_loss = self.average_loss.detach().cpu()
        else:
            average_loss = self.average_loss
        state = {
            "step": self.step,
            "nadir_vector": nadir_vector,
            "average_loss": average_loss,
            "average_loss_count": self.average_loss_count,
            "_nadir_pending": self._nadir_pending,
            "_prev_shard": self._prev_shard,
        }
        return state

    def deserialize_weighting_state(self, state: dict) -> None:
        assert all(key in state for key in ["step", "nadir_vector", "average_loss", "average_loss_count"])
        self.step = state["step"]
        self.average_loss_count = state["average_loss_count"]
        self.average_loss = state["average_loss"]
        self._nadir_pending = state.get("_nadir_pending", False)
        self._prev_shard = state.get("_prev_shard", -1)
        if state["nadir_vector"] is not None:
            self.nadir_vector = state["nadir_vector"].to(self.device)
        else:
            self.nadir_vector = None

    def _is_calibration_shard(self, warmup_shard, nadir_refresh):
        """Check if current shard is a nadir (re)calibration shard."""
        if self.shard == warmup_shard:
            return True
        if nadir_refresh > 0 and self.shard > warmup_shard and (self.shard - warmup_shard) % nadir_refresh == 0:
            return True
        return False

    def backward(self, losses, **kwargs):
        self.step += 1
        mu = kwargs["STCH_mu"]
        ramp = kwargs.get("STCH_ramp", False)
        use_log = kwargs.get("STCH_log", True)
        total_shards = kwargs.get("STCH_total_shards")
        warmup_shard = kwargs["STCH_warmup_shard"]
        nadir_refresh = kwargs.get("STCH_nadir_refresh", 0)
        stats_losses = kwargs.get("global_losses")
        if stats_losses is None:
            stats_losses = losses

        is_calibration = self._is_calibration_shard(warmup_shard, nadir_refresh)

        # On shard transition into a recalibration shard, reset the accumulator
        if self.shard != self._prev_shard:
            self._prev_shard = self.shard
            if is_calibration and self.shard > warmup_shard:
                self.average_loss = 0.0
                self.average_loss_count = 0
                self._nadir_pending = True

        if self.shard < warmup_shard:
            loss = torch.log(losses + 1e-10).sum() if use_log else losses.sum()
            self._accumulate_weights(losses.new_full((losses.numel(),), 1.0 / losses.numel()))
        elif is_calibration:
            loss = torch.log(losses + 1e-10).sum() if use_log else losses.sum()
            self.average_loss += stats_losses.detach()
            self.average_loss_count += 1
            self._accumulate_weights(losses.new_full((losses.numel(),), 1.0 / losses.numel()))
        else:
            if self.nadir_vector is None or self._nadir_pending:
                self.nadir_vector = self.average_loss / self.average_loss_count
                self._nadir_pending = False
            if ramp and total_shards is not None:
                mu_start = mu
                mu_end = kwargs.get("STCH_mu_end", 0.3)
                ramp_shards = max(1, total_shards - warmup_shard - 1)
                position = self.shard - warmup_shard - 1
                if ramp_shards == 1:
                    progress = 1.0
                else:
                    progress = min(max(position / (ramp_shards - 1), 0.0), 1.0)
                mu = mu_start - (mu_start - mu_end) * progress  # Decrease from mu_start to 0.1
            losses = torch.log(losses / self.nadir_vector + 1e-10) if use_log else (losses / self.nadir_vector)
            # Apply preference vector: shift log-losses by log(pref) before softmax
            pref_vector = kwargs.get("STCH_pref_vector")
            if pref_vector is not None:
                losses = losses + torch.log(pref_vector)
            max_term = torch.max(losses.data).detach()
            reg_losses = losses - max_term
            loss = mu * torch.logsumexp(reg_losses / mu, dim=0) * losses.numel()
            self._accumulate_weights(torch.softmax(reg_losses / mu, dim=0))
        extra_losses = kwargs.get("extra_losses")
        if extra_losses is not None:
            loss = loss + extra_losses.sum()
        loss.backward()
        return loss
