import os

import torch
import torch.distributed as dist

import wandb


def init_wandb(
    hp,
    run_id: str | None,
    enc_params: int,
    dec_params: int,
    n_decs: int,
    is_main_process: bool,
) -> None:
    """Initialize Weights & Biases logging.

    Args:
        hp: Hyperparameters namespace.
        run_id: Existing run ID for resuming, or None for new run.
        enc_params: Number of encoder parameters.
        dec_params: Number of per-decoder parameters.
        n_decs: Number of decoders.
        is_main_process: Whether this is the main process.
    """
    from monroe.config import to_dict

    wandb.init(
        id=run_id if is_main_process else None,
        save_code=True,
        project=os.environ.get("WANDB_PROJECT", "monroe"),
        resume="must" if hp.load_ckpt and is_main_process else None,
        config=to_dict(hp),
        entity=os.environ.get("WANDB_ENTITY") or None,
        mode="online" if (hp.wandb and is_main_process) else "disabled",
        settings=wandb.Settings(code_dir="."),
    )
    wandb.summary.update(
        {
            "n_decoders_pretrain": n_decs,
            "encoder_params": enc_params,
            "decoder_params_pretrain": dec_params,
        }
    )


class Tracker:
    def __init__(self, task_dict, device):
        self.task_dict = task_dict
        self.metric_names = {task: task_dict[task]["metric_name"] for task in task_dict}
        self.metrics = {task: task_dict[task]["metrics_fn"] for task in task_dict}
        self.losses = {task: task_dict[task]["loss_fn"] for task in task_dict}
        self.device = device

    @staticmethod
    def _to_tensor(values: list, device: torch.device) -> torch.Tensor:
        if len(values) == 0:
            return torch.zeros(1, device=device, dtype=torch.float32)
        if isinstance(values[0], torch.Tensor):
            return torch.stack(values).to(device=device, dtype=torch.float32)
        return torch.tensor(values, device=device, dtype=torch.float32)

    @staticmethod
    def _weighted_sum_and_count(record: list, bs: list, device: torch.device):
        """For losses: record contains per-batch means, bs contains batch sizes."""
        if len(record) == 0:
            zero = torch.zeros(1, device=device, dtype=torch.float32)
            return zero, zero
        record_t = Tracker._to_tensor(record, device)
        bs_t = Tracker._to_tensor(bs, device) if bs else torch.ones_like(record_t)
        finite = torch.isfinite(record_t)
        record_t = torch.where(finite, record_t, record_t.new_tensor(0.0))
        bs_t = torch.where(finite, bs_t, bs_t.new_tensor(0.0))
        return (record_t * bs_t).sum(), bs_t.sum()

    @staticmethod
    def _raw_sum_and_count(record: list, bs: list, device: torch.device):
        """For metrics: record contains per-batch sums, bs contains per-batch counts."""
        if len(record) == 0:
            zero = torch.zeros(1, device=device, dtype=torch.float32)
            return zero, zero
        record_t = Tracker._to_tensor(record, device)
        bs_t = Tracker._to_tensor(bs, device) if bs else torch.ones_like(record_t)
        return record_t.sum(), bs_t.sum()

    @staticmethod
    def _distributed_reduce(sum_tensor: torch.Tensor, count_tensor: torch.Tensor):
        if dist.is_available() and dist.is_initialized():
            stacked = torch.stack([sum_tensor, count_tensor])
            dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
            return stacked[0], stacked[1]
        return sum_tensor, count_tensor

    @staticmethod
    def _safe_mean(total: torch.Tensor, count: torch.Tensor) -> float:
        if count.item() == 0:
            return 0.0
        return float((total / count).item())

    def update(self, preds, y, batch, **kwargs):
        losses = torch.zeros(len(self.task_dict)).to(self.device)
        for tn, task in enumerate(self.task_dict):
            y_task = y[task]
            mask = None
            if torch.is_floating_point(y_task):
                mask = ~torch.isnan(y_task)

            losses[tn] = self.losses[task]._update_loss(preds[task], y_task, batch=batch, mask=mask, **kwargs)
            with torch.no_grad():
                self.metrics[task].update_fun(preds[task], y_task, batch=batch, mask=mask)
        return losses

    def compute_losses(self, preds, y, batch, **kwargs):
        """Compute losses without updating internal tracking state (for FAMO update_w)."""
        losses = torch.zeros(len(self.task_dict)).to(self.device)
        for tn, task in enumerate(self.task_dict):
            y_task = y[task]
            mask = None
            if torch.is_floating_point(y_task):
                mask = ~torch.isnan(y_task)
            losses[tn] = self.losses[task].compute_loss(preds[task], y_task, batch=batch, mask=mask, **kwargs)
        return losses

    def score(self):
        mean_losses = {}
        mean_metrics = {}

        for task in self.task_dict:
            loss_total, loss_count = self._weighted_sum_and_count(
                self.losses[task].record, self.losses[task].bs, self.device
            )
            loss_total, loss_count = self._distributed_reduce(loss_total, loss_count)
            mean_losses[task] = self._safe_mean(loss_total, loss_count)

            metric = self.metrics[task]
            if metric.record:
                # Incremental metric: per-batch sums accumulate in `record`,
                # per-batch counts in `bs`. Global mean = sum(record) / sum(bs).
                metric_total, metric_count = self._raw_sum_and_count(
                    metric.record, metric.bs, self.device
                )
                metric_total, metric_count = self._distributed_reduce(metric_total, metric_count)
                metric_value = self._safe_mean(metric_total, metric_count)
            else:
                # Accumulating metric (e.g. SparseFocalAUROC): its update_fun
                # buffers raw pred/gt chunks and the metric is only defined at
                # score_fun() time. Compute per-rank then all-reduce-mean across
                # ranks — ranks see disjoint molecules so per-rank AUROC is a
                # reasonable approximation; averaging gives a global summary.
                metric_value = float(metric.score_fun()[0])
                if dist.is_available() and dist.is_initialized():
                    t = torch.tensor([metric_value], device=self.device, dtype=torch.float32)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    metric_value = float((t / dist.get_world_size()).item())
            mean_metrics[task] = metric_value

        mean_total_loss = sum(mean_losses.values()) / len(mean_losses)
        return mean_total_loss, mean_losses, mean_metrics

    def log(
        self,
        split: str,
        shard: int,
        update_dict: dict | None = None,
        log_to_wandb: bool = True,
        mean_losses: dict | None = None,
        mean_metrics: dict | None = None,
    ):
        update_dict = update_dict or {}
        losses_for_log = mean_losses or {task: loss._average_loss() for task, loss in self.losses.items()}
        metrics_for_log = mean_metrics or {task: fn.score_fun()[0] for task, fn in self.metrics.items()}
        update_dict.update({f"{split}_loss/{task}": value for task, value in losses_for_log.items()})
        update_dict.update(
            {f"{split}_{self.metric_names[task]}/{task}": value for task, value in metrics_for_log.items()}
        )
        if log_to_wandb:
            wandb.log(update_dict, step=shard)
        self.reset()

    def log_shard_results(
        self,
        shard_idx: int,
        train_loss: float,
        per_task_losses: dict,
        train_metrics: dict,
        lr: float,
        is_main_process: bool,
        val_results=None,
        ema_val_results=None,
        pcba_val_results=None,
        ema_pcba_val_results=None,
        task_weights=None,
    ):
        """Log training (and optionally validation + EMA validation) results for a shard."""
        from monroe.utils import printf

        train_update = {"train_loss": train_loss, "lr": lr}
        if task_weights:
            train_update.update({f"task_weight/{task}": w for task, w in task_weights.items()})

        self.log(
            split="train",
            shard=shard_idx,
            update_dict=train_update,
            log_to_wandb=is_main_process,
            mean_losses=per_task_losses,
            mean_metrics=train_metrics,
        )

        if val_results:
            val_loss, val_losses, val_metrics = val_results
            self.log(
                split="val",
                shard=shard_idx + 1,
                update_dict={"val_loss": val_loss, "lr": lr},
                log_to_wandb=is_main_process,
                mean_losses=val_losses,
                mean_metrics=val_metrics,
            )
            if pcba_val_results:
                pcba_loss, pcba_losses, pcba_metrics = pcba_val_results
                self.log(
                    split="val_pcba",
                    shard=shard_idx + 1,
                    update_dict={"val_pcba_loss": pcba_loss},
                    log_to_wandb=is_main_process,
                    mean_losses=pcba_losses,
                    mean_metrics=pcba_metrics,
                )
            if ema_val_results:
                ema_loss, ema_losses, ema_metrics = ema_val_results
                self.log(
                    split="val_ema",
                    shard=shard_idx + 1,
                    update_dict={"val_ema_loss": ema_loss},
                    log_to_wandb=is_main_process,
                    mean_losses=ema_losses,
                    mean_metrics=ema_metrics,
                )
                if ema_pcba_val_results:
                    ema_pcba_loss, ema_pcba_losses, ema_pcba_metrics = ema_pcba_val_results
                    self.log(
                        split="val_pcba_ema",
                        shard=shard_idx + 1,
                        update_dict={"val_pcba_ema_loss": ema_pcba_loss},
                        log_to_wandb=is_main_process,
                        mean_losses=ema_pcba_losses,
                        mean_metrics=ema_pcba_metrics,
                    )
            if is_main_process:
                msg = f"Shard {shard_idx + 1}, train loss: {train_loss:.4f}, val loss: {val_loss:.4f}, "
                if ema_val_results:
                    msg += f"val_ema loss: {ema_loss:.4f}, "
                if pcba_val_results:
                    msg += f"val_pcba loss: {pcba_loss:.4f}, "
                msg += f"lr: {lr:.6f}"
                printf(msg)
        else:
            if is_main_process:
                printf(f"Shard {shard_idx + 1}, train loss: {train_loss:.4f}, lr: {lr:.6f}")

    def reset(self):
        for task in self.task_dict:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
