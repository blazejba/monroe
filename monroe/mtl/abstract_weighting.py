import torch
import torch.nn as nn


class AbsWeighting(nn.Module):
    r"""An abstract class for weighting strategies."""

    def __init__(self, task_types, encoder_class, decoders, device, dataset_heads=None,
                 pcba_assay_slices=None, **kwargs):
        super().__init__()

        self.task_types = task_types
        self.task_name = list(task_types.keys())
        self.task_num = len(self.task_name)
        self.encoder_class = encoder_class
        self.decoders = decoders
        # Optional ModuleDict of per-dataset heads (currently PCBA only).
        # Predictions from these heads are merged into the same dict as PM6 decoder outputs.
        self.dataset_heads = dataset_heads if dataset_heads is not None else nn.ModuleDict()
        # Optional mapping {task_name: column_idx} for per-assay PCBA STCH mode.
        # When non-empty, forward() slices dataset_heads["pcba"]'s [B, n_assays]
        # output into per-task [B, 1] predictions so STCH weights each assay
        # independently. The linear probe weights are still shared (a single
        # Linear(hidden_dim, n_assays) module) — only the STCH view is per-task.
        self.pcba_assay_slices = pcba_assay_slices or {}
        self.device = device
        self.kwargs = kwargs
        self._weight_sum = None
        self._weight_count = 0

    def forward(self, inputs, node_idxs=None, task_name=None):
        r"""
        Args:
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): Ignored parameter kept for API compatibility.

        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        graph_level, node_level = self.encoder(inputs, node_idxs)
        self._last_graph_emb = graph_level
        preds = self.decoders(graph_level, node_level)
        # Call dataset-level heads sequentially; each produces a wide output tensor.
        for name, head in self.dataset_heads.items():
            out = head(graph_level)
            if name == "pcba" and self.pcba_assay_slices:
                # Per-assay STCH mode: carve the [B, n_assays] head output into
                # [B, 1] per-task views. These are tensor slices (no copy).
                for task_name, idx in self.pcba_assay_slices.items():
                    preds[task_name] = out[:, idx:idx + 1]
            else:
                preds[name] = out
        return preds

    def update_train_loss_buffer(self, shard: int, losses: dict[str, float]):
        r"""Update the training loss buffer."""
        self.train_loss_buffer[:, shard] = torch.tensor(
            [losses[task] for task in self.task_name],
            dtype=torch.float32,
        ).to(self.device)

    def get_share_params(self):
        r"""Return the shared parameters of the model."""
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero."""
        self.encoder.zero_grad(set_to_none=False)

    def init_param(self):
        r"""Define and initialize some trainable parameters required by specific weighting methods."""
        pass

    def _accumulate_weights(self, weights):
        """Accumulate per-task weights for shard-level averaging."""
        w = weights.detach()
        if self._weight_sum is None:
            self._weight_sum = w.clone()
        else:
            self._weight_sum.add_(w)
        self._weight_count += 1

    def get_task_weights(self):
        """Return shard-averaged per-task weights as a dict."""
        if self._weight_sum is None or self._weight_count == 0:
            return {}
        avg = self._weight_sum / self._weight_count
        return dict(zip(self.task_name, avg.cpu().tolist()))

    def reset_task_weights(self):
        """Reset weight accumulators (call after logging)."""
        self._weight_sum = None
        self._weight_count = 0

    def serialize_weighting_state(self) -> dict:
        r"""Return a serializable snapshot of the weighting-specific state."""
        return {}

    def deserialize_weighting_state(self, state: dict) -> None:
        r"""Restore the weighting-specific state from ``state``."""
        if not state:
            return

    def backward(self, losses, **kwargs):
        r"""
        Args:
            losses (list): A list of losses of each task.
            kwargs (dict): A dictionary of hyperparameters of weighting methods.
        """
        pass
