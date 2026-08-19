"""Tests for multi-source pre-training components.

Covers:
    - SparseFocalLoss (new loss function)
    - SparseFocalAUROC (new metric)
    - DatasetHead (wide-output MLP)
    - FixedCountBatchSampler (cycling sampler for resident data)
    - concat_multi_source (combined batch + NaN-padded targets)
    - CSRMapDataset has_pm6_pos=False (forces q=0 to avoid NaN pos blending)
"""
import pytest
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, Data

from monroe.model.heads import DatasetHead, build_dataset_heads
from monroe.train.dataset import (
    FixedCountBatchSampler,
    add_multisource_tasks,
    concat_multi_source,
)
from monroe.train.loss import KabschAlignedCoordLoss, SparseFocalLoss
from monroe.train.metrics import SparseFocalAUROC


class TestSparseFocalLoss:
    def test_shape_and_grad(self):
        torch.manual_seed(0)
        pred = torch.randn(4, 10, requires_grad=True)
        gt = torch.randint(0, 2, (4, 10)).float()
        gt[torch.rand(4, 10) > 0.5] = float("nan")
        loss = SparseFocalLoss().compute_loss(pred, gt)
        assert loss.dim() == 0
        assert torch.isfinite(loss)
        loss.backward()
        assert torch.isfinite(pred.grad).all()

    def test_gamma_zero_recovers_bce(self):
        """With gamma=0 and alpha=None, focal loss equals standard BCE."""
        torch.manual_seed(1)
        pred = torch.randn(4, 10)
        gt = torch.randint(0, 2, (4, 10)).float()
        focal = SparseFocalLoss(gamma=0.0, alpha=None).compute_loss(pred, gt)
        bce = F.binary_cross_entropy_with_logits(pred, gt, reduction="mean")
        torch.testing.assert_close(focal, bce, rtol=1e-5, atol=1e-6)

    def test_all_nan_returns_zero(self):
        pred = torch.randn(4, 10, requires_grad=True)
        gt = torch.full((4, 10), float("nan"))
        loss = SparseFocalLoss().compute_loss(pred, gt)
        assert loss.item() == 0.0

    def test_bfloat16_input(self):
        """Focal loss must survive bfloat16 autocast."""
        pred = torch.randn(4, 10, dtype=torch.bfloat16, requires_grad=True)
        gt = torch.randint(0, 2, (4, 10)).float()
        loss = SparseFocalLoss().compute_loss(pred, gt)
        assert torch.isfinite(loss)


class TestSparseFocalAUROC:
    def test_random_predictions_near_half(self):
        """Mean AUROC of random logits vs random labels should be near 0.5."""
        torch.manual_seed(42)
        metric = SparseFocalAUROC(min_positives=5)
        N, C = 500, 4
        for _ in range(4):
            pred = torch.randn(N // 4, C)
            gt = torch.randint(0, 2, (N // 4, C)).float()
            metric.update_fun(pred, gt)
        score = metric.score_fun()[0]
        assert 0.35 < score < 0.65, f"Expected AUROC near 0.5, got {score}"

    def test_perfect_predictions_scores_one(self):
        """AUROC should be 1.0 when logits perfectly separate classes."""
        metric = SparseFocalAUROC(min_positives=5)
        torch.manual_seed(0)
        gt = torch.randint(0, 2, (200, 3)).float()
        # Logit = 2*gt - 1 makes positives higher than negatives
        pred = (2 * gt - 1) + 0.01 * torch.randn_like(gt)
        metric.update_fun(pred, gt)
        score = metric.score_fun()[0]
        assert score > 0.95

    def test_skips_assays_without_positives(self):
        """Assays with fewer than min_positives positives must be skipped."""
        metric = SparseFocalAUROC(min_positives=5)
        gt = torch.zeros(100, 2)  # all negatives -> both assays skipped
        pred = torch.randn(100, 2)
        metric.update_fun(pred, gt)
        score = metric.score_fun()[0]
        assert score == 0.0  # fallback when no scorable assays

    def test_reinit_clears_state(self):
        metric = SparseFocalAUROC()
        metric.update_fun(torch.randn(10, 2), torch.randint(0, 2, (10, 2)).float())
        metric.reinit()
        assert metric.pred_chunks == []
        assert metric.gt_chunks == []


class TestDatasetHead:
    def test_shape(self):
        h = DatasetHead(in_dim=720, output_dim=1328, hidden_dim=340, num_layers=1)
        x = torch.randn(8, 720)
        y = h(x)
        assert y.shape == (8, 1328)

    def test_zero_hidden_layers_is_linear(self):
        h = DatasetHead(in_dim=720, output_dim=1328, hidden_dim=340, num_layers=0)
        x = torch.randn(4, 720)
        y = h(x)
        assert y.shape == (4, 1328)
        # With num_layers=0, net is just a single nn.Linear
        assert len(h.net) == 1

    def test_build_dataset_heads_filters(self):
        """Only dataset-level tasks get heads; PM6 per-task entries are ignored."""
        td = {
            "pcba": {"n_outputs": 1328},
            "pm6_task": {"n_outputs": 1},  # ignored
        }
        heads = build_dataset_heads(td, in_dim=720, hidden_dim=340, num_layers=1, dropout=0.1)
        assert set(heads.keys()) == {"pcba"}

    def test_build_dataset_heads_respects_allow_list(self):
        """A dataset absent from dataset_task_names gets no head."""
        td = {"pcba": {"n_outputs": 1328}}
        heads = build_dataset_heads(
            td, in_dim=720, hidden_dim=340, num_layers=1, dropout=0.1,
            dataset_task_names=(),
        )
        assert set(heads.keys()) == set()

    def test_pcba_num_layers_override_yields_linear_probe(self):
        """pcba_num_layers=0 must yield a pure linear probe for PCBA, overriding
        the shared decoder depth."""
        import torch.nn as nn
        td = {"pcba": {"n_outputs": 1328}}
        heads = build_dataset_heads(
            td, in_dim=720, hidden_dim=340, num_layers=1, dropout=0.1,
            pcba_num_layers=0,
        )
        pcba_children = list(heads["pcba"].net.children())
        assert len(pcba_children) == 1
        assert isinstance(pcba_children[0], nn.Linear)
        assert pcba_children[0].in_features == 720
        assert pcba_children[0].out_features == 1328

    def test_pcba_num_layers_none_falls_back_to_shared(self):
        """Default (pcba_num_layers=None) must use the shared num_layers for PCBA."""
        td = {"pcba": {"n_outputs": 10}}
        heads = build_dataset_heads(
            td, in_dim=32, hidden_dim=16, num_layers=2, dropout=0.0,
        )
        # 2 hidden blocks (4 modules each) + 1 output Linear = 9 children
        assert len(list(heads["pcba"].net.children())) == 2 * 4 + 1


class TestFixedCountBatchSampler:
    def test_basic(self):
        s = FixedCountBatchSampler(dataset_size=100, batch_size=10, num_batches=5)
        batches = list(s)
        assert len(batches) == 5
        assert all(len(b) == 10 for b in batches)

    def test_ddp_split(self):
        """world_size=4 gives each rank num_batches batches of bs//world_size molecules."""
        s = FixedCountBatchSampler(dataset_size=100, batch_size=16, num_batches=10, world_size=4, rank=0)
        batches = list(s)
        assert len(batches) == 10
        assert all(len(b) == 4 for b in batches)

    def test_cycles_when_small(self):
        """When num_batches * bs > dataset_size, sampler must cycle without error."""
        s = FixedCountBatchSampler(dataset_size=10, batch_size=4, num_batches=20)
        batches = list(s)
        assert len(batches) == 20

    def test_reproducible(self):
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        s1 = FixedCountBatchSampler(dataset_size=50, batch_size=5, num_batches=3, generator=g1)
        s2 = FixedCountBatchSampler(dataset_size=50, batch_size=5, num_batches=3, generator=g2)
        assert list(s1) == list(s2)


class TestConcatMultiSource:
    @staticmethod
    def _make_batch(n_graphs: int, y_cols: int) -> tuple:
        """Build a tiny PyG Batch with unique per-graph targets."""
        data = []
        for i in range(n_graphs):
            d = Data(
                x=torch.zeros(2, 1),
                edge_index=torch.tensor([[0, 1], [1, 0]]),
                pos=torch.zeros(2, 3),
            )
            data.append(d)
        batch = Batch.from_data_list(data)
        y = torch.ones(n_graphs, y_cols) * 0.5
        return batch, y

    def test_nan_padding(self):
        """Each dataset's targets must be NaN outside its row range."""
        pm6_batch, pm6_y = self._make_batch(n_graphs=3, y_cols=5)
        pcba_batch, pcba_y = self._make_batch(n_graphs=2, y_cols=8)
        pcba_y = pcba_y * 2  # distinguishable values

        combined, y_dict = concat_multi_source(pm6_batch, pm6_y, {"pcba": (pcba_batch, pcba_y)})
        B_total = 3 + 2

        # Combined batch has 5 graphs
        assert combined.num_graphs == B_total

        # pm6 target: first 3 rows finite, last 2 NaN
        assert y_dict["pm6"].shape == (B_total, 5)
        assert torch.isfinite(y_dict["pm6"][:3]).all()
        assert torch.isnan(y_dict["pm6"][3:]).all()

        # pcba target: first 3 rows NaN, last 2 finite
        assert y_dict["pcba"].shape == (B_total, 8)
        assert torch.isnan(y_dict["pcba"][:3]).all()
        assert torch.isfinite(y_dict["pcba"][3:]).all()

    def test_offset_for_multiple_extras(self):
        """Three datasets: each gets its own non-overlapping valid row range.

        ``concat_multi_source`` is dataset-agnostic, so a second extra is used
        here purely to exercise the offset arithmetic. Only PCBA is wired up in
        the training entry point today.
        """
        pm6_b, pm6_y = self._make_batch(2, 3)
        pcba_b, pcba_y = self._make_batch(4, 5)
        other_b, other_y = self._make_batch(1, 7)
        extras = {"pcba": (pcba_b, pcba_y), "other": (other_b, other_y)}

        combined, y_dict = concat_multi_source(pm6_b, pm6_y, extras)
        assert combined.num_graphs == 7

        # Extras are processed in sorted key order: sorted(["pcba", "other"])
        # == ["other", "pcba"], so the row layout is
        # [pm6(0:2), other(2:3), pcba(3:7)].
        assert torch.isfinite(y_dict["other"][2:3]).all()
        assert torch.isnan(y_dict["other"][:2]).all()
        assert torch.isnan(y_dict["other"][3:]).all()

        assert torch.isfinite(y_dict["pcba"][3:7]).all()
        assert torch.isnan(y_dict["pcba"][:3]).all()


class TestMultisourceTaskDict:
    def test_adds_pcba_entry(self):
        td = {}
        td = add_multisource_tasks(td, pcba_n_assays=1328)
        assert "pcba" in td and td["pcba"]["n_outputs"] == 1328

    def test_none_skips(self):
        """Passing None for pcba_n_assays must add no entry."""
        td = {}
        td = add_multisource_tasks(td, pcba_n_assays=None)
        assert set(td.keys()) == set()


class TestKabschNaNHandling:
    """KabschAlignedCoordLoss must auto-derive a mask from NaN gt so PCBA rows
    in a combined multi-source batch don't break structure_pred loss."""

    def _make_batch(self):
        """Two graphs, 3 atoms each. Graph 1 has all-NaN ground-truth coords
        (simulating a PCBA molecule whose .POS is all NaN)."""
        torch.manual_seed(42)
        M = 6
        pos_pred = torch.randn(M, 3, requires_grad=True)
        pos_true = torch.randn(M, 3)
        pos_true[3:] = float("nan")  # graph 1 is all-NaN
        graph_id = torch.tensor([0, 0, 0, 1, 1, 1])
        return pos_pred, pos_true, graph_id

    def test_all_nan_graph_produces_finite_loss(self):
        pos_pred, pos_true, graph_id = self._make_batch()
        loss_fn = KabschAlignedCoordLoss(loss_type="huber")
        loss = loss_fn.compute_loss(pos_pred, pos_true, batch=graph_id)
        assert torch.isfinite(loss), f"loss should be finite, got {loss}"

    def test_matches_valid_graph_alone(self):
        """Loss on (valid + all-NaN) graphs must equal loss on just the valid
        graph — the NaN graph should contribute zero."""
        pos_pred, pos_true, graph_id = self._make_batch()
        loss_fn = KabschAlignedCoordLoss(loss_type="huber")

        # Mixed batch: graph 0 valid, graph 1 all-NaN
        loss_mixed = loss_fn.compute_loss(pos_pred, pos_true, batch=graph_id)
        # Reference: just graph 0
        loss_valid_only = loss_fn.compute_loss(
            pos_pred[:3], pos_true[:3], batch=graph_id[:3]
        )
        # Mixed loss divides by G=2 (mean over graphs, with graph 1 contributing
        # 0 since all its atoms are masked out with denominator clamped to 1.0).
        # Valid-only divides by G=1. So mixed should be ~half of valid-only.
        torch.testing.assert_close(
            loss_mixed, loss_valid_only / 2.0, rtol=1e-5, atol=1e-6,
        )

    def test_all_nan_everywhere_returns_zero(self):
        """If every node is NaN, loss should be 0.0 (not NaN)."""
        torch.manual_seed(0)
        pos_pred = torch.randn(6, 3, requires_grad=True)
        pos_true = torch.full((6, 3), float("nan"))
        graph_id = torch.tensor([0, 0, 0, 1, 1, 1])
        loss_fn = KabschAlignedCoordLoss(loss_type="huber")
        loss = loss_fn.compute_loss(pos_pred, pos_true, batch=graph_id)
        assert loss.item() == 0.0

    def test_gradients_flow_to_valid_rows_only(self):
        """Gradient w.r.t. pos_pred must be non-zero on valid rows and zero on
        NaN rows (those rows contribute zero weight to the loss)."""
        pos_pred, pos_true, graph_id = self._make_batch()
        loss_fn = KabschAlignedCoordLoss(loss_type="huber")
        loss = loss_fn.compute_loss(pos_pred, pos_true, batch=graph_id)
        loss.backward()
        grad = pos_pred.grad
        assert torch.isfinite(grad).all()
        # Graph 1 (all-NaN) rows should have zero gradient
        assert torch.allclose(grad[3:], torch.zeros(3, 3))
        # Graph 0 rows should have non-zero gradient somewhere
        assert grad[:3].abs().sum() > 0


class TestTrackerAURoCIntegration:
    """Verifies Tracker.score() surfaces SparseFocalAUROC's end-of-shard AUROC
    instead of returning 0 (bug pre-fix: Tracker read metric.record which is
    never populated by SparseFocalAUROC, only pred_chunks are)."""

    def test_tracker_score_calls_score_fun_for_accumulating_metric(self):
        import torch

        from monroe.train.tracker import Tracker

        torch.manual_seed(0)
        # One perfectly-predictable binary task: logits correlate strongly with gt.
        # With min_positives=10 default we need enough positives in the batch.
        n_samples, n_assays = 200, 3
        gt = torch.randint(0, 2, (n_samples, n_assays)).float()
        # Perfect predictions: use gt * 10 - 5 so logits are large positive for
        # positives, large negative for negatives — AUROC should be ~1.0.
        pred = gt * 10 - 5

        task_dict = {
            "pcba": dict(
                metric_name="auroc",
                n_outputs=n_assays,
                loss_fn=SparseFocalLoss(),
                metrics_fn=SparseFocalAUROC(min_positives=1),
                higher_is_better=True,
                task_type="graph_level",
                loss_type="focal",
            )
        }
        device = torch.device("cpu")
        tracker = Tracker(task_dict, device)
        # Feed the batch through tracker.update in one chunk
        tracker.update({"pcba": pred}, {"pcba": gt}, batch=None)
        # Now score — before fix this returned 0.0 because record is empty;
        # after fix, score_fun() is called and returns the near-perfect AUROC.
        _, _, metrics = tracker.score()
        assert metrics["pcba"] > 0.95, f"expected perfect AUROC ~1.0, got {metrics['pcba']}"

    def test_tracker_score_returns_zero_for_accumulating_metric_with_no_positives(self):
        """Edge case: if no assay has enough positives, score_fun returns 0.0
        (not NaN) and Tracker.score surfaces that verbatim."""
        import torch

        from monroe.train.tracker import Tracker

        task_dict = {
            "pcba": dict(
                metric_name="auroc",
                n_outputs=3,
                loss_fn=SparseFocalLoss(),
                metrics_fn=SparseFocalAUROC(min_positives=100),
                higher_is_better=True,
                task_type="graph_level",
                loss_type="focal",
            )
        }
        tracker = Tracker(task_dict, torch.device("cpu"))
        # Only a few samples, nowhere near min_positives=100
        tracker.update(
            {"pcba": torch.randn(5, 3)},
            {"pcba": torch.zeros(5, 3)},
            batch=None,
        )
        _, _, metrics = tracker.score()
        assert metrics["pcba"] == 0.0


class TestPCBAPerAssaySTCH:
    """add_multisource_tasks(pcba_per_assay=True) + build_dataset_heads + forward."""

    def test_per_assay_emits_one_task_per_assay(self):
        td = {}
        td = add_multisource_tasks(
            td,
            pcba_n_assays=3,
            pcba_per_assay=True,
            pcba_assay_names=["assayID-1", "assayID-7", "assayID-42"],
        )
        assert set(td.keys()) == {"pcba/assayID-1", "pcba/assayID-7", "pcba/assayID-42"}
        assert td["pcba/assayID-7"]["n_outputs"] == 1
        assert td["pcba/assayID-7"]["pcba_assay_idx"] == 1
        assert td["pcba/assayID-42"]["pcba_assay_idx"] == 2

    def test_per_assay_without_names_raises(self):
        td = {}
        with pytest.raises(ValueError, match="pcba_assay_names"):
            add_multisource_tasks(td, pcba_n_assays=3, pcba_per_assay=True)

    def test_aggregate_default_unchanged(self):
        td = {}
        td = add_multisource_tasks(td, pcba_n_assays=1328)
        assert "pcba" in td and td["pcba"]["n_outputs"] == 1328
        assert all(info.get("pcba_assay_idx") is None for info in td.values())

    def test_build_dataset_heads_shares_single_linear_probe_in_per_assay_mode(self):
        import torch.nn as nn
        td = {
            f"pcba/a{i}": {"n_outputs": 1, "pcba_assay_idx": i}
            for i in range(5)
        }
        heads = build_dataset_heads(td, in_dim=32, hidden_dim=16, num_layers=1, dropout=0.0)
        # ONE pcba head (shared linear probe across the 5 assay tasks)
        assert set(heads.keys()) == {"pcba"}
        # Its output width matches the number of per-assay tasks
        output_linear = [m for m in heads["pcba"].net if isinstance(m, nn.Linear)][-1]
        assert output_linear.out_features == 5


class TestPCBAAssayFilter:
    """Covers ``monroe.preprocessing.pcba_prep.filter_assays`` thresholds."""

    @staticmethod
    def _stat(name, n_pos, n_neg, n_nan=0):
        n_valid = n_pos + n_neg
        return {
            "assay_id": name,
            "n_valid": n_valid,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_nan": n_nan,
            "pos_frac": (n_pos / n_valid) if n_valid else 0.0,
        }

    def test_keeps_balanced_assays(self):
        from monroe.preprocessing.pcba_prep import filter_assays
        stats = [self._stat("a1", 100, 500), self._stat("a2", 200, 400)]
        kept, dropped = filter_assays(stats, min_pos=30, min_neg=30, min_valid=500)
        assert [s["assay_id"] for s in kept] == ["a1", "a2"]
        assert dropped == []

    def test_drops_all_one_class(self):
        from monroe.preprocessing.pcba_prep import filter_assays
        # all-positive assay (no negatives): drop by min_neg
        stats = [self._stat("pos_only", 1000, 0)]
        kept, dropped = filter_assays(stats, min_pos=30, min_neg=30, min_valid=500)
        assert kept == []
        assert "n_neg=0<30" in dropped[0]["drop_reason"]
        # all-negative assay
        stats = [self._stat("neg_only", 0, 1000)]
        kept, dropped = filter_assays(stats, min_pos=30, min_neg=30, min_valid=500)
        assert kept == []
        assert "n_pos=0<30" in dropped[0]["drop_reason"]

    def test_drops_too_sparse(self):
        from monroe.preprocessing.pcba_prep import filter_assays
        stats = [self._stat("sparse", 50, 50)]  # n_valid = 100 < 500
        kept, dropped = filter_assays(stats, min_pos=30, min_neg=30, min_valid=500)
        assert kept == []
        assert "n_valid=100<500" in dropped[0]["drop_reason"]

    def test_drops_too_skewed(self):
        from monroe.preprocessing.pcba_prep import filter_assays
        # plenty of negatives but only 10 positives — below min_pos
        stats = [self._stat("skewed", 10, 10_000)]
        kept, dropped = filter_assays(stats, min_pos=30, min_neg=30, min_valid=500)
        assert kept == []
        assert "n_pos=10<30" in dropped[0]["drop_reason"]

    def test_reports_multiple_reasons(self):
        from monroe.preprocessing.pcba_prep import filter_assays
        # fails all three
        stats = [self._stat("bad", 1, 1)]
        kept, dropped = filter_assays(stats, min_pos=30, min_neg=30, min_valid=500)
        assert kept == []
        reason = dropped[0]["drop_reason"]
        assert "n_pos=" in reason and "n_neg=" in reason and "n_valid=" in reason


class TestPCBAAssayStatsWrite:
    """Round-trip of write_pcba_stats."""

    def test_roundtrip(self, tmp_path):
        import json

        from monroe.preprocessing.pcba_prep import write_pcba_stats
        kept = [
            {"assay_id": "a1", "n_valid": 600, "n_pos": 60, "n_neg": 540,
             "n_nan": 100, "pos_frac": 0.1},
        ]
        dropped = [
            {"assay_id": "a0", "n_valid": 10, "n_pos": 1, "n_neg": 9,
             "n_nan": 990, "pos_frac": 0.1, "drop_reason": "n_pos=1<30;n_valid=10<500"},
        ]
        path = write_pcba_stats(str(tmp_path), kept, dropped)
        with open(path) as fh:
            data = json.load(fh)
        # Sorted by assay_id
        assert [d["assay_id"] for d in data] == ["a0", "a1"]
        # Kept flag set and drop_reason=None for kept entries
        kept_entry = next(d for d in data if d["assay_id"] == "a1")
        assert kept_entry["kept"] is True and kept_entry["drop_reason"] is None
        dropped_entry = next(d for d in data if d["assay_id"] == "a0")
        assert dropped_entry["kept"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
