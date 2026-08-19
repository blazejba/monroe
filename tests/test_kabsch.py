"""Tests for batched Kabsch alignment (monroe/train/kabsch.py)."""

import pytest
import torch

from monroe.train.kabsch import kabsch_align


def _random_rotation(seed=0):
    """Return a random 3x3 rotation matrix via QR decomposition."""
    rng = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=rng)
    Q, R = torch.linalg.qr(A)
    # Ensure det > 0 (proper rotation, not reflection)
    Q = Q * torch.sign(torch.diag(R))
    if torch.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    return Q


class TestKabschIdentity:
    """When pred == true the alignment should be a no-op."""

    def test_single_graph(self):
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        graph_id = torch.zeros(3, dtype=torch.long)

        aligned, R, t = kabsch_align(pos, pos, graph_id)

        torch.testing.assert_close(aligned, pos, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(R.squeeze(0), torch.eye(3), atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(t.squeeze(0), torch.zeros(3), atol=1e-5, rtol=1e-5)

    def test_two_atoms(self):
        """Minimal case: two atoms."""
        pos = torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        graph_id = torch.zeros(2, dtype=torch.long)

        aligned, R, t = kabsch_align(pos, pos, graph_id)

        torch.testing.assert_close(aligned, pos, atol=1e-5, rtol=1e-5)


class TestKabschRotation:
    """Alignment should recover original positions after a known rotation."""

    def test_recover_from_rotation(self):
        torch.manual_seed(42)
        pos_true = torch.randn(10, 3)
        graph_id = torch.zeros(10, dtype=torch.long)

        R_applied = _random_rotation(seed=7)
        pos_pred = pos_true @ R_applied.T  # rotate

        aligned, R, t = kabsch_align(pos_pred, pos_true, graph_id)

        torch.testing.assert_close(aligned, pos_true, atol=1e-4, rtol=1e-4)

    def test_recover_from_rotation_and_translation(self):
        torch.manual_seed(42)
        pos_true = torch.randn(8, 3)
        graph_id = torch.zeros(8, dtype=torch.long)

        R_applied = _random_rotation(seed=13)
        shift = torch.tensor([5.0, -3.0, 7.0])
        pos_pred = pos_true @ R_applied.T + shift

        aligned, R, t = kabsch_align(pos_pred, pos_true, graph_id)

        torch.testing.assert_close(aligned, pos_true, atol=1e-4, rtol=1e-4)


class TestKabschReflection:
    """det(R) should always be +1 (proper rotation, no reflection)."""

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_det_positive(self, seed):
        torch.manual_seed(seed)
        pos_true = torch.randn(6, 3)
        # Reflect across XY plane to force potential det < 0
        pos_pred = pos_true.clone()
        pos_pred[:, 2] *= -1
        graph_id = torch.zeros(6, dtype=torch.long)

        _, R, _ = kabsch_align(pos_pred, pos_true, graph_id)

        det = torch.linalg.det(R.squeeze(0))
        assert det > 0, f"det(R) = {det.item()}, expected > 0"


class TestKabschBatched:
    """Multiple graphs in a single call."""

    def test_two_graphs_independent(self):
        # Graph 0: 4 atoms, graph 1: 3 atoms
        pos_true = torch.randn(7, 3)
        graph_id = torch.tensor([0, 0, 0, 0, 1, 1, 1])

        R0 = _random_rotation(seed=10)
        R1 = _random_rotation(seed=20)
        shift0 = torch.tensor([1.0, 2.0, 3.0])
        shift1 = torch.tensor([-1.0, 0.0, 5.0])

        pos_pred = pos_true.clone()
        pos_pred[:4] = pos_true[:4] @ R0.T + shift0
        pos_pred[4:] = pos_true[4:] @ R1.T + shift1

        aligned, R, t = kabsch_align(pos_pred, pos_true, graph_id)

        assert R.shape == (2, 3, 3)
        assert t.shape == (2, 3)
        torch.testing.assert_close(aligned, pos_true, atol=1e-4, rtol=1e-4)

    def test_many_graphs(self):
        """Five graphs with varying sizes."""
        sizes = [3, 5, 2, 7, 4]
        total = sum(sizes)
        pos_true = torch.randn(total, 3)
        graph_id = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])

        # Apply different rotations per graph
        pos_pred = pos_true.clone()
        offset = 0
        for i, s in enumerate(sizes):
            R_i = _random_rotation(seed=100 + i)
            pos_pred[offset : offset + s] = pos_true[offset : offset + s] @ R_i.T
            offset += s

        aligned, R, t = kabsch_align(pos_pred, pos_true, graph_id)

        assert R.shape == (5, 3, 3)
        torch.testing.assert_close(aligned, pos_true, atol=1e-4, rtol=1e-4)


class TestKabschOutputShapes:
    """Verify output tensor shapes and dtypes."""

    def test_shapes(self):
        M, G = 12, 3
        pos = torch.randn(M, 3)
        graph_id = torch.tensor([0] * 4 + [1] * 4 + [2] * 4)

        aligned, R, t = kabsch_align(pos, pos, graph_id)

        assert aligned.shape == (M, 3)
        assert R.shape == (G, 3, 3)
        assert t.shape == (G, 3)
        assert aligned.dtype == torch.float32
        assert R.dtype == torch.float32
        assert t.dtype == torch.float32

    def test_bfloat16_input_promoted(self):
        """Inputs in bfloat16 should be promoted to float32 internally."""
        pos = torch.randn(5, 3, dtype=torch.bfloat16)
        graph_id = torch.zeros(5, dtype=torch.long)

        aligned, R, t = kabsch_align(pos, pos, graph_id)

        assert aligned.dtype == torch.float32


class TestKabschRMSD:
    """After alignment, RMSD should be near zero for rigid transforms."""

    def test_rmsd_near_zero(self):
        torch.manual_seed(99)
        pos_true = torch.randn(20, 3)
        graph_id = torch.zeros(20, dtype=torch.long)

        R_applied = _random_rotation(seed=55)
        pos_pred = pos_true @ R_applied.T + torch.tensor([10.0, -10.0, 10.0])

        aligned, _, _ = kabsch_align(pos_pred, pos_true, graph_id)

        rmsd = ((aligned - pos_true) ** 2).mean().sqrt()
        assert rmsd < 1e-4, f"RMSD = {rmsd.item()}, expected < 1e-4"
