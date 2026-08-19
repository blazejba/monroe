"""Precompute RRWP for evaluation benchmark molecules.

Produces a .pt file mapping SMILES -> {rrwp_nodes, rrwp_edges, log_deg} for a given walk_len.
This avoids recomputing dense matrix powers on every eval run.

Usage:
    python scripts/preprocessing/precompute_eval_rrwp.py --walk-len 16
    python scripts/preprocessing/precompute_eval_rrwp.py --walk-len 16 --benchmark polaris
"""

import argparse
import time
from pathlib import Path

import torch
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from monroe.eval.dataset import _ensure_precomputed_graphs, precomputed_graphs
from monroe.model.constants import BOND_TYPE_FEAT_IDX, BOND_TYPE_OTHER_CODE


def compute_rrwp(edge_index, edge_codes, num_nodes, walk_len):
    """Compute RRWP from real bonds only (stereo edges excluded)."""
    is_real = edge_codes[:, BOND_TYPE_FEAT_IDX] != BOND_TYPE_OTHER_CODE
    real_ei = edge_index[:, is_real]

    ei_rw = to_undirected(real_ei, num_nodes=num_nodes)
    adj = torch.zeros(num_nodes, num_nodes, dtype=torch.float)
    adj[ei_rw[0], ei_rw[1]] = 1.0
    adj.fill_diagonal_(1.0)
    deg = adj.sum(dim=1)
    transition = adj / deg.unsqueeze(1)

    powers = []
    current = transition
    for _ in range(walk_len):
        powers.append(current)
        current = current @ transition
    probs = torch.stack(powers, dim=-1)

    rrwp_nodes = probs.diagonal(dim1=0, dim2=1).transpose(0, 1).contiguous()
    rows, cols = edge_index
    edge_vals = probs[rows, cols, :]
    rrwp_edges = (0.5 * (edge_vals + probs[cols, rows, :])).contiguous()
    log_deg = torch.log(deg + 1)

    return rrwp_nodes, rrwp_edges, log_deg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--walk-len", type=int, default=16)
    parser.add_argument("--benchmark", choices=["polaris", "moleculeace", "both"], default="both")
    args = parser.parse_args()

    benchmarks = ["polaris", "moleculeace"] if args.benchmark == "both" else [args.benchmark]

    for bench in benchmarks:
        parquet_path = f"data/{bench}/{bench}_no_sdf.parquet"
        out_path = Path(f"data/{bench}/{bench}_rrwp_{args.walk_len}.pt")

        print(f"\n{'='*60}")
        print(f"Benchmark: {bench}, walk_len={args.walk_len}")
        print(f"{'='*60}")

        precomputed_graphs.clear()
        _ensure_precomputed_graphs(parquet_path)

        rrwp_cache = {}
        t0 = time.time()

        skipped = 0
        for smi, data in tqdm(precomputed_graphs.items(), desc=f"Computing RRWP ({bench})"):
            N = data.x.size(0)
            if data.edge_codes.dim() < 2 or data.edge_index.size(1) == 0:
                skipped += 1
                continue
            rrwp_nodes, rrwp_edges, log_deg = compute_rrwp(
                data.edge_index, data.edge_codes, N, args.walk_len
            )
            rrwp_cache[smi] = {
                "rrwp_nodes": rrwp_nodes,
                "rrwp_edges": rrwp_edges,
                "log_deg": log_deg,
            }
        if skipped:
            print(f"Skipped {skipped} molecules with no edges")

        elapsed = time.time() - t0
        print(f"Computed RRWP for {len(rrwp_cache)} molecules in {elapsed:.1f}s")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(rrwp_cache, out_path)
        print(f"Saved to {out_path} ({out_path.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
