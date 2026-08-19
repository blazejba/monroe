"""Minimal inference example: featurize one molecule and embed it.

``load_ckpt`` restores the encoder only, so a forward pass returns Monroe's
graph-level embedding rather than task predictions. That embedding is the thing
you feed to a downstream head — it is exactly what the evaluation pipeline caches
before fitting TabPFN or an MLP on top (see monroe/eval/).

Usage:
    python monroe/run_inference.py --ckpt-path /path/to/checkpoint
    python monroe/run_inference.py --ckpt-path ... --smiles "CC(=O)Oc1ccccc1C(=O)O"

A checkpoint directory is one containing config.json plus weights.pt (or
ema_weights.pt, selected with --use-ema). With no --smiles, a molecule is pulled
from a Polaris benchmark, which requires the [eval] extra.
"""
import argparse
import sys
from pathlib import Path

import datamol as dm
import numpy as np
import torch
from torch_geometric.data import Data

from monroe.model.ckpt import load_ckpt
from monroe.model.featurizer import build_single_graph

# Bundled model shipped with the repository in checkpoint/ (Git LFS) — a normal
# checkpoint directory (config.json + weights.pt). run_inference.py lives in
# monroe/, so parents[1] is the repo root.
CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoint"


def is_flat(pos: np.ndarray) -> bool:
    """True if every atom sits at z=0, i.e. a 2D layout rather than a conformer."""
    return pos.ndim == 2 and pos.shape[0] > 0 and bool(np.all(pos[:, 2] == 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt-path", default=None,
                        help="Checkpoint directory (config.json + weights.pt). "
                             "Defaults to the bundled model in checkpoint/.")
    parser.add_argument("--smiles", default=None,
                        help="Molecule to embed. Defaults to one from a Polaris benchmark.")
    parser.add_argument("--use-ema", action="store_true",
                        help="Load ema_weights.pt instead of weights.pt.")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path if args.ckpt_path is not None else str(CHECKPOINT_DIR)
    if args.ckpt_path is None and not (CHECKPOINT_DIR / "config.json").exists():
        sys.exit(
            f"No --ckpt-path given and no bundled model at {CHECKPOINT_DIR}.\n"
            "Pass --ckpt-path <checkpoint dir>, or fetch the release weights (git lfs pull)."
        )
    encoder = load_ckpt(ckpt_path, use_ema=args.use_ema)
    encoder.eval()

    smi = args.smiles
    if smi is None:
        import polaris as po
        benchmark = po.load_benchmark("polaris/adme-fang-hppb-1")
        _, test_split = benchmark.get_train_test_split()
        smi = test_split[-1]

    inchi = dm.to_inchi(smi)
    if inchi is None:
        sys.exit(f"RDKit could not parse SMILES: {smi!r}")

    # Coordinates come from RDKit (ETKDG + MMFF94s, falling back to UFF). Monroe
    # is trained to work from these rather than needing the PM6-optimised geometry.
    try:
        graph_dict = build_single_graph(inchi=inchi, symmetrize=True)
    except ValueError as exc:
        sys.exit(f"Could not featurize {smi!r}: {exc}")

    pos = np.asarray(graph_dict["pos_rdkit"], dtype=np.float32)
    if is_flat(pos):
        # build_single_graph falls back to Compute2DCoords when conformer
        # generation fails, which yields a planar molecule.
        print(
            f"WARNING: 3D conformer generation failed for {smi!r}; the featurizer fell\n"
            f"         back to a flat 2D layout (all z=0). Monroe is structure-aware, so\n"
            f"         this embedding is unreliable. scripts/preprocessing/fix_2d_molecules.py\n"
            f"         holds the retry-with-relaxed-parameters logic used to repair such\n"
            f"         molecules in bulk.",
            file=sys.stderr,
        )

    data = Data(
        x=torch.tensor(graph_dict["node_float"], dtype=torch.float32),
        node_codes=torch.tensor(graph_dict["node_codes"], dtype=torch.long),
        edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long),
        edge_codes=torch.tensor(graph_dict["edge_codes"], dtype=torch.long),
        pos_in=torch.tensor(pos, dtype=torch.float32),
    )

    device = next(encoder.parameters()).device
    data.batch = torch.zeros(data.x.size(0), dtype=torch.long)
    data = data.to(device)

    with torch.no_grad():
        graph_emb, _ = encoder(data)

    print(f"Molecule:        {smi}")
    print(f"Atoms:           {data.x.size(0)}")
    print(f"Geometry:        {'2D fallback (unreliable)' if is_flat(pos) else '3D conformer'}")
    print(f"Graph embedding: {tuple(graph_emb.shape)}")
    print(graph_emb)


if __name__ == "__main__":
    main()
