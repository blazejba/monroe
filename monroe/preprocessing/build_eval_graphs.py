"""Precompute molecular graphs for evaluation benchmarks (Polaris, MoleculeACE).

This script builds and caches molecular graphs in parquet format for use during
downstream evaluation. Run this before running evaluations.

Usage:
    python -m monroe.preprocessing.build_eval_graphs --benchmark polaris --output-path data/polaris/
    python -m monroe.preprocessing.build_eval_graphs --benchmark moleculeace --output-path data/moleculeace/
"""

import argparse
import glob
import os
import signal

import datamol as dm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

from monroe.model.featurizer import build_from_mol_and_coords, build_single_graph


def get_all_smiles(benchmark: str) -> list[str]:
    """Get all SMILES for a benchmark."""
    if benchmark == "polaris":
        from monroe.eval.polaris import get_polaris_smiles
        return get_polaris_smiles()
    elif benchmark == "moleculeace":
        from monroe.eval.moleculeace import get_moleculeace_smiles
        return get_moleculeace_smiles()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def load_cache(cache_path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load existing graphs from parquet cache."""
    if not os.path.exists(cache_path):
        print(f"Cache file {cache_path} not found.")
        return {}

    print(f"Loading cache from {cache_path}.")
    table = pq.read_table(cache_path)
    existing_graphs = {}
    for row in table.to_pylist():
        existing_graphs[row["smiles"]] = {
            "node_float": np.asarray(row["node_float"], dtype=np.float32),
            "node_codes": np.asarray(row["node_codes"], dtype=np.uint8),
            "pos_rdkit": np.asarray(row["pos_in"], dtype=np.float32),
            "edge_index": np.asarray(row["edge_index"], dtype=np.int32),
            "edge_codes": np.asarray(row["edge_codes"], dtype=np.uint8),
        }
    return existing_graphs


def reuse_cached_graphs(
    existing_graphs: dict[str, dict[str, np.ndarray]],
    all_graphs: dict[str, dict[str, np.ndarray] | None],
) -> dict[str, dict[str, np.ndarray] | None]:
    """Reuse graphs from cache where available."""
    reused = 0
    for smi, graph in existing_graphs.items():
        if smi in all_graphs and all_graphs[smi] is None:
            all_graphs[smi] = graph
            reused += 1
    if reused:
        print(f"Reusing {reused} cached molecule graphs.")
    return all_graphs


def _build_from_smiles_fallback(smi: str) -> dict[str, np.ndarray] | None:
    """Build molecular graph directly from SMILES, bypassing InChI round-trip.

    Used as a fallback when build_single_graph fails (e.g., metal-oxygen bonds
    lost during InChI conversion + rdDetermineBonds reconstruction).
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol, addCoords=False)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        # Try 3D embedding
        try:
            from monroe.model.featurizer import predict_structure
            conf = predict_structure(mol)
            pos_rdkit = np.asarray(conf.GetPositions(), dtype=np.float32)
        except Exception:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer(0)
            pos_rdkit = np.asarray(conf.GetPositions(), dtype=np.float32)
        return build_from_mol_and_coords(mol, pos_rdkit=pos_rdkit, stereo_augmentation=True, symmetrize=True)
    except Exception as exc:
        tqdm.write(f"[warn] SMILES fallback also failed for {smi}: {exc}")
        return None


def build_mol(smi: str) -> dict[str, np.ndarray] | None:
    """Build molecular graph from SMILES."""
    try:
        tqdm.write(f"[info] Building molecule graph for {smi[:50]}...")
        try:
            inchi = dm.to_inchi(smi)
        except Exception as exc:
            tqdm.write(f"[warn] Failed to convert SMILES to InChI for {smi}: {exc}")
            mol = Chem.MolFromSmiles(smi, sanitize=False)
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_PROPERTIES)
            Chem.SanitizeMol(
                mol,
                sanitizeOps=Chem.SANITIZE_SYMMRINGS
                | Chem.SANITIZE_CLEANUP
                | Chem.SANITIZE_ADJUSTHS,
            )
            Chem.Kekulize(mol, clearAromaticFlags=True)
            Chem.SanitizeMol(
                mol, sanitizeOps=Chem.SANITIZE_SETAROMATICITY | Chem.SANITIZE_KEKULIZE
            )
            Chem.SetAromaticity(mol, Chem.AromaticityModel.AROMATICITY_DEFAULT)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            inchi = Chem.MolToInchi(mol)

        # Timeout handling
        deadline_s = 10
        old_handler = None
        if hasattr(signal, "SIGALRM"):

            def _timeout_handler(signum, frame):
                raise TimeoutError("Featurization timed out")

            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(deadline_s)

        try:
            mol_graph = build_single_graph(
                inchi=inchi,
                atomic_numbers=None,
                coords=None,
                charge=None,
                stereo_augmentation=True,
                allow_2d_fallback_on_timeout=True,
                symmetrize=True,
            )
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

        # Fallback: if InChI round-trip lost all bonds (e.g., metal compounds),
        # rebuild directly from SMILES which preserves the original bond information
        if mol_graph is not None and mol_graph["edge_index"].shape[1] == 0:
            tqdm.write(f"[warn] InChI round-trip lost bonds for {smi[:50]}, trying SMILES fallback")
            mol_graph = _build_from_smiles_fallback(smi)

    except Exception as exc:
        tqdm.write(f"[warn] Failed to build graph for {smi}: {exc}")
        mol_graph = _build_from_smiles_fallback(smi)
    return mol_graph


def _merge_into_multiconf(mol_list):
    """Merge multiple conformers into a single molecule."""
    tpl = Chem.Mol(mol_list[0])
    for cid in [c.GetId() for c in tpl.GetConformers()]:
        tpl.RemoveConformer(cid)

    for m in mol_list:
        pos = m.GetConformer(0).GetPositions()
        conf = Chem.Conformer(tpl.GetNumAtoms())
        for i, (x, y, z) in enumerate(pos):
            conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(x, y, z))
        tpl.AddConformer(conf, assignId=True)
    return tpl


def _minimize_all(m):
    """Minimize all conformers with MMFF94s."""
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=500, mmffVariant="MMFF94s")
    return [t[1] for t in res]


def get_all_sdf_mols(sdf_paths: list[str]) -> dict[str, list[Chem.Mol]]:
    """Load molecules from SDF files, grouped by SMILES."""
    all_sdf_mols = {}
    for input_path in sdf_paths:
        suppl = Chem.SDMolSupplier(input_path, removeHs=False, sanitize=False)
        for mol in suppl:
            if mol is not None:
                smi = mol.GetProp("SMILES")
                if smi not in all_sdf_mols:
                    all_sdf_mols[smi] = [mol]
                else:
                    all_sdf_mols[smi].append(mol)

    print(f"Loaded {len(all_sdf_mols)} molecules from SDF files.")
    return all_sdf_mols


def build_mol_from_sdf(mol_list: list[Chem.Mol]) -> dict[str, np.ndarray] | None:
    """Build graph from SDF molecules, selecting best conformer."""
    try:
        multi = _merge_into_multiconf(mol_list)
        energies = _minimize_all(multi)
        best_idx = int(min(range(len(energies)), key=lambda i: energies[i]))
        best_conf = Chem.Conformer(multi.GetConformer(best_idx))
        best_conf.SetId(0)
        mol = Chem.Mol(multi)
        mol.RemoveAllConformers()
        mol.AddConformer(best_conf, assignId=True)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        conf = mol.GetConformer(0)
        pos = np.asarray(conf.GetPositions(), dtype=np.float32)
        mol_graph = build_from_mol_and_coords(mol, pos_rdkit=pos, stereo_augmentation=True, symmetrize=True)
    except Exception as exc:
        tqdm.write(f"[warn] Failed to build graph from SDF: {exc}")
        mol_graph = None
    return mol_graph


def main():
    parser = argparse.ArgumentParser(
        description="Precompute molecular graphs for evaluation benchmarks"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="polaris",
        choices=["polaris", "moleculeace"],
        help="Benchmark to build graphs for",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/eval_graphs",
        help="Output directory for the molecule graphs",
    )
    parser.add_argument(
        "--use-sdf",
        action="store_true",
        help="Use precomputed SDF files for molecule graphs",
    )
    parser.add_argument(
        "--sdf-dir",
        type=str,
        default="data/tdc/sdfs",
        help="Directory containing SDF files (if --use-sdf)",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards to split the work into",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="Index of the current shard (0-based)",
    )
    args = parser.parse_args()

    output_dir = args.output_path
    if output_dir.endswith(".parquet"):
        output_dir = os.path.dirname(output_dir) or "."
        print("[warn] --output-path should be a directory; using parent instead.")

    out_name = (
        f"{args.benchmark}.parquet"
        if args.use_sdf
        else f"{args.benchmark}_no_sdf.parquet"
    )
    out_path = os.path.join(output_dir, out_name)

    if args.num_shards > 1:
        base, ext = os.path.splitext(out_path)
        shard_out_path = f"{base}.part_{args.shard_id:05d}{ext}"
    else:
        shard_out_path = out_path

    # Load existing cache
    existing_graphs = load_cache(out_path)

    # Get all SMILES for benchmark
    all_graphs = {smi: None for smi in get_all_smiles(benchmark=args.benchmark)}
    all_graphs = reuse_cached_graphs(existing_graphs, all_graphs)

    # Load SDF molecules if requested
    all_sdf_mols = {}
    if args.use_sdf:
        input_paths = glob.glob(os.path.join(args.sdf_dir, "*.sdf"))
        all_sdf_mols = get_all_sdf_mols(input_paths)

    all_smi = sorted(list(all_graphs.keys()))

    # Sharding logic
    if args.num_shards > 1:
        chunk_size = len(all_smi) // args.num_shards
        remainder = len(all_smi) % args.num_shards
        start_idx = args.shard_id * chunk_size + min(args.shard_id, remainder)
        end_idx = start_idx + chunk_size + (1 if args.shard_id < remainder else 0)

        my_smi_subset = all_smi[start_idx:end_idx]
        print(
            f"Shard {args.shard_id}/{args.num_shards}: "
            f"Processing {len(my_smi_subset)} molecules (idx {start_idx} to {end_idx})"
        )
    else:
        my_smi_subset = all_smi

    all_graphs_subset = {s: all_graphs[s] for s in my_smi_subset}
    remaining_smi = [smi for smi in my_smi_subset if all_graphs_subset[smi] is None]

    if len(remaining_smi) > 0:
        print(f"Building {len(remaining_smi)} molecule graphs.")
        for smi in tqdm(remaining_smi, total=len(remaining_smi)):
            mol_graph = None
            if args.use_sdf and smi in all_sdf_mols:
                mol_list = all_sdf_mols[smi]
                mol_graph = build_mol_from_sdf(mol_list)

            if mol_graph is None:
                mol_graph = build_mol(smi)
            all_graphs_subset[smi] = mol_graph
    else:
        print("No molecules to build.")

    # Write output
    records = []
    skipped = 0
    for smi, graph in all_graphs_subset.items():
        if graph is None:
            skipped += 1
            continue

        records.append(
            dict(
                smiles=smi,
                node_float=np.asarray(graph["node_float"], dtype=np.float32).tolist(),
                node_codes=np.asarray(graph["node_codes"], dtype=np.uint8).tolist(),
                pos_in=np.asarray(graph["pos_rdkit"], dtype=np.float32).tolist(),
                edge_index=np.asarray(graph["edge_index"], dtype=np.int32).tolist(),
                edge_codes=np.asarray(graph["edge_codes"], dtype=np.uint8).tolist(),
            )
        )

    if not records:
        raise RuntimeError("No graphs were successfully built; nothing to write.")

    os.makedirs(os.path.dirname(shard_out_path) or ".", exist_ok=True)
    table = pa.Table.from_pylist(records)
    pq.write_table(table, shard_out_path)

    print(f"Wrote {len(records)} molecule graphs to {shard_out_path}")
    if skipped:
        print(f"Skipped {skipped} molecules that failed graph construction.")


if __name__ == "__main__":
    main()
