"""Fix molecules with 2D-only coordinates in eval parquets.

Retries 3D conformer generation with relaxed parameters for molecules
where the original build fell back to 2D (z=0 for all atoms).

Usage:
    # Single process (all molecules):
    python scripts/preprocessing/fix_2d_molecules.py --benchmark polaris

    # Sharded (for parallel SLURM jobs):
    python scripts/preprocessing/fix_2d_molecules.py --benchmark polaris --num-shards 20 --shard-id 0
    # ... produces data/polaris/fix2d_shard_00.pt

    # Merge shards back into parquet:
    python scripts/preprocessing/fix_2d_molecules.py --benchmark polaris --merge
"""

import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdDistGeom, rdForceFieldHelpers
from tqdm import tqdm

from monroe.model.featurizer import build_from_mol_and_coords

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")


def _obabel_3d(smi: str, rdkit_mol: "Chem.Mol | None" = None) -> np.ndarray | None:
    """Generate 3D coordinates using OpenBabel.

    If rdkit_mol is provided, validates that OBabel and RDKit produce the same
    atom count and atomic number ordering before returning coordinates.
    """
    try:
        from openbabel import openbabel as ob

        conv = ob.OBConversion()
        conv.SetInFormat("smi")
        mol = ob.OBMol()
        conv.ReadString(mol, smi)
        mol.AddHydrogens()

        builder = ob.OBBuilder()
        builder.Build(mol)

        ff = ob.OBForceField.FindForceField("MMFF94")
        if not ff.Setup(mol):
            ff = ob.OBForceField.FindForceField("UFF")
            ff.Setup(mol)
        ff.SteepestDescent(500)
        ff.GetCoordinates(mol)

        ob_atoms = []
        coords = []
        for atom in ob.OBMolAtomIter(mol):
            ob_atoms.append(atom.GetAtomicNum())
            coords.append([atom.GetX(), atom.GetY(), atom.GetZ()])

        # Validate atom ordering matches RDKit
        if rdkit_mol is not None:
            rdk_atoms = [a.GetAtomicNum() for a in rdkit_mol.GetAtoms()]
            if len(ob_atoms) != len(rdk_atoms) or ob_atoms != rdk_atoms:
                return None

        return np.array(coords, dtype=np.float32)
    except Exception:
        return None


def _rdkit_3d_worker(smi, result_dict):
    """Worker function for subprocess-based RDKit conformer generation."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    params = rdDistGeom.ETKDGv3()
    params.useRandomCoords = True
    params.maxIterations = 2000
    params.enforceChirality = False
    params.useSmallRingTorsions = False
    params.numThreads = 1

    conf_ids = rdDistGeom.EmbedMultipleConfs(mol, numConfs=10, params=params)
    if len(conf_ids) > 0:
        mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, "MMFF94s")
        if mp is not None:
            try:
                ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp)
                results = rdForceFieldHelpers.OptimizeMoleculeConfs(mol, ff, maxIters=200)
                energies = [e for _, e in results]
                best_idx = int(min(range(len(energies)), key=energies.__getitem__))
            except Exception:
                best_idx = 0
        else:
            best_idx = 0
        result_dict["pos"] = np.asarray(mol.GetConformer(int(conf_ids[best_idx])).GetPositions(), dtype=np.float32)
        return

    mol.RemoveAllConformers()
    res = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=100, useRandomCoords=True)
    if res >= 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
        result_dict["pos"] = np.asarray(mol.GetConformer(0).GetPositions(), dtype=np.float32)


def try_3d_conformer(smi: str, max_rdkit_atoms: int = 100, timeout_s: int = 60) -> np.ndarray | None:
    """Try multiple strategies to get a 3D conformer."""
    import multiprocessing as mp

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

    # Large molecules: skip RDKit entirely, go straight to OpenBabel
    if mol.GetNumHeavyAtoms() > max_rdkit_atoms:
        pos = _obabel_3d(smi, rdkit_mol=mol)
        if pos is not None:
            return pos
        return None

    # Run RDKit in a subprocess with a real timeout (SIGALRM can't interrupt C extensions)
    manager = mp.Manager()
    result_dict = manager.dict()
    proc = mp.Process(target=_rdkit_3d_worker, args=(smi, result_dict))
    proc.start()
    proc.join(timeout=timeout_s)
    if proc.is_alive():
        proc.kill()
        proc.join()

    if "pos" in result_dict:
        return result_dict["pos"]

    # Fallback: OpenBabel (handles macrocycles, timeouts, exotic chemistry)
    pos = _obabel_3d(smi, rdkit_mol=mol)
    if pos is not None:
        return pos

    return None


def find_2d_indices(rows):
    """Return indices of rows with 2D-only coordinates."""
    indices = []
    for i, r in enumerate(rows):
        pos = np.asarray(r["pos_in"], dtype=np.float32)
        if pos.ndim == 2 and pos.shape[0] > 0 and np.all(pos[:, 2] == 0):
            indices.append(i)
    return indices


def fix_shard(rows, twod_indices, shard_id, num_shards):
    """Fix a subset of 2D molecules, return dict of {index: fixed_row}."""
    chunk_size = len(twod_indices) // num_shards
    remainder = len(twod_indices) % num_shards
    start = shard_id * chunk_size + min(shard_id, remainder)
    end = start + chunk_size + (1 if shard_id < remainder else 0)
    my_indices = twod_indices[start:end]

    print(f"Shard {shard_id}/{num_shards}: {len(my_indices)} molecules (indices {start}-{end})")

    fixes = {}
    for idx in tqdm(my_indices, desc=f"Shard {shard_id}"):
        smi = rows[idx]["smiles"]
        pos_3d = try_3d_conformer(smi)
        if pos_3d is not None and not np.all(pos_3d[:, 2] == 0):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            try:
                g = build_from_mol_and_coords(mol, pos_rdkit=pos_3d, stereo_augmentation=True, symmetrize=True)
                fixes[idx] = dict(
                    smiles=smi,
                    node_float=np.asarray(g["node_float"], dtype=np.float32).tolist(),
                    node_codes=np.asarray(g["node_codes"], dtype=np.uint8).tolist(),
                    pos_in=pos_3d.tolist(),
                    edge_index=np.asarray(g["edge_index"], dtype=np.int32).tolist(),
                    edge_codes=np.asarray(g["edge_codes"], dtype=np.uint8).tolist(),
                )
            except Exception as e:
                tqdm.write(f"  Graph build failed for {smi[:50]}: {e}")

    return fixes


def merge_shards(bench_dir, rows, twod_indices):
    """Merge shard .pt files back into the parquet."""
    import glob
    shard_files = sorted(glob.glob(os.path.join(bench_dir, "fix2d_shard_*.pt")))
    if not shard_files:
        print(f"No shard files found in {bench_dir}")
        return 0

    total_fixed = 0
    for sf in shard_files:
        fixes = torch.load(sf, weights_only=False)
        for idx, row in fixes.items():
            rows[idx] = row
            total_fixed += 1
        print(f"  Loaded {len(fixes)} fixes from {os.path.basename(sf)}")

    if total_fixed > 0:
        path = os.path.join(bench_dir, f"{os.path.basename(bench_dir)}_no_sdf.parquet")
        new_table = pa.Table.from_pylist(rows)
        pq.write_table(new_table, path)
        print(f"Wrote {len(rows)} rows to {path} ({total_fixed} fixed)")

    # Cleanup shard files
    for sf in shard_files:
        os.remove(sf)
    print(f"Cleaned up {len(shard_files)} shard files")

    return total_fixed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["polaris", "moleculeace", "both"], default="both")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()

    benchmarks = ["polaris", "moleculeace"] if args.benchmark == "both" else [args.benchmark]

    for bench in benchmarks:
        bench_dir = f"data/{bench}"
        path = f"{bench_dir}/{bench}_no_sdf.parquet"
        print(f"\n{'='*60}")
        print(f"{'Merging' if args.merge else 'Fixing'} {bench}")
        print(f"{'='*60}")

        table = pq.read_table(path)
        rows = table.to_pylist()
        twod_indices = find_2d_indices(rows)
        print(f"Found {len(twod_indices)} 2D-only molecules out of {len(rows)}")

        if args.merge:
            merge_shards(bench_dir, rows, twod_indices)
            continue

        if len(twod_indices) == 0:
            print("Nothing to fix")
            continue

        fixes = fix_shard(rows, twod_indices, args.shard_id, args.num_shards)
        print(f"Fixed {len(fixes)} molecules")

        if args.num_shards > 1:
            out = os.path.join(bench_dir, f"fix2d_shard_{args.shard_id:02d}.pt")
            torch.save(fixes, out)
            print(f"Saved to {out}")
        else:
            for idx, row in fixes.items():
                rows[idx] = row
            new_table = pa.Table.from_pylist(rows)
            pq.write_table(new_table, path)
            print(f"Wrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
