#!/usr/bin/env python3
"""Build CSR training shards from prepared parquet shards.

Shared by both pre-training sources — run it after the source-specific prep step:

    pm6_prep.py   -> parquet -> build_shards.py -> CSR   (PM6, has PM6-optimised geometry)
    pcba_prep.py  -> parquet -> build_shards.py -> CSR   (PCBA, --smiles-only)

With --smiles-only, molecules carry no PM6 coordinates: conformers are generated
with RDKit (ETKDG + MMFF94s, falling back to UFF) and POS is written as NaN, so
training reads positions from POS_RDKIT and forces q=0.

Shards are claimed through a lock directory, so N copies of this command can run
concurrently without coordination; each claims the next unbuilt shard. Pass
--exit-after-one to build a single shard and exit (used by SLURM array jobs).

Usage:
    python -m monroe.preprocessing.build_shards \\
        --shards-dir data/pm6_resharded/shards --out-dir data/pm6_monroe/ --use-node-features

    python -m monroe.preprocessing.build_shards \\
        --shards-dir data/pcba_monroe/train/shards \\
        --out-dir data/pcba_monroe/train/built --smiles-only
"""
import argparse
import gc
import glob
import json
import multiprocessing as mp
import os
import re
import signal
import sys
import traceback
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from monroe.model.featurizer import build_single_graph


def slurm_cpu_count() -> int:
    v = os.environ.get("SLURM_CPUS_PER_TASK")
    return int(v) if v else None

def log(msg: str):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

def list_shard_files(shards_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(shards_dir, "*.parquet")))

def shard_id_from_name(path: str) -> Optional[int]:
    m = re.search(r"(\d+)(?=\.parquet$)", os.path.basename(path))
    return int(m.group(1)) if m else None

def claim_next_shard(shard_paths: List[str], out_dir: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    lock_dir = os.path.join(out_dir, "_locks")
    for spath in shard_paths:
        sid = shard_id_from_name(spath)
        prefix = os.path.join(out_dir, f"{sid:03d}")  # output prefix (no extension)
        if os.path.exists(f"{prefix}.node_ptr.npy"):
            continue
        lpath = os.path.join(lock_dir, f"{sid:03d}.lock")
        try:
            fd = os.open(lpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"{os.getpid()}".encode())
            finally:
                os.close(fd)
            return spath, prefix, sid
        except FileExistsError:
            continue
    return None, None, None

def release_lock(lock_path: str):
    try:
        if lock_path and os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass

@contextmanager
def suppress_stderr():
    """Temporarily silence noisy native warnings in worker processes."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd = os.dup(2)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(saved_fd)
        os.close(devnull_fd)


def _build_graph_task(args: Tuple[int, str, list, list, int]):
    idx, inchi, atomic_numbers, coords, charge = args
    deadline_s = 10
    old_handler = None
    try:
        if hasattr(signal, "SIGALRM"):
            def _timeout_handler(signum, frame):
                raise TimeoutError("Skipping - build_single_graph timed out")
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(deadline_s)
        with suppress_stderr():
            # When atomic_numbers/coords/charge are None, build_single_graph uses the
            # inference path: rebuild mol from InChI and generate ETKDG+MMFF94 conformer.
            g = build_single_graph(
                inchi=inchi,
                atomic_numbers=atomic_numbers,
                coords=coords,
                charge=int(charge) if charge is not None else None,
                stereo_augmentation=True,
                symmetrize=True,
            )
        return idx, g
    except Exception as e:
        print(e)
        return idx, None
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def pack_and_save(
    prefix: str,
    graphs: List[dict],
    graph_targets: np.ndarray,
    graph_cols: List[str],
    inchis: List[str],
    node_targets: Optional[np.ndarray] = None,
    node_cols: Optional[List[str]] = None,
):
    G = len(graphs)
    if G == 0:
        raise RuntimeError("No graphs built in this shard")

    Ns = np.fromiter((g["node_float"].shape[0] for g in graphs), count=G, dtype=np.uint64)
    Es = np.fromiter((g["edge_index"].shape[1] for g in graphs), count=G, dtype=np.uint64)

    node_ptr = np.zeros(G + 1, dtype=np.uint64)
    node_ptr[1:] = np.cumsum(Ns, dtype=np.uint64)
    edge_ptr = np.zeros(G + 1, dtype=np.uint64)
    edge_ptr[1:] = np.cumsum(Es, dtype=np.uint64)

    sumN = int(node_ptr[-1])
    sumE = int(edge_ptr[-1])

    NF  = np.empty((sumN, graphs[0]["node_float"].shape[1]), dtype=np.float32)
    NC  = np.empty((sumN, graphs[0]["node_codes"].shape[1]), dtype=np.uint8)
    POS = np.empty((sumN, 3), dtype=np.float32)
    POS_RDKIT = np.empty((sumN, 3), dtype=np.float32)
    EI  = np.empty((2, sumE), dtype=np.int32)
    EC  = np.empty((sumE, graphs[0]["edge_codes"].shape[1]), dtype=np.uint8)

    n_off = e_off = 0
    for g in graphs:
        n = g["node_float"].shape[0]
        e = g["edge_index"].shape[1]
        NF[n_off:n_off+n] = g["node_float"]
        NC[n_off:n_off+n] = g["node_codes"]
        # For SMILES-only molecules (no PM6 coords), g["pos"] is None -> fill with NaN.
        # Structure-prediction losses mask these out via ~isnan.
        if g["pos"] is None:
            POS[n_off:n_off+n] = np.nan
        else:
            POS[n_off:n_off+n] = g["pos"]
        POS_RDKIT[n_off:n_off+n] = g["pos_rdkit"]
        EI[:, e_off:e_off+e] = (g["edge_index"].astype(np.int64) + n_off).astype(np.int32)
        EC[e_off:e_off+e] = g["edge_codes"]
        n_off += n
        e_off += e

    def _save(name, arr):
        tmp = f"{prefix}.{name}.npy.tmp"
        with open(tmp, "wb") as f:
            np.save(f, arr)
        os.replace(tmp, f"{prefix}.{name}.npy")

    if len(inchis) != G:
        raise RuntimeError(f"Expected {G} inchis; got {len(inchis)}")
    graph_targets = np.asarray(graph_targets, dtype=np.float32)
    _save("NF", NF)
    _save("NC", NC)
    _save("POS", POS)
    _save("POS_RDKIT", POS_RDKIT)
    _save("EI", EI)
    _save("EC", EC)
    _save("node_ptr", node_ptr)
    _save("edge_ptr", edge_ptr)
    _save("Y_graph", graph_targets)

    with open(f"{prefix}.Y_graph_cols.json.tmp", "w") as f:
        json.dump(graph_cols, f)
    os.replace(f"{prefix}.Y_graph_cols.json.tmp", f"{prefix}.Y_graph_cols.json")

    if node_targets is not None:
        node_targets = np.asarray(node_targets, dtype=np.float32)
        _save("Y_node", node_targets)
        with open(f"{prefix}.Y_node_cols.json.tmp", "w") as f:
            json.dump(node_cols, f)
        os.replace(f"{prefix}.Y_node_cols.json.tmp", f"{prefix}.Y_node_cols.json")

    inchis_tmp = f"{prefix}.inchis.tmp"
    with open(inchis_tmp, "w") as f:
        for inchi in inchis:
            f.write(f"{inchi}\n")
    os.replace(inchis_tmp, f"{prefix}.inchis")

def process_shard(
    use_node_features: bool,
    shard_path: str,
    out_prefix: str,
    single_thread: bool,
    stats_dir: str,
    err_dir: str,
    locks_dir: str,
    smiles_only: bool = False,
):
    sid_str = os.path.basename(out_prefix)
    stats_path = os.path.join(stats_dir, f"{sid_str}.json")
    err_path = os.path.join(err_dir, f"{sid_str}.txt")
    lock_path = os.path.join(locks_dir, f"{sid_str}.lock")

    try:
        pf = pq.ParquetFile(shard_path)
        names = pf.schema_arrow.names
        if smiles_only:
            # SMILES-only shards carry only 'inchi' + 'SMILES' + target columns.
            # No PM6 coords, so atomic_numbers/coords/charge are absent.
            required_cols = ["inchi"]
            for c in required_cols:
                if c not in names:
                    raise KeyError(f"Missing required column '{c}' in {shard_path}")
            meta_cols = {"inchi", "SMILES"}
            graph_cols = [c for c in names if c not in meta_cols]
            node_cols = []  # SMILES-only datasets have no node-level labels
        else:
            required_cols = ["inchi", "atomic_numbers", "coords", "charge"]
            for c in required_cols:
                if c not in names:
                    raise KeyError(f"Missing required column '{c}' in {shard_path}")
            graph_cols = [c for c in names if c.startswith("graph_")]
            node_cols = [c for c in names if c.startswith("node_")] if use_node_features else []
        cols = required_cols + graph_cols + node_cols

        rows_in = pf.metadata.num_rows if pf.metadata is not None else 0
        n_graph_cols = len(graph_cols)
        n_node_cols = len(node_cols)
        graph_targets_buf = np.empty((rows_in, n_graph_cols), dtype=np.float32) if n_graph_cols else None

        def iter_rows(batch_size: int = 4092):
            idx = 0
            for batch in pf.iter_batches(columns=cols, batch_size=batch_size, use_threads=False):
                bd = batch.to_pydict()
                for j in range(len(bd["inchi"])):
                    if graph_targets_buf is not None:
                        graph_targets_buf[idx] = np.asarray([bd[c][j] for c in graph_cols], dtype=np.float32)
                    node_vals = None
                    if n_node_cols:
                        n_nodes = len(bd["atomic_numbers"][j])
                        non_h_indices = [k for k, z in enumerate(bd["atomic_numbers"][j]) if z != 1]
                        cols_np = []
                        for c in node_cols:
                            v = bd[c][j]
                            if v is None:
                                col = np.full((n_nodes,), np.nan, dtype=np.float32)
                            else:
                                raw = np.asarray(v, dtype=np.float32).reshape(-1)
                                if raw.shape[0] == len(non_h_indices):
                                    col = np.full((n_nodes,), np.nan, dtype=np.float32)
                                    col[np.asarray(non_h_indices, dtype=np.int64)] = raw
                                elif raw.shape[0] == n_nodes:
                                    col = raw
                                else:
                                    col = np.full((n_nodes,), np.nan, dtype=np.float32)
                            cols_np.append(col)
                        node_vals = np.column_stack(cols_np)
                    if smiles_only:
                        an, crd, chg = None, None, None
                    else:
                        an = bd["atomic_numbers"][j]
                        crd = bd["coords"][j]
                        chg = bd["charge"][j]
                    yield (
                        idx,
                        bd["inchi"][j],
                        an,
                        crd,
                        chg,
                        node_vals,
                    )
                    idx += 1

        graphs = []
        kept_inchis = []
        kept_idx = []
        kept_node_targets: List[np.ndarray] = [] if n_node_cols else []

        def _collect(i, g, inchi, node_tgts):
            """Shared bookkeeping for a successfully-built graph."""
            graphs.append(g)
            kept_idx.append(i)
            kept_inchis.append(inchi)
            if n_node_cols:
                if node_tgts is None:
                    node_tgts = np.full((g["node_float"].shape[0], n_node_cols), np.nan, dtype=np.float32)
                kept_node_targets.append(node_tgts)

        if single_thread:
            iterator = tqdm(
                iter_rows(),
                total=rows_in,
                desc=f"shard {sid_str}",
                unit="mol",
                smoothing=0.01,
            )
            for i, inchi, atomic_numbers, coords, charge, node_tgts in iterator:
                _, g = _build_graph_task((i, inchi, atomic_numbers, coords, charge))
                if g is not None:
                    _collect(i, g, inchi, node_tgts)
        else:
            _cpu = slurm_cpu_count() or os.cpu_count() or 2
            workers = max(1, _cpu - 1)
            mp_ctx = mp.get_context("spawn")
            max_inflight = max(4, workers * 4)

            def task_iter():
                for i, inchi, atomic_numbers, coords, charge, node_tgts in iter_rows():
                    yield i, inchi, atomic_numbers, coords, charge, node_tgts

            it = task_iter()
            pending = {}
            pending_node_targets = {} if n_node_cols else None

            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp_ctx,
                max_tasks_per_child=200,
                ) as ex:
                futures = set()

                def submit_one():
                    try:
                        i, inchi, an, crd, chg, node_tgts = next(it)
                    except StopIteration:
                        return None
                    pending[i] = inchi
                    if pending_node_targets is not None:
                        pending_node_targets[i] = node_tgts
                    return ex.submit(_build_graph_task, (i, inchi, an, crd, chg))

                # Prime the window
                for _ in range(max_inflight):
                    f = submit_one()
                    if f is None:
                        break
                    futures.add(f)

                pbar = tqdm(total=rows_in, desc=f"shard {sid_str}",
                            unit="mol", mininterval=30.0, miniters=500, smoothing=0.0)

                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for f in done:
                        futures.remove(f)
                        i, g = f.result()
                        pbar.update(1)
                        inchi = pending.pop(i, None)
                        node_tgt = pending_node_targets.pop(i, None) if pending_node_targets is not None else None
                        if g is not None and inchi is not None:
                            _collect(i, g, inchi, node_tgt)
                        nf = submit_one()
                        if nf is not None:
                            futures.add(nf)
                pbar.close()

        failed = rows_in - len(graphs)

        gc.collect()

        graph_targets = (
            graph_targets_buf[np.asarray(kept_idx, dtype=np.int64)]
            if graph_targets_buf is not None else np.empty((len(graphs), 0), dtype=np.float32)
        )
        node_targets = np.concatenate(kept_node_targets, axis=0) if n_node_cols else None
        pack_and_save(out_prefix, graphs, graph_targets, graph_cols, kept_inchis, node_targets, node_cols)

        with open(stats_path, "w") as f:
            json.dump(
                {
                    "in_shard_path": shard_path,
                    "out_prefix": out_prefix,
                    "graph_cols": graph_cols,
                    "node_cols": node_cols,
                    "rows": rows_in,
                    "kept": len(graphs),
                    "failed": failed,
                    "finished_at": datetime.now().isoformat(timespec="seconds"),
                },
                f, indent=2,
            )

    except Exception as e:
        try:
            with open(err_path, "w") as f:
                f.write("".join(traceback.format_exception(e)))
        except Exception:
            pass
        raise
    finally:
        release_lock(lock_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--single-thread", action="store_true")
    parser.add_argument("--exit-after-one", action="store_true")
    parser.add_argument("--use-node-features", action="store_true")
    parser.add_argument(
        "--smiles-only",
        action="store_true",
        help="Build from SMILES/InChI only (e.g. PCBA). Generates ETKDG+MMFF94 conformers; POS is NaN.",
    )
    args = parser.parse_args()

    if not args.single_thread:
        mp.set_start_method("spawn", force=True)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "_locks"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "stats"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "error"), exist_ok=True)

    shard_paths = list_shard_files(args.shards_dir)
    if not shard_paths:
        log("No shards found")
        sys.exit(1)

    while True:
        spath, out_prefix, sid = claim_next_shard(shard_paths, args.out_dir)
        if spath is None:
            break
        log(f"Claimed shard {sid} -> {spath}")
        try:
            process_shard(
                use_node_features=args.use_node_features,
                shard_path=spath,
                out_prefix=out_prefix,
                single_thread=args.single_thread,
                stats_dir=os.path.join(args.out_dir, "stats"),
                err_dir=os.path.join(args.out_dir, "error"),
                locks_dir=os.path.join(args.out_dir, "_locks"),
                smiles_only=args.smiles_only,
            )
            log(f"Finished shard {sid}")
        except Exception as e:
            with open(os.path.join(args.out_dir, "error", f"{sid:03d}.txt"), "w") as f:
                f.write("".join(traceback.format_exception(e)))
            log(f"Failed shard {sid}: {e}")
        if args.exit_after_one:
            break
