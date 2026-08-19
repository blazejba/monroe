#!/usr/bin/env python3
"""Consolidate multiple small CSR shards into fewer larger shards.

This script merges N consecutive shards into one, reducing the total number of
shards while maintaining the same data. Useful for reducing filesystem overhead
when you have many small shards.

Usage:
    python scripts/preprocessing/consolidate_shards.py \
        --consolidation-factor 10 \
        --input-dir data/pm6_monroe/ \
        --output-dir data/pm6/ \
        --include-rrwp

Arguments:
    --consolidation-factor: Number of input shards to merge into one output shard
    --input-dir: Directory containing the input CSR shards
    --output-dir: Directory for consolidated output shards
    --include-rrwp: Also consolidate RRWP files if present
"""
import argparse
import glob
import json
import os
import warnings

import numpy as np
from tqdm import tqdm


def list_shard_prefixes(directory: str):
    paths = sorted(glob.glob(os.path.join(directory, "*.NF.npy")))
    return [p[: -len(".NF.npy")] for p in paths]


def load_inchis(path: str):
    with open(path) as f:
        return [line.rstrip("\n") for line in f]


def save_inchis(path: str, inchis):
    with open(path, "w") as f:
        for inchi in inchis:
            f.write(f"{inchi}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--consolidation-factor", type=int, required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-rrwp", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    prefixes = list_shard_prefixes(args.input_dir)

    names = ["NF", "NC", "POS", "POS_RDKIT", "EI", "EC"]

    for i in tqdm(range(0, len(prefixes), args.consolidation_factor)):
        group = prefixes[i : i + args.consolidation_factor]
        out_prefix = os.path.join(args.output_dir, f"{i // args.consolidation_factor:03d}")

        arrays = {k: [] for k in names}
        node_ptr = [0]
        edge_ptr = [0]
        inchis = []
        y_graph_cols = None
        y_node_cols = None
        has_node_targets = False
        node_offset = 0
        edge_offset = 0
        if args.include_rrwp:
            log_deg_arrays = []
            rrwp_nodes = {}
            rrwp_edges = {}

        for prefix in group:
            for k in names:
                arr = np.load(f"{prefix}.{k}.npy")
                if k == "EI":
                    arr = (arr + node_offset).astype(np.int32)
                arrays[k].append(arr)

            np_node_ptr = np.load(f"{prefix}.node_ptr.npy")
            np_edge_ptr = np.load(f"{prefix}.edge_ptr.npy")
            node_ptr.extend((np_node_ptr[1:] + node_offset).tolist())
            edge_ptr.extend((np_edge_ptr[1:] + edge_offset).tolist())
            node_offset += int(np_node_ptr[-1])
            edge_offset += int(np_edge_ptr[-1])

            arrays.setdefault("Y_graph", []).append(np.load(f"{prefix}.Y_graph.npy"))
            node_path = f"{prefix}.Y_node.npy"
            if os.path.exists(node_path):
                has_node_targets = True
                arrays.setdefault("Y_node", []).append(np.load(node_path))
            inchis.extend(load_inchis(f"{prefix}.inchis"))
            if y_graph_cols is None:
                with open(f"{prefix}.Y_graph_cols.json") as f:
                    y_graph_cols = json.load(f)
            if has_node_targets and y_node_cols is None:
                y_node_cols_path = f"{prefix}.Y_node_cols.json"
                if os.path.exists(y_node_cols_path):
                    with open(y_node_cols_path) as f:
                        y_node_cols = json.load(f)
            if args.include_rrwp:
                log_deg_arrays.append(np.load(f"{prefix}.log_deg.npy"))
                for path in sorted(glob.glob(f"{prefix}.*.rrwp_nodes.npy")):
                    step_str = path[len(prefix) + 1 : -len(".rrwp_nodes.npy")]
                    try:
                        step = int(step_str)
                    except (ValueError, TypeError):
                        continue
                    rrwp_nodes.setdefault(step, {})[prefix] = np.load(path)
                for path in sorted(glob.glob(f"{prefix}.*.rrwp_edges.npy")):
                    step_str = path[len(prefix) + 1 : -len(".rrwp_edges.npy")]
                    try:
                        step = int(step_str)
                    except (ValueError, TypeError):
                        continue
                    rrwp_edges.setdefault(step, {})[prefix] = np.load(path)

        np.save(f"{out_prefix}.NF.npy", np.concatenate(arrays["NF"], axis=0))
        np.save(f"{out_prefix}.NC.npy", np.concatenate(arrays["NC"], axis=0))
        np.save(f"{out_prefix}.POS.npy", np.concatenate(arrays["POS"], axis=0))
        np.save(f"{out_prefix}.POS_RDKIT.npy", np.concatenate(arrays["POS_RDKIT"], axis=0))
        np.save(f"{out_prefix}.EI.npy", np.concatenate(arrays["EI"], axis=1))
        np.save(f"{out_prefix}.EC.npy", np.concatenate(arrays["EC"], axis=0))
        np.save(f"{out_prefix}.node_ptr.npy", np.asarray(node_ptr, dtype=np.uint64))
        np.save(f"{out_prefix}.edge_ptr.npy", np.asarray(edge_ptr, dtype=np.uint64))
        np.save(f"{out_prefix}.Y_graph.npy", np.concatenate(arrays["Y_graph"], axis=0))
        if has_node_targets:
            np.save(f"{out_prefix}.Y_node.npy", np.concatenate(arrays["Y_node"], axis=0))
        save_inchis(f"{out_prefix}.inchis", inchis)
        with open(f"{out_prefix}.Y_graph_cols.json", "w") as f:
            json.dump(y_graph_cols, f)
        if has_node_targets:
            with open(f"{out_prefix}.Y_node_cols.json", "w") as f:
                json.dump(y_node_cols or [], f)
        if args.include_rrwp:
            np.save(f"{out_prefix}.log_deg.npy", np.concatenate(log_deg_arrays, axis=0))
            all_steps = sorted(set(rrwp_nodes.keys()) | set(rrwp_edges.keys()))
            for step in all_steps:
                nodes_by_prefix = rrwp_nodes.get(step, {})
                edges_by_prefix = rrwp_edges.get(step, {})
                missing = [
                    prefix
                    for prefix in group
                    if prefix not in nodes_by_prefix or prefix not in edges_by_prefix
                ]
                if missing:
                    warnings.warn(
                        f"Skipping RRWP step {step} for {out_prefix} (missing {len(missing)} shards)"
                    )
                    continue
                nodes_concat = np.concatenate(
                    [nodes_by_prefix[prefix] for prefix in group], axis=0
                )
                edges_concat = np.concatenate(
                    [edges_by_prefix[prefix] for prefix in group], axis=0
                )
                np.save(f"{out_prefix}.{step}.rrwp_nodes.npy", nodes_concat)
                np.save(f"{out_prefix}.{step}.rrwp_edges.npy", edges_concat)
