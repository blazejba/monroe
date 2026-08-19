import argparse
import os
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_undirected
from tqdm import tqdm

from monroe.model.constants import (
    BOND_TYPES,
    EDGE_FEAT_LIST_ONE_HOT,
    NODE_FEAT_LIST_FLOAT,
    NODE_FEAT_LIST_ONE_HOT,
)
from monroe.model.grit import GritTransformer
from monroe.train.dataset import CSRMapDataset, NodeBudgetBatchSampler, _find_shard_prefixes

# Stereo edge identification constants (derived from constants.py)
BOND_TYPE_IDX = list(EDGE_FEAT_LIST_ONE_HOT.keys()).index("bond_type")
STEREO_CODE = len(BOND_TYPES)  # "other" bucket for stereo edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", required=True)
    parser.add_argument("--walk-len", type=int, required=True)
    parser.add_argument("--max-nodes", type=int, default=500)
    parser.add_argument("--n-flushes-per-shard", type=int, default=10)
    parser.add_argument("--order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--shard-prefix", type=int, help="Process only shards ending with this number (e.g. 001)")
    args = parser.parse_args()

    hp = dict(hidden_dim=32, num_layers=1, num_heads=2, emb_dim=16,
              walk_len=args.walk_len, rbf_dim=2)

    model = GritTransformer(
        node_feature_vocab=NODE_FEAT_LIST_ONE_HOT,
        edge_feature_vocab=EDGE_FEAT_LIST_ONE_HOT,
        node_float_dim=len(NODE_FEAT_LIST_FLOAT),
        **hp,
    )
    model.eval()
    
    shard_prefixes = _find_shard_prefixes(args.shard_dir)

    if args.shard_prefix is not None:
        target_suffix = f"{args.shard_prefix:03d}"
        shard_prefixes = [p for p in shard_prefixes if p.endswith(target_suffix)]
        if not shard_prefixes:
            raise ValueError(f"No shard found ending with {target_suffix} in {args.shard_dir}")
    elif args.order == "asc":
        shard_prefixes = sorted(shard_prefixes)
    else:
        shard_prefixes = sorted(shard_prefixes, reverse=True)
    
    for si, shard_prefix in enumerate(shard_prefixes, 1):
        print(f"[{si}/{len(shard_prefixes)}] {shard_prefix}")

        # compute sizes directly from ptr arrays, not from CSRMapDataset
        node_ptr = np.load(f"{shard_prefix}.node_ptr.npy", mmap_mode="r")
        edge_ptr = np.load(f"{shard_prefix}.edge_ptr.npy", mmap_mode="r")
        total_nodes = int(node_ptr[-1])
        total_edges = int(edge_ptr[-1])

        node_path = f"{shard_prefix}.{args.walk_len}.rrwp_nodes.npy"
        edge_path = f"{shard_prefix}.{args.walk_len}.rrwp_edges.npy"
        logd_path = f"{shard_prefix}.log_deg.npy"
        node_shape = (total_nodes, args.walk_len)
        edge_shape = (total_edges, args.walk_len)
        logd_shape = (total_nodes,)
        dtype = np.float32

        if all(os.path.exists(p) for p in (node_path, edge_path, logd_path)):
            print("\tfiles exist - skipping computation")
            continue

        dataset = CSRMapDataset(shard_prefix)
        sampler = NodeBudgetBatchSampler(dataset=dataset, max_nodes=args.max_nodes, shuffle=False)
        loader = DataLoader(dataset, batch_sampler=sampler)

        Path(node_path).parent.mkdir(parents=True, exist_ok=True)
        
        node_path_tmp = node_path + ".tmp"
        edge_path_tmp = edge_path + ".tmp"
        logd_path_tmp = logd_path + ".tmp"

        try:
            rrwp_nodes = open_memmap(node_path_tmp, mode="w+", dtype=dtype, shape=node_shape)
            rrwp_edges = open_memmap(edge_path_tmp, mode="w+", dtype=dtype, shape=edge_shape)
            log_degs   = open_memmap(logd_path_tmp,  mode="w+", dtype=dtype, shape=logd_shape)

            node_wp = edge_wp = 0
            tgt_node_flush = total_nodes // args.n_flushes_per_shard
            buf_nodes, buf_edges, buf_logd = [], [], []
            buf_nodes_count = buf_edges_count = 0

            for (batch, _) in tqdm(loader, desc="batches", total=len(loader)):
                n = batch.num_nodes
                m = batch.edge_index.size(1)

                # Filter out stereo edges for RRWP computation.
                # Stereo edges encode stereochemistry, not molecular connectivity,
                # so they should not influence random walk probabilities.
                edge_codes = batch.edge_codes  # [E, num_features]
                is_real_bond = edge_codes[:, BOND_TYPE_IDX] != STEREO_CODE
                real_edge_index = batch.edge_index[:, is_real_bond]

                # Build transition matrix from real bonds only
                ei_rw = to_undirected(real_edge_index, num_nodes=n)
                adj = torch.zeros(n, n, dtype=torch.float)
                adj[ei_rw[0], ei_rw[1]] = 1.0
                adj.fill_diagonal_(1.0)
                deg = adj.sum(dim=1)
                log_deg = torch.log(deg + 1)
                transition = adj / deg.unsqueeze(1)
                
                # Compute powers of transition matrix
                powers = []
                current = transition
                for _ in range(args.walk_len):
                    powers.append(current)
                    current = current @ transition
                probs = torch.stack(powers, dim=-1)
                
                # Node RRWP: diagonal of the probability matrix
                node_rrwp = probs.diagonal(dim1=0, dim2=1).transpose(0, 1)
                
                # Edge RRWP for ALL edges (including stereo) by indexing into probs
                rows, cols = batch.edge_index
                edge_vals = probs[rows, cols, :]
                edge_vals = 0.5 * (edge_vals + probs[cols, rows, :])
                edge_rrwp = edge_vals.contiguous()

                buf_nodes.append(node_rrwp.numpy().astype(dtype, copy=False))
                buf_edges.append(edge_rrwp.numpy().astype(dtype, copy=False))
                buf_logd.append(log_deg.numpy().reshape(-1).astype(dtype, copy=False))
                buf_nodes_count += n
                buf_edges_count += m

                if buf_nodes_count >= tgt_node_flush:
                    cat_nodes = np.concatenate(buf_nodes, 0)
                    cat_edges = np.concatenate(buf_edges, 0)
                    cat_logd  = np.concatenate(buf_logd, 0)
                    end_node = node_wp + cat_nodes.shape[0]
                    end_edge = edge_wp + cat_edges.shape[0]
                    rrwp_nodes[node_wp:end_node] = cat_nodes
                    rrwp_edges[edge_wp:end_edge] = cat_edges
                    log_degs[node_wp:end_node]   = cat_logd
                    node_wp, edge_wp = end_node, end_edge
                    buf_nodes.clear()
                    buf_edges.clear()
                    buf_logd.clear()
                    buf_nodes_count = buf_edges_count = 0
                    rrwp_nodes.flush()
                    rrwp_edges.flush()
                    log_degs.flush()

            if buf_nodes_count:
                cat_nodes = np.concatenate(buf_nodes, 0)
                cat_edges = np.concatenate(buf_edges, 0)
                cat_logd  = np.concatenate(buf_logd, 0)
                rrwp_nodes[node_wp:node_wp+cat_nodes.shape[0]] = cat_nodes
                rrwp_edges[edge_wp:edge_wp+cat_edges.shape[0]] = cat_edges
                log_degs[node_wp:node_wp+cat_nodes.shape[0]]   = cat_logd

            rrwp_nodes.flush()
            rrwp_edges.flush()
            log_degs.flush()
            del rrwp_nodes, rrwp_edges, log_degs
            
            os.replace(node_path_tmp, node_path)
            os.replace(edge_path_tmp, edge_path)
            os.replace(logd_path_tmp, logd_path)
            
        except Exception as e:
            print(f"Failed to process {shard_prefix}: {e}")
            # If something failed, we want to clean up the partial files
            # First ensure they are closed
            if 'rrwp_nodes' in locals():
                del rrwp_nodes
            if 'rrwp_edges' in locals():
                del rrwp_edges
            if 'log_degs' in locals():
                del log_degs
            
            for p in (node_path_tmp, edge_path_tmp, logd_path_tmp):
                if os.path.exists(p):
                    os.remove(p)
            
            # Leave final paths absent so the next run retries this shard
            for p in (node_path, edge_path, logd_path):
                if os.path.exists(p):
                    os.remove(p)

