import glob
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from monroe.train.loss import (
    KabschAlignedCoordLoss,
    SparseBetaNLLLoss,
    SparseFocalLoss,
    SparseHuberLoss,
    SparseLogHuberLoss,
    SparseLogRatioSquaredLoss,
    SparseOrdinalLoss,
)
from monroe.train.metrics import (
    PairwiseDistanceDummyMetric,
    SparseBetaMAE,
    SparseFocalAUROC,
    SparseL1,
    SparseLogitMAE,
    SparseLogMAE,
    SparseOrdinalAcc,
)
from monroe.utils import printf

if TYPE_CHECKING:
    from monroe.train.distributed import DistributedConfig


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass
class TaskInfo:
    """Task configuration extracted from task_dict."""

    task_dict: dict
    graph_tasks: list
    node_tasks: list
    graph_target_indices: list
    node_target_indices: list


# -----------------------------------------------------------------------------
# Batch Utilities
# -----------------------------------------------------------------------------


def to_bfloat16_batch(batch, device, non_blocking: bool = True):
    """Move batch to device and cast floating tensors to bfloat16.

    Positional tensors (pos, pos_in) are kept in float32 so that edge RBF
    distances are computed at full precision, matching the eval pipeline.

    Args:
        batch: PyG Data batch object.
        device: Target device.
        non_blocking: Use non-blocking transfers.

    Returns:
        Batch with tensors moved and cast.
    """

    def convert(t):
        if torch.is_tensor(t):
            if t.is_floating_point():
                return t.to(device=device, dtype=torch.bfloat16, non_blocking=non_blocking)
            return t.to(device=device, non_blocking=non_blocking)
        return t

    batch = batch.apply(convert)

    # Keep positions in float32 for edge distance precision. bfloat16 has only
    # 7-bit mantissa (~0.04 Å error at 10 Å) vs float32's 23-bit, and the edge
    # RBF bins are only 0.14 Å wide. Under autocast, element-wise ops (sub,
    # norm, exp) preserve the input dtype, so float32 positions yield float32
    # bond lengths and RBF activations.
    if hasattr(batch, "pos_in") and batch.pos_in is not None:
        batch.pos_in = batch.pos_in.float()
    if hasattr(batch, "pos") and batch.pos is not None:
        batch.pos = batch.pos.float()

    return batch


def build_target_indices(task_names: List[str], y_cols: List[str]) -> List[int]:
    """Map task names to column indices in the target tensor.

    Args:
        task_names: List of task names to map.
        y_cols: Column names from the dataset.

    Returns:
        List of indices corresponding to each task name.

    Raises:
        KeyError: If any task names are not found in y_cols.
    """
    col_index = {name: i for i, name in enumerate(y_cols)}
    missing = [name for name in task_names if name not in col_index]
    if missing:
        raise KeyError(f"Missing targets in dataset: {missing}")
    return [col_index[name] for name in task_names]


def targets_to_dict(
    y: torch.Tensor, task_names: List[str], target_indices: List[int]
) -> Dict[str, torch.Tensor]:
    """Convert target tensor to dict keyed by task name.

    Args:
        y: Target tensor of shape [batch, num_targets].
        task_names: Names for each task.
        target_indices: Column indices for each task.

    Returns:
        Dict mapping task name to its target column.
    """
    return {task: y[:, idx] for task, idx in zip(task_names, target_indices)}


# -----------------------------------------------------------------------------
# Shard Utilities
# -----------------------------------------------------------------------------


def get_shard_prefix(data_dir: str, shard_idx: int) -> str:
    """Get the file prefix for a given shard index.

    Args:
        data_dir: Directory containing shard files.
        shard_idx: Index of the shard to get.

    Returns:
        Full path prefix for the shard files.
    """
    prefixes = sorted(glob.glob(os.path.join(data_dir, "*.node_ptr.npy")))
    prefixes = [p[: -len(".node_ptr.npy")] for p in prefixes]
    prefixes = sorted(prefixes, key=lambda p: int(os.path.basename(p)))
    # Reserve last shard for validation
    return prefixes[shard_idx % (len(prefixes) - 1)]


def _cleanup_shm_list(shm_names: List[str]):
    """Unlinks a list of shared memory blocks."""
    for name in shm_names:
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error cleaning up shm {name}: {e}")


def load_shard_to_shm(prefix: str, walk_len: Optional[int] = None, use_node_labels: bool = False) -> Dict[str, Any]:
    """
    Loads shard data from disk and copies it into newly allocated SharedMemory.
    Returns metadata dict: { key: (shm_name, shape, dtype_str) }.
    """
    metadata = {}
    
    # Helper to load array, create SHM, copy, and record metadata
    def load_and_share(key: str, path: str, mmap_mode=None):
        if not os.path.exists(path):
            return
        
        # Load into RAM then copy to SHM.
        # Direct read to buffer is not easily supported by np.load for .npy files.
        arr = np.load(path)
        
        shm_name = f"monroe_shm_{os.path.basename(prefix)}_{key}_{os.getpid()}_{random.randint(0, 2**31)}"
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes, name=shm_name)
        
        # Create numpy wrapper around buffer
        shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        shm_arr[:] = arr[:] # Copy data
        
        metadata[key] = (shm_name, arr.shape, str(arr.dtype))
        
        del arr
        shm.close() # Close handle in this process
    
    # Define files to load - same logic as CSRShard __init__
    load_and_share("NF", f"{prefix}.NF.npy")
    load_and_share("NC", f"{prefix}.NC.npy")
    load_and_share("EI", f"{prefix}.EI.npy")
    load_and_share("EC", f"{prefix}.EC.npy")
    load_and_share("POS", f"{prefix}.POS.npy")
    load_and_share("POS_RDKIT", f"{prefix}.POS_RDKIT.npy")
    load_and_share("node_ptr", f"{prefix}.node_ptr.npy")
    load_and_share("edge_ptr", f"{prefix}.edge_ptr.npy")
    load_and_share("Y_graph", f"{prefix}.Y_graph.npy")
    
    suffix = f".{walk_len}" if walk_len is not None else ""
    load_and_share("rrwp_nodes", f"{prefix}{suffix}.rrwp_nodes.npy")
    load_and_share("rrwp_edges", f"{prefix}{suffix}.rrwp_edges.npy")
    load_and_share("log_deg", f"{prefix}.log_deg.npy")
    
    if use_node_labels:
        load_and_share("Y_node", f"{prefix}.Y_node.npy")
    
    return metadata


class ShardLoader:
    """Manages double-buffered loading of shards to shared memory.

    This class handles asynchronous prefetching of data shards into shared
    memory, coordinating across distributed processes. Local rank 0 on each
    node does the actual loading; other ranks read from shared memory.
    """

    def __init__(self, hp, dist_cfg: "DistributedConfig"):
        """Initialize the shard loader.

        Args:
            hp: Hyperparameters namespace with pretrain.load_to_memory,
                pretrain.data_dir, pretrain.n_shards, encoder.walk_len,
                pretrain.use_node_labels.
            dist_cfg: Distributed configuration.
        """
        self.hp = hp
        self.dist_cfg = dist_cfg
        self.executor = (
            ThreadPoolExecutor(max_workers=1)
            if dist_cfg.is_local_main and hp.pretrain.load_to_memory
            else None
        )
        self.next_future = None
        self.shm_to_cleanup: List[str] = []

    def prefetch_first_shard(self, start_shard: int) -> None:
        """Start loading the first shard in the background.

        Args:
            start_shard: Index of the first shard to load.
        """
        if not self.hp.pretrain.load_to_memory:
            return
        if self.executor is None:
            return

        if start_shard < self.hp.pretrain.n_shards:
            prefix = get_shard_prefix(self.hp.pretrain.data_dir, start_shard)
            self.next_future = self.executor.submit(
                load_shard_to_shm,
                prefix=prefix,
                walk_len=self.hp.encoder.walk_len,
                use_node_labels=self.hp.pretrain.use_node_labels,
            )

    def get_shard_metadata(self, shard_idx: int, next_shard_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Get metadata for current shard, prefetch next shard.

        Local rank 0 waits for the background load to complete, then
        broadcasts metadata to all ranks via ``dist.broadcast_object_list``.

        Args:
            shard_idx: Index of the shard to get.
            next_shard_id: Data shard ID to prefetch next, or None to skip prefetch.

        Returns:
            Metadata dict for shared memory arrays, or None if not loading to memory.
        """
        if not self.hp.pretrain.load_to_memory:
            return None  # No SHM mode — no communication needed

        current_metadata = None

        # Only the local main rank actually loads from disk/future
        if self.dist_cfg.is_local_main and self.executor is not None:
            if self.next_future:
                try:
                    current_metadata = self.next_future.result()
                except Exception as e:
                    printf(f"Background loading failed: {e}")
                    raise

                # Prefetch next shard
                if next_shard_id is not None:
                    prefix = get_shard_prefix(self.hp.pretrain.data_dir, next_shard_id)
                    self.next_future = self.executor.submit(
                        load_shard_to_shm,
                        prefix=prefix,
                        walk_len=self.hp.encoder.walk_len,
                        use_node_labels=self.hp.pretrain.use_node_labels,
                    )
                else:
                    self.next_future = None

        # ALL ranks must participate in the broadcast
        if self.dist_cfg.is_distributed:
            payload = [current_metadata]
            dist.broadcast_object_list(payload, src=0)
            current_metadata = payload[0]

        return current_metadata

    def cleanup_previous_shard(self, current_metadata: Optional[Dict[str, Any]]) -> None:
        """Clean up shared memory from previous shard.

        Args:
            current_metadata: Metadata from current shard (to mark for next cleanup).
        """
        if not self.dist_cfg.is_local_main:
            return

        if self.shm_to_cleanup:
            _cleanup_shm_list(self.shm_to_cleanup)
            self.shm_to_cleanup = []

        if current_metadata:
            self.shm_to_cleanup = [v[0] for v in current_metadata.values()]


def _find_shard_prefixes(root: str) -> List[str]:
    node_ptr_files = sorted(glob.glob(os.path.join(root, "*.node_ptr.npy")))
    return [p[:-len(".node_ptr.npy")] for p in node_ptr_files]


class CSRShard:

    def __init__(
        self,
        prefix: str,
        walk_len: Optional[int] = None,
        use_node_labels: bool = False,
        load_to_memory: bool = False,
        shm_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.walk_len = walk_len
        self.shm_blocks = [] # Keep references to prevent early gc/close if needed, though mostly needed for unlinking.
        
        # Helper to load from SHM or Disk
        def load_array(key: str, filename: str):
            if shm_metadata and key in shm_metadata:
                name, shape, dtype_str = shm_metadata[key]
                try:
                    shm = shared_memory.SharedMemory(name=name)
                    self.shm_blocks.append(shm)
                    
                    # Create ndarray on the buffer
                    # Note: we must keep 'shm' alive as long as we use this array
                    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
                    return arr
                except FileNotFoundError:
                    print(f"Warning: SHM block {name} not found for {key}, falling back to disk.")
            
            # Fallback to disk
            mmap_mode = "r" if not load_to_memory else None
            if os.path.exists(filename):
                return np.load(filename, mmap_mode=mmap_mode)
            else:
                return None

        self.NF  = load_array("NF", f"{prefix}.NF.npy")
        self.NC  = load_array("NC", f"{prefix}.NC.npy")
        self.EI  = load_array("EI", f"{prefix}.EI.npy")
        self.EC  = load_array("EC", f"{prefix}.EC.npy")
        self.POS = load_array("POS", f"{prefix}.POS.npy")
        self.POS_RDKIT = load_array("POS_RDKIT", f"{prefix}.POS_RDKIT.npy")
        self.node_ptr = load_array("node_ptr", f"{prefix}.node_ptr.npy")
        self.edge_ptr = load_array("edge_ptr", f"{prefix}.edge_ptr.npy")
        
        suffix = f".{walk_len}" if walk_len is not None else ""
        self.rrwp_nodes = load_array("rrwp_nodes", f"{prefix}{suffix}.rrwp_nodes.npy")
        self.rrwp_edges = load_array("rrwp_edges", f"{prefix}{suffix}.rrwp_edges.npy")
        self.log_deg    = load_array("log_deg",     f"{prefix}.log_deg.npy")
        
        self.rrwp_precomputed = (
            (self.rrwp_nodes is not None)
            and (self.rrwp_edges is not None)
            and (self.log_deg is not None)
        )

        self.Y_graph = load_array("Y_graph", f"{prefix}.Y_graph.npy")
        
        with open(f"{prefix}.Y_graph_cols.json") as f:
            self.Y_graph_cols: List[str] = json.load(f)
            
        if use_node_labels:
            self.Y_node = load_array("Y_node", f"{prefix}.Y_node.npy")
            if self.Y_node is not None:
                with open(f"{prefix}.Y_node_cols.json") as f:
                    self.Y_node_cols: List[str] = json.load(f)
            else:
                self.Y_node_cols = []
        else:
            self.Y_node = None
            self.Y_node_cols: List[str] = []

        self._node_counts = (self.node_ptr[1:] - self.node_ptr[:-1]).astype(np.int64)
        self._edge_counts = (self.edge_ptr[1:] - self.edge_ptr[:-1]).astype(np.int64)

    def close(self):
        """Close shared memory handles."""
        for shm in self.shm_blocks:
            shm.close()
        self.shm_blocks = []

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return int(self.node_ptr.shape[0] - 1)

    @property
    def node_counts(self) -> np.ndarray:
        return self._node_counts

    @property
    def edge_counts(self) -> np.ndarray:
        return self._edge_counts

    def get(self, i: int) -> Dict[str, np.ndarray]:
        n0, n1 = int(self.node_ptr[i]), int(self.node_ptr[i+1])
        e0, e1 = int(self.edge_ptr[i]), int(self.edge_ptr[i+1])

        nf  = self.NF[n0:n1]                             # float32
        nc  = self.NC[n0:n1]                             # uint8
        pos = self.POS[n0:n1]                            # float16
        pos_rdkit = self.POS_RDKIT[n0:n1]                # float16
        ei  = self.EI[:, e0:e1]                          # int32 (global)
        ec  = self.EC[e0:e1]                             # uint8
        if self.rrwp_precomputed:
            rrwp_nodes = self.rrwp_nodes[n0:n1].astype(np.float32, copy=False) # float16 -> float32
            rrwp_edges = self.rrwp_edges[e0:e1].astype(np.float32, copy=False) # float16 -> float32
            log_deg    = self.log_deg[n0:n1].astype(np.float32, copy=False)     # float16 -> float32
        else:
            rrwp_nodes = None
            rrwp_edges = None
            log_deg    = None

        ei_local = (ei.astype(np.int64) - n0).astype(np.int32, copy=False)
        y_node = None
        if self.Y_node is not None:
            y_node = self.Y_node[n0:n1]
        return dict(
            node_float=nf,
            node_codes=nc,
            pos=pos.astype(np.float32, copy=False),       # float16 -> float32
            pos_rdkit=pos_rdkit.astype(np.float32, copy=False), # float16 -> float32
            edge_index=ei_local,
            edge_codes=ec,
            y_node=y_node,
            rrwp_nodes=rrwp_nodes,
            rrwp_edges=rrwp_edges,
            log_deg=log_deg,
        )


class CSRMapDataset(Dataset):
    def __init__(
        self,
        shard_prefix: str,
        subset_ratio: Optional[float] = None,
        q: Optional[float] = None,
        walk_len: Optional[int] = None,
        use_node_labels: bool = False,
        load_to_memory: bool = False,
        shm_metadata: Optional[Dict[str, Any]] = None,
        has_pm6_pos: bool = True,
    ):
        super().__init__()
        self.shard = CSRShard(
            shard_prefix,
            walk_len=walk_len,
            use_node_labels=use_node_labels,
            load_to_memory=load_to_memory,
            shm_metadata=shm_metadata
        )
        self.Y_graph_cols = self.shard.Y_graph_cols
        self.Y_node_cols = self.shard.Y_node_cols
        self.subset_ratio = float(subset_ratio) if subset_ratio is not None else 1.0
        if not (0 < self.subset_ratio <= 1.0):
            raise ValueError("subset_ratio must be in the interval (0, 1].")

        total_graphs = len(self.shard)
        subset_size = max(1, min(total_graphs, int(total_graphs * self.subset_ratio)))
        self.indices = list(range(total_graphs))[:subset_size]
        self.node_counts = [int(self.shard.node_counts[i]) for i in self.indices]
        self.edge_counts = [int(self.shard.edge_counts[i]) for i in self.indices]
        self.Y_graph_float = np.array(
            self.shard.Y_graph[self.indices],
            dtype=np.float32,
        )
        self.q = q
        # When False (e.g. PCBA), POS is NaN and _sample_q is forced to 0.0 so
        # pos_in always equals pos_rdkit — prevents NaN leaking through blending.
        self.has_pm6_pos = has_pm6_pos

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_q(self) -> float:
        """Sample or reuse the positional mixing factor ``q``."""
        # For SMILES-only molecules we have no PM6 coords; RDKit conformer is the only
        # usable position. Force q=0 to avoid blending NaN positions.
        if not self.has_pm6_pos:
            return 0.0

        if self.q is not None:
            return float(self.q)

        r = random.random()
        if r < 0.1:
            return 1.0
        if r < 0.9:
            return 0.0
        return random.uniform(0.4, 0.6)

    @staticmethod
    def _blend_positions(pos: np.ndarray, pos_bad: np.ndarray, q: float) -> np.ndarray:
        if q == 1.0:
            return pos
        if q == 0.0:
            return pos_bad

        q_arr = np.asarray(q, dtype=pos.dtype)
        one = np.asarray(1.0, dtype=pos.dtype)
        return q_arr * pos + (one - q_arr) * pos_bad

    def __getitem__(self, idx: int):
        g = self.shard.get(self.indices[int(idx)])

        pos = g["pos"]                          # good conformer [N,3]
        pos_bad = g["pos_rdkit"]                # bad conformer [N,3]
        q = self._sample_q()
        pos_in = self._blend_positions(pos, pos_bad, q)

        rrwp_nodes = g["rrwp_nodes"]
        rrwp_edges = g["rrwp_edges"]
        log_deg = g["log_deg"]

        data = Data(
            x=torch.from_numpy(g["node_float"]).to(dtype=torch.float32, copy=False),
            node_codes=torch.from_numpy(g["node_codes"]).to(dtype=torch.long, copy=False),
            pos=torch.from_numpy(pos).to(dtype=torch.float32, copy=False),
            pos_in=torch.from_numpy(pos_in).to(dtype=torch.float32, copy=False),
            edge_index=torch.from_numpy(g["edge_index"]).to(dtype=torch.long, copy=False),
            edge_codes=torch.from_numpy(g["edge_codes"]).to(dtype=torch.long, copy=False),
            rrwp_nodes=(
                torch.from_numpy(rrwp_nodes).to(dtype=torch.float32, copy=False)
                if rrwp_nodes is not None else None
            ),
            rrwp_edges=(
                torch.from_numpy(rrwp_edges).to(dtype=torch.float32, copy=False)
                if rrwp_edges is not None else None
            ),
            log_deg=torch.from_numpy(log_deg).to(dtype=torch.float32, copy=False) if log_deg is not None else None,
        )
        if g["y_node"] is not None:
            data.node_targets = torch.from_numpy(g["y_node"]).to(dtype=torch.float32, copy=False)
        return data, torch.from_numpy(self.Y_graph_float[int(idx)])


class NodeBudgetBatchSampler(Sampler[List[int]]):
    """Greedy node-budget batching with linear complexity and optional DDP splitting.

    Supports an optional edge budget to prevent OOM on graphs with high edge density
    (e.g. when edges are pre-symmetrized during preprocessing).
    """

    def __init__(
        self,
        dataset: CSRMapDataset,
        max_nodes: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        if max_nodes <= 0:
            raise ValueError("max_nodes must be a positive integer")
        self.dataset = dataset
        self.max_nodes = int(max_nodes)
        self.shuffle = shuffle
        self.generator = generator
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)
        self._cached_batches: Optional[List[List[int]]] = None

    def __iter__(self):
        self._ensure_batches()
        try:
            batches = self._cached_batches or []
            if self.world_size > 1:
                batches = batches[self.rank :: self.world_size]
            for batch in batches:
                yield batch
        finally:
            self._cached_batches = None

    def __len__(self) -> int:
        """Return number of batches. Approximate until first iteration completes."""
        if self._cached_batches is not None:
            n_batches = len(self._cached_batches)
        else:
            # Estimate without triggering a full batch build.
            # This may differ from the actual count yielded by __iter__
            # because greedy bin-packing produces a different number of
            # batches than a simple ceiling division.
            total_nodes = sum(self.dataset.node_counts)
            n_batches = max(1, (total_nodes + self.max_nodes - 1) // self.max_nodes)
        if self.world_size > 1:
            return (n_batches + self.world_size - 1) // self.world_size
        return n_batches

    def _ensure_batches(self) -> None:
        if self._cached_batches is None:
            self._cached_batches = self._build_batches()

    def _build_batches(self) -> List[List[int]]:
        n = len(self.dataset)
        if n == 0:
            return []

        graph_sizes = list(enumerate(self.dataset.node_counts))

        if self.shuffle and n > 1:
            perm = self._randperm(n)
            graph_sizes = [graph_sizes[i] for i in perm]

        batches: List[List[int]] = []
        current: List[int] = []
        current_nodes = 0

        for idx, node_count in graph_sizes:
            node_count = int(node_count)

            # Oversized single graph: flush current batch, emit as its own batch
            if node_count >= self.max_nodes:
                if current:
                    batches.append(current)
                    current = []
                    current_nodes = 0
                batches.append([idx])
                continue

            # Check if adding this graph would exceed the node budget
            if current and current_nodes + node_count > self.max_nodes:
                batches.append(current)
                current = []
                current_nodes = 0

            current.append(idx)
            current_nodes += node_count

        if current:
            batches.append(current)

        if self.shuffle and len(batches) > 1:
            perm = self._randperm(len(batches))
            batches = [batches[i] for i in perm]

        if self.world_size > 1 and batches:
            remainder = len(batches) % self.world_size
            if remainder:
                pad = self.world_size - remainder
                # Sample random batches for padding to reduce bias (instead of duplicating last)
                pad_indices = [i % len(batches) for i in range(pad)]
                batches.extend([list(batches[i]) for i in pad_indices])

        return batches

    def _randperm(self, n: int) -> List[int]:
        if self.generator is not None:
            return torch.randperm(n, generator=self.generator).tolist()
        return torch.randperm(n).tolist()


class FixedCountBatchSampler(Sampler[List[int]]):
    """Cycling batch sampler that emits fixed-size batches indefinitely.

    Used for resident datasets (PCBA) that pair up with PM6 shards: whenever
    the PM6 dataloader yields a batch, we draw a fixed number of molecules from
    each resident dataset. When the dataset is exhausted we reshuffle and continue.

    Supports DDP splitting via interleaved slicing (``indices[rank::world_size]``).

    Args:
        dataset_size: Number of molecules in the resident dataset.
        batch_size: Fixed molecules per batch.
        num_batches: How many batches to yield per epoch (matches PM6 batches per shard).
        shuffle: Reshuffle after each exhaustion (recommended).
        generator: Optional torch.Generator for reproducibility.
        world_size: DDP world size.
        rank: DDP rank.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        num_batches: int,
        shuffle: bool = True,
        generator: Optional[torch.Generator] = None,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.shuffle = shuffle
        self.generator = generator
        self.world_size = max(1, int(world_size))
        self.rank = int(rank)

    def __iter__(self):
        # Split batch_size across ranks so each rank sees `batch_size // world_size`
        # unique molecules per step. We pre-build num_batches * world_size
        # per-rank batches and interleave across ranks — same idiom as NodeBudgetBatchSampler.
        per_rank_bs = max(1, self.batch_size // self.world_size)
        total_batches = self.num_batches * self.world_size

        # Lazily reshuffle the index pool as we exhaust it.
        pool: List[int] = []

        def refill():
            if self.shuffle and self.dataset_size > 1:
                if self.generator is not None:
                    perm = torch.randperm(self.dataset_size, generator=self.generator).tolist()
                else:
                    perm = torch.randperm(self.dataset_size).tolist()
            else:
                perm = list(range(self.dataset_size))
            return perm

        pool = refill()
        batches: List[List[int]] = []
        while len(batches) < total_batches:
            if len(pool) < per_rank_bs:
                pool.extend(refill())
            batch = pool[:per_rank_bs]
            pool = pool[per_rank_bs:]
            batches.append(batch)

        # Take this rank's slice
        my_batches = batches[self.rank :: self.world_size]
        for batch in my_batches:
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def load_pm6_dataset(
    split: str,
    load_from: str,
    shard_id: Optional[int] = None,
    subset_ratio: Optional[float] = None,
    walk_len: Optional[int] = None,
    q: Optional[float] = None,
    use_node_labels: bool = False,
    load_to_memory: bool = False,
    shm_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[CSRMapDataset, CSRMapDataset]:

    assert split in ["train", "val"]
    assert shard_id is not None if split == "train" else True, "train split requires shard_id"

    prefixes = _find_shard_prefixes(load_from)
    if len(prefixes) < 2:
        raise ValueError(f"Expected >= 2 shards in {load_from}, found {len(prefixes)}")

    prefixes = sorted(prefixes, key=lambda p: int(os.path.basename(p)))

    if split == "val":
        prefix = prefixes[-1]  # reserve last shard for validation
    else:
        prefix = prefixes[shard_id % (len(prefixes) - 1)]

    return CSRMapDataset(
        prefix,
        subset_ratio=subset_ratio,
        walk_len=walk_len,
        q=q,
        use_node_labels=use_node_labels,
        load_to_memory=load_to_memory,
        shm_metadata=shm_metadata
    )


def _default_stats_path() -> str:
    """Return the path to the bundled PM6 stats file."""
    import importlib.resources
    with importlib.resources.as_file(
        importlib.resources.files("monroe.assets").joinpath("pm6_stats.json")
    ) as p:
        return str(p)


def build_task_dict(
    use_node_labels: bool = False,
    structure_loss: bool = False,
    norm_stats_path: str | None = None,
    beta_nll_softplus: bool = True,
    per_task_ordinal_k: bool = False,
    exclude_loss_types: list[str] | None = None,
    exclude_task_states: list[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Build task dictionary from stats.json with loss-driven configuration.

    Each task in stats.json should have a "loss" key specifying one of:
    - huber: Huber loss, MAE metric, 1 output
    - log_huber: Log-space Huber loss, MAE metric, 1 output
    - log_ratio_squared: Logit-space MSE for [0,1] values, 1 output
    - beta_nll: Beta distribution NLL for [0,1] values, 2 outputs
    - ordinal: Ordinal regression for count data, 1 output (accuracy metric)

    Args:
        per_task_ordinal_k: If True, use per-task K (n_unique from stats) for
            each ordinal loss instead of a global K=max(n_unique)+1=82.
        exclude_loss_types: List of loss type names to skip (e.g. ["ordinal"]).
        exclude_task_states: List of electronic states to exclude (e.g. ["anion", "cation", "T0"]).
    """
    # Loss name -> (loss_class, metric_class, n_outputs, metric_name, higher_is_better)
    LOSS_CONFIG = {
        "huber":             (SparseHuberLoss,          SparseL1,        1, "mae", False),
        "log_huber":         (SparseLogHuberLoss,       SparseLogMAE,    1, "mae", False),
        "log_ratio_squared": (SparseLogRatioSquaredLoss, SparseLogitMAE, 1, "mae", False),
        "beta_nll":          (SparseBetaNLLLoss,        SparseBetaMAE,   2, "mae", False),
        "ordinal":           (SparseOrdinalLoss,        SparseOrdinalAcc, 1, "acc", True),
    }

    if norm_stats_path is None:
        norm_stats_path = _default_stats_path()
    assert os.path.exists(norm_stats_path), f"norm_stats_path must exist: {norm_stats_path}"
    with open(norm_stats_path) as f:
        label_stats = json.load(f)

    # Derive ordinal K: either per-task from n_unique, or global max
    global_ordinal_K = max(
        (s.get("n_unique", 0) for s in label_stats.values() if s.get("loss") == "ordinal"),
        default=0,
    ) + 1

    exclude = set(exclude_loss_types or [])
    if exclude:
        print(f"[build_task_dict] Excluding loss types: {exclude}")

    ex_states = set(exclude_task_states or [])
    if ex_states:
        print(f"[build_task_dict] Excluding task states: {ex_states}")

    task_dict: Dict[str, Dict[str, Any]] = {}

    for task, stats in label_stats.items():
        is_node_task = task.startswith("node_")
        if is_node_task and not use_node_labels:
            continue

        # Skip tasks belonging to excluded electronic states
        if ex_states and any(f"_{state}_" in task for state in ex_states):
            continue

        # Get loss type from config
        loss_name = stats.get("loss")
        if loss_name is None:
            raise ValueError(f"Task '{task}' missing 'loss' key in stats.json")
        if loss_name in exclude:
            continue
        if loss_name not in LOSS_CONFIG:
            raise ValueError(f"Unknown loss type '{loss_name}' for task '{task}'. "
                           f"Available: {list(LOSS_CONFIG.keys())}")

        loss_cls, metric_cls, n_outputs, metric_name, higher_is_better = LOSS_CONFIG[loss_name]
        task_type = "node_level" if is_node_task else "graph_level"

        if loss_name == "ordinal":
            K = stats.get("n_unique", global_ordinal_K) if per_task_ordinal_k else global_ordinal_K
            loss_kwargs = {"num_classes": K}
        elif loss_name == "beta_nll":
            loss_kwargs = {"use_softplus": beta_nll_softplus}
        else:
            loss_kwargs = {}
        metric_kwargs = {"num_classes": K} if loss_name == "ordinal" else {}

        entry = dict(
            metric_name=metric_name,
            n_outputs=n_outputs,
            metrics_fn=metric_cls(**metric_kwargs),
            loss_fn=loss_cls(**loss_kwargs),
            higher_is_better=higher_is_better,
            task_type=task_type,
            loss_type=loss_name,
        )

        task_dict[task] = entry

    if structure_loss:
        task_dict["structure_pred"] = dict(
            metric_name="mae",
            n_outputs=3,  # Always 3D coordinates for Kabsch alignment
            metrics_fn=PairwiseDistanceDummyMetric(),
            loss_fn=KabschAlignedCoordLoss(),
            higher_is_better=False,
            task_type="node_level",
            loss_type="kabsch",
        )

    return task_dict


def add_multisource_tasks(
    task_dict: Dict[str, Dict[str, Any]],
    pcba_n_assays: Optional[int] = None,
    pcba_per_assay: bool = False,
    pcba_assay_names: Optional[list] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extend ``task_dict`` with PCBA dataset-level tasks.

    These tasks are NOT served by ``BatchedDecoders``; they route through
    ``dataset_heads`` in ``AbsWeighting.forward``. Each head produces a single
    wide output tensor; loss and metric aggregate across all outputs.

    Focal loss uses ``SparseFocalLoss``'s own defaults (gamma=2.0, alpha=0.25),
    which are the published values.

    Args:
        task_dict: Existing PM6 task dict (will be mutated).
        pcba_n_assays: Number of PCBA bioassays (columns). None disables PCBA task.
        pcba_per_assay: If True, emit one task entry per surviving PCBA assay
            (each with ``n_outputs=1`` and a ``pcba_assay_idx`` field). STCH then
            weights each assay independently. If False (default), emit a single
            ``pcba`` task that aggregates all assays via mean focal loss.
        pcba_assay_names: Required when ``pcba_per_assay=True``. Ordered list of
            assay column names (from ``Y_graph_cols.json``) used to name the
            per-assay task entries as ``pcba/<assay_id>``.

    Returns:
        The mutated task_dict with new entries added in-place.
    """
    if pcba_n_assays is not None:
        if pcba_per_assay:
            if pcba_assay_names is None or len(pcba_assay_names) != pcba_n_assays:
                raise ValueError(
                    f"pcba_per_assay=True requires pcba_assay_names of length "
                    f"pcba_n_assays={pcba_n_assays}, got "
                    f"{None if pcba_assay_names is None else len(pcba_assay_names)}"
                )
            for i, assay_name in enumerate(pcba_assay_names):
                task_dict[f"pcba/{assay_name}"] = dict(
                    metric_name="auroc",
                    n_outputs=1,
                    # Shared-shape losses/metrics: one Python instance per task,
                    # the linear-probe weight matrix is still shared via the
                    # single dataset_heads["pcba"] module — these slots only
                    # carve out per-assay views for STCH weighting + logging.
                    loss_fn=SparseFocalLoss(),
                    metrics_fn=SparseFocalAUROC(min_positives=10),
                    higher_is_better=True,
                    task_type="graph_level",
                    loss_type="focal",
                    pcba_assay_idx=i,
                )
        else:
            task_dict["pcba"] = dict(
                metric_name="auroc",
                n_outputs=pcba_n_assays,
                loss_fn=SparseFocalLoss(),
                metrics_fn=SparseFocalAUROC(min_positives=10),
                higher_is_better=True,
                task_type="graph_level",
                loss_type="focal",
            )
    return task_dict



def dataloader_factory(
    task_dict: Dict[str, Dict[str, Any]],
    dataset_dir: str,
    split: str,
    batch_node_budget: int,
    subset_ratio: Optional[float] = None,
    shard_id: Optional[int] = None,
    n_workers: Optional[int] = 0,
    seed: Optional[int] = None,
    walk_len: Optional[int] = None,
    q: Optional[float] = None,
    use_node_labels: bool = False,
    load_to_memory: bool = False,
    world_size: int = 1,
    rank: int = 0,
    shm_metadata: Optional[Dict[str, Any]] = None,
) -> DataLoader:

    assert split in ("train", "val"), "split must be 'train' or 'val'"

    ds = load_pm6_dataset(
        split=split,
        load_from=dataset_dir,
        shard_id=shard_id,
        subset_ratio=subset_ratio,
        walk_len=walk_len,
        q=q,
        use_node_labels=use_node_labels,
        load_to_memory=load_to_memory,
        shm_metadata=shm_metadata,
    )

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed + (shard_id or 0)))

    sampler = NodeBudgetBatchSampler(
        dataset=ds,
        max_nodes=int(batch_node_budget),
        shuffle=(split == "train"),
        generator=generator,
        world_size=world_size,
        rank=rank,
    )
    return DataLoader(
        ds,
        batch_sampler=sampler,
        pin_memory=True,
        num_workers=n_workers,
        persistent_workers=bool(n_workers and n_workers > 0),
        prefetch_factor=4 if (n_workers or 0) > 0 else None,
    )


# -----------------------------------------------------------------------------
# Multi-source (resident) dataset utilities
# -----------------------------------------------------------------------------


@dataclass
class ResidentDataset:
    """A dataset loaded once into shared memory and reused every PM6 shard.

    Attributes:
        name: Dataset identifier (currently only ``pcba``).
        dataset: The wrapping ``CSRMapDataset`` (uses ``has_pm6_pos=False``).
        shm_metadata: SHM metadata dict (never cleaned up until training ends).
    """

    name: str
    dataset: "CSRMapDataset"
    shm_metadata: Optional[Dict[str, Any]]


def _load_resident_shard(
    name: str,
    shard_prefix: str,
    walk_len: Optional[int],
    load_to_memory: bool,
    dist_cfg: "DistributedConfig",
) -> ResidentDataset:
    """Load one resident shard into SHM (local rank 0) and broadcast metadata.

    Every rank ends up with a ``CSRMapDataset`` backed by the same SHM blocks.
    """
    shm_metadata: Optional[Dict[str, Any]] = None
    if load_to_memory:
        if dist_cfg.is_local_main:
            shm_metadata = load_shard_to_shm(
                prefix=shard_prefix,
                walk_len=walk_len,
                use_node_labels=False,
            )
        # All ranks must participate in the broadcast — do NOT guard with rank check.
        if dist_cfg.is_distributed:
            payload = [shm_metadata]
            dist.broadcast_object_list(payload, src=0)
            shm_metadata = payload[0]

    ds = CSRMapDataset(
        shard_prefix,
        walk_len=walk_len,
        use_node_labels=False,
        load_to_memory=load_to_memory,
        shm_metadata=shm_metadata,
        has_pm6_pos=False,  # PCBA has no PM6 coords — forces q=0 and uses RDKit pos.
    )
    return ResidentDataset(name=name, dataset=ds, shm_metadata=shm_metadata)


def load_resident_datasets(
    hp,
    dist_cfg: "DistributedConfig",
) -> Dict[str, ResidentDataset]:
    """Load all configured resident datasets (currently PCBA) once into SHM.

    The loaded SHM is never cleaned up during training (contrast with PM6 shard
    rotation). This function returns a dict so callers can build per-dataset
    dataloaders and sample mini-batches every step.

    Args:
        hp: Hyperparameters namespace with ``pretrain.pcba_dir`` and
            ``encoder.walk_len``, ``pretrain.load_to_memory``.
        dist_cfg: Distributed configuration.

    Returns:
        Dict mapping dataset name -> ResidentDataset. Empty if no paths configured.
    """
    out: Dict[str, ResidentDataset] = {}
    walk_len = hp.encoder.walk_len
    load_to_memory = hp.pretrain.load_to_memory

    cfg_map = {
        "pcba": getattr(hp.pretrain, "pcba_dir", None),
    }
    for name, root in cfg_map.items():
        if not root:
            continue
        prefixes = _find_shard_prefixes(root)
        if not prefixes:
            raise FileNotFoundError(
                f"No CSR shards found for resident dataset '{name}' under {root}"
            )
        if len(prefixes) > 1:
            # The resident path loads a single shard into SHM and keeps it
            # there for the whole run. Multi-shard residency would need the
            # loader to concatenate or rotate, which is not implemented.
            # Prep the dataset with --num-train-shards 1 (the default for
            # pcba_prep.py) or extend this function before enabling multi-shard.
            raise RuntimeError(
                f"Resident dataset '{name}' under {root} has {len(prefixes)} "
                f"shards; only a single shard is supported. Re-run prep with "
                f"--num-train-shards 1."
            )
        prefix = prefixes[0]
        out[name] = _load_resident_shard(
            name=name,
            shard_prefix=prefix,
            walk_len=walk_len,
            load_to_memory=load_to_memory,
            dist_cfg=dist_cfg,
        )
    return out


def build_resident_dataloaders(
    residents: Dict[str, ResidentDataset],
    mols_per_batch: Dict[str, int],
    num_batches: int,
    world_size: int,
    rank: int,
    n_workers: int = 0,
    seed: Optional[int] = None,
) -> Dict[str, DataLoader]:
    """Wrap each ``ResidentDataset`` in a DataLoader with a ``FixedCountBatchSampler``.

    Args:
        residents: Output of ``load_resident_datasets``.
        mols_per_batch: Dict mapping dataset name -> molecules per batch.
        num_batches: How many batches to yield per PM6 shard (matches PM6 dataloader).
        world_size, rank: DDP coordinates.
        n_workers: DataLoader workers; 0 since data is in SHM already.
        seed: Optional seed for reproducible shuffling.

    Returns:
        Dict name -> DataLoader with the same number of batches as PM6.
    """
    loaders: Dict[str, DataLoader] = {}
    for name, res in residents.items():
        bs = int(mols_per_batch.get(name, 0))
        if bs <= 0:
            continue
        gen = None
        if seed is not None:
            gen = torch.Generator()
            gen.manual_seed(int(seed) + hash(name) % 10_000)
        sampler = FixedCountBatchSampler(
            dataset_size=len(res.dataset),
            batch_size=bs,
            num_batches=num_batches,
            shuffle=True,
            generator=gen,
            world_size=world_size,
            rank=rank,
        )
        loaders[name] = DataLoader(
            res.dataset,
            batch_sampler=sampler,
            pin_memory=True,
            num_workers=n_workers,
            persistent_workers=bool(n_workers and n_workers > 0),
            prefetch_factor=4 if (n_workers or 0) > 0 else None,
        )
    return loaders


def build_pcba_val_dataloader(
    shard_dir: str,
    batch_node_budget: int,
    walk_len: int,
    world_size: int = 1,
    rank: int = 0,
    n_workers: int = 0,
) -> DataLoader:
    """One-shot DataLoader over the PCBA val CSR shard, for PCBA-only val eval.

    Does not load into shared memory (one-pass read; IO is not the bottleneck
    for val). Uses ``has_pm6_pos=False`` so ``q=0`` and positions come from
    ``POS_RDKIT`` — PCBA has no PM6 coordinates.

    The val dir must contain exactly one CSR shard (merge with
    ``scripts/preprocessing/consolidate_shards.py`` if you have more than one —
    val is already a single shard by default).
    """
    prefixes = _find_shard_prefixes(shard_dir)
    if not prefixes:
        raise FileNotFoundError(f"No CSR shards under {shard_dir}")
    if len(prefixes) > 1:
        raise RuntimeError(
            f"pcba_val_dir {shard_dir} has {len(prefixes)} shards; expected 1. "
            f"Consolidate before enabling val eval."
        )
    ds = CSRMapDataset(
        prefixes[0],
        walk_len=walk_len,
        use_node_labels=False,
        q=0.0,
        load_to_memory=False,
        has_pm6_pos=False,
    )
    sampler = NodeBudgetBatchSampler(
        dataset=ds,
        max_nodes=int(batch_node_budget),
        shuffle=False,
        world_size=world_size,
        rank=rank,
    )
    return DataLoader(
        ds,
        batch_sampler=sampler,
        pin_memory=True,
        num_workers=n_workers,
        persistent_workers=bool(n_workers and n_workers > 0),
    )


def concat_multi_source(
    pm6_batch,
    pm6_y: torch.Tensor,
    extra: Dict[str, Tuple[Any, torch.Tensor]],
) -> Tuple[Any, Dict[str, torch.Tensor]]:
    """Concatenate a PM6 PyG Batch with one mini-batch per resident dataset.

    Each per-dataset target tensor is NaN-padded to cover the full combined batch
    so that sparse loss functions mask out non-relevant molecules automatically.

    Args:
        pm6_batch: PyG Batch from the PM6 dataloader.
        pm6_y: PM6 graph-level target tensor ``[B_pm6, num_pm6_cols]``.
        extra: Dict mapping dataset name -> (pyg_batch, y_tensor of shape [B_ds, cols_ds]).

    Returns:
        (combined_batch, y_dict)
            combined_batch: PyG Batch concatenating all molecules in order
                [pm6, (sorted keys of extra...)]. PM6 molecules carry valid POS;
                others have NaN POS (flows through from shards).
            y_dict: {"pm6": [B_total, pm6_cols], "pcba": [B_total, pcba_cols], ...}
                Each tensor is NaN outside its owning dataset's row range. Loss
                functions see these as regular sparse targets.
    """
    from torch_geometric.data import Batch

    B_pm6 = int(pm6_y.shape[0])
    sizes = {"pm6": B_pm6}
    sorted_extra_names = sorted(extra.keys())
    for name in sorted_extra_names:
        sizes[name] = int(extra[name][1].shape[0])
    B_total = sum(sizes.values())

    # Concatenate Data objects for the combined PyG Batch
    all_data = pm6_batch.to_data_list()
    for name in sorted_extra_names:
        sub_batch, _ = extra[name]
        all_data.extend(sub_batch.to_data_list())
    combined = Batch.from_data_list(all_data)

    def _pad(y: torch.Tensor, offset: int) -> torch.Tensor:
        n, c = int(y.shape[0]), int(y.shape[1])
        y_float = y.float() if not y.is_floating_point() else y
        padded = torch.full((B_total, c), float("nan"), dtype=y_float.dtype, device=y_float.device)
        padded[offset:offset + n] = y_float
        return padded

    y_dict: Dict[str, torch.Tensor] = {"pm6": _pad(pm6_y, 0)}
    offset = B_pm6
    for name in sorted_extra_names:
        _, y = extra[name]
        y_dict[name] = _pad(y, offset)
        offset += sizes[name]

    return combined, y_dict

