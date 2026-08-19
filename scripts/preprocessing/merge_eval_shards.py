"""Merge sharded parquet files produced by build_eval_graphs.py.

Usage:
    python scripts/preprocessing/merge_eval_shards.py data/polaris/
    python scripts/preprocessing/merge_eval_shards.py data/moleculeace/
    python scripts/preprocessing/merge_eval_shards.py data/polaris/ data/moleculeace/
"""

import argparse
import glob
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def merge_shards(directory: str) -> None:
    parts = sorted(glob.glob(os.path.join(directory, "*.part_*.parquet")))
    if not parts:
        print(f"No shard files found in {directory}")
        return

    # Derive output name from the first shard (strip .part_NNNNN)
    base = os.path.basename(parts[0])
    out_name = base.split(".part_")[0] + ".parquet"
    out_path = os.path.join(directory, out_name)

    tables = [pq.read_table(p) for p in parts]
    merged = pa.concat_tables(tables)

    # Deduplicate by smiles (keep first occurrence)
    smiles = merged.column("smiles").to_pylist()
    seen = set()
    keep = []
    for i, s in enumerate(smiles):
        if s not in seen:
            seen.add(s)
            keep.append(i)
    if len(keep) < len(smiles):
        print(f"  Deduplicating: {len(smiles)} -> {len(keep)} rows")
        merged = merged.take(keep)

    pq.write_table(merged, out_path)
    print(f"  Wrote {len(keep)} rows to {out_path} (from {len(parts)} shards)")


def main():
    parser = argparse.ArgumentParser(description="Merge eval graph shards")
    parser.add_argument("dirs", nargs="+", help="Directories containing .part_*.parquet shards")
    args = parser.parse_args()

    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"Skipping {d}: not a directory", file=sys.stderr)
            continue
        print(f"Merging shards in {d}")
        merge_shards(d)


if __name__ == "__main__":
    main()
