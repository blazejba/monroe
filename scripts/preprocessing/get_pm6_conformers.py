#!/usr/bin/env python3
"""Download PM6 conformer geometries from HuggingFace and export to parquet.

This script downloads the PubChemQC PM6 dataset from HuggingFace, filters for
S0 (ground state) conformers, and exports to parquet files for use with pm6_prep.py.

Dataset: molssiai-hub/pubchemqc-pm6 (pm6opt configuration)
Output:  data/pm6_hf/pm6opt_s0_parquet/*.parquet

Requirements:
    pip install datasets
"""
import os

from datasets import DownloadConfig, load_dataset

if __name__ == "__main__":

    CACHE_DIR = "./data/pm6_hf"
    N_OUTPUT_FILES = 500
    NUM_PROC = 32
    
    ds = load_dataset(
        "molssiai-hub/pubchemqc-pm6",
        name="pm6opt",
        split="train",
        cache_dir=CACHE_DIR,
        data_files="data/pm6opt/train/*.json",
        num_proc=NUM_PROC,
        download_config=DownloadConfig(resume_download=True, max_retries=10),
        trust_remote_code=True,
    )
    _is_state_s0 = lambda example: example.get("state") == "S0"
    ds = ds.filter(_is_state_s0, num_proc=NUM_PROC)
    ds = ds.rename_columns({
        "pubchem-inchi": "inchi",
        "atomic-numbers": "atomic_numbers",
        "coordinates": "coords",
    })
    ds = ds.select_columns(["inchi", "atomic_numbers", "coords", "charge"])

    total_rows = len(ds)
    rows_per_file = total_rows // N_OUTPUT_FILES
    out_dir = os.path.join(CACHE_DIR, "pm6opt_s0_parquet")
    os.makedirs(out_dir, exist_ok=True)

    for shard_idx, start in enumerate(range(0, total_rows, rows_per_file)):
        stop = min(start + rows_per_file, total_rows)
        shard = ds.select(range(start, stop))
        shard_path = os.path.join(out_dir, f"pm6opt_s0_{shard_idx:05d}.parquet")
        print(f"Saving rows {start}:{stop} to {shard_path}")
        shard.to_parquet(shard_path)
