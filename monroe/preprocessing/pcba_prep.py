#!/usr/bin/env python3
"""PCBA preprocessing: SMILES -> InChI -> benchmark filter -> train/val split -> sharded parquets.

PCBA (PubChem BioAssay) is 1.56M molecules x 1,328 binary bioassays.
Source: https://zenodo.org/records/8024997 (pcba_1328.zip -> PCBA_1328_1564k.parquet).

Output layout:
    out_dir/
        train/
            shards/pcba_shard_000.parquet, ..., pcba_shard_NNN.parquet
        val/
            shards/pcba_shard_000.parquet
        manifest.json
"""
import argparse
import json
import os
from datetime import datetime

import datamol as dm
import pandas as pd

from monroe.preprocessing.pm6_prep import (
    _suppress_rdkit_logs,
    get_benchmark_inchis,
    initialize_duckdb,
    register_benchmark_inchis,
)


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def _inchi_worker(smiles_list):
    """Worker function for parallel InChI conversion. Each process initializes its own RDKit logs."""
    _suppress_rdkit_logs()
    out = []
    for s in smiles_list:
        try:
            out.append(dm.to_inchi(s) if s else None)
        except Exception:
            out.append(None)
    return out


def canonicalize_to_inchi(
    df: pd.DataFrame,
    smiles_col: str = "SMILES",
    n_jobs: int = None,
    chunk_size: int = 5000,
) -> pd.DataFrame:
    """Add an `inchi` column derived from SMILES. Drops rows where conversion fails.

    Parallelizes via multiprocessing when ``n_jobs > 1`` (default: all CPUs).
    """
    n = len(df)
    log(f"Canonicalizing {n:,} SMILES -> InChI...")

    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 1) - 1)

    smiles = df[smiles_col].tolist()
    if n_jobs <= 1 or n <= chunk_size:
        _suppress_rdkit_logs()
        inchis = _inchi_worker(smiles)
    else:
        from concurrent.futures import ProcessPoolExecutor
        chunks = [smiles[i:i + chunk_size] for i in range(0, n, chunk_size)]
        inchis = []
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for idx, result in enumerate(ex.map(_inchi_worker, chunks)):
                inchis.extend(result)
                done = min((idx + 1) * chunk_size, n)
                if (idx + 1) % 10 == 0 or done == n:
                    log(f"  {done:,}/{n:,} done")

    df = df.copy()
    df["inchi"] = inchis
    n_before = len(df)
    df = df[df["inchi"].notna()].reset_index(drop=True)
    log(f"InChI conversion: {len(df):,}/{n_before:,} succeeded")
    return df


def ensure_output_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(out_dir, "train", "shards")
    val_dir = os.path.join(out_dir, "val", "shards")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "pcba_prep.duckdb")
    return db_path, train_dir, val_dir


def register_pcba(con, df: pd.DataFrame, assay_cols: list):
    """Register the cleaned PCBA dataframe as a DuckDB table (InChI-deduped, unfiltered)."""
    log(f"Registering PCBA DataFrame ({len(df):,} rows, {len(assay_cols)} assay cols)...")
    con.register("pcba_raw_df", df)
    con.execute("CREATE OR REPLACE TABLE pcba_raw AS SELECT * FROM pcba_raw_df")
    # Dedup by InChI (keep first row). Benchmark overlap filtering is deferred to
    # split_train_val so overlap molecules can be routed into val.
    con.execute("""
        CREATE OR REPLACE TABLE pcba_dedup AS
        SELECT *
        FROM pcba_raw
        WHERE inchi IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY inchi ORDER BY inchi) = 1
    """)


def split_train_val(con, val_frac: float, seed: int, num_train_shards: int):
    """Hash-based train/val split with benchmark-overlap molecules routed into val.

    Benchmark-overlap molecules are excluded from training (no downstream
    contamination) but added to val so they contribute a monitoring signal on the
    exact molecules used in Polaris/MoleculeACE test sets.
    """
    val_threshold_pct = int(val_frac * 100)
    log(f"Splitting train/val (val_frac={val_frac:.2%}), benchmark overlap -> val")
    con.execute(f"""
        CREATE OR REPLACE TABLE pcba_split AS
        SELECT
            d.*,
            (b.inchi IS NOT NULL
                OR (abs(hash(d.inchi || '{seed}')) % 100) < {val_threshold_pct}
            ) AS is_val,
            CASE
                WHEN b.inchi IS NOT NULL
                     OR (abs(hash(d.inchi || '{seed}')) % 100) < {val_threshold_pct}
                    THEN 0
                ELSE (abs(hash(d.inchi)) % {num_train_shards})::INTEGER
            END AS shard_id
        FROM pcba_dedup d
        LEFT JOIN benchmark_inchis b USING (inchi)
    """)


def write_shards(con, out_dir: str, assay_cols: list, num_train_shards: int):
    """Write train and val parquet shards."""
    # Columns to keep: inchi, SMILES, assay cols
    cols_sql = ", ".join([f'"{c}"' for c in ["inchi", "SMILES", *assay_cols]])
    width = max(3, len(str(num_train_shards - 1)))

    train_dir = os.path.join(out_dir, "train", "shards")
    val_dir = os.path.join(out_dir, "val", "shards")

    # Val shard
    val_path = os.path.join(val_dir, "pcba_shard_000.parquet")
    log(f"Writing val shard -> {val_path}")
    con.execute(f"""
        COPY (
            SELECT {cols_sql}
            FROM pcba_split
            WHERE is_val = TRUE
            ORDER BY inchi
        ) TO '{val_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD')
    """)

    # Train shards
    log(f"Writing {num_train_shards} train shards -> {train_dir}")
    for sid in range(num_train_shards):
        out_path = os.path.join(train_dir, f"pcba_shard_{sid:0{width}d}.parquet")
        con.execute(f"""
            COPY (
                SELECT {cols_sql}
                FROM pcba_split
                WHERE is_val = FALSE AND shard_id = {sid}
                ORDER BY inchi
            ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD')
        """)
        if (sid + 1) % 10 == 0 or sid == num_train_shards - 1:
            log(f"  {sid + 1}/{num_train_shards} shards written")


def compute_stats(con, assay_cols: list):
    total_before = con.execute("SELECT COUNT(*) FROM pcba_raw").fetchone()[0]
    total_unique = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT inchi FROM pcba_raw WHERE inchi IS NOT NULL)"
    ).fetchone()[0]
    total_dedup = con.execute("SELECT COUNT(*) FROM pcba_dedup").fetchone()[0]

    n_train = con.execute("SELECT COUNT(*) FROM pcba_split WHERE is_val = FALSE").fetchone()[0]
    n_val = con.execute("SELECT COUNT(*) FROM pcba_split WHERE is_val = TRUE").fetchone()[0]
    n_val_overlap = con.execute("""
        SELECT COUNT(*) FROM pcba_split s
        WHERE s.is_val = TRUE AND s.inchi IN (SELECT inchi FROM benchmark_inchis)
    """).fetchone()[0]

    return {
        "num_assays": len(assay_cols),
        "total_rows_before": int(total_before),
        "total_unique_inchi_before": int(total_unique),
        "total_after_dedup": int(total_dedup),
        "train_rows": int(n_train),
        "val_rows": int(n_val),
        "val_rows_from_benchmark_overlap": int(n_val_overlap),
    }


def compute_assay_stats(con, assay_cols: list) -> list:
    """Per-assay label statistics computed over TRAIN rows only.

    Labels are binary float32 in {0.0, 1.0} with NaN for missing. For each assay
    returns n_pos, n_neg, n_valid (= n_pos + n_neg), n_nan, and pos_frac.

    All aggregates are emitted in a single SQL pass over pcba_split, so the cost
    is one scan of the training rows regardless of assay count.
    """
    log(f"Computing per-assay stats over training rows ({len(assay_cols)} assays)...")
    total_train = con.execute(
        "SELECT COUNT(*) FROM pcba_split WHERE is_val = FALSE"
    ).fetchone()[0]

    # Two aggregates per assay: counts of strictly 1.0 and strictly 0.0.
    # NaN/NULL fail both predicates and thus fall out of n_valid automatically.
    parts = []
    for c in assay_cols:
        q = f'"{c}"'
        parts.append(f'SUM(CASE WHEN {q} = 1.0 THEN 1 ELSE 0 END)')
        parts.append(f'SUM(CASE WHEN {q} = 0.0 THEN 1 ELSE 0 END)')
    query = f"SELECT {', '.join(parts)} FROM pcba_split WHERE is_val = FALSE"
    row = con.execute(query).fetchone()

    stats = []
    for i, c in enumerate(assay_cols):
        n_pos = int(row[2 * i] or 0)
        n_neg = int(row[2 * i + 1] or 0)
        n_valid = n_pos + n_neg
        n_nan = int(total_train) - n_valid
        pos_frac = (n_pos / n_valid) if n_valid > 0 else 0.0
        stats.append({
            "assay_id": c,
            "n_valid": n_valid,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_nan": n_nan,
            "pos_frac": pos_frac,
        })
    return stats


def filter_assays(stats: list, min_pos: int, min_neg: int, min_valid: int):
    """Split per-assay stats into (kept, dropped).

    An assay is dropped if it fails ANY threshold. Dropped entries are annotated
    with ``drop_reason`` summarising which thresholds they missed.
    """
    kept = []
    dropped = []
    for s in stats:
        reasons = []
        if s["n_pos"] < min_pos:
            reasons.append(f"n_pos={s['n_pos']}<{min_pos}")
        if s["n_neg"] < min_neg:
            reasons.append(f"n_neg={s['n_neg']}<{min_neg}")
        if s["n_valid"] < min_valid:
            reasons.append(f"n_valid={s['n_valid']}<{min_valid}")
        if reasons:
            dropped.append({**s, "drop_reason": ";".join(reasons)})
        else:
            kept.append(s)
    return kept, dropped


def log_assay_histogram(stats: list, kept: list, dropped: list) -> None:
    """Log compact quantile summaries so the operator can eyeball thresholds."""
    n_total = len(stats)
    if n_total == 0:
        return
    n_valid_sorted = sorted(s["n_valid"] for s in stats)
    n_pos_sorted = sorted(s["n_pos"] for s in stats)
    pos_fracs_sorted = sorted(s["pos_frac"] for s in stats if s["n_valid"] > 0)

    def q(arr, p):
        if not arr:
            return 0
        i = min(len(arr) - 1, max(0, int(p * (len(arr) - 1))))
        return arr[i]

    log(f"Per-assay stats over {n_total:,} input assays:")
    log(
        f"  n_valid:  min={n_valid_sorted[0]:,}  "
        f"p25={q(n_valid_sorted, 0.25):,}  "
        f"median={q(n_valid_sorted, 0.5):,}  "
        f"p75={q(n_valid_sorted, 0.75):,}  "
        f"max={n_valid_sorted[-1]:,}"
    )
    log(
        f"  n_pos:    min={n_pos_sorted[0]:,}  "
        f"p25={q(n_pos_sorted, 0.25):,}  "
        f"median={q(n_pos_sorted, 0.5):,}  "
        f"p75={q(n_pos_sorted, 0.75):,}  "
        f"max={n_pos_sorted[-1]:,}"
    )
    if pos_fracs_sorted:
        log(
            f"  pos_frac: min={pos_fracs_sorted[0]:.4f}  "
            f"p25={q(pos_fracs_sorted, 0.25):.4f}  "
            f"median={q(pos_fracs_sorted, 0.5):.4f}  "
            f"p75={q(pos_fracs_sorted, 0.75):.4f}  "
            f"max={pos_fracs_sorted[-1]:.4f}"
        )
    log(f"  kept:     {len(kept):,} / {n_total:,}    dropped: {len(dropped):,}")

    # Drop-reason breakdown
    if dropped:
        from collections import Counter
        reason_counts = Counter()
        for d in dropped:
            for r in d["drop_reason"].split(";"):
                # Bucket by threshold type, stripping numerics for grouping
                key = r.split("=")[0]
                reason_counts[key] += 1
        log(f"  drop_reason buckets: {dict(reason_counts)}")


def write_pcba_stats(out_dir: str, kept: list, dropped: list) -> str:
    """Write pcba_stats.json containing every input assay with a ``kept`` flag."""
    all_entries = [{**s, "kept": True, "drop_reason": None} for s in kept]
    all_entries += [{**s, "kept": False} for s in dropped]
    all_entries.sort(key=lambda d: d["assay_id"])
    path = os.path.join(out_dir, "pcba_stats.json")
    with open(path, "w") as f:
        json.dump(all_entries, f, indent=2)
    log(f"Wrote per-assay stats -> {path}")
    return path


if __name__ == "__main__":
    """
    Example:
    python -m monroe.preprocessing.pcba_prep \
        --pcba-parquet data/pcba_raw/pcba_1328/PCBA_1328_1564k.parquet \
        --out-dir data/pcba_monroe \
        --num-train-shards 20 \
        --val-frac 0.10 \
        --seed 1

    The default filter thresholds reproduce the published dataset: 1,089 of the
    1,328 assays are kept.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pcba-parquet", required=True, help="PCBA_1328_1564k.parquet from Zenodo")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--num-train-shards", type=int, default=20,
        help="Number of train parquet shards. Sharding exists so the CSR build "
             "step can be parallelised across shards; they are consolidated back "
             "into a single resident shard by consolidate_shards.py before "
             "training, so this value does not affect the trained model.",
    )
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8)))
    # Assay filtering thresholds — applied to training rows only, before writing shards.
    parser.add_argument(
        "--min-pos-per-assay", type=int, default=100,
        help="Drop assays with fewer than this many positive training examples.",
    )
    parser.add_argument(
        "--min-neg-per-assay", type=int, default=100,
        help="Drop assays with fewer than this many negative training examples.",
    )
    parser.add_argument(
        "--min-valid-per-assay", type=int, default=1000,
        help="Drop assays with fewer than this many non-NaN training examples.",
    )
    parser.add_argument(
        "--min-kept-assays", type=int, default=50,
        help="Sanity floor: error out if filtering leaves fewer than this many assays.",
    )
    args = parser.parse_args()

    db_path, train_dir, val_dir = ensure_output_dirs(args.out_dir)

    # Load PCBA parquet, identify assay columns
    log(f"Loading PCBA parquet: {args.pcba_parquet}")
    df = pd.read_parquet(args.pcba_parquet)
    log(f"Loaded: {df.shape}")

    assay_cols = [c for c in df.columns if c.startswith("assayID-")]
    log(f"Identified {len(assay_cols)} assay columns")

    # Keep only needed columns (SMILES + assays). CID/SID/Unnamed: 0 dropped.
    df = df[["SMILES", *assay_cols]].copy()

    # Downcast assay cols to float32 (they're binary, but NaN means missing)
    for c in assay_cols:
        df[c] = df[c].astype("float32")

    # Canonicalize SMILES -> InChI
    df = canonicalize_to_inchi(df, smiles_col="SMILES")

    # DuckDB pipeline
    benchmark_inchis = get_benchmark_inchis()
    con = initialize_duckdb(db_path, args.threads)
    register_benchmark_inchis(con, benchmark_inchis)
    register_pcba(con, df, assay_cols)
    split_train_val(con, args.val_frac, args.seed, args.num_train_shards)

    stats = compute_stats(con, assay_cols)
    log(
        f"Rows: before={stats['total_rows_before']:,}, unique={stats['total_unique_inchi_before']:,}, "
        f"after_dedup={stats['total_after_dedup']:,}, "
        f"train={stats['train_rows']:,}, val={stats['val_rows']:,} "
        f"(of which {stats['val_rows_from_benchmark_overlap']:,} routed from benchmark overlap)"
    )

    # Per-assay stats + filtering (training rows only).
    assay_stats = compute_assay_stats(con, assay_cols)
    kept_stats, dropped_stats = filter_assays(
        assay_stats,
        min_pos=args.min_pos_per_assay,
        min_neg=args.min_neg_per_assay,
        min_valid=args.min_valid_per_assay,
    )
    log_assay_histogram(assay_stats, kept_stats, dropped_stats)

    if len(kept_stats) < args.min_kept_assays:
        raise SystemExit(
            f"ERROR: filter left only {len(kept_stats)} assays (< --min-kept-assays "
            f"= {args.min_kept_assays}). Lower the per-assay thresholds or investigate "
            f"the input data."
        )

    kept_assay_cols = [s["assay_id"] for s in kept_stats]
    write_pcba_stats(args.out_dir, kept_stats, dropped_stats)

    write_shards(con, args.out_dir, kept_assay_cols, args.num_train_shards)
    con.close()

    manifest = {
        **stats,
        "num_train_shards": args.num_train_shards,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "assay_cols": kept_assay_cols,
        "n_input_assays": len(assay_cols),
        "n_kept_assays": len(kept_assay_cols),
        "n_dropped_assays": len(dropped_stats),
        "filter_thresholds": {
            "min_pos_per_assay": args.min_pos_per_assay,
            "min_neg_per_assay": args.min_neg_per_assay,
            "min_valid_per_assay": args.min_valid_per_assay,
        },
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    log("Done.")
