#!/usr/bin/env python3
import argparse
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import datamol as dm
import duckdb
import pandas as pd


def log(msg):
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def _suppress_rdkit_logs():
    """Suppress RDKit logging and return file descriptor info for restoration."""
    try:
        if hasattr(dm, "disable_rdkit_log"):
            dm.disable_rdkit_log()
        from rdkit import RDLogger, rdBase
        RDLogger.DisableLog("rdApp.*")
        rdBase.BlockLogs()
    except Exception:
        pass


def _smiles_to_inchis(smis: list) -> set:
    """Convert SMILES to InChI, suppressing stderr."""
    inchis = set()
    try:
        fd = sys.stderr.fileno()
        saved = os.dup(fd)
        with open(os.devnull, "w") as fnull:
            os.dup2(fnull.fileno(), fd)
            for s in smis:
                try:
                    i = dm.to_inchi(s)
                    if i:
                        inchis.add(i)
                except Exception:
                    pass
    finally:
        try:
            os.dup2(saved, fd)
            os.close(saved)
        except Exception:
            pass
    return inchis

def get_polaris_inchis():
    """Load test split InChIs from all Polaris benchmarks to exclude from pretraining."""
    import polaris as po

    from monroe.eval.polaris import polaris_benchmarks
    
    log("Loading Polaris test split SMILES...")
    _suppress_rdkit_logs()
    
    smis = []
    for benchmark_name in polaris_benchmarks.keys():
        try:
            benchmark = po.load_benchmark(benchmark_name)
            _, test_split = benchmark.get_train_test_split()
            for item in test_split:
                if isinstance(item, tuple):
                    smis.append(str(item[0]))
                else:
                    smis.append(str(item))
        except Exception as e:
            log(f"  Warning: Failed to load {benchmark_name}: {e}")
    
    log(f"Polaris test rows: {len(smis)}; unique SMILES: {len(set(smis))}")
    inchis = _smiles_to_inchis(smis)
    log(f"Unique Polaris InChI: {len(inchis)}")
    return list(inchis)

def get_moleculeace_inchis():
    """Load test split InChIs from all MoleculeACE datasets to exclude from pretraining."""
    from MoleculeACE import Data
    from MoleculeACE import datasets as moleculeace_datasets
    
    log("Loading MoleculeACE test split SMILES...")
    _suppress_rdkit_logs()
    
    smis = []
    for dataset_name in moleculeace_datasets:
        try:
            data = Data(dataset_name)
            smis.extend([str(s) for s in data.smiles_test])
        except Exception as e:
            log(f"  Warning: Failed to load {dataset_name}: {e}")
    
    log(f"MoleculeACE test rows: {len(smis)}; unique SMILES: {len(set(smis))}")
    inchis = _smiles_to_inchis(smis)
    log(f"Unique MoleculeACE InChI: {len(inchis)}")
    return list(inchis)

def get_benchmark_inchis():
    """Get all InChIs from test splits of Polaris and MoleculeACE benchmarks."""
    polaris_inchis = set(get_polaris_inchis())
    moleculeace_inchis = set(get_moleculeace_inchis())
    all_inchis = polaris_inchis | moleculeace_inchis
    log(f"Total unique benchmark InChIs to exclude: {len(all_inchis)} "
        f"(Polaris: {len(polaris_inchis)}, MoleculeACE: {len(moleculeace_inchis)}, "
        f"overlap: {len(polaris_inchis & moleculeace_inchis)})")
    return list(all_inchis)

def find_geom_parquets(geom_dir):
    pat = os.path.join(geom_dir, "**/*.parquet")
    files = glob.glob(pat, recursive=True)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No Parquet files found under --geom-dir='{geom_dir}'. "
            f"Expected pre-exported pm6opt S0 Parquets with columns: inchi, atomic_numbers, coords, charge."
        )
    return pat, len(files)

def register_hf_geom(con, geom_parquet_glob):
    log("Registering HF geometry Parquets...")
    pat_escaped = geom_parquet_glob.replace("'", "''")
    con.execute(f"""
        CREATE OR REPLACE TABLE hf_geom AS
        WITH src AS (
            SELECT
                inchi,
                CAST(atomic_numbers AS INTEGER[]) AS atomic_numbers,
                /* assume DOUBLE[] coords; cast enforces element type */
                CAST(coords AS DOUBLE[])          AS coords,
                CAST(charge AS INTEGER)           AS charge
            FROM parquet_scan('{pat_escaped}')
            WHERE inchi IS NOT NULL
        ),
        scored AS (
            SELECT
                inchi, atomic_numbers, coords, charge,
                hash(
                    to_json(atomic_numbers) ||
                    to_json(coords) ||
                    COALESCE(CAST(charge AS VARCHAR), 'NA')
                ) AS h
            FROM src
        )
        SELECT
            inchi,
            arg_min(atomic_numbers, h) AS atomic_numbers,
            arg_min(coords, h)         AS coords,
            arg_min(charge, h)         AS charge
        FROM scored
        GROUP BY inchi
    """)
    n = con.execute("SELECT COUNT(*) FROM hf_geom").fetchone()[0]
    log(f"HF geometry rows after dedup: {n}")
    return n

def enrich_with_hf_geom(con):
    log("Joining HF geometry into PM6...")
    con.execute("""
        CREATE OR REPLACE VIEW pm6_with_geom AS
        SELECT d.*, g.atomic_numbers, g.coords, g.charge
        FROM pm6_dedup_filtered d
        LEFT JOIN hf_geom g USING (inchi)
    """)

def build_final_enriched_views(con):
    con.execute("""
        CREATE OR REPLACE TABLE geom_all AS
        SELECT * FROM hf_geom
    """)
    con.execute("""
        CREATE OR REPLACE VIEW pm6_enriched AS
        SELECT d.*, gg.atomic_numbers, gg.coords, gg.charge
        FROM pm6_dedup_filtered d
        LEFT JOIN geom_all gg USING (inchi)
    """)
    con.execute("""
        CREATE OR REPLACE VIEW pm6_enriched_geomonly AS
        SELECT *
        FROM pm6_enriched
        WHERE atomic_numbers IS NOT NULL AND coords IS NOT NULL
    """)

def write_one(db_path, out_path, sid, threads, seed):
    con = duckdb.connect(db_path)
    con.execute(f"PRAGMA threads={threads}")
    con.execute(f"""
        COPY (
          SELECT *
          FROM pm6_dedup_shuffled
          WHERE shard_id = {sid}
          ORDER BY rnd
        )
        TO '{out_path}'
        (FORMAT PARQUET, COMPRESSION 'UNCOMPRESSED')
    """)
    con.close()

def ensure_output_dirs(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    shards_dir = os.path.join(out_dir, "shards")
    os.makedirs(shards_dir, exist_ok=True)
    db_path = os.path.join(out_dir, "pm6_prep.duckdb")
    return db_path, shards_dir

def initialize_duckdb(db_path, threads):
    log("Initializing DuckDB and materializing stage table...")
    con = duckdb.connect(db_path)
    con.execute(f"PRAGMA threads={threads}")
    con.execute("PRAGMA temp_directory='./.duckdb_tmp'")
    con.execute("PRAGMA enable_progress_bar")
    con.execute("PRAGMA progress_bar_time=1000")
    return con

def register_benchmark_inchis(con, benchmark_inchis):
    benchmark_df = pd.DataFrame({"inchi": benchmark_inchis})
    con.register("benchmark_df", benchmark_df)
    con.execute("""
        CREATE OR REPLACE TABLE benchmark_inchis AS
        SELECT DISTINCT inchi FROM benchmark_df WHERE inchi IS NOT NULL
    """)

def prepare_pm6_views(con, pattern, num_target_shards):
    log(f"Scanning PM6 shards with pattern: {pattern}")
    num_input = len(glob.glob(pattern))
    log(f"Input shard files found: {num_input}")
    pat_escaped = pattern.replace("'", "''")
    con.execute(f"CREATE OR REPLACE VIEW pm6_raw AS SELECT * FROM parquet_scan('{pat_escaped}')")
    log("Deduplicating by InChI and removing benchmark test set overlap...")
    con.execute(f"""
        CREATE OR REPLACE VIEW pm6_dedup_filtered AS
        SELECT
            *,
            (abs(hash(inchi)) % {int(num_target_shards)})::INTEGER AS shard_id
        FROM (
            SELECT *
            FROM pm6_raw
            WHERE inchi IS NOT NULL
            QUALIFY row_number() OVER (PARTITION BY inchi ORDER BY inchi) = 1
        ) d
        ANTI JOIN benchmark_inchis USING (inchi)
    """)
    return num_input

def materialize_shuffled_table(con, seed, base_view="pm6_dedup_filtered"):
    log("Materializing shuffled table for concurrent writers...")
    con.execute(f"""
        CREATE OR REPLACE TABLE pm6_dedup_shuffled AS
        SELECT
            *,
            hash(inchi || '{seed}') AS rnd
        FROM {base_view}
    """)

def compute_stats(con, num_input, base_view_for_write):
    log("Computing stats...")
    total_before = con.execute("SELECT COUNT(*) FROM pm6_raw").fetchone()[0]
    total_unique = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT inchi FROM pm6_raw WHERE inchi IS NOT NULL)"
    ).fetchone()[0]
    total_after = con.execute("SELECT COUNT(*) FROM pm6_dedup_filtered").fetchone()[0]
    overlap_unique_after = con.execute(
        "SELECT COUNT(*) FROM (SELECT DISTINCT inchi FROM pm6_dedup_filtered)"
    ).fetchone()[0]
    overlap_count = total_unique - overlap_unique_after
    counts_df = con.execute("""
        SELECT shard_id, COUNT(*) AS n
        FROM pm6_dedup_filtered
        GROUP BY shard_id
        ORDER BY shard_id
    """).df()

    # Geometry stats
    try:
        hf_geom_rows = con.execute("SELECT COUNT(*) FROM hf_geom").fetchone()[0]
    except Exception:
        hf_geom_rows = 0
    try:
        gen_geom_rows = con.execute("SELECT COUNT(*) FROM gen_geom").fetchone()[0]
    except Exception:
        gen_geom_rows = 0
    try:
        with_geom = con.execute(
            "SELECT COUNT(*) FROM pm6_enriched WHERE atomic_numbers IS NOT NULL AND coords IS NOT NULL"
        ).fetchone()[0]
    except Exception:
        with_geom = 0
    try:
        with_charge = con.execute(
            "SELECT COUNT(*) FROM pm6_enriched WHERE charge IS NOT NULL"
        ).fetchone()[0]
    except Exception:
        with_charge = 0
    rows_written = con.execute(f"SELECT COUNT(*) FROM {base_view_for_write}").fetchone()[0]

    stats = {
        "num_input_shards": int(num_input),
        "total_rows_before": int(total_before),
        "total_unique_inchi_before": int(total_unique),
        "removed_benchmark_overlap_est": int(overlap_count),
        "total_rows_after": int(total_after),
        "per_shard_counts": counts_df.to_dict(orient="records"),
        "hf_geom_rows": int(hf_geom_rows),
        "rdkit_generated_geom_rows": int(gen_geom_rows),
        "rows_with_geometry": int(with_geom),
        "rows_with_charge": int(with_charge),
        "rows_written_after_geometry_filter": int(rows_written),
    }
    summary = (
        f"Rows before: {total_before} | unique InChI before: {total_unique} | "
        f"unique removed by benchmark est: {overlap_count} | rows after dedup: {total_after} | "
        f"HF geom: {hf_geom_rows} | RDKit geom: {gen_geom_rows} | with geometry: {with_geom} | "
        f"with charge: {with_charge} | rows to write: {rows_written}"
    )
    return stats, summary

def write_manifest(manifest_path, manifest):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

def prepare_shard_paths(shards_dir, num_shards):
    width = max(3, len(str(num_shards - 1)))
    paths = []
    for sid in range(num_shards):
        out_path = os.path.join(shards_dir, f"pm6_shard_{sid:0{width}d}.parquet")
        if os.path.exists(out_path):
            os.remove(out_path)
        paths.append((sid, out_path))
    return paths

def write_shards(db_path, paths, concurrency, threads, seed, shards_dir):
    num_shards = len(paths)
    threads_per_writer = max(1, threads // max(1, concurrency))
    log(f"Writing {num_shards} shards in parallel (concurrency={concurrency}) to: {shards_dir}")
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(write_one, db_path, out_path, sid, threads_per_writer, seed)
            for sid, out_path in paths
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            fut.result()
            if i % 10 == 0 or i == num_shards:
                log(f"Written {i}/{num_shards} files")


if __name__ == "__main__":
    """
    Example:
    python -m monroe.preprocessing.pm6_prep \
      --out-dir data/pm6_resharded \
      --pm6-pattern "data/pm6_raw/pm6_processed_*.parquet" \
      --geom-dir "./data/pm6_hf/pm6opt_s0_parquet" \
      --num-target-shards 1010 \
      --concurrency 8 \
      --threads 16 \
      --seed 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--pm6-pattern", required=True)
    parser.add_argument("--geom-dir", default="./data/pm6_hf/pm6opt_s0_parquet",
                        help="Directory containing pre-exported pm6opt S0 Parquets")
    parser.add_argument("--num-target-shards", type=int, required=True)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8)))
    parser.add_argument("--concurrency", type=int, default=5, help="parallelism for RDKit and writers")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    db_path, shards_dir = ensure_output_dirs(args.out_dir)

    # Load benchmark InChIs (Polaris + MoleculeACE test splits)
    benchmark_inchis = get_benchmark_inchis()

    # Init DuckDB
    con = initialize_duckdb(db_path, args.threads)
    register_benchmark_inchis(con, benchmark_inchis)

    # PM6 dedup + anti-join benchmark test sets
    num_input = prepare_pm6_views(con, args.pm6_pattern, args.num_target_shards)

    # Geometry enrichment
    geom_glob, geom_files = find_geom_parquets(args.geom_dir)
    log(f"Geometry Parquet files found: {geom_files} under {args.geom_dir}")
    register_hf_geom(con, geom_glob)
    enrich_with_hf_geom(con)
    build_final_enriched_views(con)

    # Shuffle from geometry-only view
    materialize_shuffled_table(con, args.seed, base_view="pm6_enriched_geomonly")

    # Stats and close
    stats, stats_summary = compute_stats(con, num_input, base_view_for_write="pm6_enriched_geomonly")
    con.close()

    manifest = {
        **stats,
        "num_target_shards": int(args.num_target_shards),
        "shuffle_seed": args.seed,
        "concurrency": args.concurrency,
        "threads": args.threads,
    }
    write_manifest(os.path.join(args.out_dir, "manifest.json"), manifest)
    log(stats_summary)

    # Write shards
    paths = prepare_shard_paths(shards_dir, int(args.num_target_shards))
    write_shards(db_path, paths, args.concurrency, args.threads, args.seed, shards_dir)

    log("Done.")