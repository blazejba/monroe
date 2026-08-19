"""Evaluate a checkpoint on the MoleculeACE and Polaris ADMET benchmarks."""

import argparse
import json
from pathlib import Path

import torch

from monroe.analysis.leaderboard import compute_mean_rank
from monroe.eval import moleculeace, polaris
from monroe.eval.tabpfn import default_ensemble_specs
from monroe.model.ckpt import _resolve_ckpt_dir, load_ckpt
from monroe.utils import printf


def _load_existing_results(path: Path) -> dict | None:
    """Load existing results JSON, returning None if absent or corrupt."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def evaluate_with_seeds(
    encoder: torch.nn.Module,
    benchmark: str,
    device: torch.device,
    num_seeds: int,
    base_seed: int = 42,
    existing_results: dict | None = None,
    n_estimators: int | None = None,
    softmax_temperature: float | None = None,
) -> dict:
    """Run evaluation with multiple seeds, skipping seeds already present.

    Args:
        encoder: Pre-trained encoder for embeddings
        benchmark: "polaris" or "moleculeace"
        device: Torch device
        num_seeds: Number of seeds to run
        base_seed: Starting seed value
        existing_results: Previously saved results dict (seed-keyed) to resume from
        n_estimators: Override TabPFN n_estimators (None = use default)
        softmax_temperature: Override TabPFN softmax_temperature (None = use default)

    Returns:
        Dict mapping seed to results
    """
    ensemble_specs = default_ensemble_specs()
    if n_estimators is not None:
        ensemble_specs = [{**s, "n_estimators": n_estimators} for s in ensemble_specs]
    if softmax_temperature is not None:
        ensemble_specs = [{**s, "softmax_temperature": softmax_temperature} for s in ensemble_specs]
    all_results = dict(existing_results) if existing_results else {}

    eval_fn = polaris.evaluate if benchmark == "polaris" else moleculeace.evaluate

    first_new = True
    for i in range(num_seeds):
        seed = base_seed + i
        if str(seed) in all_results:
            printf(f"\nSkipping {benchmark} seed {seed} (already exists)")
            continue

        printf(f"\nEvaluating {benchmark} with seed {seed}...")

        results = eval_fn(
            featuriser=encoder,
            device=device,
            ensemble_specs=ensemble_specs,
            seed=seed,
            clear_cache=first_new,
        )
        first_new = False

        all_results[str(seed)] = results

    return all_results


def _result_filename(benchmark: str, use_ema: bool) -> str:
    """Return the result JSON filename, e.g. results_polaris.json or results_ema_polaris.json."""
    prefix = "results_ema_" if use_ema else "results_"
    return f"{prefix}{benchmark}.json"


def evaluate_checkpoint(
    ckpt_path: str,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run all requested benchmarks for a single checkpoint."""
    output_dir.mkdir(parents=True, exist_ok=True)
    use_ema = args.use_ema

    # Load encoder for TabPFN evaluations
    weights_label = "EMA weights" if use_ema else "weights"
    printf(f"\nLoading {weights_label} from {ckpt_path}...")
    encoder = load_ckpt(ckpt_path, use_ema=use_ema)
    encoder = encoder.to(device)
    encoder.eval()

    # 1. MoleculeACE Evaluation
    moleculeace_path = output_dir / _result_filename("moleculeace", use_ema)
    if not args.skip_moleculeace:
        existing_mace = _load_existing_results(moleculeace_path)
        moleculeace_results = evaluate_with_seeds(
            encoder=encoder,
            benchmark="moleculeace",
            device=device,
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            existing_results=existing_mace,
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
        )

        with moleculeace_path.open("w") as f:
            json.dump(moleculeace_results, f, indent=2)
        printf(f"\nSaved MoleculeACE results to {moleculeace_path}")

    # 2. Polaris ADMET Evaluation
    polaris_path = output_dir / _result_filename("polaris", use_ema)
    if not args.skip_polaris:
        existing_polaris = _load_existing_results(polaris_path)
        # Strip _summary before passing as existing (it's recomputed below)
        if existing_polaris:
            existing_polaris.pop("_summary", None)

        polaris_results = evaluate_with_seeds(
            encoder=encoder,
            benchmark="polaris",
            device=device,
            num_seeds=args.num_seeds,
            base_seed=args.base_seed,
            existing_results=existing_polaris,
            n_estimators=args.n_estimators,
            softmax_temperature=args.softmax_temperature,
        )

        # Compute mean rank using leaderboard data
        rank_info = compute_mean_rank(polaris_results)
        polaris_results["_summary"] = rank_info

        printf(
            f"\nPolaris Mean Rank: {rank_info['mean_rank']:.2f} "
            f"(across {rank_info['n_tasks_ranked']} tasks)"
        )
        printf("Per-task ranks:")
        for task, rank in sorted(rank_info["task_ranks"].items(), key=lambda x: x[1]):
            score = rank_info["task_scores"].get(task, float("nan"))
            printf(f"  {task}: rank {rank}, score {score:.4f}")

        with polaris_path.open("w") as f:
            json.dump(polaris_results, f, indent=2)
        printf(f"\nSaved Polaris results to {polaris_path}")

    # 3. MLP Head Evaluation
    if getattr(args, "mlp_head", False):
        _run_mlp_head_eval(encoder, output_dir, args, device)


def _run_mlp_head_eval(
    encoder: torch.nn.Module,
    output_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    """Run MLP head evaluation on frozen embeddings."""
    use_ema = args.use_ema
    tag = ""

    for i in range(args.num_seeds):
        seed = args.base_seed + i

        if not args.skip_polaris:
            pol_path = output_dir / f"results{'_ema' if use_ema else ''}_mlp{tag}_polaris.json"
            existing = _load_existing_results(pol_path) or {}
            if str(seed) not in existing:
                printf(f"\nMLP Head Polaris seed {seed}...")
                results = polaris.evaluate_mlp_head(encoder, device, seed=seed)
                existing[str(seed)] = results
                # Compute rank
                from monroe.analysis.leaderboard import compute_mean_rank
                existing["_summary"] = compute_mean_rank(existing)
                with pol_path.open("w") as f:
                    json.dump(existing, f, indent=2)
                printf(f"Saved MLP Polaris to {pol_path}")
            else:
                printf(f"\nSkipping MLP Polaris seed {seed} (exists)")

        if not args.skip_moleculeace:
            mace_path = output_dir / f"results{'_ema' if use_ema else ''}_mlp{tag}_moleculeace.json"
            existing = _load_existing_results(mace_path) or {}
            if str(seed) not in existing:
                printf(f"\nMLP Head MoleculeACE seed {seed}...")
                results = moleculeace.evaluate_mlp_head(encoder, device, seed=seed)
                existing[str(seed)] = results
                with mace_path.open("w") as f:
                    json.dump(existing, f, indent=2)
                printf(f"Saved MLP MoleculeACE to {mace_path}")
            else:
                printf(f"\nSkipping MLP MoleculeACE seed {seed} (exists)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoint on the MoleculeACE and Polaris benchmarks"
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to a checkpoint directory or experiment root (evaluates all checkpoints)",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds for TabPFN evaluation",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed value",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results JSON files (default: <ckpt-path> or <ckpt-path>/<checkpoint-N>)",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Override TabPFN n_estimators (default: use default_ensemble_specs)",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=None,
        help="Override TabPFN softmax_temperature (default: use default_ensemble_specs)",
    )
    parser.add_argument(
        "--skip-moleculeace",
        action="store_true",
        help="Skip MoleculeACE evaluation",
    )
    parser.add_argument(
        "--skip-polaris",
        action="store_true",
        help="Skip Polaris ADMET evaluation",
    )
    parser.add_argument(
        "--use-ema",
        action="store_true",
        help="Load ema_weights.pt instead of weights.pt; saves results as results_ema_*.json",
    )
    parser.add_argument("--mlp-head", action="store_true", help="Evaluate with per-task MLP heads on frozen embeddings (MiniMol-style)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    printf(f"Using device: {device}")

    ckpt_dir = _resolve_ckpt_dir(Path(args.ckpt_path))
    if not (ckpt_dir / "config.json").exists():
        printf(f"No checkpoint (config.json) found at {ckpt_dir}")
        return

    printf("\n" + "#" * 60)
    printf(f"Evaluating checkpoint: {ckpt_dir}")
    printf("#" * 60)

    out = Path(args.output_dir) if args.output_dir else ckpt_dir
    evaluate_checkpoint(str(ckpt_dir), out, args, device)

    printf("\n" + "=" * 60)
    printf("Evaluation Complete!")
    printf("=" * 60)


if __name__ == "__main__":
    main()
