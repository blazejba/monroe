"""Polaris ADMET benchmark evaluation."""

import json
import logging
import time
from pathlib import Path

import numpy as np
import polaris as po
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from monroe.eval.dataset import embedded_mols, lookup_embeddings
from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn
from monroe.utils import printf

# Suppress polaris rich progress bars and verbose INFO/WARNING logs
logging.getLogger("polaris").setLevel(logging.ERROR)
try:
    from polaris.utils.context import progress_instance
    _progress = progress_instance.get()
    _progress.disable = True
    _progress.console.quiet = True
except Exception:
    pass


def _load_benchmark_with_retry(dataset_name: str, max_retries: int = 3, base_delay: float = 5.0):
    """Load a Polaris benchmark with retry on network errors."""
    for attempt in range(max_retries):
        try:
            return po.load_benchmark(dataset_name)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            printf(f"Failed to load {dataset_name} (attempt {attempt + 1}/{max_retries}): {e}")
            printf(f"Retrying in {delay:.0f}s...")
            time.sleep(delay)

# Load benchmark definitions from JSON assets
_ASSETS_DIR = Path(__file__).parent.parent / "assets"

with open(_ASSETS_DIR / "polaris_benchmarks.json") as f:
    polaris_benchmarks: dict[str, str] = json.load(f)

# Metric direction: True = higher is better, False = lower is better
metric_direction: dict[str, bool] = {
    "roc_auc": True,
    "pr_auc": True,
    "spearmanr": True,
    "spearman": True,
    "pearsonr": True,
    "r2": True,
    "accuracy": True,
    "cohen_kappa": True,
    "f1": True,
    "mcc": True,
    "explained_var": True,
    "mean_absolute_error": False,
    "mean_squared_error": False,
    "rmse": False,
    "mae": False,
    "overall_test_rmse": False,
    "cliff_test_rmse": False,
    "noncliff_test_rmse": False,
    "cliff_delta_rmse": False,
}


def get_polaris_smiles(
    by_split: bool = False,
    by_task: bool = False,
) -> dict[str, list[str]] | list[str]:
    """Return the union of all SMILES used by Polaris benchmarks.

    Args:
        by_split: If True, return dict with 'train_val' and 'test' keys
        by_task: If True, return dict with task name keys

    Returns:
        List of SMILES or dict mapping to SMILES lists
    """
    if by_split:
        all_smi: dict[str, set[str]] = {split: set() for split in ["test", "train_val"]}
    elif by_task:
        all_smi: dict[str, set[str]] = {task: set() for task in polaris_benchmarks.keys()}
    else:
        all_smi_set: set[str] = set()

    for dataset_name in polaris_benchmarks.keys():
        try:
            benchmark = _load_benchmark_with_retry(dataset_name)
        except Exception as e:
            printf(f"Failed to load benchmark {dataset_name}: {e}")
            continue

        train, test = benchmark.get_train_test_split()

        def extract_smiles(iterator):
            smis = []
            for item in iterator:
                if isinstance(item, (tuple, list)) and len(item) >= 1:
                    smis.append(item[0])
                else:
                    smis.append(item)
            return smis

        train_smis = extract_smiles(train)
        test_smis = extract_smiles(test)

        if by_split:
            all_smi["train_val"].update(train_smis)
            all_smi["test"].update(test_smis)
        elif by_task:
            all_smi[dataset_name].update(train_smis)
            all_smi[dataset_name].update(test_smis)
        else:
            all_smi_set.update(train_smis)
            all_smi_set.update(test_smis)

    if by_split:
        return {split: sorted(smis) for split, smis in all_smi.items()}
    elif by_task:
        return {task: sorted(smis) for task, smis in all_smi.items()}
    return sorted(all_smi_set)


def evaluate(
    featuriser: torch.nn.Module | None,
    device: torch.device,
    ensemble_specs: list[dict] | None = None,
    seed: int = 42,
    clear_cache: bool = False,
) -> dict[str, dict[str, float]]:
    """Evaluate on Polaris ADMET benchmarks using TabPFN.

    Args:
        featuriser: Pre-trained encoder model. If None, uses cached embeddings only.
        device: Torch device
        ensemble_specs: TabPFN ensemble configuration
        seed: Random seed
        clear_cache: Whether to clear embedding cache

    Returns:
        Dict mapping dataset_name -> {metric_name: score}
    """
    if ensemble_specs is None:
        ensemble_specs = default_ensemble_specs()

    # Precompute embeddings for all molecules
    if featuriser is not None:
        all_smiles = get_polaris_smiles()
        lookup_embeddings(
            smis=all_smiles,
            featuriser=featuriser,
            batch_size=256,
            device=device,
            cache_path=None,
            clear_cache=clear_cache,
        )

    eval_results = {}

    for dataset_name in polaris_benchmarks.keys():
        benchmark = _load_benchmark_with_retry(dataset_name)
        target_metric = polaris_benchmarks.get(dataset_name, "mean_absolute_error")
        is_classification = target_metric in ["roc_auc", "pr_auc", "accuracy"]

        train_split, test_split = benchmark.get_train_test_split()

        # Extract data
        train_x, train_y = [], []
        for x, y, *_ in train_split:
            train_x.append(x)
            train_y.append(y)

        test_x = []
        for item in test_split:
            if isinstance(item, tuple):
                test_x.append(item[0])
            else:
                test_x.append(item)

        # Prepare embeddings
        X_train, y_train, train_mask = _prepare_data(train_x, train_y)
        X_test, _, test_mask = _prepare_data(test_x)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            printf(f"Skipping {dataset_name}: insufficient embeddings")
            eval_results[dataset_name] = {target_metric: float("nan")}
            continue

        # Run TabPFN
        # Use median predictions for MAE tasks (median minimizes L1 loss)
        is_mae = target_metric == "mean_absolute_error"
        if is_classification:
            preds_valid, proba_valid = fit_predict_tabpfn(
                X_train, y_train, X_test,
                is_classification=True,
                ensemble_specs=ensemble_specs,
                seed=seed,
            )
            # Map back to full test set
            final_pred = _map_preds(preds_valid, test_mask, default_val=0)
            final_proba = _map_preds(proba_valid, test_mask, default_val=0.5)
            results = benchmark.evaluate(y_pred=final_pred, y_prob=final_proba)
        else:
            preds_valid = fit_predict_tabpfn(
                X_train, y_train, X_test,
                is_classification=False,
                ensemble_specs=ensemble_specs,
                seed=seed,
                output_type="median" if is_mae else "mean",
            )
            final_pred = _map_preds(preds_valid, test_mask, default_val=np.mean(y_train))
            results = benchmark.evaluate(y_pred=final_pred)

        eval_results[dataset_name] = {}
        for _, row in results.results.iterrows():
            eval_results[dataset_name][row["Metric"]] = row["Score"]

    return eval_results


def _compute_metric(y_true, y_pred, y_prob, metric_name):
    """Compute a single metric given true values and predictions."""
    if metric_name in ("roc_auc",):
        return roc_auc_score(y_true, y_prob)
    elif metric_name in ("pr_auc",):
        from sklearn.metrics import average_precision_score
        return average_precision_score(y_true, y_prob)
    elif metric_name == "pearsonr":
        return pearsonr(y_true, y_pred)[0]
    elif metric_name in ("spearmanr", "spearman"):
        return spearmanr(y_true, y_pred)[0]
    elif metric_name == "mean_absolute_error":
        return mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


def _prepare_data(
    smiles_list: list[str],
    targets_list: list | None = None,
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    """Prepare embeddings and targets, tracking valid indices."""
    X, y = [], []
    valid_mask = []

    for i, smi in enumerate(smiles_list):
        emb = embedded_mols.get(smi)
        if emb is not None:
            X.append(emb.numpy() if isinstance(emb, torch.Tensor) else emb)
            if targets_list is not None:
                y.append(targets_list[i])
            valid_mask.append(True)
        else:
            valid_mask.append(False)

    X = np.array(X) if X else np.zeros((0, 0))
    y = np.array(y) if y else np.zeros((0,))
    return X, y, valid_mask


def _map_preds(
    valid_preds: np.ndarray,
    mask: list[bool],
    default_val: float = 0.0,
) -> np.ndarray:
    """Map predictions from valid subset back to full array."""
    full = np.full(len(mask), default_val)
    full[mask] = valid_preds
    return full


def evaluate_mlp_head(
    featuriser: torch.nn.Module,
    device: torch.device,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Evaluate Polaris using per-task MLP heads on frozen embeddings."""
    from monroe.eval.mlp_head import predict_mlp, train_mlp_head

    # Extract embeddings for all molecules
    all_smiles = get_polaris_smiles()
    lookup_embeddings(smis=all_smiles, featuriser=featuriser, batch_size=256, device=device, cache_path=None, clear_cache=False)

    eval_results = {}
    for dataset_name in polaris_benchmarks.keys():
        benchmark = _load_benchmark_with_retry(dataset_name)
        target_metric = polaris_benchmarks.get(dataset_name, "mean_absolute_error")
        is_classification = target_metric in ["roc_auc", "pr_auc", "accuracy"]

        train_split, test_split = benchmark.get_train_test_split()

        train_x, train_y = [], []
        for x, y, *_ in train_split:
            train_x.append(x)
            train_y.append(y)
        test_x = []
        for item in test_split:
            test_x.append(item[0] if isinstance(item, tuple) else item)

        X_train, y_train, train_mask = _prepare_data(train_x, train_y)
        X_test, _, test_mask = _prepare_data(test_x)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            eval_results[dataset_name] = {target_metric: float("nan")}
            continue

        model, y_mean, y_std = train_mlp_head(
            X_train, y_train, is_classification=is_classification, seed=seed, device=device,
        )
        preds_valid, probs_valid = predict_mlp(model, X_test, y_mean, y_std, device=device)

        if is_classification:
            final_pred = _map_preds(preds_valid, test_mask, default_val=0)
            final_proba = _map_preds(probs_valid, test_mask, default_val=0.5)
            results = benchmark.evaluate(y_pred=final_pred, y_prob=final_proba)
        else:
            final_pred = _map_preds(preds_valid, test_mask, default_val=np.mean(y_train))
            results = benchmark.evaluate(y_pred=final_pred)

        eval_results[dataset_name] = {}
        for _, row in results.results.iterrows():
            eval_results[dataset_name][row["Metric"]] = row["Score"]
        printf(f"  {dataset_name}: {target_metric}={eval_results[dataset_name].get(target_metric, 'N/A'):.4f}")

        del model
        torch.cuda.empty_cache()

    return eval_results
