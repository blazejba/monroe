"""MoleculeACE activity cliff benchmark evaluation."""

import numpy as np
import pandas as pd
import torch
from MoleculeACE import Data, calc_cliff_rmse, calc_rmse, datasets

from monroe.eval.dataset import _ensure_precomputed_graphs, get_embeddings, precomputed_graphs
from monroe.eval.tabpfn import default_ensemble_specs, fit_predict_tabpfn
from monroe.utils import printf


def get_moleculeace_smiles() -> list[str]:
    """Return all unique SMILES from MoleculeACE benchmark datasets."""
    smiles: list[str] = []
    for dataset in datasets:
        df = pd.read_csv(
            "https://raw.githubusercontent.com/molML/MoleculeACE/"
            "7e6de0bd2968c56589c580f2a397f01c531ede26/"
            f"MoleculeACE/Data/benchmark_data/{dataset}.csv"
        )
        smiles.extend(df["smiles"].astype(str).tolist())

    return sorted(set(smiles))


def evaluate(
    featuriser: torch.nn.Module | None,
    device: torch.device,
    ensemble_specs: list[dict] | None = None,
    seed: int = 42,
    clear_cache: bool = False,
    parquet_path: str = "data/moleculeace/moleculeace_no_sdf.parquet",
) -> dict[str, dict[str, float]]:
    """Evaluate on MoleculeACE benchmark datasets using TabPFN.

    Args:
        featuriser: Pre-trained encoder model
        device: Torch device
        ensemble_specs: TabPFN ensemble configuration
        seed: Random seed
        clear_cache: Whether to clear cached graphs/embeddings
        parquet_path: Path to precomputed graphs parquet

    Returns:
        Dict mapping dataset_name -> {overall_test_rmse, cliff_test_rmse, noncliff_test_rmse}
    """
    if ensemble_specs is None:
        ensemble_specs = default_ensemble_specs()

    # Load precomputed graphs
    if clear_cache:
        _ensure_precomputed_graphs(parquet_path, clear=True)
    elif not precomputed_graphs:
        _ensure_precomputed_graphs(parquet_path, clear=False)

    eval_results = {}

    for dataset_name in datasets:
        data = Data(dataset_name)

        # Filter to molecules with valid graphs
        smiles_train, y_train = [], []
        for smi, y in zip(data.smiles_train, data.y_train):
            if precomputed_graphs.get(smi) is not None:
                smiles_train.append(smi)
                y_train.append(y)

        smiles_test, y_test, cliff_mask_test = [], [], []
        for smi, y, cliff in zip(data.smiles_test, data.y_test, data.cliff_mols_test):
            if precomputed_graphs.get(smi) is not None:
                smiles_test.append(smi)
                y_test.append(y)
                cliff_mask_test.append(cliff)

        # Generate embeddings
        X_train = _get_embedding_array(featuriser, smiles_train, device)
        X_test = _get_embedding_array(featuriser, smiles_test, device)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            printf(
                f"Skipping {dataset_name}: insufficient embeddings "
                f"(train={X_train.shape[0]}, test={X_test.shape[0]})"
            )
            continue

        # Run TabPFN regression
        y_hat = fit_predict_tabpfn(
            X_train,
            np.asarray(y_train),
            X_test,
            is_classification=False,
            ensemble_specs=ensemble_specs,
            seed=seed,
        )

        # Compute metrics
        y_test = np.asarray(y_test)
        rmse = calc_rmse(y_test, y_hat)
        noncliff_mask_test = [not p for p in cliff_mask_test]
        rmse_cliff = calc_cliff_rmse(
            y_test_pred=y_hat, y_test=y_test, cliff_mols_test=cliff_mask_test
        )
        rmse_noncliff = calc_cliff_rmse(
            y_test_pred=y_hat, y_test=y_test, cliff_mols_test=noncliff_mask_test
        )

        eval_results[dataset_name] = {
            "overall_test_rmse": rmse,
            "cliff_test_rmse": rmse_cliff,
            "noncliff_test_rmse": rmse_noncliff,
        }
        printf(
            f"  {dataset_name}: RMSE={rmse:.4f}, "
            f"Cliff={rmse_cliff:.4f}, NonCliff={rmse_noncliff:.4f}"
        )

    return eval_results


def evaluate_mlp_head(
    featuriser: torch.nn.Module,
    device: torch.device,
    seed: int = 42,
    parquet_path: str = "data/moleculeace/moleculeace_no_sdf.parquet",
) -> dict[str, dict[str, float]]:
    """Evaluate MoleculeACE using per-task MLP heads on frozen embeddings."""
    from monroe.eval.mlp_head import predict_mlp, train_mlp_head

    if not precomputed_graphs:
        _ensure_precomputed_graphs(parquet_path, clear=False)

    eval_results = {}
    for dataset_name in datasets:
        data = Data(dataset_name)

        smiles_train, y_train = [], []
        for smi, y in zip(data.smiles_train, data.y_train):
            if precomputed_graphs.get(smi) is not None:
                smiles_train.append(smi)
                y_train.append(y)

        smiles_test, y_test, cliff_mask_test = [], [], []
        for smi, y, cliff in zip(data.smiles_test, data.y_test, data.cliff_mols_test):
            if precomputed_graphs.get(smi) is not None:
                smiles_test.append(smi)
                y_test.append(y)
                cliff_mask_test.append(cliff)

        X_train = _get_embedding_array(featuriser, smiles_train, device)
        X_test = _get_embedding_array(featuriser, smiles_test, device)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            printf(f"Skipping {dataset_name}: insufficient embeddings")
            continue

        model, y_mean, y_std = train_mlp_head(
            X_train, np.asarray(y_train), is_classification=False, seed=seed, device=device,
        )
        y_hat, _ = predict_mlp(model, X_test, y_mean, y_std, device=device)

        y_test = np.asarray(y_test)
        rmse = calc_rmse(y_test, y_hat)
        noncliff_mask_test = [not p for p in cliff_mask_test]
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, cliff_mols_test=cliff_mask_test)
        rmse_noncliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, cliff_mols_test=noncliff_mask_test)

        eval_results[dataset_name] = {
            "overall_test_rmse": rmse,
            "cliff_test_rmse": rmse_cliff,
            "noncliff_test_rmse": rmse_noncliff,
        }
        printf(f"  {dataset_name}: RMSE={rmse:.4f}, Cliff={rmse_cliff:.4f}, NonCliff={rmse_noncliff:.4f}")

        del model
        torch.cuda.empty_cache()

    return eval_results


def _get_embedding_array(
    model: torch.nn.Module | None,
    smiles: list[str],
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    """Generate embeddings for a list of SMILES and return as numpy array."""
    if model is None:
        return np.empty((0, 0))
    embs_dict = get_embeddings(smiles, model, device, batch_size=batch_size)
    if not embs_dict:
        return np.empty((0, 0))
    embs = [embs_dict[s] for s in smiles if s in embs_dict]
    if not embs:
        return np.empty((0, 0))
    return torch.stack(embs).numpy()


def _compute_mace_metrics(y_test, y_hat, cliff_mask):
    """Compute MoleculeACE metrics."""
    rmse = calc_rmse(y_test, y_hat)
    noncliff_mask = [not c for c in cliff_mask]
    rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, cliff_mols_test=cliff_mask)
    rmse_noncliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, cliff_mols_test=noncliff_mask)
    return {
        "overall_test_rmse": rmse,
        "cliff_test_rmse": rmse_cliff,
        "noncliff_test_rmse": rmse_noncliff,
    }
