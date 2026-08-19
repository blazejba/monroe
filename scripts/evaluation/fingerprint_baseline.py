"""
Classical QSAR baseline.

Writes `results/polaris/gbm_ecfp_desc.json` and `results/moleculeace/gbm_ecfp_desc.json`.
"""
import json
from pathlib import Path

import numpy as np

from monroe.eval.fingerprint_baseline import featurise, fit_predict
from monroe.eval.polaris import _load_benchmark_with_retry, _map_preds, polaris_benchmarks
from monroe.utils import printf


def run_polaris(seed: int) -> dict[str, dict[str, float]]:
    results = {}
    for dataset_name in polaris_benchmarks:
        benchmark = _load_benchmark_with_retry(dataset_name)
        target_metric = polaris_benchmarks[dataset_name]
        is_classification = target_metric in ("roc_auc", "pr_auc", "accuracy")

        train_split, test_split = benchmark.get_train_test_split()
        train_x, train_y = [], []
        for x, y, *_ in train_split:
            train_x.append(x)
            train_y.append(y)
        test_x = [item[0] if isinstance(item, tuple) else item for item in test_split]

        X_train, train_mask = featurise(train_x)
        X_test, test_mask = featurise(test_x)
        y_train = np.asarray([y for y, keep in zip(train_y, train_mask) if keep])

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            printf(f"Skipping {dataset_name}: featurisation failed")
            results[dataset_name] = {target_metric: float("nan")}
            continue

        if is_classification:
            y_train = y_train.astype(int)

        preds, probs = fit_predict(
            X_train, y_train, X_test,
            is_classification=is_classification,
            target_metric=target_metric,
            seed=seed, n_iter=20,
        )

        if is_classification:
            final_pred = _map_preds(preds, test_mask, default_val=0)
            final_proba = _map_preds(probs, test_mask, default_val=0.5)
            out = benchmark.evaluate(y_pred=final_pred, y_prob=final_proba)
        else:
            final_pred = _map_preds(preds, test_mask, default_val=float(np.mean(y_train)))
            out = benchmark.evaluate(y_pred=final_pred)

        results[dataset_name] = {row["Metric"]: row["Score"] for _, row in out.results.iterrows()}
        score = results[dataset_name].get(target_metric, float("nan"))
        printf(f"  {dataset_name}: {target_metric}={score:.4f}")

    return results


def run_moleculeace(seed: int) -> dict[str, dict[str, float]]:
    from MoleculeACE import Data, calc_cliff_rmse, calc_rmse, datasets

    results = {}
    for dataset_name in datasets:
        data = Data(dataset_name)

        X_train, train_mask = featurise(data.smiles_train)
        X_test, test_mask = featurise(data.smiles_test)
        y_train = np.asarray([y for y, k in zip(data.y_train, train_mask) if k])
        y_test = np.asarray([y for y, k in zip(data.y_test, test_mask) if k])
        cliffs = [c for c, k in zip(data.cliff_mols_test, test_mask) if k]

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            printf(f"Skipping {dataset_name}: featurisation failed")
            continue

        y_hat, _ = fit_predict(
            X_train, y_train, X_test,
            is_classification=False,
            target_metric="overall_test_rmse",
            seed=seed, n_iter=20,
        )

        rmse = calc_rmse(y_test, y_hat)
        noncliff = [not c for c in cliffs]
        rmse_cliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, cliff_mols_test=cliffs)
        rmse_noncliff = calc_cliff_rmse(y_test_pred=y_hat, y_test=y_test, cliff_mols_test=noncliff)

        results[dataset_name] = {
            "overall_test_rmse": rmse,
            "cliff_test_rmse": rmse_cliff,
            "noncliff_test_rmse": rmse_noncliff,
        }
        printf(f"  {dataset_name}: RMSE={rmse:.4f}, Cliff={rmse_cliff:.4f}")

    return results


#: Seeds behind the shipped results/{polaris,moleculeace}/gbm_ecfp_desc.json
NUM_SEEDS = 5


def _merge(path: Path, seed: int, payload: dict) -> None:
    existing = json.loads(path.read_text()) if path.exists() else {}
    existing[str(seed)] = payload
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2))
    printf(f"Wrote {path}")


def main():
    method = "gbm_ecfp_desc"
    out = Path("results")

    for seed in range(42, 42 + NUM_SEEDS):
        printf(f"=== {method} | seed {seed} ===")
        printf("Polaris:")
        _merge(out / "polaris" / f"{method}.json", seed, run_polaris(seed))
        printf("MoleculeACE:")
        _merge(out / "moleculeace" / f"{method}.json", seed, run_moleculeace(seed))


if __name__ == "__main__":
    main()
