"""Tuned fingerprint / descriptor baselines (classical QSAR reference point).

ECFP4 count fingerprints, optionally concatenated with RDKit 2D descriptors, fed to a
per-task tuned LightGBM or random forest. Model family and hyperparameters are selected
by randomised search with cross-validation on the *training* split only; the benchmark
test split is never used for selection.

This is the "default baseline for such benchmarks" that a reader expects alongside the
foundation-model comparison, and it mirrors the recipe used by strong classical entries
on the TDC ADMET leaderboard (fingerprints + descriptors -> gradient boosting).
"""

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV

from monroe.utils import printf

RDLogger.DisableLog("rdApp.*")

FP_RADIUS = 2
FP_BITS = 2048

_fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
_feature_cache: dict[tuple[str, str], np.ndarray | None] = {}
_DESC_NAMES: list[str] | None = None


# ---------------------------------------------------------------------------
# Featurisation
# ---------------------------------------------------------------------------

def _descriptor_names() -> list[str]:
    global _DESC_NAMES
    if _DESC_NAMES is None:
        _DESC_NAMES = [name for name, _ in Descriptors.descList]
    return _DESC_NAMES


def _featurise_one(smi: str) -> np.ndarray | None:
    """ECFP4 counts concatenated with RDKit 2D descriptors."""
    if smi in _feature_cache:
        return _feature_cache[smi]

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        _feature_cache[smi] = None
        return None

    fp = _fp_gen.GetCountFingerprintAsNumPy(mol).astype(np.float32)
    desc = np.empty(len(_descriptor_names()), dtype=np.float32)
    for i, (_, fn) in enumerate(Descriptors.descList):
        try:
            desc[i] = fn(mol)
        except Exception:
            desc[i] = np.nan
    # Trees handle raw scales, but not inf; NaN is left for LightGBM.
    desc[~np.isfinite(desc)] = np.nan
    feat = np.concatenate([fp, desc])

    _feature_cache[smi] = feat
    return feat


def featurise(smiles_list: list[str]) -> tuple[np.ndarray, list[bool]]:
    """Featurise SMILES, tracking which ones RDKit could parse.

    Returns:
        X: [n_valid, n_features] float32 array.
        valid_mask: per-input flag, aligned with *smiles_list*.
    """
    feats, valid_mask = [], []
    for smi in smiles_list:
        f = _featurise_one(smi)
        if f is None:
            valid_mask.append(False)
        else:
            feats.append(f)
            valid_mask.append(True)

    X = np.stack(feats) if feats else np.zeros((0, 0), dtype=np.float32)
    return X, valid_mask


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _spearman(y_true, y_pred):
    r = spearmanr(y_true, y_pred).statistic
    return 0.0 if np.isnan(r) else r


def _pearson(y_true, y_pred):
    r = pearsonr(y_true, y_pred).statistic
    return 0.0 if np.isnan(r) else r


def scorer_for(target_metric: str):
    """sklearn scorer matching the benchmark's own target metric.

    Selecting hyperparameters under the metric the benchmark reports is what makes
    this a *tuned* baseline rather than a default-settings one.
    """
    if target_metric == "roc_auc":
        return "roc_auc"
    if target_metric == "pr_auc":
        return "average_precision"
    if target_metric == "accuracy":
        return "accuracy"
    if target_metric in ("mean_absolute_error", "mae"):
        return "neg_mean_absolute_error"
    if target_metric in ("mean_squared_error", "rmse", "overall_test_rmse"):
        return "neg_root_mean_squared_error"
    if target_metric == "spearmanr":
        return make_scorer(_spearman)
    if target_metric == "pearsonr":
        return make_scorer(_pearson)
    if target_metric == "r2":
        return "r2"
    raise ValueError(f"No scorer for metric: {target_metric}")


# ---------------------------------------------------------------------------
# Tuned model
# ---------------------------------------------------------------------------

_LGBM_SPACE = {
    "n_estimators": [200, 400, 800, 1500],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63, 127],
    "min_child_samples": [5, 10, 20, 40],
    "subsample": [0.6, 0.8, 1.0],
    "subsample_freq": [1],
    "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 10.0],
}

def fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    is_classification: bool,
    target_metric: str,
    seed: int = 42,
    n_iter: int = 30,
    n_jobs: int = -1,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Tune LightGBM hyperparameters on the training split, then predict.

    Returns:
        preds: point predictions (class labels for classification).
        probs: positive-class probabilities, or None for regression.
    """
    scoring = scorer_for(target_metric)
    n = X_train.shape[0]
    # Large tasks get fewer folds: 5-fold on the 12k-molecule CYP sets costs more than
    # it buys, and the selection signal is already stable there.
    n_splits = 3 if n > 2000 else (5 if n >= 250 else 3)

    if is_classification:
        # Guard against folds that would contain a single class.
        min_class = int(np.bincount(y_train.astype(int)).min())
        n_splits = max(2, min(n_splits, min_class))

    import lightgbm as lgb

    # Single-threaded estimator; parallelism is applied across search candidates,
    # which keeps the cores busy without oversubscribing them.
    estimator = (
        lgb.LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1)
        if is_classification
        else lgb.LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1)
    )
    search = RandomizedSearchCV(
        estimator,
        _LGBM_SPACE,
        n_iter=n_iter,
        scoring=scoring,
        cv=n_splits,
        random_state=seed,
        n_jobs=n_jobs,
        refit=True,
        error_score=-np.inf,
    )
    search.fit(X_train, y_train)
    if verbose:
        printf(f"    lightgbm: cv={search.best_score_:.4f}")

    best_est = search.best_estimator_
    preds = best_est.predict(X_test)
    probs = best_est.predict_proba(X_test)[:, 1] if is_classification else None
    return preds, probs
