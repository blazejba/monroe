"""Reusable TabPFN ensemble predictor for downstream evaluation."""

import numpy as np
import torch
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier, TabPFNRegressor


def fit_predict_tabpfn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    is_classification: bool,
    ensemble_specs: list[dict],
    seed: int = 42,
    output_type: str = "mean",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Fit TabPFN ensemble and return predictions.

    Args:
        X_train: Training features (N_train, D)
        y_train: Training labels (N_train,)
        X_test: Test features (N_test, D)
        is_classification: If True, use classifier; else regressor
        ensemble_specs: List of dicts with keys:
            - pca_evr: float, PCA explained variance ratio (1.0 = no PCA)
            - n_estimators: int, number of TabPFN estimators
            - softmax_temperature: float, temperature for classifier
        seed: Random seed
        output_type: TabPFNRegressor prediction type ("mean" or "median").
            Use "median" for MAE-evaluated tasks (median minimizes L1 loss).

    Returns:
        For classification: tuple of (predictions, probabilities)
        For regression: predictions array
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    preds_accum = np.zeros(len(X_test))
    preds_proba_accum = np.zeros(len(X_test)) if is_classification else None
    valid_specs_count = 0

    for spec in ensemble_specs:
        pca_evr = spec.get("pca_evr", 1.0)
        n_estimators = spec.get("n_estimators", 8)
        softmax_temperature = spec.get("softmax_temperature", 1.0)

        X_train_fit = X_train.copy()
        X_test_fit = X_test.copy()

        if pca_evr != 1.0:
            n_comp = pca_evr
            # Cap integer components to min(n_samples, n_features) - 1
            if isinstance(n_comp, int):
                max_comp = min(X_train_fit.shape[0], X_train_fit.shape[1]) - 1
                n_comp = min(n_comp, max(1, max_comp))
            pca = PCA(n_components=n_comp)
            X_train_fit = pca.fit_transform(X_train_fit)
            X_test_fit = pca.transform(X_test_fit)

        # Clear CUDA cache to prevent memory exhaustion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if is_classification:
            model = TabPFNClassifier(
                device=device_str,
                n_estimators=n_estimators,
                softmax_temperature=softmax_temperature,
                random_state=seed,
            )
            model.fit(X_train_fit, y_train)
            proba = model.predict_proba(X_test_fit)[:, 1]
            preds_proba_accum += proba
            preds_accum += (proba > 0.5).astype(float)
        else:
            model = TabPFNRegressor(
                device=device_str,
                n_estimators=n_estimators,
                softmax_temperature=softmax_temperature,
                random_state=seed,
            )
            model.fit(X_train_fit, y_train)
            preds_accum += model.predict(X_test_fit, output_type=output_type)

        del model
        valid_specs_count += 1

    if valid_specs_count == 0:
        if is_classification:
            return np.full(len(X_test), 0), np.full(len(X_test), 0.5)
        return np.full(len(X_test), np.mean(y_train))

    preds_accum /= valid_specs_count

    if is_classification:
        preds_proba_accum /= valid_specs_count
        final_pred = (preds_proba_accum > 0.5).astype(int)
        return final_pred, preds_proba_accum

    return preds_accum


def default_ensemble_specs() -> list[dict]:
    """Return default ensemble configuration."""
    return [
        {"pca_evr": 1.0, "n_estimators": 8, "softmax_temperature": 0.9},
    ]
