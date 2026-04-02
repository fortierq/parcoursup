"""Regression models for predicting MPI performance from lycée grades."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _get_coef(model: object) -> np.ndarray:
    """Extract coefficients or feature importances from a fitted model."""
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_)
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    raise AttributeError(f"{type(model).__name__} has no coef_ or feature_importances_")


def _get_final_model(estimator: object) -> object:
    """Get the last step if Pipeline, otherwise return as-is."""
    if isinstance(estimator, Pipeline):
        return estimator[-1]
    return estimator


def learn_mpi_model(
    dataset: pd.DataFrame,
    features: list[str],
    target: str = "mpi_moy",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit OLS, NNLS, and Random Forest; return both NNLS and RF results.

    Returns
    -------
    nnls_coef : DataFrame with NNLS coefficients.
    rf_coef : DataFrame with Random Forest feature importances.
    scored : training rows with predictions from both models.
    comparison : DataFrame comparing train R² and cross-validated R².
    """
    train = dataset.dropna(subset=[target]).copy()

    available = [c for c in features if c in train.columns]
    X = train[available].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable features.")

    # Mutually exclusive specialties: NaN means "didn't take it" → 0
    optional = ["nsi_term", "nsi_prem", "pc_term", "pc_prem"]
    for col in optional:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # Drop rows with remaining NaN (missing data, not optional subjects)
    mask = X.notna().all(axis=1)
    X = X.loc[mask]
    y = train.loc[mask, target].to_numpy(dtype=float)

    # ── Fit models ────────────────────────────────────────────────────────
    n = len(X)
    cv = LeaveOneOut() if n < 50 else min(5, n // 10)
    if isinstance(cv, int):
        cv = max(cv, 3)

    models = {
        "OLS": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
        "NNLS": Pipeline([("scaler", StandardScaler()), ("model", LinearRegression(positive=True))]),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=10,
            min_samples_leaf=2, random_state=1,
        ),
    }

    rows: list[dict] = []
    for name, estimator in models.items():
        estimator.fit(X, y)
        train_r2 = estimator.score(X, y)
        cv_r2 = cross_val_score(estimator, X, y, cv=cv, scoring="r2").mean()
        coef = _get_coef(_get_final_model(estimator))
        rows.append({
            "model": name,
            "train_r2": round(train_r2, 4),
            "cv_r2": round(cv_r2, 4),
            "n_negative": int((coef < 0).sum()),
            "n_samples": n,
        })

    comparison = pd.DataFrame(rows)

    # ── NNLS coefficients (on standardized scale) ─────────────────────────
    nnls_model = _get_final_model(models["NNLS"])
    nnls_coef = pd.DataFrame({
        "feature": X.columns,
        "coefficient": nnls_model.coef_,
    }).sort_values("coefficient", ascending=False).reset_index(drop=True)

    # ── RF feature importances ────────────────────────────────────────────
    rf_coef = pd.DataFrame({
        "feature": X.columns,
        "importance": models["Random Forest"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Scored dataset ────────────────────────────────────────────────────
    scored = train.loc[mask].copy()
    scored["prediction_nnls"] = models["NNLS"].predict(X)
    scored["prediction_rf"] = models["Random Forest"].predict(X)

    return nnls_coef, rf_coef, scored, comparison
