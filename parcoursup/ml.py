"""Linear regression for predicting MPI performance from lycée grades."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def learn_mpi_model(
    dataset: pd.DataFrame,
    features: list[str],
    target: str = "mpi_moy",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Fit a standardized linear regression predicting *target*.

    Parameters
    ----------
    dataset : DataFrame with target and feature columns.
    features : columns to use (numeric grades and boolean has_ flags).
    target : column to predict.

    Returns
    -------
    coefficients : DataFrame with feature names and standardized coefficients.
    scored : training rows with ``prediction`` and ``residual`` columns.
    metrics : dict with ``n_eleves``, ``r2``, ``correlation``, ``intercept``.
    """
    train = dataset.dropna(subset=[target]).copy()

    available = [c for c in features if c in train.columns]
    X = train[available].apply(pd.to_numeric, errors="coerce")

    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("No usable features.")

    y = train[target].to_numpy(dtype=float)

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        # ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    pipeline.fit(X, y)

    predictions = pipeline.predict(X)
    model = pipeline.named_steps["model"]
    r2 = pipeline.score(X, y)

    coefficients = pd.DataFrame({
        "feature": X.columns,
        "coefficient": model.coef_,
    }).sort_values("coefficient", ascending=False, key=abs).reset_index(drop=True)

    scored = train.copy()
    scored["prediction"] = predictions
    scored["residual"] = scored[target] - predictions

    metrics = {
        "n_eleves": len(scored),
        "r2": round(r2, 4),
        "correlation": round(float(pd.Series(predictions).corr(pd.Series(y))), 4),
        "intercept": round(float(model.intercept_), 4),
    }

    return coefficients, scored, metrics
