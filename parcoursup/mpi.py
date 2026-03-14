"""Analyse des résultats MPI à partir des données Parcoursup."""

from __future__ import annotations

import csv
import unicodedata
from pathlib import Path
from typing import TypedDict

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .load import load

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FEATURES: dict[str, tuple[str, int, str]] = {
    "math_spe_terminale": ("math_spe", 0, "moyenne"),
    "pc_terminale": ("pc", 0, "moyenne"),
    "nsi_terminale": ("nsi", 0, "moyenne"),
    "math_expertes_terminale": ("math_expertes", 0, "moyenne"),
    "fr_terminale": ("fr", 0, "moyenne"),
    "lva_terminale": ("lva", 0, "moyenne"),
    "math_spe_premiere": ("math_spe", -1, "moyenne"),
    "pc_premiere": ("pc", -1, "moyenne"),
    "nsi_premiere": ("nsi", -1, "moyenne"),
    "fr_premiere": ("fr", -1, "moyenne"),
}

SUBJECT_PLOT_SPECS: tuple[dict[str, object], ...] = (
    {
        "matiere": "Mathématiques",
        "mpi_column": "maths_mpi",
        "lycee_columns": (
            ("math_spe_premiere", "Maths spé première"),
            ("math_spe_terminale", "Maths spé terminale"),
            ("math_expertes_terminale", "Maths expertes terminale"),
        ),
    },
    {
        "matiere": "Physique-chimie",
        "mpi_column": "physique_mpi",
        "lycee_columns": (
            ("pc_premiere", "PC première"),
            ("pc_terminale", "PC terminale"),
        ),
    },
    {
        "matiere": "Informatique",
        "mpi_column": "informatique_mpi",
        "lycee_columns": (
            ("nsi_premiere", "NSI première"),
            ("nsi_terminale", "NSI terminale"),
        ),
    },
    {
        "matiere": "Français / philosophie",
        "mpi_column": "fr_mpi",
        "lycee_columns": (
            ("fr_premiere", "Français première"),
            ("fr_terminale", "Français terminale"),
        ),
    },
    {
        "matiere": "Anglais",
        "mpi_column": "anglais_mpi",
        "lycee_columns": (("lva_terminale", "LVA terminale"),),
    },
)

LYCEE_COLUMN_LABELS: dict[str, str] = {column: label for spec in SUBJECT_PLOT_SPECS for column, label in spec["lycee_columns"]}
MPI_COLUMN_LABELS: dict[str, str] = {str(spec["mpi_column"]): f"{spec['matiere']} MPI" for spec in SUBJECT_PLOT_SPECS}


class MPIAnalysis(TypedDict):
    dataset: pd.DataFrame
    correlations: pd.DataFrame
    coefficients: pd.DataFrame
    scored: pd.DataFrame
    metrics: dict[str, float]
    subject_correlation_matrix: pd.DataFrame
    subject_scatter: pd.DataFrame
    projection_line: pd.DataFrame


def _normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""

    normalized = unicodedata.normalize("NFKD", str(value).strip())
    without_accents = "".join(char for char in normalized if not unicodedata.combining(char))
    return "".join(char for char in without_accents.casefold() if char.isalnum())


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    return float(text.replace(",", "."))


MAPPING = {
    "mathematiquesmpi": "maths_mpi",
    "mathematiquesmpi*": "maths_mpi",
    "physiquempi": "pc_mpi",
    "physiquempi*": "pc_mpi",
    "informatiquempi": "info_mpi",
    "informatiquempi*": "info_mpi",
    "anglaismpi": "lv1_mpi",
    "anglaismpi*": "lv1_mpi",
    "francaisphilompi": "fr_mpi",
    "francaisphilompi*": "fr_mpi",
    "bilan": "mpi",
}


def _parse_mpi_export(path: Path, annee: int) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=";")
        rows = [row for row in reader]

    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]

    header = rows[0]
    subject_positions = [index for index, value in enumerate(header) if value.strip()]
    classe = "MPI*" if "mpii" in path.stem else "MPI"

    records: list[dict[str, object]] = []
    for row in rows[3:]:
        nom = row[0].strip()
        prenom = row[1].strip()
        if not nom or not prenom:
            continue
        if any(s in _normalize_text(nom) for s in ["bidon", "fictif"]):
            continue
        record: dict[str, object] = {
            "annee": annee,
            "classe": classe,
            "nom": _normalize_text(nom),
            "prenom": _normalize_text(prenom),
        }

        for i in subject_positions:
            h = _normalize_text(header[i])
            if h not in MAPPING:
                continue
            subject = MAPPING[h]
            record[f"{subject}_moyenne"] = _to_float(row[i])
            if i + 1 < width:
                record[f"{subject}_rang"] = _to_float(row[i + 1])

        records.append(record)

    return pd.DataFrame.from_records(records)


def load_mpi_notes(annees: tuple[int, ...] = (2023, 2024)) -> pd.DataFrame:
    frames = []
    for annee in annees:
        short = annee % 100
        for path in sorted((ROOT / "data" / "notes_mpi" / f"{short}").glob("*.csv")):
            frames.append(_parse_mpi_export(path, annee))

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True).groupby(["annee", "classe", "nom", "prenom"], dropna=False).mean(numeric_only=True).reset_index()


def load_classement(annee: int) -> pd.DataFrame:
    short = annee % 100
    path = ROOT / "data" / "parcoursup" / f"{short}" / f"classement_{short}.csv"
    ranking = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    ranking = ranking.rename(
        columns={
            "Candidat - Code": "code",
            "Candidat - Nom": "nom",
            "Candidat - Prénom": "prenom",
            "Classement": "classement",
            "Classement internat": "classement_internat",
            "pointsFormule": "points_formule",
        }
    )
    ranking["code"] = pd.to_numeric(ranking["code"], errors="coerce")
    ranking["classement_num"] = pd.to_numeric(ranking["classement"], errors="coerce")
    ranking["points_formule"] = pd.to_numeric(ranking["points_formule"], errors="coerce")
    ranking["nom"] = ranking["nom"].map(_normalize_text)
    ranking["prenom"] = ranking["prenom"].map(_normalize_text)
    ranking["annee"] = annee
    return ranking


def _build_feature_frame(notes: pd.DataFrame, annee: int) -> pd.DataFrame:
    features = pd.DataFrame(index=notes.index)
    for feature_name, (matiere, annee_offset, stat) in DEFAULT_FEATURES.items():
        key = (matiere, annee + annee_offset, stat)
        if key in notes.columns:
            features[feature_name] = pd.to_numeric(notes[key], errors="coerce")
        else:
            features[feature_name] = float("nan")
    return features


def load_mpi(annees: tuple[int, ...] = (2023, 2024)) -> pd.DataFrame:
    mpi_summary = load_mpi_notes(annees)
    datasets: list[pd.DataFrame] = []

    for annee in annees:
        _, notes, _ = load(annee)
        ranking = load_classement(annee)
        features = _build_feature_frame(notes, annee)
        cohort = ranking.join(features, on="code")
        unique_cohort = cohort.groupby(["nom", "prenom"]).filter(lambda group: len(group) == 1)
        cohort_columns = [
            "annee",
            "code",
            "nom",
            "prenom",
            "classement",
            "classement_num",
            "points_formule",
            *features.columns.tolist(),
        ]
        cohort_unique = unique_cohort.loc[:, cohort_columns]
        mpi_annee = mpi_summary.loc[mpi_summary["annee"] == annee]
        merged = mpi_annee.merge(cohort_unique, on=["annee", "nom", "prenom"], how="left", suffixes=("_mpi", "_parcoursup"))
        datasets.append(merged)

    return pd.concat(datasets, ignore_index=True)


def compute_correlations(dataset: pd.DataFrame, target: str = "mpi_moyenne") -> pd.DataFrame:
    numeric_columns = dataset.select_dtypes(include=["number", "bool"]).columns
    correlations = (
        dataset.loc[:, numeric_columns].corr(numeric_only=True)[target].dropna().sort_values(ascending=False).rename("correlation").to_frame()
    )
    correlations.index.name = "variable"
    return correlations


def compute_subject_correlation_matrix(dataset: pd.DataFrame) -> pd.DataFrame:
    lycee_columns = [column for column in LYCEE_COLUMN_LABELS if column in dataset.columns]
    mpi_columns = [column for column in MPI_COLUMN_LABELS if column in dataset.columns]

    matrix = pd.DataFrame(index=[LYCEE_COLUMN_LABELS[column] for column in lycee_columns])
    for mpi_column in mpi_columns:
        values = []
        for lycee_column in lycee_columns:
            pair = dataset.loc[:, [lycee_column, mpi_column]].dropna()
            if len(pair) < 2:
                values.append(float("nan"))
            else:
                values.append(float(pair[lycee_column].corr(pair[mpi_column])))
        matrix[MPI_COLUMN_LABELS[mpi_column]] = values
    matrix.index.name = "Matière lycée"
    return matrix


def build_subject_scatter_data(dataset: pd.DataFrame) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    student_columns = ["annee", "classe", "nom_mpi", "prenom_mpi"]

    for spec in SUBJECT_PLOT_SPECS:
        mpi_column = str(spec["mpi_column"])
        if mpi_column not in dataset.columns:
            continue
        for lycee_column, serie in spec["lycee_columns"]:
            if lycee_column not in dataset.columns:
                continue
            frame = (
                dataset.loc[:, [*student_columns, lycee_column, mpi_column]]
                .dropna()
                .rename(columns={lycee_column: "note_lycee", mpi_column: "note_mpi"})
            )
            if frame.empty:
                continue
            frame["matiere"] = str(spec["matiere"])
            frame["serie"] = serie
            frame["variable_lycee"] = lycee_column
            frame["variable_mpi"] = mpi_column
            records.append(frame)

    if not records:
        return pd.DataFrame(
            columns=[
                *student_columns,
                "note_lycee",
                "note_mpi",
                "matiere",
                "serie",
                "variable_lycee",
                "variable_mpi",
            ]
        )

    return pd.concat(records, ignore_index=True)


def build_projection_line(scored: pd.DataFrame, intercept: float) -> pd.DataFrame:
    if scored.empty or "projection_mpi" not in scored.columns:
        return pd.DataFrame(columns=["projection_mpi", "prediction_mpi"])

    projection = pd.to_numeric(scored["projection_mpi"], errors="coerce").dropna()
    if projection.empty:
        return pd.DataFrame(columns=["projection_mpi", "prediction_mpi"])

    line = pd.DataFrame({"projection_mpi": [projection.min(), projection.max()]})
    line["prediction_mpi"] = intercept + line["projection_mpi"]
    return line


def learn_selection_model(
    dataset: pd.DataFrame,
    feature_names: tuple[str, ...] | None = None,
    target: str = "mpi_moyenne",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    if dataset.empty:
        raise ValueError("Le jeu de données d'entraînement est vide.")

    feature_names = feature_names or tuple(DEFAULT_FEATURES)
    train = dataset.dropna(subset=[target]).copy()
    X = train.loc[:, feature_names].apply(pd.to_numeric, errors="coerce")
    X = X.dropna(axis=1, how="all")
    if X.empty:
        raise ValueError("Aucune feature numérique exploitable pour l'apprentissage.")

    y = train[target].to_numpy(dtype=float)

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("linear_regression", LinearRegression()),
        ]
    )
    pipeline.fit(X, y)

    imputed = pipeline.named_steps["imputer"].transform(X)
    X_scaled = pipeline.named_steps["scaler"].transform(imputed)
    model = pipeline.named_steps["linear_regression"]
    predictions = pipeline.predict(X)
    projection = X_scaled @ model.coef_
    r2 = float(model.score(X_scaled, y))

    used_features = tuple(X.columns)
    standardized = pd.Series(model.coef_, index=used_features, name="coefficient_standardise")
    positive_weights = standardized.clip(lower=0)
    if positive_weights.sum() == 0:
        positive_weights[:] = 1.0
    normalized_weights = (positive_weights / positive_weights.sum()).rename("poids_selection")

    coefficient_table = pd.concat([standardized, normalized_weights], axis=1).sort_values("poids_selection", ascending=False)

    scored = train.copy()
    X_imputed = pd.DataFrame(imputed, columns=used_features, index=train.index)
    scored["projection_mpi"] = projection
    scored["prediction_mpi"] = predictions
    scored["score_selection_appris"] = X_imputed.mul(normalized_weights, axis=1).sum(axis=1)

    metrics = {
        "n_eleves": float(len(scored)),
        "r2": r2,
        "correlation_prediction": float(pd.Series(predictions).corr(pd.Series(y))),
        "intercept": float(model.intercept_),
    }
    return coefficient_table, scored, metrics


def analyse_mpi(annees: tuple[int, ...] = (2023, 2024)) -> MPIAnalysis:
    dataset = load_mpi(annees)
    correlations = compute_correlations(dataset)
    coefficients, scored, metrics = learn_selection_model(dataset)
    subject_correlation_matrix = compute_subject_correlation_matrix(dataset)
    subject_scatter = build_subject_scatter_data(dataset)
    projection_line = build_projection_line(scored, metrics["intercept"])
    return {
        "dataset": dataset,
        "correlations": correlations,
        "coefficients": coefficients,
        "scored": scored,
        "metrics": metrics,
        "subject_correlation_matrix": subject_correlation_matrix,
        "subject_scatter": subject_scatter,
        "projection_line": projection_line,
    }
