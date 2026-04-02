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
    "math_spe_term": ("math_spe", 0, "moy"),
    "pc_term": ("pc", 0, "moy"),
    "nsi_term": ("nsi", 0, "moy"),
    "math_expertes_term": ("math_expertes", 0, "moy"),
    "fr_term": ("fr", 0, "moy"),
    "lva_term": ("lva", 0, "moy"),
    "math_spe_prem": ("math_spe", -1, "moy"),
    "pc_prem": ("pc", -1, "moy"),
    "nsi_prem": ("nsi", -1, "moy"),
    "fr_prem": ("fr", -1, "moy"),
}

SUBJECT_PLOT_SPECS: tuple[dict[str, object], ...] = (
    {
        "matiere": "Mathématiques",
        "mpi_column": "math_mpi",
        "lycee_columns": (
            ("math_spe_prem", "Maths spé première"),
            ("math_spe_term", "Maths spé term"),
            ("math_expertes_term", "Maths expertes term"),
        ),
    },
    {
        "matiere": "Physique-chimie",
        "mpi_column": "physique_mpi",
        "lycee_columns": (
            ("pc_prem", "PC première"),
            ("pc_term", "PC term"),
        ),
    },
    {
        "matiere": "Informatique",
        "mpi_column": "informatique_mpi",
        "lycee_columns": (
            ("nsi_prem", "NSI première"),
            ("nsi_term", "NSI term"),
        ),
    },
    {
        "matiere": "Français / philosophie",
        "mpi_column": "fr_mpi",
        "lycee_columns": (
            ("fr_prem", "Français première"),
            ("fr_term", "Français term"),
        ),
    },
    {
        "matiere": "Anglais",
        "mpi_column": "anglais_mpi",
        "lycee_columns": (("lva_term", "LVA term"),),
    },
)

LYCEE_COLUMN_LABELS: dict[str, str] = {
    column: label for spec in SUBJECT_PLOT_SPECS for column, label in spec["lycee_columns"]}
MPI_COLUMN_LABELS: dict[str, str] = {
    str(spec["mpi_column"]): f"{spec['matiere']} MPI" for spec in SUBJECT_PLOT_SPECS}


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
    without_accents = "".join(
        char for char in normalized if not unicodedata.combining(char))
    return "".join(char for char in without_accents.casefold() if char.isalnum())


def _to_float(value: object) -> float:
    if value is None:
        return float("nan")

    text = str(value).strip()
    if not text:
        return float("nan")

    return float(text.replace(",", "."))


MAPPING = {
    "mathematiquesmpi": "math_mpi",
    "mathematiquesmpi*": "math_mpi",
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
    subject_positions = [index for index,
                         value in enumerate(header) if value.strip()]
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
            record[f"{subject}_moy"] = _to_float(row[i])
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

    notes = pd.concat(frames, ignore_index=True).groupby(
        ["annee", "classe", "nom", "prenom"], dropna=False).mean(numeric_only=True).reset_index()
    moy_columns = [column for column in notes.columns if str(
        column).endswith("_moy")]
    notes.loc[:, moy_columns] = notes.loc[:, moy_columns].round(1)
    return notes


def load_classement(annee: int) -> pd.DataFrame:
    short = annee % 100
    path = ROOT / "data" / "parcoursup" / \
        f"{short}" / f"classement_{short}.csv"
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
    ranking["points_formule"] = pd.to_numeric(
        ranking["points_formule"], errors="coerce").round(1)
    ranking["nom"] = ranking["nom"].map(_normalize_text)
    ranking["prenom"] = ranking["prenom"].map(_normalize_text)
    ranking["annee"] = annee
    return ranking


def _build_feature_frame(notes: pd.DataFrame, annee: int) -> pd.DataFrame:
    features = pd.DataFrame(index=notes.index)
    for feature_name, (matiere, annee_offset, stat) in DEFAULT_FEATURES.items():
        key = (matiere, annee + annee_offset, stat)
        if key in notes.columns:
            features[feature_name] = pd.to_numeric(
                notes[key], errors="coerce").round(1)
        else:
            features[feature_name] = float("nan")
    return features


def load_mpi(annees: tuple[int, ...] = (2023, 2024)) -> pd.DataFrame:
    mpi_summary = load_mpi_notes(annees)
    datasets: list[pd.DataFrame] = []

    for annee in annees:
        eleves, notes, lycees = load(annee)
        ranking = load_classement(annee)
        features = _build_feature_frame(notes, annee)
        profils = eleves.reset_index().loc[:, ["code", "fille", "boursier", "uai"]]
        if "pourcentage_tb" in lycees.columns:
            profils = profils.merge(lycees[["uai", "pourcentage_tb"]], on="uai", how="left")
        else:
            profils["pourcentage_tb"] = float("nan")
        cohort = ranking.join(features, on="code").merge(
            profils, on="code", how="left")
        unique_cohort = cohort.groupby(["nom", "prenom"]).filter(
            lambda group: len(group) == 1)
        cohort_columns = [
            "annee",
            "code",
            "nom",
            "prenom",
            "fille",
            "boursier",
            "pourcentage_tb",
            "classement",
            "points_formule",
            * features.columns.tolist(),
        ]
        cohort_unique = unique_cohort.loc[:, cohort_columns]
        mpi_annee = mpi_summary.loc[mpi_summary["annee"] == annee]
        merged = mpi_annee.merge(cohort_unique, on=[
                                 "annee", "nom", "prenom"], how="left", suffixes=("_mpi", "_parcoursup"))
        merged["fille"] = merged["fille"].fillna(False).astype(
            bool).map({True: "oui", False: "non"})
        merged["boursier"] = merged["boursier"].fillna(
            False).astype(bool).map({True: "oui", False: "non"})
        datasets.append(merged)

    return pd.concat(datasets, ignore_index=True)
