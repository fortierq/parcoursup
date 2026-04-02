"""Transformation des fichiers CSV Parcoursup en DataFrames structurés."""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

FICHE_AVENIR_MATIERES = [
    "Philosophie",
    "Langue vivante A",
    "Mathématiques Spécialité",
    "Physique-Chimie Spécialité",
    "Sciences de la vie et de la Terre Spécialité",
    "Sciences de l'ingénieur et sciences physiques",
    "Numérique et Sciences Informatiques",
    "Mathématiques Expertes",
]

# ── Répertoire racine du projet ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


# ── Mapping matières CSV → clé courte ────────────────────────────────────────
MATIERES: dict[str, str] = {
    "Philosophie": "fr",
    "Langue vivante A": "lva",
    "Français": "fr",
    "Mathématiques Spécialité": "math_spe",
    "Physique-Chimie Spécialité": "pc",
    "Numérique et Sciences Informatiques": "nsi",
    "Mathématiques Expertes": "math_expertes",
}

# Statistiques extraites pour chaque matière / trimestre
STATS: dict[str, str] = {
    "Moyenne du Candidat": "moy",
    "Rang Candidat": "rang",
    "Effectif Classe": "effectif",
    "Moyenne classe Candidat": "moy_classe",
}


def _year_str(annee: int, offset: int = 0) -> str:
    """Retourne ``'2024/2025'`` pour *annee=2026, offset=-1*."""
    a = annee + offset
    return f"{a - 1}/{a}"


# ── Chargement du CSV brut ───────────────────────────────────────────────────


def _read_csv(annee: int) -> pd.DataFrame:
    short = annee % 100
    path = ROOT / "data" / "parcoursup" / str(short) / f"mp2i_{short}.csv"
    try:
        return pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    except pd.errors.ParserError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            return pd.read_csv(path, sep=";", encoding="utf-8", engine="python", on_bad_lines="warn")


def _read_ival_csv() -> pd.DataFrame:
    available = sorted((ROOT / "data" / "ival").glob("ival_*.csv"))
    if not available:
        return pd.DataFrame()
    path = available[-1]

    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8",
        low_memory=False,
        dtype={
            "UAI": "string",
            "Code commune": "string",
            "Code departement": "string",
        },
    )


# ── Construction du DataFrame élèves ─────────────────────────────────────────


def _build_eleves(df: pd.DataFrame, annee: int) -> pd.DataFrame:
    ys = _year_str(annee)
    eleves = pd.DataFrame(index=df["Candidat - Code"])
    eleves.index.name = "code"

    eleves["nom"] = df["Candidat - Nom"].values
    eleves["prenom"] = df["Candidat - Prénom"].values
    eleves["fille"] = df["Sexe"].values == "Féminin"
    eleves["profil"] = df["Profil Candidat - Libellé"].values

    birth_year = pd.to_datetime(df["Date Naissance"], format="%d/%m/%Y").dt.year
    eleves["annees_avance"] = 18 - (annee - birth_year.values)

    eleves["boursier"] = df["Candidat boursier - Libellé"].str.startswith("Boursier").values
    eleves["revenu"] = _to_numeric(df["Revenu brut global"]).values
    eleves["distance"] = _to_numeric(df["Distance domicile-établissement(Km)"]).values
    eleves["internat"] = df["Demande Internat - Libellé"].values == "Oui"
    eleves["niveau_classe"] = df["Niveau Classe - Libellé"].values
    eleves["motivation"] = df["Lettre de motivation"].values if "Lettre de motivation" in df.columns else None
    eleves["avis_ce"] = (
        df["Avis CE sur la capacité à réussir - Libellé"].values if "Avis CE sur la capacité à réussir - Libellé" in df.columns else None
    )

    # Colonnes liées à l'établissement d'origine (année courante)

    col = f"UAI Etablissement origine {ys}"
    eleves["uai"] = df[col].values if col in df.columns else None

    col = f"Département Etablissement origine - Code {ys}"
    eleves["departement"] = df[col].values if col in df.columns else None

    col = f"Pays Etablissement origine - Libellé {ys}"
    eleves["pays"] = df[col].values if col in df.columns else None

    col = f"Type de contrat établissement d'origine - Libellé {ys}"
    eleves["public"] = (df[col] == "Public").values if col in df.columns else False

    return eleves


def _build_lycees(eleves: pd.DataFrame) -> pd.DataFrame:
    ival = _read_ival_csv()

    if ival.empty:
        lycees = eleves.dropna(subset=["uai"]).groupby("uai", as_index=False).agg(public=("public", "max"), departement=("departement", "first"))
        lycees["nom"] = None
        lycees["pourcentage_tb"] = float("nan")
        lycees["commune"] = None
        return lycees[["uai", "nom", "pourcentage_tb", "public", "commune", "departement"]]

    mentions_tb = _to_numeric(ival["Nombre de mentions TB avec félicitations - G"]).fillna(0) + _to_numeric(
        ival["Nombre de mentions TB sans félicitations - G"]
    ).fillna(0)
    presents_g = _to_numeric(ival["Présents - Gnle"])
    ival = ival.assign(pourcentage_tb=(100 * mentions_tb / presents_g).where(presents_g > 0))

    latest = (
        ival.sort_values("Année")
        .dropna(subset=["UAI"])
        .drop_duplicates(subset=["UAI"], keep="last")
        .loc[:, ["UAI", "Etablissement", "Secteur", "Commune", "Code departement"]]
        .rename(
            columns={
                "UAI": "uai",
                "Etablissement": "nom",
                "Secteur": "public",
                "Commune": "commune",
                "Code departement": "departement",
            }
        )
    )
    latest["public"] = latest["public"].fillna("").str.casefold().eq("public")

    pourcentage_tb = ival.dropna(subset=["UAI"]).groupby("UAI")["pourcentage_tb"].mean().rename("pourcentage_tb")

    lycees = latest.merge(pourcentage_tb, left_on="uai", right_index=True, how="left")
    lycees = lycees[
        [
            "uai",
            "nom",
            "pourcentage_tb",
            "public",
            "commune",
            "departement",
        ]
    ]
    return lycees.sort_values(["departement", "commune", "nom"], na_position="last").reset_index(drop=True)


# ── Construction du DataFrame notes ──────────────────────────────────────────
#
# Le CSV contient deux blocs de bulletins (Terminale puis Première).
# Pandas renomme les colonnes dupliquées du 2e bloc avec un suffixe ".1".
# On accède donc aux colonnes directement par leur nom :
#   - Bloc Terminale : "Moyenne du Candidat - Philosophie - Trimestre 1"
#   - Bloc Première  : "Moyenne du Candidat - Philosophie - Trimestre 1.1"


def _to_numeric(s: pd.Series) -> pd.Series:
    """Convertit une série en numérique, en gérant les décimales françaises (virgules)."""
    if pd.api.types.is_string_dtype(s):
        s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def _legacy_fiche_avenir_column(stat_csv: str, matiere_csv: str) -> str | None:
    if matiere_csv not in FICHE_AVENIR_MATIERES:
        return None

    index = FICHE_AVENIR_MATIERES.index(matiere_csv)
    suffix = "" if index == 0 else f".{index}"
    return f"{stat_csv}{suffix}"


def _candidate_note_columns(stat_csv: str, matiere_csv: str, trimestre: int, suffix: str) -> list[str]:
    candidates = [f"{stat_csv} - {matiere_csv} - Trimestre {trimestre}{suffix}"]

    if stat_csv in {"Moyenne du Candidat", "Moyenne classe Candidat"}:
        candidates.append(f"{stat_csv} en {matiere_csv} pour trimestre {trimestre}{suffix}")

    if suffix == "" and stat_csv in {"Rang Candidat", "Effectif Classe"}:
        candidates.append(f"{stat_csv} - {matiere_csv}")
        legacy = _legacy_fiche_avenir_column(stat_csv, matiere_csv)
        if legacy is not None:
            candidates.append(legacy)

    return candidates


def _get_col(df: pd.DataFrame, stat_csv: str, matiere_csv: str, trimestre: int, suffix: str = "") -> pd.Series:
    """Accède à une colonne, avec gestion des variantes historiques de nommage."""
    for target in _candidate_note_columns(stat_csv, matiere_csv, trimestre, suffix):
        if target in df.columns:
            return _to_numeric(df[target])
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")


def _build_notes(df: pd.DataFrame, annee: int) -> pd.DataFrame:
    """Construit le DataFrame notes avec un MultiIndex à 3 niveaux (matière, année_short, stat)."""

    tuples: list[tuple[str, int, str]] = []
    data: dict[tuple[str, int, str], pd.Series] = {}

    year_configs = [
        (annee, ""), # Terminale
        (annee - 1, ".1"), # Première
    ]

    for matiere_csv, matiere_short in MATIERES.items():
        for year_short, suffix in year_configs:
            if matiere_csv == "Français" and suffix == "":
                continue
            if matiere_csv == "Philosophie" and suffix == ".1":
                continue
            for stat_csv, stat_short in STATS.items():
                cols_t = []
                for t in (1, 2, 3):
                    series = _get_col(df, stat_csv, matiere_csv, t, suffix)
                    if series.notna().any():
                        cols_t.append(series)

                key = (matiere_short, year_short, stat_short)
                if cols_t:
                    data[key] = pd.concat(cols_t, axis=1).mean(axis=1).values
                else:
                    data[key] = pd.Series([float("nan")] * len(df)).values
                tuples.append(key)

    for col_csv, stat_name in [
        ("Note de l'épreuve - Français écrit", "bac_ecrit"),
        ("Note de l'épreuve - Français oral", "bac_oral"),
    ]:
        if col_csv in df.columns:
            key = ("fr", annee, stat_name)
            data[key] = _to_numeric(df[col_csv]).values
            tuples.append(key)
            print(key)

    multi_idx = pd.MultiIndex.from_tuples(tuples, names=["matiere", "annee", "stat"])
    notes = pd.DataFrame(data, index=df["Candidat - Code"], columns=multi_idx)
    notes.index.name = "code"

    return notes


def _add_specialite_flags(eleves: pd.DataFrame, notes: pd.DataFrame, df: pd.DataFrame, annee: int) -> pd.DataFrame:
    for niveau, year in (("term", annee), ("prem", annee - 1)):
        ys = _year_str(year)

        for matiere in ("math_spe", "pc", "nsi"):
            eleves[f"has_{matiere}_{niveau}"] = notes[(matiere, year, "moy")].notna().values

        opt1 = f"Option facultative 1 Scolarité - Libellé {ys}"
        opt2 = f"Option facultative 2 Scolarité - Libellé {ys}"
        has_me = pd.Series(False, index=df.index)

        for opt_col in (opt1, opt2):
            if opt_col in df.columns:
                has_me = has_me | (df[opt_col] == "Mathématiques Expertes")

        eleves[f"has_math_expertes_{niveau}"] = has_me.values

    return eleves


# ── API publique ─────────────────────────────────────────────────────────────


def load(annee: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge et transforme le fichier CSV Parcoursup pour l'année donnée.

    Parameters
    ----------
    annee : int
        Année de la campagne (ex. 2026).

    Returns
    -------
    eleves : pd.DataFrame
        Informations personnelles des candidats (indexé par *code*).
    notes : pd.DataFrame
        Moyennes annuelles avec ``MultiIndex`` ``(matière, année_short, stat)``.
    lycees : pd.DataFrame
        Informations établissements enrichies par IVAL et distance moyenne des candidats.
    """
    df = _read_csv(annee)
    eleves = _build_eleves(df, annee)
    notes = _build_notes(df, annee)
    eleves = _add_specialite_flags(eleves, notes, df, annee)
    lycees = _build_lycees(eleves)
    return eleves, notes, lycees
