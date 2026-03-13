"""Transformation des fichiers CSV Parcoursup en DataFrames structurés."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# ── Répertoire racine du projet ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent


# ── Mapping matières CSV → clé courte ────────────────────────────────────────
MATIERES: dict[str, str] = {
    "Philosophie": "philo",
    "Langue vivante A": "lva",
    "Français": "fr",
    "Mathématiques Spécialité": "math_spe",
    "Physique-Chimie Spécialité": "pc",
    "Numérique et Sciences Informatiques": "nsi",
    "Mathématiques Expertes": "math_expertes",
}

# Statistiques extraites pour chaque matière / trimestre
STATS: dict[str, str] = {
    "Moyenne du Candidat": "moyenne",
    "Rang Candidat": "rang",
    "Effectif Classe": "effectif",
    "Moyenne classe Candidat": "moyenne_classe",
}


def _year_str(annee: int, offset: int = 0) -> str:
    """Retourne ``'2024/2025'`` pour *annee=2026, offset=-1*."""
    a = annee + offset
    return f"{a - 1}/{a}"


# ── Chargement du CSV brut ───────────────────────────────────────────────────

def _read_csv(annee: int) -> pd.DataFrame:
    short = annee % 100
    path = ROOT / "data" / "parcoursup" / str(short) / f"mp2i_{short}.csv"
    return pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)


# ── Construction du DataFrame élèves ─────────────────────────────────────────

def _build_eleves(df: pd.DataFrame, annee: int) -> pd.DataFrame:
    ys = _year_str(annee)
    eleves = pd.DataFrame(index=df["Candidat - Code"])
    eleves.index.name = "code"

    eleves["nom"] = df["Candidat - Nom"].values
    eleves["prenom"] = df["Candidat - Prénom"].values
    eleves["fille"] = (df["Sexe"].values == "Féminin")

    birth_year = pd.to_datetime(
        df["Date Naissance"], format="%d/%m/%Y").dt.year
    eleves["annees_avance"] = 18 - (annee - birth_year.values)

    eleves["terminale_france"] = (
        df["Profil Candidat - Libellé"].values == "En terminale")
    eleves["boursier"] = df["Candidat boursier - Libellé"].str.startswith(
        "Boursier").values
    eleves["revenu"] = _to_numeric(df["Revenu brut global"]).values
    eleves["distance"] = _to_numeric(
        df["Distance domicile-établissement(Km)"]).values
    eleves["internat"] = (df["Demande Internat - Libellé"].values == "Oui")
    eleves["niveau_classe"] = df["Niveau Classe - Libellé"].values

    # Colonnes liées à l'établissement d'origine (année courante)
    col = f"Type de contrat établissement d'origine - Libellé {ys}"
    eleves["public"] = (
        df[col] == "Public").values if col in df.columns else False

    col = f"UAI Etablissement origine {ys}"
    eleves["uai"] = df[col].values if col in df.columns else None

    col = f"Département Etablissement origine - Code {ys}"
    eleves["departement"] = df[col].values if col in df.columns else None

    col = f"Pays Etablissement origine - Libellé {ys}"
    eleves["pays"] = df[col].values if col in df.columns else None

    col = f"Scolarisation sur l'année - Libellé {ys}"
    eleves["scolarisation"] = (
        df[col].str.contains("scolarisé sur cette année",
                             case=False, na=False).values
        if col in df.columns
        else False
    )

    return eleves


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


def _get_col(df: pd.DataFrame, col_name: str, suffix: str = "") -> pd.Series:
    """Accède à une colonne, avec gestion du suffixe pandas pour les doublons."""
    target = col_name + suffix
    if target in df.columns:
        return _to_numeric(df[target])
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")


def _build_notes(df: pd.DataFrame, annee: int) -> pd.DataFrame:
    """Construit le DataFrame notes avec un MultiIndex à 3 niveaux (matière, année_short, stat)."""

    tuples: list[tuple[str, int, str]] = []
    data: dict[tuple[str, int, str], pd.Series] = {}

    # Bloc 0 → Terminale (annee, suffix=""), Bloc 1 → Première (annee-1, suffix=".1")
    year_configs = [
        (annee, ""),      # Terminale
        (annee - 1, ".1"),  # Première
    ]

    for matiere_csv, matiere_short in MATIERES.items():
        for year_short, suffix in year_configs:
            for stat_csv, stat_short in STATS.items():
                cols_t = []
                for t in (1, 2, 3):
                    col_name = f"{stat_csv} - {matiere_csv} - Trimestre {t}"
                    series = _get_col(df, col_name, suffix)
                    if series.notna().any():
                        cols_t.append(series)

                key = (matiere_short, year_short, stat_short)
                if cols_t:
                    data[key] = pd.concat(cols_t, axis=1).mean(axis=1).values
                else:
                    data[key] = pd.Series([float("nan")] * len(df)).values
                tuples.append(key)

    # Français : ajouter écrit et oral (épreuves du bac)
    for col_csv, stat_name in [
        ("Note de l'épreuve - Français écrit", "ecrit"),
        ("Note de l'épreuve - Français oral", "oral"),
    ]:
        if col_csv in df.columns:
            key = ("fr", 2026, stat_name)
            data[key] = _to_numeric(df[col_csv]).values
            tuples.append(key)

    # ── Métadonnées par année (info) ─────────────────────────────────────
    for offset in (0, -1):
        ys = _year_str(annee, offset)
        yr = 2026 + offset

        # lva
        col = f"Langue vivante A scolarité - Libellé {ys}"
        key = ("info", yr, "lva")
        data[key] = df[col].values if col in df.columns else pd.Series(
            [None] * len(df)).values
        tuples.append(key)

        # math_expertes (booléen)
        opt1 = f"Option facultative 1 Scolarité - Libellé {ys}"
        opt2 = f"Option facultative 2 Scolarité - Libellé {ys}"
        has_me = pd.Series([False] * len(df))
        for opt_col in (opt1, opt2):
            if opt_col in df.columns:
                has_me = has_me | (df[opt_col] == "Mathématiques Expertes")
        key = ("info", yr, "math_expertes")
        data[key] = has_me.values
        tuples.append(key)

    # Construire le MultiIndex et le DataFrame
    multi_idx = pd.MultiIndex.from_tuples(
        tuples, names=["matiere", "annee", "stat"])
    notes = pd.DataFrame(data, index=df["Candidat - Code"], columns=multi_idx)
    notes.index.name = "code"

    return notes


# ── API publique ─────────────────────────────────────────────────────────────

def load(annee: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    """
    df = _read_csv(annee)
    eleves = _build_eleves(df, annee)
    notes = _build_notes(df, annee)
    return eleves, notes
