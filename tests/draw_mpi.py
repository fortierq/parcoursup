from __future__ import annotations

import altair as alt
import marimo as mo
import pandas as pd

ROW_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Points formule", ("points_formule",)),
    ("Mathématiques", ("math_spe_prem",
                       "math_spe_term", "math_expertes_term")),
    ("Physique-chimie", ("pc_prem", "pc_term")),
    ("Informatique", ("nsi_prem", "nsi_term")),
    ("Français", ("fr_prem", "fr_term")),
    ("Anglais", ("lva_term",)),
)

COL_SPECS: tuple[tuple[str, str], ...] = (
    ("Moyenne", "mpi_moy"),
    ("Mathématiques", "math_mpi_moy"),
    ("Physique", "pc_mpi_moy"),
    ("Informatique", "info_mpi_moy"),
    ("Français / philo", "fr_mpi_moy"),
    ("Anglais", "lv1_mpi_moy"),
)


def build_mpi_chart(dataset: pd.DataFrame) -> mo.ui.altair_chart:
    return mo.ui.altair_chart(
        alt.Chart(dataset)
        .mark_point()
        .encode(
            x=alt.X("points_formule", scale=alt.Scale(domain=[44, 80])),
            y=alt.Y("mpi_moy", scale=alt.Scale(domain=[0, 20])),
            color="fille",
            shape="boursier",
        )
        .properties(width=1000)
    )


def build_selection_summary(selection: pd.DataFrame) -> pd.DataFrame:
    selected_columns = [
        column for column in selection.columns if any(token in column for token in ("moy", "term", "prem"))
    ]
    return (
        selection.loc[:, selected_columns]
        .mean()
        .round(1)
        .rename("moy")
        .reset_index()
        .rename(columns={"index": "variable"})
    )


def build_correlation_long(selection: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_order = [label for label, _ in ROW_SPECS]
    col_order = [label for label, _ in COL_SPECS]

    for row_label, row_columns in ROW_SPECS:
        available_row_columns = [
            column for column in row_columns if column in selection.columns]
        if available_row_columns:
            row_values = selection.loc[:, available_row_columns].mean(
                axis=1, skipna=True)
        else:
            row_values = pd.Series(float("nan"), index=selection.index)

        for col_label, col_name in COL_SPECS:
            if col_name not in selection.columns:
                correlation = float("nan")
            else:
                pair = pd.concat(
                    [row_values.rename("row"), selection[col_name].rename("col")], axis=1).dropna()
                correlation = pair["row"].corr(pair["col"]) if len(
                    pair) >= 2 else float("nan")

            rows.append(
                {
                    "variable_ligne": row_label,
                    "variable_colonne": col_label,
                    "correlation": correlation,
                }
            )

    corr_long = pd.DataFrame(rows)
    corr_long["variable_ligne"] = pd.Categorical(
        corr_long["variable_ligne"], categories=row_order, ordered=True)
    corr_long["variable_colonne"] = pd.Categorical(
        corr_long["variable_colonne"], categories=col_order, ordered=True)
    return corr_long


def build_correlation_heatmap(corr_long: pd.DataFrame) -> alt.LayerChart:
    row_order = [label for label, _ in ROW_SPECS]
    col_order = [label for label, _ in COL_SPECS]
    base = alt.Chart(corr_long.dropna())

    heatmap = base.mark_rect().encode(
        x=alt.X("variable_ligne:N", title="Lycée", sort=row_order),
        y=alt.Y("variable_colonne:N", title="MPI", sort=col_order),
        color=alt.Color(
            "correlation:Q",
            title="Corrélation",
            scale=alt.Scale(domain=[-1, 1], scheme="redblue"),
        ),
        tooltip=[
            alt.Tooltip("variable_ligne:N", title="Ligne"),
            alt.Tooltip("variable_colonne:N", title="Colonne"),
            alt.Tooltip("correlation:Q", title="Corrélation", format=".2f"),
        ],
    ).properties(width=620, height=320)

    labels = base.mark_text(baseline="middle", fontSize=11).encode(
        x=alt.X("variable_ligne:N", sort=row_order),
        y=alt.Y("variable_colonne:N", sort=col_order),
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.value("black"),
    )
    return heatmap + labels
