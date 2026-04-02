import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    from parcoursup.mpi import load_mpi
    from parcoursup.ml import learn_mpi_model
    import marimo as mo
    import altair as alt
    import pandas as pd

    mpi = load_mpi()
    features = [
        "pourcentage_tb",
        "math_spe_prem", "math_spe_term", "math_expertes_term",
        "pc_term",
        "nsi_term",
        "fr_term",
        "lva_term",
    ]

    nnls_coef, rf_coef, scored, comparison = learn_mpi_model(mpi, features)
    return alt, comparison, mo, nnls_coef, pd, rf_coef, scored


@app.cell
def _(alt, comparison, mo, nnls_coef, pd, rf_coef, scored):
    # ── Comparison table ──────────────────────────────────────────────────
    comparison_table = mo.ui.table(comparison, show_column_summaries=False)

    # ── NNLS bar chart ────────────────────────────────────────────────────
    nnls_chart = (
        alt.Chart(nnls_coef)
        .mark_bar()
        .encode(
            x=alt.X("coefficient:Q", title="Coefficient"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.value("#4c78a8"),
        )
        .properties(width=400, height=250, title="NNLS — coefficients")
    )

    # ── RF bar chart ──────────────────────────────────────────────────────
    rf_chart = (
        alt.Chart(rf_coef)
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.value("#e6550d"),
        )
        .properties(width=400, height=250, title="Random Forest — importances")
    )

    # ── Scatter: NNLS ─────────────────────────────────────────────────────
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 20], "y": [0, 20]}))
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )
    scatter_nnls = (
        alt.Chart(scored[["mpi_moy", "prediction_nnls"]].dropna())
        .mark_point(opacity=0.4)
        .encode(
            x=alt.X("mpi_moy:Q", title="MPI réel", scale=alt.Scale(domain=[0, 20])),
            y=alt.Y("prediction_nnls:Q", title="Prédit NNLS", scale=alt.Scale(domain=[0, 20])),
        )
        .properties(width=400, height=350, title="NNLS — réel vs prédit")
    )

    # ── Scatter: RF ───────────────────────────────────────────────────────
    scatter_rf = (
        alt.Chart(scored[["mpi_moy", "prediction_rf"]].dropna())
        .mark_point(opacity=0.4)
        .encode(
            x=alt.X("mpi_moy:Q", title="MPI réel", scale=alt.Scale(domain=[0, 20])),
            y=alt.Y("prediction_rf:Q", title="Prédit RF", scale=alt.Scale(domain=[0, 20])),
        )
        .properties(width=400, height=350, title="Random Forest — réel vs prédit")
    )

    mo.vstack([
        mo.md("## Comparaison des modèles"),
        comparison_table,
        mo.hstack([nnls_chart, rf_chart], justify="space-around"),
        mo.hstack([scatter_nnls + diag, scatter_rf + diag], justify="space-around"),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
