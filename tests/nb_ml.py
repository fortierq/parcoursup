import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    from parcoursup.mpi import load_mpi
    from parcoursup.ml import learn_mpi_model
    from tests.draw_mpi import ROW_SPECS
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
        "has_nsi_prem", "has_pc_prem",
    ]

    coefficients, scored, metrics = learn_mpi_model(mpi, features)
    return alt, coefficients, metrics, mo, pd, scored


@app.cell
def _(alt, coefficients, metrics, mo, pd, scored):
    metrics_md = mo.md(f"""
    ## Régression linéaire — prédiction de `mpi_moy`

    | Métrique | Valeur |
    |---|---|
    | **R²** | {metrics['r2']} |
    | **Corrélation** | {metrics['correlation']} |
    | **Élèves** | {int(metrics['n_eleves'])} |
    | **Intercept** | {metrics['intercept']} |
    """)

    coeff_chart = (
        alt.Chart(coefficients)
        .mark_bar()
        .encode(
            x=alt.X("coefficient:Q", title="Coefficient standardisé"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.condition(
                alt.datum.coefficient > 0,
                alt.value("#4c78a8"),
                alt.value("#e45756"),
            ),
        )
        .properties(width=500, height=300)
    )

    scatter = (
        alt.Chart(scored[["mpi_moy", "prediction"]].dropna())
        .mark_point(opacity=0.5)
        .encode(
            x=alt.X("mpi_moy:Q", title="MPI réel", scale=alt.Scale(domain=[0, 20])),
            y=alt.Y("prediction:Q", title="MPI prédit", scale=alt.Scale(domain=[0, 20])),
        )
        .properties(width=500, height=400)
    )
    diag = (
        alt.Chart(pd.DataFrame({"x": [0, 20], "y": [0, 20]}))
        .mark_line(color="red", strokeDash=[5, 5])
        .encode(x="x:Q", y="y:Q")
    )

    mo.vstack([
        metrics_md,
        mo.hstack([coeff_chart, scatter + diag], justify="space-around"),
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
