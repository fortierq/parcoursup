# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas>=2.0",
#     "altair>=6.0.0",
#     "marimo>=0.20.4",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import pandas as pd
    import altair as alt

    from parcoursup.load import load

    annee = 2026
    eleves, notes, lycees = load(annee)
    return alt, annee, eleves, mo, notes, pd


@app.cell
def _(eleves):
    eleves
    return


@app.cell
def _(notes):
    notes
    return


@app.cell
def _(eleves, pd):
    stats = pd.DataFrame(
        {
            "Filles": [eleves["fille"].sum()],
            "Garçons": [(~eleves["fille"]).sum()],
            "Boursiers": [eleves["boursier"].sum()],
            "Internat demandé": [eleves["internat"].sum()],
            "NSI term": [eleves["has_nsi_term"].sum()],
            "NSI première": [eleves["has_nsi_prem"].sum()],
            "PC term": [eleves["has_pc_term"].sum()],
            "PC première": [eleves["has_pc_prem"].sum()],
        }
    ).T
    stats.columns = ["Effectif"]
    stats["Proportion"] = (stats["Effectif"] / len(eleves) * 100).round(1).astype(str) + " %"
    stats
    return


@app.cell
def stats_notes(annee, mo, notes, pd):
    short = annee % 100
    matieres = ["philo", "fr", "math_spe", "pc", "nsi", "math_expertes", "lva"]
    summary_rows = []
    for m in matieres:
        if (m, short, "moy") in notes.columns:
            col = notes[(m, short, "moy")]
            summary_rows.append(
                {
                    "Matière": m,
                    "Moyenne": col.mean(),
                    "Médiane": col.median(),
                    "Écart-type": col.std(),
                    "Min": col.min(),
                    "Max": col.max(),
                    "N": col.count(),
                    "Moyenne classe": notes[(m, short, "moy_classe")].mean(),
                }
            )
    summary = pd.DataFrame(summary_rows).round(2)
    mo.md(f"### Moyennes annuelles — Terminale ({short})")
    return (summary,)


@app.cell
def table_notes(mo, summary):
    mo.ui.table(summary)
    return


@app.cell
def distance_revenu(eleves, mo):
    mo.md("### Distance et revenu")
    desc = eleves[["distance", "revenu"]].describe().round(1)
    return (desc,)


@app.cell
def table_distance(desc, mo):
    mo.ui.table(desc.reset_index().rename(columns={"index": ""}))
    return


@app.cell
def avance_retard(eleves, mo):
    mo.md("### Avance / retard scolaire")
    eleves["annees_avance"].value_counts().sort_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Géographie
    """)
    return


@app.cell
def distance(alt, eleves):
    _data = eleves[["distance"]].dropna().copy()
    _data["distance"] = _data["distance"].clip(upper=1000)
    _chart = (
        alt.Chart(_data)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#6366f1")
        .encode(
            x=alt.X("distance:Q", bin=alt.Bin(step=10, extent=[0, 1000]), title="Distance (km)"),
            y=alt.Y("count():Q", title="Effectif"),
            tooltip=[
                alt.Tooltip("distance:Q", bin=alt.Bin(step=10), title="Distance (km)"),
                alt.Tooltip("count():Q", title="Effectif"),
            ],
        )
        .properties(width=1000, height=300, title="Distribution des distances des candidats")
        .interactive()
    )
    _chart
    return


if __name__ == "__main__":
    app.run()
