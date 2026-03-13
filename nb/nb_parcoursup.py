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

    from parcoursup.processing import load

    return alt, load, mo, pd


@app.cell
def chargement(load, mo):
    annee = 2026
    eleves, notes = load(annee)
    mo.md(
        f"## Données Parcoursup {annee} — MP2I\n\n**{len(eleves)}** candidats chargés")
    return annee, eleves, notes


@app.cell
def _(notes):
    notes
    return


@app.cell
def _(eleves, mo, pd):
    stats = pd.DataFrame({
        "Filles": [eleves["fille"].sum()],
        "Garçons": [(~eleves["fille"]).sum()],
        "Boursiers": [eleves["boursier"].sum()],
        "Terminale France": [eleves["terminale_france"].sum()],
        "Internat demandé": [eleves["internat"].sum()],
    }).T
    stats.columns = ["Effectif"]
    stats["Proportion"] = (stats["Effectif"] / len(eleves)
                           * 100).round(1).astype(str) + " %"
    mo.md("### Profil des candidats")
    return (stats,)


@app.cell
def table_stats(mo, stats):
    mo.ui.table(stats.reset_index().rename(columns={"index": "Critère"}))
    return


@app.cell
def stats_notes(annee, mo, notes, pd):
    short = annee % 100
    matieres = ["philo", "fr", "math_spe", "pc", "nsi", "math_expertes", "lva"]
    summary_rows = []
    for m in matieres:
        if (m, short, "moyenne") in notes.columns:
            col = notes[(m, short, "moyenne")]
            summary_rows.append({
                "Matière": m,
                "Moyenne": col.mean(),
                "Médiane": col.median(),
                "Écart-type": col.std(),
                "Min": col.min(),
                "Max": col.max(),
                "N": col.count(),
                "Moyenne classe": notes[(m, short, "moyenne_classe")].mean()
            })
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
    distrib = eleves["annees_avance"].value_counts().sort_index()
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
            x=alt.X("distance:Q", bin=alt.Bin(
                step=10, extent=[0, 1000]), title="Distance (km)"),
            y=alt.Y("count():Q", title="Effectif"),
            tooltip=[
                alt.Tooltip("distance:Q", bin=alt.Bin(
                    step=10), title="Distance (km)"),
                alt.Tooltip("count():Q", title="Effectif"),
            ],
        )
        .properties(width=1000, height=300, title="Distribution des distances des candidats")
        .interactive()
    )
    _chart
    return


@app.cell
def _():
    return


@app.cell
def dropdown_attribut(mo):
    mo.md("---\n## 🔍 Exploration par attribut")

    ATTRIBUTS = {
        "fille": "Sexe (fille)",
        "boursier": "Boursier",
        "terminale_france": "Terminale France",
        "internat": "Internat demandé",
        "public": "Établissement public",
        "scolarisation": "Scolarisé cette année",
        "annees_avance": "Années d'avance",
        "distance": "Distance (km)",
        "revenu": "Revenu brut global",
        "departement": "Département",
        "niveau_classe": "Niveau classe",
    }

    dropdown = mo.ui.dropdown(
        options=ATTRIBUTS.keys(),
        value="fille",
        label="Attribut à explorer",
    )
    dropdown
    return (dropdown,)


@app.cell(hide_code=True)
def exploration(alt, dropdown, eleves, mo, pd):
    def _():
        attr = dropdown.value
        if attr is None:
            mo.md("*Sélectionnez un attribut ci-dessus.*")

        col = eleves[attr]
        is_bool = col.dtype == "bool"
        is_numeric = pd.api.types.is_numeric_dtype(col)

        # ── Tableau de statistiques ──────────────────────────────────────────
        if is_bool:
            n_true = col.sum()
            n_false = (~col).sum()
            tbl = pd.DataFrame({
                "Valeur": ["Oui", "Non"],
                "Effectif": [n_true, n_false],
                "Proportion": [
                    f"{n_true / len(col) * 100:.1f} %",
                    f"{n_false / len(col) * 100:.1f} %",
                ],
            })
        elif is_numeric:
            desc = col.describe().round(2)
            tbl = desc.reset_index()
            tbl.columns = ["Statistique", "Valeur"]
        else:
            vc = col.value_counts().head(20)
            tbl = pd.DataFrame({"Valeur": vc.index, "Effectif": vc.values})
            tbl["Proportion"] = (tbl["Effectif"] / len(col)
                                 * 100).round(1).astype(str) + " %"

        # ── Graphique Altair ─────────────────────────────────────────────────
        label = dropdown.selected_key if hasattr(
            dropdown, "selected_key") else attr

        if is_bool:
            plot_df = pd.DataFrame({
                label: ["Oui", "Non"],
                "Effectif": [n_true, n_false],
            })
            chart = (
                alt.Chart(plot_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X(f"{label}:N", axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("Effectif:Q"),
                    color=alt.Color(f"{label}:N", scale=alt.Scale(
                        scheme="tableau10"), legend=None),
                    tooltip=[f"{label}:N", "Effectif:Q"],
                )
                .properties(width=300, height=300, title=f"Répartition — {label}")
            )
        elif is_numeric:
            plot_df = col.dropna().reset_index(drop=True).to_frame(name=label)
            chart = (
                alt.Chart(plot_df)
                .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                .encode(
                    x=alt.X(f"{label}:Q", bin=alt.Bin(
                        maxbins=40), title=label),
                    y=alt.Y("count():Q", title="Effectif"),
                    tooltip=[
                        alt.Tooltip(f"{label}:Q", bin=alt.Bin(
                            maxbins=40), title=label),
                        alt.Tooltip("count():Q", title="Effectif"),
                    ],
                    color=alt.value("#6366f1"),
                )
                .properties(width=500, height=300, title=f"Distribution — {label}")
            )
        else:
            vc_top = col.value_counts().head(15)
            plot_df = pd.DataFrame(
                {"Valeur": vc_top.index, "Effectif": vc_top.values})
            chart = (
                alt.Chart(plot_df)
                .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
                .encode(
                    x=alt.X("Effectif:Q"),
                    y=alt.Y("Valeur:N", sort="-x"),
                    color=alt.value("#8b5cf6"),
                    tooltip=["Valeur:N", "Effectif:Q"],
                )
                .properties(width=500, height=max(200, len(vc_top) * 28), title=f"Top valeurs — {label}")
            )
        return mo.vstack([
            mo.md(f"### Statistiques : {label}"),
            mo.ui.table(tbl),
            mo.as_html(chart),
        ])

    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
