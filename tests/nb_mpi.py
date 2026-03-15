# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas>=2.0",
#     "altair>=6.0.0",
#     "marimo>=0.20.4",
#     "pyarrow>=18.0.0",
#     "pyzmq>=27.1.0",
#     "scikit-learn>=1.7.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from parcoursup.mpi import load_mpi
    from tests.draw_mpi import build_mpi_chart, build_correlation_heatmap, build_correlation_long, build_selection_summary
    import marimo as mo


    mpi = load_mpi()[["nom", "prenom", "fille", "boursier", "mpi_moy", "math_mpi_moy", "pc_mpi_moy", "info_mpi_moy",
                      "points_formule", "nsi_prem", "nsi_term", "pc_prem", "pc_term", "math_spe_prem", "math_spe_term", "math_expertes_term", "fr_prem", "fr_term", "lva_term"]]
    table = mo.ui.table(mpi, selection="multi")
    return (
        build_correlation_heatmap,
        build_correlation_long,
        build_mpi_chart,
        build_selection_summary,
        mo,
        mpi,
        table,
    )


@app.cell
def _(build_mpi_chart, mpi, table):
    table_selection = mpi if table.value.empty else table.value
    chart = build_mpi_chart(table_selection)
    return chart, table_selection


@app.cell
def _(
    build_correlation_heatmap,
    build_correlation_long,
    build_selection_summary,
    chart,
    mo,
    table,
    table_selection,
):

    chart_selection = chart.apply_selection(table_selection)
    moys_selection = build_selection_summary(chart_selection)
    corr_long = build_correlation_long(chart_selection)
    correlation_chart = build_correlation_heatmap(corr_long)

    mo.vstack([
        table,
        mo.hstack([chart, moys_selection]),
        mo.hstack([chart_selection, correlation_chart]),
    ])
    return


if __name__ == "__main__":
    app.run()
