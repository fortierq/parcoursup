import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    from parcoursup.mpi import load_mpi
    from tests.draw_mpi import build_mpi_chart, build_correlation_heatmap, build_correlation_long, build_selection_summary
    import marimo as mo

    mpi = load_mpi()[["nom", "prenom", "fille", "boursier", "mpi_moy", "math_mpi_moy", "pc_mpi_moy", "info_mpi_moy", "lv1_mpi_moy", "fr_mpi_moy",
                      "points_formule", "nsi_prem", "nsi_term", "pc_prem", "pc_term", "math_spe_prem", "math_spe_term", "math_expertes_term", "fr_prem", "fr_term", "lva_term", "pourcentage_tb"]]
    table = mo.ui.table(mpi, selection="multi",show_column_summaries=False)
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
    mpi,
    table,
    table_selection,
):

    chart_selection = chart.apply_selection(table_selection)
    moys_selection = build_selection_summary(
        chart_selection, mpi)
    corr_long = build_correlation_long(chart_selection)
    correlation_chart = build_correlation_heatmap(corr_long)

    mo.vstack([
            table,
            mo.hstack([chart, mo.vstack([moys_selection], align="center")], justify="space-around"),
            mo.hstack([chart_selection], justify="space-around"),
            mo.vstack([correlation_chart], align="center"),
        ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
