# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pandas>=2.0",
#     "altair>=6.0.0",
#     "marimo>=0.20.4",
#     "pyzmq>=27.1.0",
#     "scikit-learn>=1.7.0",
# ]
# ///

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import altair as alt
    import marimo as mo

    from parcoursup.mpi import load_mpi

    mpi = load_mpi()
    mpi
    return alt, mo, mpi


@app.cell
def _(alt, mo, mpi):
    from numpy import shape
    chart = mo.ui.altair_chart(alt.Chart(mpi).mark_point().encode(
        x="points_formule", 
        y="mpi_moyenne", 
        color="fille",
        shape="boursier"
        )
    )
    return (chart,)


@app.cell
def _(chart, mo):
    mo.vstack([chart, mo.ui.table(chart.value)])
    return


@app.cell
def _(chart):
    print(chart.to_json())

    return


if __name__ == "__main__":
    app.run()
