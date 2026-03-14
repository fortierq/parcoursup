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
    chart = mo.ui.altair_chart(alt.Chart(mpi).mark_point().encode(x="Horsepower", y="Miles_per_Gallon", color="Origin"))

    return


if __name__ == "__main__":
    app.run()
