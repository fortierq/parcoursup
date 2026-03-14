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


app._unparsable_cell(
    r"""
    import altair as alt
    import marimo as mo

    from parcoursup.mpi import load_mpi

    mpi = load_mpi()

    chart = mo.ui.altair_chart(
        alt.Chart(mpi)
        .mark_point()
        .encode(
            x=alt.X("points_formule", scale=alt.Scale(domain=[44, 80])),
            y="mpi_moyenne",
            color="fille",
            shape="boursier",
        )
    )
    )
    """,
    name="_"
)


@app.cell
def _(chart, mo, mpi):
    selection = chart.apply_selection(mpi)
    df = selection[[m for m in selection.columns if any(s in m for s in ["moyenne", "terminale", "premiere"])]]
    moyennes_selection = (
        df.mean()
        .round(2)
        .rename("moyenne")
        .reset_index()
        .rename(columns={"index": "variable"})
    )
    mo.vstack([mo.hstack([chart, mo.ui.table(moyennes_selection)]), chart.apply_selection(mpi)])   
    return


if __name__ == "__main__":
    app.run()
