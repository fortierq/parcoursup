import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    from parcoursup.load import load

    annee = 2026
    eleves, notes, lycees = load(annee)
    return eleves, lycees


@app.cell
def _(eleves, lycees):
    lycees["nombre_eleves"] = lycees["uai"].map(eleves["uai"].value_counts()).fillna(0).astype(int)
    lycees.sort_values("nombre_eleves", ascending=False)
    return


if __name__ == "__main__":
    app.run()
