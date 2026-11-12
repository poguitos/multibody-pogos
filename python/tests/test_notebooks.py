"""The notebooks of python/notebooks/ (plan task 6.6, D58), run from top to
bottom as Jupyter runs them: their code cells in order, in one namespace.

A notebook checks its own numbers against closed forms: its `check` calls
stop it when a difference passes the allowance derived beside it. A notebook
that runs through is therefore a passed test. Plots are drawn off screen.
The Rerun viewer is not opened: each replay is written to a recording in a
temporary folder instead, and Rerun must not warn about any of them."""

import json
import warnings
from pathlib import Path

import pytest

import multibody as mbd

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("*.ipynb"))


def test_the_three_notebooks_are_there():
    assert [path.name for path in NOTEBOOKS] == ["four_bar.ipynb", "pendulum.ipynb", "quarter_car.ipynb"]
    for path in NOTEBOOKS:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["cells"][0]["cell_type"] == "markdown"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_a_notebook_runs_and_its_checks_pass(path, tmp_path, monkeypatch, capfd):
    for needed in ("matplotlib", "pandas", "rerun"):
        pytest.importorskip(needed)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recordings = []

    def to_a_file(view):
        def written(*args, **kwargs):
            kwargs["path"] = tmp_path / f"{len(recordings)}.rrd"
            recordings.append(kwargs["path"])
            return view(*args, **kwargs)
        return written

    monkeypatch.setattr(mbd, "replay", to_a_file(mbd.replay))
    monkeypatch.setattr(mbd, "plot", to_a_file(mbd.plot))
    monkeypatch.chdir(tmp_path)                       # anything a notebook writes lands here
    namespace = {"__name__": "__main__"}
    try:
        with warnings.catch_warnings():               # a notebook's filters end with it
            for k, cell in enumerate(json.loads(path.read_text(encoding="utf-8"))["cells"]):
                if cell["cell_type"] == "code":
                    exec(compile("".join(cell["source"]), f"{path.name}, cell {k + 1}", "exec"), namespace)
    finally:
        plt.close("all")
    assert recordings and all(recording.stat().st_size > 0 for recording in recordings)
    assert "WARN" not in capfd.readouterr().err
