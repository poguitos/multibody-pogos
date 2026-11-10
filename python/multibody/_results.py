"""Results as tables (plan task 6.4, D56).

A recorded run (``sim.run_recorded``) or a ``Recorder`` becomes a pandas
DataFrame indexed by time; ``write_results`` and ``read_results`` keep such a
table in Parquet or CSV, every number read back as the same double.

The columns of a run have two levels: the vector (``q``, ``v``, ``x``) and the
coordinate, named after the body (``df["q"]["crank"]``, or
``df[("q", "crank")]``). A joint with several coordinates numbers them
(``block[0]`` to ``block[6]`` for a free joint: x, y, z, then the quaternion's
x, y, z, w); a body without a name is ``body<i>``, and a name several bodies
share is followed by ``#<i>``. The states of state elements are numbered.

pandas is needed here only, and pyarrow for Parquet only: ``import multibody``
needs neither.
"""

from pathlib import Path

import numpy as np

from ._errors import MbdError


def _pandas():
    try:
        import pandas
    except ImportError as missing:
        raise ImportError("MBD-P012: results as tables need pandas (pip install pandas, and pyarrow for Parquet; "
                          "or the package's extra, pip install -e \".[results]\")") from missing
    return pandas


def coordinate_names(model, kind="q"):
    """The names of a model's coordinates (kind "q") or velocities ("v"), in
    the order of the state vectors; see the module's docstring."""
    if kind not in ("q", "v"):
        raise ValueError("kind is 'q' or 'v'")
    names = [model.name[b] or f"body{b}" for b in range(model.nbodies())]
    shared = {n for n in names[1:] if names[1:].count(n) > 1}
    counts = model.nqs if kind == "q" else model.nvs
    out = []
    for b in range(1, model.nbodies()):
        label = f"{names[b]}#{b}" if names[b] in shared else names[b]
        out += [label] if counts[b] == 1 else [f"{label}[{k}]" for k in range(counts[b])]
    return out


def _model_of(source):
    from ._mbd import Model, Simulator, System
    if isinstance(source, Simulator):
        return source.system.model
    if isinstance(source, System):
        return source.model
    if isinstance(source, Model):
        return source
    raise TypeError("the names come from a Simulator, a System or a Model")


def dataframe(source, names_from=None):
    """A table indexed by time.

    ``dataframe(sim.run_recorded(T, dt), sim)``: a run, its columns (vector,
    coordinate), the names taken from ``names_from`` (the simulator, its
    system or its model). ``dataframe(recorder)``: a Recorder's channels, one
    column each."""
    pd = _pandas()
    from ._mbd import Recorder
    if isinstance(source, Recorder):
        data = source.as_dict()
        time = data.pop("time")
        return pd.DataFrame(data, index=pd.Index(time, name="time"))
    if not (isinstance(source, dict) and {"time", "q", "v", "x"} <= source.keys()):
        raise TypeError("dataframe() takes a run (sim.run_recorded(...)) with the simulator, or a Recorder")
    if names_from is None:
        raise TypeError("a run's table takes its names from the simulator: dataframe(run, sim)")
    model = _model_of(names_from)
    columns, blocks = [], []
    for kind in ("q", "v"):
        names = coordinate_names(model, kind)
        block = np.asarray(source[kind])
        if block.shape[1] != len(names):
            raise MbdError(f"MBD-P013: the run has {block.shape[1]} values of {kind} per sample; the model "
                           f"{len(names)}: it is not this model's run")
        columns += [(kind, n) for n in names]
        blocks.append(block)
    x = np.asarray(source["x"])
    columns += [("x", str(k)) for k in range(x.shape[1])]
    blocks.append(x)
    return pd.DataFrame(np.hstack(blocks), index=pd.Index(np.asarray(source["time"]), name="time"),
                        columns=pd.MultiIndex.from_tuples(columns, names=["vector", "coordinate"]))


def _format(path):
    suffix = Path(path).suffix.lower()
    if suffix not in (".parquet", ".csv"):
        raise MbdError(f"MBD-P014: a results file ends in .parquet or .csv; '{path}' does not")
    return suffix


def write_results(frame, path):
    """Write a table (from dataframe()) to `path`: Parquet or CSV by its
    ending. CSV numbers are written in full, so that they read back exactly."""
    if _format(path) == ".parquet":
        try:
            frame.to_parquet(path)
        except ImportError as missing:
            raise ImportError("MBD-P012: Parquet needs pyarrow (pip install pyarrow)") from missing
    else:
        frame.to_csv(path)


def read_results(path):
    """The table in a results file written by write_results() (or a
    Recorder's write_csv()), as it was written."""
    pd = _pandas()
    if _format(path) == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError as missing:
            raise ImportError("MBD-P012: Parquet needs pyarrow (pip install pyarrow)") from missing
    with open(path, encoding="utf-8") as f:
        two_levels = f.readline().startswith("vector,")
    # Every number read back as the same double: pandas 3 parses so by
    # default (and deprecates the option); pandas 2 needs "round_trip".
    exact = {} if int(pd.__version__.split(".")[0]) >= 3 else {"float_precision": "round_trip"}
    return pd.read_csv(path, header=[0, 1] if two_levels else 0, index_col=0, **exact)
