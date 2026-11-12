# Data: Phase 6, viewing (task 6.5)

Raw measurements behind the journal entry `2026-11-12_phase-6-viewing.md`. The
laptop (i7-13700H, 16 GB, Windows 11), on battery for the builds and runs
below (`Win32_Battery.BatteryStatus` 1, 72 %).

## Rerun

`conda install -n <env> -c conda-forge rerun-sdk=0.43.0` with NumPy, pandas,
pyarrow and Python pinned, after a dry run listing only new packages
(rerun-sdk 0.43.0, 114.2 MB, the build `py311h3a8d568_0`, for Python 3.11
and later; anywidget, ipywidgets and others): `mbd-python` and `mbd-try`,
rerun 0.43.0 in both, NumPy 2.5.4, pandas 3.1.1 and pyarrow 26.0.1 unchanged.
PyPI's newest was 0.43.1.

What 0.43.0 offers, read from its signatures (`rerun_probe*.py`, scratch):
`RecordingStream` with `save`, `spawn`, `send_columns`, `send_blueprint`;
`Transform3D.columns(translation, quaternion, ...)`, `Boxes3D`, `Points3D`,
`LineStrips3D.columns(...).partition(...)` (one strip per time step, tried),
`Scalars`, `ViewCoordinates.RIGHT_HAND_{X,Y,Z}_{UP,DOWN}`,
`escape_entity_path_part`; the blueprint's `Horizontal`, `Vertical`,
`Spatial3DView`, `TimeSeriesView`. No `rerun.dataframe` (ModuleNotFoundError)
and no public methods on `rerun.experimental.Recording`: no reader of `.rrd`
files was found in the package.

## The suite

`scripts\build.ps1 -Test` (the bindings and the model files' test
recompiled): 188 s; 542 tests passed, the suite 74.49 s; `test_viewer.py`
4.07 s, `test_parity.py` 10.58 s, `test_files.py` 4.34 s; the new C++ test
"the HMMWV, its tyres left out, survives a round trip" 0.05 s.
`pytest -rs test_viewer.py`: 7 passed, none skipped.

## The car, recorded and replayed

`scripts\build.ps1 -Target mbd_hmmwv`: 33 s. `mbd_hmmwv.exe record <folder>`:

| Variant | Samples | Coordinates | Steps (Radau IIA, tol 1e-6) | Events | Time | Files |
|---|---|---|---|---|---|---|
| joints | 501 | 32 | 2,253 | 24 | 1.6 s | `hmmwv.json` 20,343 bytes, `hmmwv_run.csv` 365,660 bytes |
| bushings | 501 | 44 | 2,065 | 24 | 2.4 s | `hmmwv_bushed.json` 24,883 bytes, `hmmwv_bushed_run.csv` 504,037 bytes |

Each without its 4 tyres. Replayed to recordings (`replay_all.py`, scratch,
calling the two examples with a path):

| Recording | Bytes | Time | What was drawn |
|---|---|---|---|
| four-bar (`replay_four_bar.py`) | 507,770 | 2.3 s (the run included) | |
| HMMWV with joints | 922,011 | 0.1 s | 301 frames; 14 bodies, each a box of its inertia; 12 lines (springs, shocks, tie rods); up Z; the chassis from x = 0.000 to 21.450 m |
| HMMWV with bushings | 1,161,338 | 0.1 s | the same; 20 lines (the bushings' spring-dampers added) |

The recordings were not looked at: the assistant has no display. They are
for the author to open (`rerun <file>`).

## A warning, then none

The first replays wrote, on the process's stderr, one warning for each
closure's or contact's point, such as:

```
WARN re_log_types::path::parse_path: When parsing the entity path "world/ground/points/0 constraint\\ 0\\ revolute\\ closure": Unescaped whitespace. The path will be interpreted as /world/ground/points/0\ constraint\ 0\ revolute\ closure
```

(the index before the escaped label was outside the escaping). Escaped
whole, the same replays wrote nothing on stderr. The recordings' test now
requires no "WARN" on the process's stderr (pytest's `capfd`); with the old
path restored in the build's copy it fails, quoting those lines.

## The author's environment

`scripts\build.ps1 -Install -Python ...mbd-try\python.exe`: 94 s. Then
`python -m pytest python/tests` stopped at the viewer's first test with
"Windows fatal exception: code 0xc06d007f" in `numpy.linalg.eigh` (from
`_viewer._inertia_box`). `conda list -n mbd-try`: libblas, libcblas,
liblapack `3.11.0-12_..._mkl`, `mkl 2026.1.0`, there since the environment's
creation (its `conda-meta/history`: `create -n mbd-try ... python=3.14 numpy
matplotlib ipykernel pytest`, MKL's packages added then). `mbd-python` has
the OpenBLAS builds (its `environment.yml` pins them for this reason).

`conda install -n mbd-try "libblas=*=*openblas"` (with the other pins), after
a dry run: libopenblas 0.3.34 added; libblas, libcblas, liblapack to their
`_openblas` builds; MKL removed. Then `numpy.linalg.eigh(numpy.eye(3))`:
`[1. 1. 1.]`; `python -m pytest python/tests`: 60 passed, 10 skipped (the
parity tests); `replay_four_bar.py` to a file: 507,825 bytes.

The documented command for such an environment, with the pin, as a dry run
(`conda create -n mbd-docs-check ... "libblas=*=*openblas" --dry-run`):
python 3.14.10, NumPy 2.5.4, pandas 3.1.1, pyarrow 26.0.1, rerun-sdk 0.43.0,
libblas `_openblas`; no MKL.
