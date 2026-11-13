# Data: Phase 6, Rerun's viewer inside notebooks (D59)

Machine: the development laptop, 13 November 2026.

## The package, before installing it

```
mbd-try\python.exe -m pip install --dry-run "rerun-notebook==0.43.0"
...
Would install rerun-notebook-0.43.0
```

(every requirement already satisfied: anywidget, ipykernel, jupyter-ui-poll,
ipywidgets 8.1.9 and their own).

`pip download --no-deps rerun-notebook==0.43.0` into an empty folder,
unpacked, read only:

| File | Bytes |
|---|---|
| `rerun_notebook-0.43.0-py2.py3-none-any.whl` | 18,019,710 |
| `rerun_notebook/static/re_viewer_bg.wasm` | 70,282,095 |
| `rerun_notebook/static/widget.js` | 554,296 |
| `rerun_notebook/__init__.py` | 11,607 |

The viewer gzipped: 17,806,905 bytes; gzipped and base64-encoded, as the
`inline` option sends it: 23,742,540 bytes.

`curl -I https://app.rerun.io/version/0.43.0/notebook/widget.js`: 200,
`text/javascript`.

rerun-sdk 0.43.0's metadata: `Requires-Dist: rerun-notebook==0.43.0 ; extra
== 'notebook'`.

## The install, as the author would type it

A scratch `.bat` run with `cmd /c`:

```bat
call C:\Users\margk\miniforge3\Scripts\activate.bat C:\Users\margk\miniforge3
set PSExecutionPolicyPreference=
call conda activate mbd-try
pip install rerun-notebook==0.43.0
python -c "import rerun_notebook, rerun; print('rerun-notebook', rerun_notebook.__version__, 'rerun-sdk', rerun.__version__)"
```

Its last lines:

```
Successfully installed rerun-notebook-0.43.0
rerun-notebook 0.43.0 rerun-sdk 0.43.0
```

## A replay in a real kernel

`python/notebooks/pendulum.ipynb` run cell by cell in a Jupyter kernel of
`mbd-try` (`jupyter_client`, no front end), `replay` not redirected. The
replay cell's display messages:

```
[widget eddf1619 ] HTML(value='<div id="d0bb5ffe-2e67-4cc2-afa0-56886bfe2c86"><style onload="eval(a
[widget 2aa2f44f ] VBox(children=(HTML(value=''), <rerun_notebook.Viewer object at 0x0000020D2F950C
```

then the run ended (`ran 10 code cells, 3 figures`), 21.8 s for the whole
notebook, kernel start included; the replay cell's wait for a viewer that
never answers is at most 10 s by the package's code, not timed apart. No
timeout message appeared in the kernel's output.

## Tests

`python -m pytest python/tests/test_viewer.py` in `mbd-try`: 9 passed in
13.99 s.

## The suite, at the gate

`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1 -Test`
(from cmd, output to a file), with the version raised to 0.7.0, the Python
how-to examples and the page check that reads them: 544 of 544 tests passed
in 297.76 s, the laptop on battery (Win32_Battery status 1, discharging, 77
%); 111.77 s the day before on the same suite less one file, power state not
recorded. `Python: test_howto.py` 3.03 s, `Python: test_notebooks.py`
18.04 s, `Python: test_viewer.py` 4.29 s, the page check 0.05 s.

The page check against a changed quote: the guide's `python-run` block
edited (`DormandPrince` to `RK4`), `test_howto.exe "How-to guides*"` failed
at `python.md, line 35: example python-run` (1 of 243 assertions); restored,
all 243 passed.

## The author's environment, at version 0.7.0

`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1
-Install -Python C:\Users\margk\miniforge3\envs\mbd-try\python.exe` ended
with `Successfully uninstalled multibody-0.6.0` and `Successfully installed
multibody-0.7.0`. Then, in `mbd-try`: `import multibody` gives `0.7.0
C:\dev\multibody\python\multibody\__init__.py`, and `python -m pytest
python/tests -q` 72 passed, 10 skipped (the parity tests) in 13.27 s.
