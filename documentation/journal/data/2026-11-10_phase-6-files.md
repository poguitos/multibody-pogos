# Data: Phase 6, files (task 6.4)

Raw measurements behind the journal entry `2026-11-10_phase-6-files.md`. The
laptop (i7-13700H, 16 GB, Windows 11, MSVC, one compiler); on mains power
from the first build on (`Win32_Battery.BatteryStatus` 2), unless said.

## The dependency

nlohmann/json 3.12.0: `https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz`,
114,576 bytes, SHA-256
`42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa`
(`sha256sum` of the download), 69 entries, licence MIT (`LICENSE.MIT`).

## The environments

`conda install -n <env> -c conda-forge pandas=3.1.1 pyarrow=26.0.1
numpy=2.5.4 python=<its version>`, after a dry run listing only new
packages (pandas 3.1.1, pyarrow 26.0.1, pyarrow-core 26.0.1 and their
libraries; nothing updated, downgraded or removed):

| Environment | Result |
|---|---|
| `mbd-python` (the suite's, Python 3.12.16) | first attempt: `CondaHTTPError: HTTP 000 CONNECTION FAILED` for `libopentelemetry-cpp-1.30.0`; retried: installed; NumPy 2.5.4, pandas 3.1.1, pyarrow 26.0.1 |
| `mbd-try` (the author's, Python 3.14.10) | installed; NumPy 2.5.4, pandas 3.1.1, pyarrow 26.0.1 |

## The builds

| Run | Result |
|---|---|
| 1: `scripts\build.ps1 -Test` (reconfigure, nlohmann/json fetched; the headers' accessors rebuilding most of the library and tests) | stopped after 447 s at `tests/io/test_model_files.cpp`: C2665, no `Simulator` constructor for a `const System` (the constructor takes `System&`). `src/io/model_file.cpp` had compiled |
| 2: the same, the test's systems not const | 239 s; 540 tests, 538 passed. Failed: the catalogue's two tests: "Documented but used nowhere: MBD-I003, MBD-I005, MBD-I006, MBD-I007" and "Without a code: model_file.cpp:628 model_file.cpp:997" |
| 3: codes written in full at each call, the catalogue's rule extended | 61 s; 540 tests passed, the suite in 36.94 s; `Python: test_parity.py` 5.87 s, `test_files.py` 1.68 s; the five model-file tests 0.01 to 0.06 s each |
| 4: `-Test -Filter Python`, after `read_results` stopped passing `float_precision` to pandas 3 | the 8 Python files passed (`test_parity.py` 6.35 s) |

The two lines the catalogue's test found without a code are the rethrows of
a constructor's error with its place in the file, and of a load's error with
the path; codes 3 to 7 were built as `"MBD-" + code` inside `fail()`.

## The round trip

- The C++ test's system of every kind (eight bodies, one per joint; every
  primitive constraint, every closure, a set made by hand and one named like
  a closure; five kinds of force element; a joint force): its text written,
  read and written again, the same; `describe()` the same; the
  accelerations at a generic state (`Rng(29)`, t = 0.3 s) identical bit for
  bit, and finite.
- A four-bar on a spring and a damper, saved to a file and loaded: the file's
  text equals `model_to_json`; 1 s of RK4 at 1 ms from crank angle 1.2 rad,
  q and v identical bit for bit.

What the comparison sees (`files_mutations.py`, scratch): the parallelogram
of the Python tests with a nonlinear spring and a linear one, 96 lines and
2,697 characters of text; read back from altered text, then written and run
0.5 s from the same state:

| Text read | Written again the same | Coordinates of q that differ, of 1,503 |
|---|---|---|
| unaltered | yes | 0 |
| the spring's `preload` removed (as a reader that dropped it) | no | 1,290 |
| an anchor's 0.5 one ulp up (`0x1.0000000000001p-1`) | no | 1,290 |

## Results as tables

`python/tests/test_files.py`, 9 tests in `mbd-python` (pandas 3.1.1, pyarrow
26.0.1): all passed, none skipped. With `float_precision="round_trip"`, pandas
3.1.1 warned twice: "Pandas4Warning: The 'float_precision' argument is
deprecated. Use the default float precision instead." Without it, the CSV
round trip stayed identical to the bit (-0.0 and 0.30000000000000004
included); the option is kept for pandas 2.

## The author's environment

`scripts\build.ps1 -Install -Python %USERPROFILE%\miniforge3\envs\mbd-try\python.exe`
(its own folder, `build-python\ninja-cp314-cp314-win_amd64`, the library
rebuilt after the headers' change): 1,251 s; `python -m pytest python/tests`:
53 passed, 10 skipped (the parity tests, without the suite's build).
