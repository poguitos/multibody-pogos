# Data: Phase 6, the author's first install

Raw observations behind the journal entry `2026-11-10_phase-6-first-install.md`.
The laptop: i7-13700H, 16 GB, Windows 11 Home in Spanish, Miniforge 26.9.4 in
`C:\Users\<you>\miniforge3`, CMake 4.2.0 in `C:\Program Files\CMake`.

## The failure, as the author reported it

`pip install -e .` typed in a miniforge3 Prompt, in the base environment and
in `mbd-try`. The end of the output, as pasted:

```
C:/Program Files/CMake/share/cmake-4.2/Modules/FetchContent.cmake:2394 (__FetchContent_Populate)
  CMakeLists.txt:56 (FetchContent_MakeAvailable)

-- Configuring incomplete, errors occurred!

*** CMake configuration failed
[end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed building editable for multibody
Failed to build multibody
error: failed-wheel-build-for-install
```

Line 56 of `CMakeLists.txt` is `FetchContent_MakeAvailable(eigen)`, the first
dependency fetched.

## The folder it failed in

`build-python\` held one folder, `cp314-cp314-win_amd64` (both environments
run Python 3.14.10, so both builds went there). In it:

| File | Last written | What it says |
|---|---|---|
| `CMakeCache.txt` | 16:06:42 | `CMAKE_GENERATOR:INTERNAL=Visual Studio 18 2026`, `CMAKE_COMMAND` = `C:/Program Files/CMake/bin/cmake.exe` |
| `.skbuild-info.json`, `CMakeInit.txt` | 16:06:38, 16:06:39 | written by the author's run |
| `build.ninja`, `cmake_install.cmake` | 11:20:01 | the script's install of the morning (Ninja) |
| `.ninja_log`, `install_manifest_python.txt` | 11:22:17 | the same |
| `_deps\eigen-subbuild\CMakeCache.txt` | 09:48:19 | `CMAKE_GENERATOR:INTERNAL=Ninja`; the folder itself touched 16:06:42 |
| `_deps\pybind11-subbuild\CMakeCache.txt` | 09:48:22 | `Ninja` |
| `_deps\spdlog-subbuild\CMakeCache.txt` | 09:48:29 | `Ninja` |

- `C:\Users\<you>\miniforge3\python.exe -m pip show multibody`: "Package(s)
  not found: multibody" (nothing was installed in base).
- `mbd-try` still imported the morning's install: `multibody.__file__` in
  the repository, `multibody._mbd.__file__` in its `site-packages`.

The CMake message the log carried above the lines pasted was not seen. Its
form, reproduced in an empty project (`project(genclash LANGUAGES NONE)`)
configured with Visual Studio's generator and then with Ninja in the same
folder:

```
CMake Error: Error: generator : Ninja
Does not match the generator used previously: Visual Studio 18 2026
Either remove the CMakeCache.txt file and CMakeFiles directory or choose a different binary directory.
```

## The execution policy

`Get-ExecutionPolicy -List`: MachinePolicy, UserPolicy, CurrentUser and
LocalMachine `Undefined`; Process `Bypass`, which is the assistant's session
only (kept in the variable `PSExecutionPolicyPreference`, inherited by the
processes it starts). With that variable cleared, as in the author's prompts:

- `powershell -NoProfile -Command Get-ExecutionPolicy`: `Restricted`;
- `powershell -NoProfile -File hello.ps1`: "No se puede cargar el archivo
  ...\hello.ps1 porque la ejecución de scripts está deshabilitada en este
  sistema." (`UnauthorizedAccess`);
- `powershell -NoProfile -ExecutionPolicy Bypass -File hello.ps1`: ran.

The Start menu's shortcut "miniforge3 Prompt":
`C:\Windows\System32\cmd.exe /K C:\Users\<you>\miniforge3\Scripts\activate.bat C:\Users\<you>\miniforge3`.

## The steps of `docs/python.md`, tried as written

The folder `build-python\cp314-cp314-win_amd64` deleted first. A batch file
did what the steps say, from a state like a new miniforge3 Prompt
(`PSExecutionPolicyPreference`, `CMAKE_GENERATOR` and `PYTHONPATH` cleared;
`cl.exe` not on the PATH):

```bat
call C:\Users\<you>\miniforge3\Scripts\activate.bat C:\Users\<you>\miniforge3
call conda activate mbd-try
cd /d C:\dev\multibody
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1 -Install -Python "%CONDA_PREFIX%\python.exe"
python -c "import multibody as mbd; print(mbd.__version__, mbd.__file__)"
python -m pytest python/tests -q
```

(`call` is needed in a batch file, not at the prompt.)

- Started 16:14:28.99; the install ended 16:25:26.89, exit code 0: 10 min
  58 s. From the build folder's file times: pip's build requirements to
  16:14:42 (scikit-build-core 1.1.2, packaging 26.3, pathspec 1.1.1; 14 s);
  the configuration with the downloads of Eigen, pybind11 and spdlog to
  16:15:19 (37 s); Ninja's 59 steps 600.5 s (`.ninja_log`); the install to
  16:25:26.
- The longest steps: linking `_mbd.cp314-win_amd64.pyd` 62.1 s;
  `simulator_implicit.cpp` 55.1 s; `linearization.cpp` 54.0 s;
  `statics.cpp` 31.7 s; `bind_analysis.cpp` 30.2 s.
- The cache: `CMAKE_GENERATOR` Ninja, `MBD_COMPILE_JOBS=1`, Release.
- The laptop was on battery: `Win32_Battery.BatteryStatus` 1 (discharging),
  76 %; power plan "SAMSUNG MODE"; processor load 5 % just after. The
  morning's first install, 3 min 41 s with 58 steps (data of the packaging
  entry), was not noted for power.
- The output: "Installing the package multibody into
  C:\Users\<you>\miniforge3\envs\mbd-try\python.exe" first; "Successfully
  uninstalled multibody-0.6.0", "Successfully installed multibody-0.6.0" at
  the end; the log 167 lines with the batch file's own.
- The check: `0.6.0 C:\dev\multibody\python\multibody\__init__.py`;
  `43 passed in 0.37s`.
- `build-python\` then held only `ninja-cp314-cp314-win_amd64`.

The PowerShell form, from PowerShell with the policy variable cleared:
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build.ps1
-Install -Python $env:USERPROFILE\miniforge3\envs\mbd-try\python.exe`:
exit 0 in 37 s, nothing compiled (`[0/2] Re-checking globbed directories`,
`[1/2] Copying the multibody package's Python files`).

## The failure messages of the steps, each caused once

| Cause | Message |
|---|---|
| A plain Command Prompt (PATH without Miniforge), `conda activate mbd-try` | `"conda" no se reconoce como un comando interno o externo, programa o archivo por lotes ejecutable.` |
| Step 3 left out (prompt in `C:\Users\<you>`) | `El argumento 'scripts\build.ps1' para el parámetro -File no existe. Proporcione la ruta de acceso a un archivo '.ps1' existente como argumento para el parámetro -File.` |
| No environment active (`CONDA_PREFIX` unset) | `No Python interpreter at '%CONDA_PREFIX%\python.exe'.` (after the script loaded MSVC) |
| A folder configured by the other generator | the CMake error above |
