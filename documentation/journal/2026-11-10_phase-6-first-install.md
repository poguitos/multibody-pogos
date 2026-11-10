# Phase 6: the author's first install

- **Dates:** 10 November 2026
- **Plan:** after 6.2 and the tuning; D53 (amended)
- **Commits:** see git log
- **Written:** at the time

## Goal

The author tried to install the package to use it, following
`docs/python.md`, and it failed. They asked for more detailed instructions.
Done when the laptop's steps are written out from the prompt the author
uses, and work when followed exactly as written.

## Starting point

Task 6.2 (D53) made the package installable with `pip install -e .`, building
in a folder kept between installs, `build-python/<wheel tag>/`. On the laptop
the rule was to install through `scripts\build.ps1 -Install`, which sets
Ninja so that the limit of one compiler holds; plain pip on Windows uses
Visual Studio's generator, which ignores it. `docs/python.md` showed
`pip install -e .` first and the script's line further down. Every install
so far had been run by the assistant, from its own PowerShell session.

## What was done

1. **The cause, from the folder's state.** Both of the author's environments
   run Python 3.14, so both builds went to `build-python/cp314-cp314-win_amd64/`,
   where the script's install had built with Ninja in the morning. The
   top-level cache had been written again at the author's run for Visual
   Studio's generator, beside the morning's Ninja files; the three
   dependencies' sub-build caches still said Ninja. Line 56, where the build
   stopped, fetches the first dependency, Eigen: its sub-build is configured
   with the parent's generator in the folder the morning had made for Ninja.
   Nothing had been installed in base, and `mbd-try` still had the morning's
   install.
2. **A second trap, found while reading the instructions as the author
   would.** The laptop's execution policy is Windows' default, Restricted:
   `scripts\build.ps1` typed in a PowerShell window is refused ("la
   ejecución de scripts está deshabilitada en este sistema"). The
   assistant's own session runs with a process-wide Bypass, so this never
   showed. The README's build instructions had the same trap.
3. **The fix.** The script's install builds in its own folder,
   `build-python/ninja-<wheel tag>/` (`-C build-dir=...`), and prints the
   interpreter it installs into. The mixed folder was deleted.
4. **The instructions.** `docs/python.md`'s "Installing it" now opens with
   the laptop's steps from a miniforge3 Prompt: activate an environment
   (which ones not to use, and why), go to the folder, start the script
   with `powershell -NoProfile -ExecutionPolicy Bypass -File` (each part
   explained), what the output ends with and how long it takes, how to check
   the install, VS Code, what to do after a change, the same from
   PowerShell, and the five failures met or caused, each with its message as
   this Windows prints it. Plain pip follows, for other machines. The README,
   the troubleshooting page and `CLAUDE.md` say the same in brief.
5. **Tried as written.** A batch file ran the steps from a state like a new
   miniforge3 Prompt, the policy variable cleared; then the PowerShell form;
   then each failure was caused once to read its message.

## Decisions

D53 amended: the script's install in a folder of its own; the laptop's
instructions as steps from the author's prompt, tried as written. Rejected:
FetchContent without sub-builds (policy CMP0168, CMake 3.30, above the
project's minimum of 3.25; the top-level cache would still change generator
at every switch of route).

## What went wrong, and how it was found

- **Two routes shared one build folder.** D53 kept one folder per wheel tag
  and set the generator per route: plain pip's on Windows is Visual Studio's,
  the script's Ninja. Each route was tested alone (CI's runners start empty;
  the laptop had only ever seen the script), so the clash could only show
  when a person used both. Found by the author.
- **The instructions were written for the assistant's shell.** The
  assistant ran the script from a PowerShell session where scripts are
  allowed; the author's prompts run none. The documented line would have
  been refused even without the folder's clash. Found by reading the steps
  from the author's side and checking the policy with the session's
  variable cleared.
- **A wrong guess in the first draft of the failures.** It said that a
  prompt in the wrong folder gives pip's "does not appear to be a Python
  project". The script installs from its own folder wherever it is started,
  so the failure is PowerShell's: the argument to `-File` does not exist.
  Found by causing it.
- **The plain pip that failed would have broken the laptop's rule had it
  worked:** Visual Studio's generator ignores the limit of one compiler. The
  failure came before any compiler started. The instructions now say first,
  in bold, not to type `pip install -e .` on the laptop.

## Verification

- The steps, as a batch file from a fresh prompt's state: installed, exit
  0; `0.6.0 C:\dev\multibody\python\multibody\__init__.py`; the package's
  tests `43 passed`. The folder used: `build-python\ninja-cp314-cp314-win_amd64`,
  generator Ninja, `MBD_COMPILE_JOBS=1`.
- The PowerShell form: installed again in 37 s, nothing compiled, the same
  folder reused.
- Each failure in the instructions caused once, its message copied into
  them: a plain Command Prompt, the folder step left out, no environment
  active, a folder configured by the other generator (in an empty project).
- Not tried on the laptop: plain `pip install -e .` in its new, empty
  folder, which would run Visual Studio's generator against the laptop's
  rule. CI's package job runs it on Windows at every push.

## Measurements

In `data/2026-11-10_phase-6-first-install.md`: the folder as found, the
policy, the batch file and its timings (10 min 58 s, of which 600.5 s for
Ninja's 59 steps, on battery), the PowerShell form (37 s), the four messages.
The morning's first install took 3 min 41 s, power not noted; installs are
compared only within a session.

## Lessons

- **Try instructions from the reader's starting point,** with the reader's
  settings, not the writer's. The assistant's session differed from the
  author's prompt in its execution policy and its environment, and both
  differences mattered.
- **A cache serves the tool that made it.** Two routes into one build folder
  need the same generator, or folders of their own.
- **Put the warning before the command it warns about.** The old page
  showed `pip install -e .` first and the laptop's exception after it.

## Open ends

- Plain pip on Windows could keep the limit too, with Visual Studio's
  generator told to build one file at a time; not tried.
- Other pages give `scripts\build.ps1 ...` as typed in PowerShell (the
  README now says how to start it under the default policy; `CLAUDE.md`'s
  commands are for the assistant's session).

## For the book

Chapter on the Python interface (6), or on working with an assistant:
instructions are code with a user, and the user's first run is their test;
the clash of two correct routes in one cache; the assistant's environment
as a hidden difference from the reader's.
