# Help: the troubleshooting guide and the debugging aids

- **Dates:** 5 October 2026
- **Plan:** tasks H.3 and H.4; decisions D23 (help grown with the program),
  D31
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session

## Goal

H.3: a troubleshooting guide, from symptom to cause to fix, for builds,
models, simulations and slow runs, and how to use the debugging tools. *Done
when every problem met so far in the journal has its entry.* H.4: debugging
aids in the program, `describe(system)`, a step trace written to CSV, a
state dump. *Done when a test shows a deliberately broken model diagnosed
from its trace alone.* The two are done together because the guide's last
section explains the tools the second one adds.

## Starting point

The help had its index (H.1) and the message catalogue (H.2), and a
placeholder for troubleshooting. Phase 3 had just been finished, adding
assembly, statics, linearisation, contact and events, each with its own
report and codes. The journal held some twenty problems met since December
2025, each with how it was found.

## What was done

1. **The trace** (`Simulator::trace`, `TraceRow`). One row at the first
   step's start and after each step: drift after the integrator's step and
   before the projection, the projection's iterations and residuals, rank,
   kinetic and gravity energies, power and work of the applied and
   constraint forces, the energy balance, the largest speed, the events. The
   work is integrated with the integrator's own stages (D31); the constraint
   forces' power `lambda . nu` is computed in every `acceleration()` (two dot
   products).
2. **`kernel/diagnostics.hpp`**: `describe(system[, q])`, `dump_state(sim)`,
   `write_trace_csv` and `read_trace_csv` (17 digits, read back exactly), and
   `diagnose(trace)` with codes K100 to K104 and A030.
3. **The troubleshooting guide** (`docs/help/troubleshooting.md`): eight
   sections and a table that maps each problem in the journal to its entry.
4. Four tests (`tests/kernel/test_kernel_diagnostics.cpp`).

## Decisions

D31: a balance of kinetic and gravity energy against the work of everything
else, so that no element needs a potential; the work by the integrator's own
rule. Energy put in by the applied forces is judged by the growth of the
energy's peaks over the run.

## What went wrong, and how it was found

- **A spring looked like an energy source.** The first version of the K102
  check flagged a run whenever the applied forces had done positive work and
  the energy had risen. A stiff spring released from compression, simulated
  correctly, was flagged: its stored energy is not in the energy column (only
  kinetic and gravity are), so its release looks like energy put in. The
  check now compares the energy's peaks in the last quarter of the run with
  those in the first; a spring cannot raise them, a wrong-signed damper does.
  Its message says that over less than one period this can still mislead.
- **A roundoff rise counted as a rise.** The same check flagged the stiff
  spring integrated with too long a step, where the energy "rose" by 1e-31 J.
  The rise must now exceed 1e-3 of the energy exchanged.
- **The trapezoidal rule for the work.** The first trace integrated the work
  by the trapezoidal rule between step ends; reasoning about its error
  (second order, about 1e-3 of the energy at omega dt = 0.1) showed it would
  raise false alarms, and it was replaced by the RK4 stages' powers before
  any run depended on it.
- **A closure's bodies listed five times.** `describe` printed each body of a
  revolute closure once per primitive; it now lists each once.
- **A claim in the guide.** The draft said a 10 kg block on contact needs a
  step under 0.6 ms "with the contact's defaults"; the default friction is
  zero, and the figure holds for mu = 0.5 and a slip speed of 1 mm/s. Found
  rereading before the commit.

## Verification

| Check | Measured | Bound, and why |
|---|---|---|
| A damped pendulum, 5 s at 1 ms: the balance while the damper takes 0.60 J out | 5e-12 J | 1e-6 J: RK4's energy error, (w dt)^4 per radian, about 1e-8 J |
| The trace written and read back | equal, bit for bit | |
| A healthy trace diagnosed | nothing to report | |
| A four-bar whose loop cannot close, diagnosed from its file | K100 (and K101, K103, K104 as consequences) | |
| A pendulum whose damper has the wrong sign | K102, no K101 | |
| A stiff spring at a step too long (w dt = 2.5) / at a step that resolves it | K101 / nothing | |
| `describe` of the closed four-bar: parts and "= 1" degree of freedom | as stated | |
| `dump_state`: bodies, residuals, last projection | as stated | |
| Every journal problem in the guide's table | 26 rows | |

426 tests pass with GCC 13 (422 before); the no-allocation checks pass.

## Measurements

[data/2026-10-05_trace.md](data/2026-10-05_trace.md): the balance, and the
diagnoses of the healthy and broken models with their full output.

## Lessons

- A diagnostic must be tested on healthy runs as much as on broken ones: both
  false alarms here were found that way, and a diagnostic that cries wolf is
  soon ignored.
- Writing the guide from the journal made the journal's value concrete: every
  entry under "What went wrong" became a symptom someone else will meet.

## Open ends

- H.5, the API reference built in CI; H.6 at the Phase 3 gate.
- A trace column for each element's power, when a model's energy needs to be
  traced to the element responsible.
- Python access to the trace and the diagnosis (Phase 6).

## For the book

- The energy balance without potentials: the work of everything else
  integrated alongside the motion, and why the integrator's own rule
  matters.
- The journal turned into a troubleshooting guide: the engineering-practice
  chapter's example of a record that pays for itself.
