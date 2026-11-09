# The regression harness

Plan task 5.5 (decision D49). Scenarios that run the library
deterministically and record named channels, compared with reference results
stored here (`data/<scenario>.csv`). A reference is this library's own
earlier result: the harness does not say a result is right (the other tests
do that), it says the result has not changed.

| Scenario | What it runs | What it covers |
|---|---|---|
| `four-bar` | A four-bar closed by a revolute closure, RK4 at 1 ms, 2 s | Loops with redundant equations, projection |
| `heavy-top` | A heavy top on a spherical closure at 30 rad/s, generalised-alpha at 1 ms, 2 s | Gyroscopic terms, quaternions, an implicit method |
| `bushed-pendulum` | The bushed double pendulum (1e6 N/m), Radau IIA at fixed steps of 1 ms, 1 s | Bushings, Radau IIA |
| `hmmwv-joints` | The HMMWV of task 5.4 over the bump, RK4 at 0.5 ms, 1.2 s | A vehicle's closed loops, springs, tyres, contact switching |
| `hmmwv-bushings` | The same with the paper's bushings, generalised-alpha at 1 ms | Stiff spring-dampers |
| `sedan-drive` | The double-wishbone sedan at rest, accelerating, steering; RK4 at 1 ms, 2.3 s | Pacejka tyres, drivetrain, steering, state elements |
| `sedan-modes` | The sedan's static equilibrium and modes at rest | Statics, linearisation |
| `wishbone-sweep` | The sports car's front double wishbone over +-0.05 m | Kinematic analysis |
| `lap-oval` | The lap speed profile of an oval | The lap model |
| `hmmwv-joints-5s`, `hmmwv-bushings-5s`, `sedan-slalom` | The paper's 5 s; a 20 s slalom | The long suite (nightly) |

## The tolerance

Every channel is held to 1e-8 of its largest magnitude in the reference. The
hidden case `[sensitivity]` measures where that sits (9 November 2026, the
author's laptop, MSVC):

| Scenario | 1e-12 change, over the tolerance | 1e-6 change, over the tolerance |
|---|---|---|
| four-bar | 6.9e-4 | 693 |
| heavy-top | 8.1e-4 | 809 |
| bushed-pendulum | 1.7e-4 | 177 |
| hmmwv-joints | 6.0e-5 | 60 |
| hmmwv-bushings | 2.4e-4 | 60 |
| sedan-drive | 1.0e-3 | 997 |
| sedan-modes | 5.1e-3 | 26 |
| wishbone-sweep | 6.8e-4 | 682 |
| lap-oval | 4.2e-5 | 41 |

A relative change of 1e-12 in one physical parameter (gravity; for the
analyses a hardpoint or the friction) stands for what rounding differences
between compilers do: it moves no channel by more than half a percent of the
tolerance. A change of 1e-6 moves every scenario by 26 times the tolerance or
more. Between Windows (MSVC) and Linux (GCC) the CI jobs report each
scenario's worst difference as an annotation.

## When a scenario fails

Either a result changed that should not have (a defect: find it), or a change
was meant (an algorithm improved, a model altered). Only in the second case,
and only knowing which scenarios moved and why:

```powershell
$env:MBD_REGRESSION_UPDATE = "1"
build\tests\test_regression.exe "[regression]~[sensitivity]"      # the fast scenarios
build\tests\test_regression.exe "[long]"                           # the long ones
Remove-Item Env:\MBD_REGRESSION_UPDATE
```

and say in the task's journal entry and commit message which scenarios
changed, by how much (the failing test's message gives every channel's
difference), and why. Never write a reference again to make a failure go away
without knowing why it failed.

## Running it

```powershell
build\tests\test_regression.exe                    # the fast scenarios, and the check that a change is caught
build\tests\test_regression.exe "[long]"           # the long ones
build\tests\test_regression.exe "[sensitivity]" -s # the table above
```

`MBD_REGRESSION_REPORT=<file>` appends each scenario's worst channel to
`<file>` (CI turns the lines into annotations).

## The nightly workflow

`.github/workflows/nightly.yml` runs the hidden long tests (`[long]`: the drift
over 1000 s, the N-four-bar with 40 loops, the long scenarios), the
sensitivity probe, and the mutation tests (`scripts/mutation_test.py`): nine
deliberate changes to the kernel's and the force elements' physics, each of
which must make a test fail. GitHub runs a schedule only from the default
branch; elsewhere the workflow runs when it or the mutation script changes.
