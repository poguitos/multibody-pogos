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
| `sedan-drive` | The sedan (simple corners: each wheel on a slider) at rest, accelerating, steering; RK4 at 1 ms, 2.3 s | Pacejka tyres, drivetrain, steering, state elements |
| `sedan-modes` | The sedan's static equilibrium and modes at rest | Statics, linearisation |
| `sports-car-turn` | The double-wishbone sports car with anti-roll bars and downforce, accelerating, turning left; RK4 at 1 ms, 2 s | Linkage corners steered through a rack, anti-roll bars, ride-height-sensitive downforce; each tyre's forces, toe and camber |
| `hatchback-brake` | The front-wheel-drive McPherson hatchback, accelerating, braking in a right turn; RK4 at 1 ms, 2 s | McPherson struts, front-wheel drive, brakes |
| `sedan-rough-road` | The sedan at 15 m/s on an ISO 8608 road of class C (seed 1), steering after 0.5 s; RK4 at 1 ms, 2 s | The road interface, a generated random road |
| `wishbone-sweep` | The sports car's front double wishbone over +-0.05 m | Kinematic analysis |
| `lap-oval` | The lap speed profile of an oval | The lap model |
| `hmmwv-joints-5s`, `hmmwv-bushings-5s`, `sedan-slalom` | The paper's 5 s; a 20 s slalom | The long suite (nightly) |

## The tolerance

Every channel is held to 1e-8 of its largest magnitude in the reference. The
hidden case `[sensitivity]` measures where that sits (9 November 2026, the
author's laptop, MSVC; the sports car and the hatchback, added for task 7.1,
on 14 November; the rough road, for task 7.2, and `sedan-modes` again, on 24
November; the vehicle scenarios again on 25 November, after task 7.3 gave the
wheels spin joints, and after task 7.4 gave them the Magic Formula tyre with
its lag; in brackets, the values before 7.4):

| Scenario | 1e-12 change, over the tolerance | 1e-6 change, over the tolerance |
|---|---|---|
| four-bar | 6.9e-4 | 693 |
| heavy-top | 8.1e-4 | 809 |
| bushed-pendulum | 1.7e-4 | 177 |
| hmmwv-joints | 6.0e-5 | 60 |
| hmmwv-bushings | 2.4e-4 | 60 |
| sedan-drive | 9.9e-4 (9.9e-4; 1.0e-3 before task 7.3) | 992 (996) |
| sedan-modes | 1.0e-2 (1.4e-2; 1.0e-2 before task 7.3; 5.1e-3 before the mirror of task 7.1) | 67 (30) |
| sports-car-turn | 4.6e-4 (4.7e-4) | 463 (471) |
| hatchback-brake | 4.1e-4 (2.3e-4) | 407 (225) |
| sedan-rough-road | 6.1e-4 (6.1e-4) | 610 (611) |
| wishbone-sweep | 6.8e-4 | 682 |
| lap-oval | 4.2e-5 | 41 |

A relative change of 1e-12 in one physical parameter (gravity; for the
analyses a hardpoint or the friction) stands for what rounding differences
between compilers do: it moves no channel by more than 1.0 percent of the
tolerance. A change of 1e-6 moves every scenario by 41 times the tolerance or
more. A channel that is zero but for rounding throughout a scenario (a
symmetric car's roll going straight) would have a tolerance of rounding
itself; scenarios avoid them (the rough road's car steers for that reason).
Nor should one channel hold values of very different scales, since its
tolerance follows the largest: `sedan-modes` keeps the wheels' spin modes
(about 3000 per second) apart from the body's (8 to 260 rad/s), which in
one channel with them moved only 8.5 times the tolerance at 1e-6. Since
task 7.4 the spins decay at about 170 per second (the tyres' damping at
rest and their rolling resistance) and share the body's channel, 1.5 to 170;
the third channel keeps any real mode above 1000 per second, none now. Between Windows (MSVC) and Linux (GCC) the CI jobs report each
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
