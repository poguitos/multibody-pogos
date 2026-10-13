# Linearisation: margins, modes and compile time, 13 October 2026

Code: branch `phase-3`, the commit of task 3.7. Laptop i7-13700H, 16 GB,
Windows 11, MSVC 19.50, RelWithDebInfo, one compile job; about 4.9 GB free.

Commands (from `build/tests`): `test_kernel.exe "[linearization]" -s`,
`test_vehicle.exe "[linearization]" -s`.

## Margins

| Case | Quantity | Measured | Bound |
|---|---|---|---|
| damped mass (m 2, k 800, c 8) | eigenvalue against -2 + 19.8997i | 2.8e-10 | 2e-7 (1e-8 relative) |
| | natural frequency against 20 rad/s | 2.1e-11 | 2e-7 |
| | damping ratio against 0.1 | 1.4e-11 | 1e-8 |
| four-bar on a spring | omega against sqrt(V'' / n^T M n) = 9.83315 rad/s | 1.7e-8 | 9.8e-5 (1e-5 relative) |
| | damping ratio | 0 | 1e-6 |
| | shape against the motion n | 3.6e-16 | 1e-6 |
| off the constraints | K080 and K081 reported; gap after | 1.1e-16 m | 1e-10 |
| quarter car, undamped | body mode against 8.52232 rad/s (1.35637 Hz) | 9.2e-11 rad/s | 8.5e-8 |
| | wheel hop against 74.2117 rad/s (11.8112 Hz) | 9.5e-10 rad/s | 7.4e-7 |
| | shape ratios x_s / x_u | 5.2e-10, 4.0e-14 | 1.1e-7, 1.5e-10 |
| | damping ratios | 3e-15, 0 | 1e-9 |
| quarter car, damped (c 1500) | det(l^2 M + l C + K) residual, relative | 3.3e-10, 6.1e-10 | 1e-8 |
| | damping ratios (body, wheel hop) | 0.294, 0.263 | (> 0) |
| sedan at its equilibrium | degrees of freedom; zero eigenvalues | 10; 3 | exact |
| | most positive real part | 1.1e-10 | 3.6e-4 (1e-6 of the largest) |

## The sedan's modes at rest

From `LinearModel::summary`, about the static equilibrium of the
double-wishbone sedan (tyre damping and the suspension dampers as built):

| Mode | eigenvalue | frequency [Hz] | damping ratio | largest shape component |
|---|---|---|---|---|
| 1 | 1.2e-11 | 0 | | neutral |
| 2 | 1.1e-10 +- 2.4e-10i | 0 | | neutral (a pair of zeros) |
| 3 | -0.048 | 0.0076 | 1 | chassis forward |
| 4 | -6.54 | 1.04 | 1 | |
| 5 | -5.81 +- 5.76i | 1.30 | 0.71 | |
| 6 | -4.97 +- 9.91i | 1.76 | 0.45 | |
| 7 | -23.3 +- 20.0i | 4.89 | 0.76 | |
| 8 | -46.3 | 7.37 | 1 | |
| 9 | -48.6 +- 55.8i | 11.77 | 0.66 | |
| 10 | -46.2 +- 70.8i | 13.46 | 0.55 | |
| 11 to 14 | -208 to -358 | 33 to 57 | 1 | uprights |

Body modes near 1.3 to 1.8 Hz and wheel hop near 12 to 13 Hz, as expected of
a passenger car, but several modes are overdamped: at rest each tyre's slip
force acts as a damper (below V_low = 1 m/s, a lateral force of 45000 N per
m/s of slip velocity), which no real tyre at standstill does. Vehicle modes
belong at speed (Phase 9).

## Compile time

`src/kernel/linearization.cpp`, which instantiates Eigen's general
`EigenSolver<MatX>`: 31.3 s in the full rebuild (the library's other new files
4.5 to 9 s). Last full rebuild: 99 objects, 5.7 min, against 8.3 min for 85
objects earlier the same day with 1.1 to 1.7 GB free instead of 4.9 GB.
