# Benchmark, pinned, 5 October 2026

- **Code:** `9472d8d` (task 2.10), RelWithDebInfo, MSVC 14.50, `/O2`
- **Machine:** i7-13700H laptop, Windows 11, normal state (see the A/B file
  for the slow state seen the same morning)
- **Command:** build with `scripts\build.ps1 -Target mbd_bench`, then in
  PowerShell:

```powershell
$psi = New-Object System.Diagnostics.ProcessStartInfo 'C:\dev\multibody\build\bench\mbd_bench.exe'
$psi.RedirectStandardOutput = $true; $psi.UseShellExecute = $false
$p = [System.Diagnostics.Process]::Start($psi)
$p.ProcessorAffinity = [IntPtr](1 -shl 2); $p.PriorityClass = 'High'
$p.StandardOutput.ReadToEnd(); $p.WaitForExit()
```

Each figure is the median of seven rounds, in microseconds per call.

| Model | Operation | Time [us] |
|---|---|---|
| revolute chain, 2 bodies       | positions and velocities         |       0.36 |
| revolute chain, 2 bodies       | mass matrix                      |       0.40 |
| revolute chain, 2 bodies       | inverse dynamics                 |       0.57 |
| revolute chain, 2 bodies       | forward dynamics                 |       0.82 |
| revolute chain, 4 bodies       | positions and velocities         |       0.73 |
| revolute chain, 4 bodies       | mass matrix                      |       1.02 |
| revolute chain, 4 bodies       | inverse dynamics                 |       1.10 |
| revolute chain, 4 bodies       | forward dynamics                 |       1.67 |
| revolute chain, 8 bodies       | positions and velocities         |       1.42 |
| revolute chain, 8 bodies       | mass matrix                      |       2.85 |
| revolute chain, 8 bodies       | inverse dynamics                 |       2.26 |
| revolute chain, 8 bodies       | forward dynamics                 |       3.46 |
| revolute chain, 16 bodies      | positions and velocities         |       2.82 |
| revolute chain, 16 bodies      | mass matrix                      |       8.23 |
| revolute chain, 16 bodies      | inverse dynamics                 |       4.55 |
| revolute chain, 16 bodies      | forward dynamics                 |       7.08 |
| revolute chain, 32 bodies      | positions and velocities         |       5.80 |
| revolute chain, 32 bodies      | mass matrix                      |      27.85 |
| revolute chain, 32 bodies      | inverse dynamics                 |       9.40 |
| revolute chain, 32 bodies      | forward dynamics                 |      14.23 |
| revolute chain, 64 bodies      | positions and velocities         |      11.49 |
| revolute chain, 64 bodies      | mass matrix                      |      97.71 |
| revolute chain, 64 bodies      | inverse dynamics                 |      18.38 |
| revolute chain, 64 bodies      | forward dynamics                 |      28.83 |
| revolute tree, 2 bodies        | positions and velocities         |       0.37 |
| revolute tree, 2 bodies        | mass matrix                      |       0.39 |
| revolute tree, 2 bodies        | inverse dynamics                 |       0.59 |
| revolute tree, 2 bodies        | forward dynamics                 |       0.79 |
| revolute tree, 4 bodies        | positions and velocities         |       0.74 |
| revolute tree, 4 bodies        | mass matrix                      |       0.94 |
| revolute tree, 4 bodies        | inverse dynamics                 |       1.10 |
| revolute tree, 4 bodies        | forward dynamics                 |       1.66 |
| revolute tree, 8 bodies        | positions and velocities         |       1.42 |
| revolute tree, 8 bodies        | mass matrix                      |       2.17 |
| revolute tree, 8 bodies        | inverse dynamics                 |       2.27 |
| revolute tree, 8 bodies        | forward dynamics                 |       3.54 |
| revolute tree, 16 bodies       | positions and velocities         |       2.94 |
| revolute tree, 16 bodies       | mass matrix                      |       4.87 |
| revolute tree, 16 bodies       | inverse dynamics                 |       4.27 |
| revolute tree, 16 bodies       | forward dynamics                 |       7.21 |
| revolute tree, 32 bodies       | positions and velocities         |       5.62 |
| revolute tree, 32 bodies       | mass matrix                      |      11.55 |
| revolute tree, 32 bodies       | inverse dynamics                 |       8.67 |
| revolute tree, 32 bodies       | forward dynamics                 |      13.99 |
| revolute tree, 64 bodies       | positions and velocities         |      11.17 |
| revolute tree, 64 bodies       | mass matrix                      |      25.48 |
| revolute tree, 64 bodies       | inverse dynamics                 |      17.42 |
| revolute tree, 64 bodies       | forward dynamics                 |      27.85 |
| double-wishbone sedan          | positions and velocities         |       2.65 |
| double-wishbone sedan          | constraint equations             |       8.53 |
| double-wishbone sedan          | mass matrix                      |       4.63 |
| double-wishbone sedan          | bias forces                      |       3.96 |
| double-wishbone sedan          | constrained forward dynamics     |      23.65 |
| double-wishbone sedan          | accelerations, with the forces   |      25.79 |
| double-wishbone sedan          | constraint projection            |      24.21 |
| double-wishbone sedan          | one RK4 step of 1 ms, everything |     127.79 |

