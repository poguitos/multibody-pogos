# Data: Phase 7, the steering (task 7.7)

The author's laptop (Windows 11, MSVC, RelWithDebInfo through
`scripts\build.ps1`, one compiler), on mains power
(Win32_Battery.BatteryStatus 2): the suite, 637 tests, all passing, 63.3 s.

## The tests' measurements

Commands: `build\tests\test_vehicle.exe "[steering]" -s` and
`build\tests\test_kernel.exe "[coupler]" -s`, the INFO lines (the
column's statics and its two modes passed within their allowances; the
modes' closed form is printed with the linear model's summary):

```
  closed form 5.6365 and 2762.35 rad/s; Linear model at t = 0 s: 2 degrees of
  freedom, 2 modes; largest acceleration at the operating point 0.00e+00.
    mode 1: lambda 2.1088e-12 + 5.6365i, 0.897077 Hz, damping ratio -3.74133e-
  rack: one piece within 2.81893e-17 m; geared within 3.90313e-18 m and 8.
  67362e-19 m/s; its rate against its travel's within 1.28233e-08 m/s (largest
  0.00631625 m/s); the left wheel's toe up to 0.0339112 rad
  parallel: toes against the closed form within 7.03304e-12 rad; at 60 mm the
  left 0.519934 rad, the right 0.527484; the axes meet at x 48.385, y -81.4508
  m; -7.16141 % of Ackermann
  outward (Ackermann): toes against the closed form within 8.84953e-12 rad; at
  60 mm the left 0.575723 rad, the right 0.505366; the axes meet at x -3.97419,
  y 8.91277 m; 56.2644 % of Ackermann
  inward (anti-Ackermann): toes against the closed form within 9.4369e-15 rad;
  at 60 mm the left 0.505365 rad, the right 0.63324; the axes meet at x 4.
  53999, y -5.0555 m; -127.28 % of Ackermann
  after 0.5 s the left wheel's toe 0.0101455 rad

  after 0.5 s the left wheel's toe 0.106028 rad
  pinion 98.0392 rad (98.0392 closed form), rack 0.980392 m

--
```

## The closed form before the test

The planar linkage's angles for the three arm geometries, worked out in
Python before the test was written (it showed that arms aimed at the rear
axle's centre give about half of Ackermann's difference with the rack ahead
of the axle; the script's "ackermann %" for the third geometry has the
wrong sign, the larger angle taken as the inner: the test's -127 % is right):

```python
import math
xf, xr, ht = 1.35, -1.35, 0.8
L = xf - xr
Ky = ht - 0.09
def solve(K, d, I0, s, prev):
    I = (I0[0], I0[1] + s)
    l2 = (K[0]+d[0]-I0[0])**2 + (K[1]+d[1]-I0[1])**2
    e = (K[0]-I[0], K[1]-I[1])
    A = e[0]*d[0] + e[1]*d[1]
    B = e[1]*d[0] - e[0]*d[1]
    C = (l2 - (d[0]**2+d[1]**2) - (e[0]**2+e[1]**2))/2
    rho = math.hypot(A, B); phi = math.atan2(B, A); c = math.acos(max(-1,min(1,C/rho)))
    cands = [phi - c, phi + c]
    cands = [math.remainder(x, 2*math.pi) for x in cands]
    return min(cands, key=lambda x: abs(x-prev))
for name, d in (("parallel", (0.12, 0.0)), ("ackermann", None), ("anti", None)):
    if d is None:
        u = (xf - xr, Ky - 0.0); n = math.hypot(*u); u = (u[0]/n, u[1]/n)
        d = (0.12*u[0], 0.12*u[1]) if name == "ackermann" else (0.12*u[0], -0.12*u[1])
    KL = (xf, Ky); KR = (xf, -Ky)
    IL = (xf + d[0], ht - 0.42); IR = (xf + d[0], -(ht - 0.42))
    dR = (d[0], -d[1])
    dl = dr = 0.0
    print(name, "arm", d)
    for s in [0.005*k for k in range(0, 13)]:
        dl = solve(KL, d, IL, s, dl); dr = solve(KR, dR, IR, s, dr)
        if s in (0.005, 0.02, 0.04, 0.06):
            # wheel centres rotate about kingpins: W0 = K + (0, 0.09) left
            WL = (KL[0] - 0.09*math.sin(dl), KL[1] + 0.09*math.cos(dl))
            WR = (KR[0] + 0.09*math.sin(dr), KR[1] - 0.09*math.cos(dr))
            nL = (-math.sin(dl), math.cos(dl)); nR = (-math.sin(dr), math.cos(dr))
            # WL + a nL = WR + b nR
            det = nL[0]*(-nR[1]) - (-nR[0])*nL[1]
            rx, ry = WR[0]-WL[0], WR[1]-WL[1]
            a = (rx*(-nR[1]) - (-nR[0])*ry)/det if abs(det)>1e-15 else float('inf')
            xc = WL[0] + a*nL[0]
            # ideal Ackermann outer angle for this inner: cot do - cot di = track / L (track between kingpins)
            t = 2*Ky
            di, do = max(dl, dr), min(dl, dr)
            do_ideal = math.atan(1/(1/math.tan(di) + t/L))
            pct = (di - do)/(di - do_ideal)*100
            print("  s %.3f  dL %.5f dR %.5f  centre x %.3f (rear %.2f)  ackermann %.1f %%" % (s, dl, dr, xc, xr, pct))
```
