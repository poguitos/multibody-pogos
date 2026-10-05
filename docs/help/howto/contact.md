# How to put a body in contact with the ground

<!-- example: tests/examples/ex_phase3.cpp#contact -->

```cpp
PlaneContactParams p;
p.stiffness = 1e5;          // [N/m] per point: a penalty spring
p.damping = 700.0;          // [N s/m], reached 0.1 mm deep
p.friction = 0.6;
p.slip_speed = 1e-3;        // [m/s]: below it friction acts as a damper
std::vector<ContactSphere> corners;
for (const Real x : {-0.2, 0.2}) {
    for (const Real y : {-0.15, 0.15}) corners.push_back({Vec3(x, y, -0.1), 0.0});   // points: radius 0
}
sys.force_elements.push_back(std::make_shared<PlaneContact>(box, corners, p));   // against z = 0
```

`PlaneContact` puts spheres of a body (radius 0 for points) against the XY
plane of a frame on any body, by default the ground's `z = 0`. The normal
force is a penalty spring (exponent 1, or 1.5 for Hertz's sphere) with a
damper that ramps in from first touch, so it does not jump; it never pulls.
Friction is Coulomb's, regularised: below `slip_speed` it acts as a damper,
so a body that would stick creeps at about that speed. Two consequences for
the time step: it must resolve the contact's stiffness (RK4 needs `omega dt`
below 2.8) and the friction's (`dt < 2.8 v_s m / (mu F_n)` per point): 0.2 ms
here.

More: [the kernel, contact](../../kernel.md#forces-and-simulation).
