# How to find natural frequencies, damping and mode shapes

<!-- example: tests/examples/ex_phase3.cpp#modes -->

```cpp
// Wheel (40 kg) and body (250 kg) on vertical sliders: suspension spring and
// damper between them, a tyre spring under the wheel.
System sys;
sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
const Transform3 up = Transform3::FromRotation(Mat3(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX()).toRotationMatrix()));
const auto slider = std::make_shared<PrismaticJointModel>();
const int wheel = sys.model.add_body(0, slider, up, up, RigidBodyInertia::from_solid_box(40.0, Vec3(0.15, 0.15, 0.15)), "wheel");
const int body = sys.model.add_body(0, slider, up, up, RigidBodyInertia::from_solid_box(250.0, Vec3(0.5, 0.2, 0.4)), "body");
sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(wheel, body, Vec3::Zero(), Vec3::Zero(), 2e4, 1500.0, 0.3));
sys.force_elements.push_back(std::make_shared<TireContactForce>(wheel, 0.35, 2e5, 0.0));

Simulator sim(sys);
sim.q << 0.34, 0.6;
static_equilibrium(sim);                 // linearise about the resting state
const Linearization lin = linearize(sim);
for (const Mode& mode : lin.modes) {
    out << mode.natural_frequency_hz << " Hz, damping ratio " << mode.damping_ratio << '\n';
}
```

`linearize` gives the state matrices `A` and `B` about the simulator's state,
in coordinates of the constraint surface, with the reduced mass, stiffness
and damping matrices. Linearise about a resting state (`static_equilibrium`
first); otherwise the eigenvalues describe the motion near the state for a
short time only, and the result says so (MBD-K080). `modes` are the
eigenvalues of `A` (natural frequency, damping ratio, shape in the velocity
coordinates, `largest_at` naming the coordinate that moves most);
`undamped` the modes without damping. Zero frequencies are free directions
(MBD-K081); a growing mode is MBD-K082.

More: [the kernel, linearisation](../../kernel.md#linearisation).
