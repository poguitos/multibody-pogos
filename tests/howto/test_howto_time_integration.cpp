// The examples of the how-to guide docs/help/howto/time-integration.md
// (task H.6), each run and checked here. The guide quotes the code between
// "// [example: name]" and "// [end example]" exactly; test_howto_pages.cpp
// checks that it does, so the guide cannot drift from code that compiles and
// works. The checks after each example are derived from the physics, not
// read from a run.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>

#include <Eigen/Geometry>

#include "mbd/analysis/recorder.hpp"
#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"

using namespace mbd;
using namespace mbd::kernel;

namespace {

/// A 1 kg rod 1 m long hanging from a hinge along Z at its top end, gravity
/// along -Y. About the hinge its moment of inertia is I = m (b^2 + w^2) / 3 +
/// m d^2 with half-length b = 0.5, half-width w = 0.02 and d = 0.5, so
/// 0.33347 kg m^2; small swings have w^2 = m g d / I.
void build_pendulum(System& sys)
{
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia rod = RigidBodyInertia::from_solid_box(1.0, Vec3(0.02, 0.5, 0.02));
    rod.com_B = Vec3(0.0, -0.5, 0.0);
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), rod, "rod");
}

Real pendulum_w()
{
    const Real I = (0.5 * 0.5 + 0.02 * 0.02) / 3.0 + 0.5 * 0.5;
    return std::sqrt(g_accel * 0.5 / I);
}

/// Generalised-alpha's recurrence (Chung and Hulbert) for q'' = -w^2 q at
/// rho_infinity rho, started from a = -w^2 q: (q, v) after n steps of h.
std::pair<Real, Real> alpha_steps(Real w, Real h, Real rho, int n, Real q, Real v)
{
    const Real am = (2.0 * rho - 1.0) / (rho + 1.0), af = rho / (rho + 1.0);
    const Real gam = 0.5 - am + af, beta = 0.25 * (gam + 0.5) * (gam + 0.5), W = w * w;
    Real a = -W * q;
    for (int k = 0; k < n; ++k) {
        const Real a1 = (-(1.0 - af) * W * (q + h * v + h * h * (0.5 - beta) * a) - am * a + af * (-W * q))
                        / ((1.0 - am) + (1.0 - af) * beta * h * h * W);
        q += h * v + h * h * ((0.5 - beta) * a + beta * a1);
        v += h * ((1.0 - gam) * a + gam * a1);
        a = a1;
    }
    return {q, v};
}

/// A 1 kg block on a slider with a spring of k N/m, no gravity.
void build_spring_block(System& sys, Real k)
{
    sys.model.gravity = Vec3::Zero();
    const int block = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                         RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)), "block");
    JointCoordinateForceParams spring;
    spring.spring = Curve::linear(k);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, block, spring));
}

// [example: actuator]
/// An actuator whose force follows its command with a first-order lag:
/// f_dot = (command - f) / tau. Its one state is the force.
class Actuator : public StateElement {
public:
    explicit Actuator(Real tau) : tau_(tau) {}
    const char* name() const override { return "actuator"; }
    int size() const override { return 1; }
    void initial_states(VecRef x) const override { x(0) = 0.0; }
    void begin_evaluation(const Simulator&, Real, ConstVecRef x) override { force = x(0); }
    void rates(const Simulator&, Real, ConstVecRef x, VecRef x_dot) override
    {
        x_dot(0) = (command - x(0)) / tau_;
    }

    Real command{0.0};   ///< Set between steps [N]
    Real force{0.0};     ///< The force of the evaluation in progress [N]

private:
    Real tau_;
};
// [end example]

} // namespace

TEST_CASE("How-to time integration: a fixed step", "[howto]")
{
    // RK4 at 1 ms on a swing of 1 mrad. Small-angle solution: q = q0 cos(w
    // t). Its error at 2 s: the swing's nonlinearity shifts the phase by 2 pi
    // (t / T) q0^2 / 16, 4.8e-7 rad, so 5e-10 rad of angle; RK4's own, about
    // t w^5 h^4 / 120 of q0, is far smaller. Allowed: 1e-9 rad.
    System sys;
    build_pendulum(sys);
    // [example: fixed-step]
    Simulator sim(sys);
    sim.method = Integrator::RK4;   // the default
    sim.q(0) = 0.001;               // the hinge's angle [rad]
    sim.initialize();
    sim.run(2.0, 0.001);            // 2 s in steps of 1 ms
    // sim.q, sim.v and sim.time now hold the state at 2 s.
    // [end example]
    const Real w = pendulum_w();
    INFO("angle " << sim.q(0) << " against " << 0.001 * std::cos(w * 2.0));
    CHECK(std::abs(sim.time - 2.0) < 1e-12);   // 2000 steps add up 2 s to rounding
    CHECK(std::abs(sim.q(0) - 0.001 * std::cos(w * 2.0)) < 1e-9);
}

TEST_CASE("How-to time integration: to a tolerance", "[howto]")
{
    // Dormand-Prince with error control at 1e-8 per step, output every 50
    // ms. The tolerances are local; over 2 s the global error stays of their
    // order (10 to 100 steps, each within the tolerance). Allowed against
    // the small-angle solution: 1e-7 rad, at the end and within the last
    // step from the continuous extension.
    System sys;
    build_pendulum(sys);
    // [example: error-control]
    Simulator sim(sys);
    sim.method = Integrator::DormandPrince;
    sim.step_control.relative_tolerance = 1e-8;   // per step; the global error follows them roughly
    sim.step_control.absolute_tolerance = 1e-8;
    sim.q(0) = 0.001;
    sim.initialize();
    for (int k = 0; k < 40; ++k) sim.step(0.05);  // output every 50 ms; the method picks its own steps
    const IntegratorStats& stats = sim.integrator_stats();   // steps, rejected, evaluations, step_size
    VecX q_mid, v_mid, x_mid;
    const Real t_mid = sim.time - 0.5 * stats.step_size;     // halfway through the last step taken
    sim.interpolate(t_mid, q_mid, v_mid, x_mid);
    // [end example]
    const Real w = pendulum_w();
    INFO(stats.steps << " steps, " << stats.rejected << " rejected, " << stats.evaluations << " evaluations; angle "
                     << sim.q(0) << " against " << 0.001 * std::cos(w * sim.time) << "; at " << t_mid << " s "
                     << q_mid(0) << " against " << 0.001 * std::cos(w * t_mid));
    CHECK(std::abs(sim.time - 2.0) < 1e-12);
    CHECK(stats.steps >= 40);   // at least one per output step
    CHECK(std::abs(sim.q(0) - 0.001 * std::cos(w * sim.time)) < 1e-7);
    CHECK(std::abs(q_mid(0) - 0.001 * std::cos(w * t_mid)) < 1e-7);
}

TEST_CASE("How-to time integration: a stiff model", "[howto]")
{
    // A spring of 1e6 N/m on 1 kg: w = 1000 rad/s, so w h = 10 at 10 ms,
    // where RK4 (stable to w h = 2.8) would grow by about 1e3 a step.
    // Generalised-alpha is stable at any step and damps such motions towards
    // rho_infinity per step. On this linear model its steps are its
    // recurrence, to Newton's tolerance (the defaults, met here in one
    // iteration to the derivatives' rounding): allowed 1e-6 of the swing,
    // 1e-9 m and 1e-6 m/s. The swing shrinks to under a tenth in 10 steps.
    System sys;
    build_spring_block(sys, 1e6);
    // [example: stiff]
    Simulator sim(sys);                          // a 1 kg block on a spring of 1e6 N/m: 1000 rad/s
    sim.method = Integrator::GeneralizedAlpha;
    sim.implicit_control.rho_infinity = 0.5;     // what is left per step of motions far too fast for it
    sim.q(0) = 0.001;
    sim.initialize();
    sim.run(0.1, 0.01);                          // w h = 10: RK4 would need steps under 2.8 ms
    // [end example]
    const auto [q_rec, v_rec] = alpha_steps(1000.0, 0.01, 0.5, 10, 0.001, 0.0);
    INFO("position " << sim.q(0) << " against " << q_rec << ", velocity " << sim.v(0) << " against " << v_rec
                     << "; Newton iterations " << sim.integrator_stats().newton_iterations);
    CHECK(std::abs(sim.q(0) - q_rec) < 1e-9);
    CHECK(std::abs(sim.v(0) - v_rec) < 1e-6);
    CHECK(std::hypot(sim.q(0), sim.v(0) / 1000.0) < 0.1 * 0.001);
}

TEST_CASE("How-to time integration: real time", "[howto]")
{
    // The linearly implicit step on the same spring at 1 ms (w h = 1): on a
    // linear model it is backward Euler, which multiplies q - i v / w by 1 /
    // (1 - i w h) each step, |.| = 1 / sqrt(2). After 10 steps the swing is
    // 1/32 of its start, to the derivatives' rounding (1e-10 of k); one
    // evaluation per step.
    System sys;
    build_spring_block(sys, 1e6);
    // [example: real-time]
    Simulator sim(sys);
    sim.method = Integrator::LinearlyImplicitEuler;
    sim.q(0) = 0.001;
    sim.initialize();
    for (int k = 0; k < 10; ++k) sim.step(0.001);   // in real time: one step per tick of the clock
    // integrator_stats().evaluations is 10: one evaluation per step, every step.
    // [end example]
    const Real amplitude = std::hypot(sim.q(0), sim.v(0) / 1000.0);
    INFO("amplitude " << amplitude << " against " << 0.001 / 32.0);
    CHECK(std::abs(amplitude - 0.001 / 32.0) < 1e-9 * 0.001);
    CHECK(sim.integrator_stats().evaluations == 10);
}

TEST_CASE("How-to time integration: states of your own", "[howto]")
{
    // The actuator (time constant 0.2 s) pushing a 2 kg block from rest
    // with a command of 3 N: f = 3 (1 - e^(-t/tau)), and the block's position
    // x = (3 / m) (t^2 / 2 - tau t + tau^2 (1 - e^(-t/tau))). RK4 at 10 ms:
    // about t (h / tau)^4 / 120 of the motion's scale (0.75 m), 3.9e-8;
    // allowed 1e-6 m and 1e-6 N. With 4 fast steps per step (multirate) the
    // state is as accurate, and the block's extrapolated motion over the step
    // costs little more on so slow a force: allowed 1e-5 m.
    System sys;
    sys.model.gravity = Vec3::Zero();
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(2.0, Vec3(0.1, 0.1, 0.1)), "block");
    const Real m = 2.0, tau_a = 0.2, t = 1.0;
    const Real f_exact = 3.0 * (1.0 - std::exp(-t / tau_a));
    const Real x_exact = 3.0 / m * (t * t / 2.0 - tau_a * t + tau_a * tau_a * (1.0 - std::exp(-t / tau_a)));
    {
        // [example: states]
        Actuator actuator(0.2);                  // a time constant of 0.2 s
        Simulator sim(sys);                      // a 2 kg block on a slider, no gravity
        sim.attach(actuator);                    // its state joins sim.x, integrated with q and v
        sim.force_callback = [&](Simulator&, Real, VecX& tau) { tau(0) += actuator.force; };
        actuator.command = 3.0;                  // [N]
        sim.run(1.0, 0.01);
        // sim.x(0) is the actuator's force at 1 s, sim.q(0) the block's position.
        // [end example]
        INFO("force " << sim.x(0) << " against " << f_exact << "; position " << sim.q(0) << " against " << x_exact);
        CHECK(std::abs(sim.x(0) - f_exact) < 1e-6);
        CHECK(std::abs(sim.q(0) - x_exact) < 1e-6);
    }
    {
        Actuator actuator(0.2);
        Simulator sim(sys);
        sim.attach(actuator);
        sim.force_callback = [&](Simulator&, Real, VecX& tau) { tau(0) += actuator.force; };
        actuator.command = 3.0;
        // [example: multirate]
        sim.method = Integrator::RK4;            // multirate works with RK4 only
        sim.multirate = 4;                       // the states take 4 steps for each step of the bodies
        sim.run(1.0, 0.01);
        // [end example]
        INFO("multirate: force " << sim.x(0) << ", position " << sim.q(0));
        CHECK(std::abs(sim.x(0) - f_exact) < 1e-6);
        CHECK(std::abs(sim.q(0) - x_exact) < 1e-5);
    }
}

TEST_CASE("How-to time integration: a closed mechanism and its drift", "[howto]")
{
    // A 1 kg rod 1 m long on a free joint, its top end held at the world's
    // origin by a spherical closure, released at rest 0.3 rad from the
    // vertical: a pendulum closed by a constraint. Every step is projected
    // back: the residual stays within constraint_tolerance (1e-10) at every
    // sample, while each step's drift before the projection, RK4's error
    // across the constraint at 1 ms, is positive and far smaller than the
    // step's motion (about (w h)^5 of the rod, w near 4 rad/s: 1e-12; allowed
    // 1e-8 m).
    System sys;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.02, 0.02, 0.5)), "rod");
    sys.constraints.push_back(spherical_closure(Marker{1, Transform3::FromTranslation(Vec3(0.0, 0.0, 0.5))},
                                                Marker{0, Transform3()}));
    const Quat tilt(Eigen::AngleAxisd(0.3, Vec3::UnitX()));
    // [example: drift]
    Simulator sim(sys);                          // a rod held at its top end by a spherical closure
    sim.q.head<3>() = tilt * Vec3(0.0, 0.0, -0.5);   // the free joint: position, then the quaternion
    sim.q.tail<4>() = tilt.coeffs();
    sim.constraint_tolerance = 1e-10;            // the default: |phi| left after each projection [m]
    sim.initialize();                            // projects onto the constraints
    Recorder rec;
    add_drift_channels(rec, sim);                // constraint_drift, constraint_residual, and the velocities'
    for (int k = 0; k < 1000; ++k) {
        sim.step(0.001);
        rec.sample(sim.time);
    }
    const ConstraintDrift& drift = sim.constraint_drift();   // max_position: the most a step left
    // [end example]
    Real largest_residual = 0.0;
    for (const Real r : rec.column("constraint_residual")) largest_residual = std::max(largest_residual, r);
    INFO("largest drift " << drift.max_position << ", largest residual " << largest_residual);
    CHECK(rec.samples() == 1000);
    CHECK(largest_residual <= 1e-10);
    CHECK(drift.max_position > 0.0);
    CHECK(drift.max_position < 1e-8);
}
