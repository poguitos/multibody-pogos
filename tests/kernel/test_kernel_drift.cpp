// Constraints per integrator (plan task 4.7, decision D37). Every evaluation
// solves the loop constraints at the acceleration level, so each step's result
// lies off them by the integrator's error across them, and the drift adds up
// unless it is removed. Every method projects after each of its steps;
// generalised-alpha, whose history a projection does not move, can instead
// solve each step on the constraints in the stabilized index-2 form
// (implicit_control.stabilized_index2). Checked on a planar four-bar with a
// pendulum on its coupler (closed in 3D: 5 equations of rank 2) and on a heavy
// top held at its tip (a free joint with a quaternion, 3 equations):
//
//   - stabilized, generalised-alpha's steps end on the constraints and
//     converge at second order (ratio 4 when the step halves), more
//     accurately than projected; left free, it drifts;
//   - with every method the state stays on the constraints, to
//     constraint_tolerance in positions and to rounding in velocities (the
//     stabilized step: to its Newton tolerance), with no projection failing;
//     the hidden case [long] runs 1000 simulated seconds: the plan's "done
//     when";
//   - the drift channel: what each step left before the projection, and the
//     residuals after, in constraint_drift(), the trace and a Recorder;
//   - a tolerance that is not positive is refused.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

#include <Eigen/Geometry>

#include "mbd/analysis/recorder.hpp"
#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using mbd_test::FourBar;

namespace {

/// The four-bar with its pendulum, crank at 1 rad turning at 2 rad/s, the
/// pendulum at 0.3 rad and at rest.
void release_four_bar(Simulator& sim)
{
    sim.q.head(3) = FourBar::configuration(1.0);
    sim.q(3) = 0.3;
    sim.v.head(3) = 2.0 * FourBar::motion(sim.q);
    sim.v(3) = 0.0;
    sim.initialize();
}

/// A heavy symmetric top: one body on a free joint, its tip held at the world
/// origin by a spherical closure, gravity along -Z. The body frame is at the
/// centre of mass with the axis of symmetry along its Z, the tip 0.1 m below
/// it (the constraint then goes through the quaternion: held at the body's
/// origin instead, it would be linear in the coordinates, and every method
/// keeps a linear constraint exactly); a flat 1 kg disc 0.2 m across. Released
/// with its axis 0.5 rad from the vertical, spinning at 30 rad/s about it,
/// the centre of mass and the tip at rest (a free joint's velocity is the
/// angular velocity and the velocity of the body's origin, in the body's
/// axes): it nutates at about 15 rad/s and precesses.
struct HeavyTop {
    static constexpr Real l = 0.1;   // tip to centre of mass [m]
    System sys;

    HeavyTop()
    {
        sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
        sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                           RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.02)), "top");
        sys.constraints.push_back(spherical_closure(Marker{1, Transform3::FromTranslation(Vec3(0.0, 0.0, -l))},
                                                    Marker{0, Transform3()}));
    }

    static void release(Simulator& sim)
    {
        const Quat tilt(Eigen::AngleAxisd(0.5, Vec3::UnitX()));
        sim.q.head<3>() = tilt * Vec3(0.0, 0.0, l);
        sim.q.tail<4>() = tilt.coeffs();
        sim.v.setZero();
        sim.v(2) = 30.0;
        sim.initialize();
    }
};

struct Method {
    Integrator method;
    const char* name;
    bool adaptive;     // error control at the default tolerances, within each output step
    bool stabilized;   // generalised-alpha in the stabilized index-2 form
};

constexpr Method kMethods[] = {{Integrator::RK4, "RK4", false, false},
                               {Integrator::SemiImplicitEuler, "semi-implicit Euler", false, false},
                               {Integrator::DormandPrince, "Dormand-Prince", true, false},
                               {Integrator::GeneralizedAlpha, "generalised-alpha", false, false},
                               {Integrator::GeneralizedAlpha, "generalised-alpha, stabilized", false, true},
                               {Integrator::SDIRK4, "SDIRK4", true, false},
                               {Integrator::RadauIIA, "Radau IIA", true, false},
                               {Integrator::LinearlyImplicitEuler, "linearly implicit Euler", false, false}};

/// `duration` seconds of `sys` with `method` at output steps of 5 ms; checks
/// that every step left the state on the constraints. Prints the drift
/// channel's maxima, and the largest drift over the first and the last tenth
/// of the run.
void check_drift(System& sys, void (*release)(Simulator&), const char* model, const Method& m, Real duration)
{
    Simulator sim(sys);
    sim.method = m.method;
    sim.step_control.adaptive = m.adaptive;
    sim.implicit_control.stabilized_index2 = m.stabilized;
    release(sim);
    const Real dt = 0.005;
    const int steps = static_cast<int>(std::round(duration / dt));
    Real early = 0.0, late = 0.0;
    for (int k = 0; k < steps; ++k) {
        sim.step(dt);
        const Real d = sim.constraint_drift().position;
        if (k < steps / 10) early = std::max(early, d);
        if (k >= steps - steps / 10) late = std::max(late, d);
    }
    const ConstraintDrift& d = sim.constraint_drift();
    INFO(model << ", " << m.name << ", " << sim.time << " s: drift at most " << d.max_position << " (first tenth "
               << early << ", last tenth " << late << ") and " << d.max_velocity << "; residuals at most "
               << d.max_position_residual << " and " << d.max_velocity_residual << "; "
               << sim.integrator_stats().evaluations << " evaluations, " << sim.integrator_stats().newton_iterations
               << " Newton iterations in " << sim.integrator_stats().steps << " steps");
    CHECK(std::abs(sim.time - duration) < 1e-9);
    CHECK(d.max_position_residual <= sim.constraint_tolerance);
    if (m.stabilized) {
        // Solved on the constraints: nothing projected, the drift is the
        // residual, J v = nu held to the Newton tolerance.
        CHECK(d.max_position == d.max_position_residual);
        CHECK(d.max_velocity_residual <= sim.step_control.absolute_tolerance);
        CHECK(sim.last_projection().iterations == 0);
    } else {
        // A projection solves J v = nu exactly: the rounding of J v, a few
        // eps times |J| |v| (|v| up to some 30 rad/s here).
        CHECK(d.max_velocity_residual < 1e-12);
    }
    CHECK(sim.projection_failures() == 0);
}

} // namespace

TEST_CASE("Constraints per integrator: stabilized, generalised-alpha ends each step on the constraints, at second order",
          "[kernel][integrators][implicit][constraints]")
{
    // The four-bar for 1 s, against Dormand-Prince at 1e-13 (held to the
    // constraints at 1e-14). Newton's tolerances at 1e-12, so that its error
    // is far below the method's. Stabilized, halving the step must divide the
    // error by 2^2 = 4, to within the next term (measured 4.09 and 4.04 at
    // these steps), and every step's end is on the constraints with nothing
    // projected. Projected after the step (the default), the method is less
    // accurate at every step, by less as the step shortens (measured 1.34,
    // 1.18 and 1.09 times here), its ratios still above 4 (4.68, 4.37): the
    // projection takes from the state, not from the method's history, what
    // each step left off the constraints. Left free, the method drifts: its
    // acceleration-like variable lags the accelerations, so each step leaves
    // the constraints by far more than its error along them.
    FourBar fb(true);
    auto run = [&](Integrator method, Real h, Real tol, bool enforce, bool stabilized) {
        auto sim = std::make_unique<Simulator>(fb.sys);
        sim->method = method;
        sim->enforce_constraints = enforce;
        sim->implicit_control.stabilized_index2 = stabilized;
        sim->step_control.relative_tolerance = method == Integrator::DormandPrince ? tol : 0.0;
        sim->step_control.absolute_tolerance = tol;
        if (method == Integrator::DormandPrince) sim->constraint_tolerance = 1e-14;
        release_four_bar(*sim);
        sim->run(1.0, h);
        return sim;
    };
    const auto reference = run(Integrator::DormandPrince, 0.01, 1e-13, true, false);
    auto error_of = [&](const Simulator& sim) {
        return (sim.q - reference->q).norm() + (sim.v - reference->v).norm();
    };
    Real error[3];
    for (int k = 0; k < 3; ++k) {
        const Real h = 0.002 / static_cast<Real>(1 << k);
        const auto sim = run(Integrator::GeneralizedAlpha, h, 1e-12, true, true);
        const auto projected = run(Integrator::GeneralizedAlpha, h, 1e-12, true, false);
        error[k] = error_of(*sim);
        const ConstraintDrift& d = sim->constraint_drift();
        INFO("h = " << h << ": error " << error[k] << " (projected " << error_of(*projected) << "); residuals at most "
                    << d.max_position_residual << " and " << d.max_velocity_residual << "; "
                    << sim->integrator_stats().newton_iterations << " Newton iterations in "
                    << sim->integrator_stats().steps << " steps (projected "
                    << projected->integrator_stats().newton_iterations << ")");
        CHECK(d.max_position_residual <= sim->constraint_tolerance);
        CHECK(d.max_velocity_residual <= 1e-12);
        CHECK(sim->last_projection().iterations == 0);
        CHECK(error_of(*projected) > error[k]);
        CHECK(projected->constraint_drift().max_position_residual <= projected->constraint_tolerance);
    }
    INFO("errors " << error[0] << ", " << error[1] << ", " << error[2] << ": ratios " << error[0] / error[1] << ", "
                   << error[1] / error[2]);
    CHECK(error[0] / error[1] > 3.95);
    CHECK(error[0] / error[1] < 4.15);
    CHECK(error[1] / error[2] > 3.95);
    CHECK(error[1] / error[2] < 4.1);

    const auto free = run(Integrator::GeneralizedAlpha, 0.002, 1e-12, false, false);
    INFO("free, at 2 ms: |phi| " << free->constraint_drift().position_residual << " after 1 s");
    CHECK(free->constraint_drift().position_residual > 1e-6);
}

TEST_CASE("Constraints per integrator: every method keeps the state on the constraints",
          "[kernel][integrators][constraints]")
{
    // Ten seconds of each model with each method; the hidden case below runs
    // a thousand.
    FourBar fb(true);
    HeavyTop top;
    for (const Method& m : kMethods) {
        check_drift(fb.sys, release_four_bar, "four-bar", m, 10.0);
        check_drift(top.sys, HeavyTop::release, "heavy top", m, 10.0);
    }
}

TEST_CASE("Constraints per integrator: the drift stays under the tolerance for 1000 simulated seconds",
          "[.][long][kernel][integrators][constraints]")
{
    // The plan's "done when" for task 4.7. Hidden: about ten minutes, most of
    // them SDIRK4's (some twenty Newton iterations a step at the default
    // tolerances). Run by hand with -s, which prints the maxima with each
    // check; the results are in the journal's data. Semi-implicit Euler and
    // the linearly implicit step leave the top out: both take the gyroscopic
    // terms explicitly, and a top spinning at 30 rad/s, stepped at 5 ms,
    // blew up within the first 100 s whatever the constraints did (the
    // projection then failed at almost every step).
    FourBar fb(true);
    HeavyTop top;
    for (const Method& m : kMethods) {
        check_drift(fb.sys, release_four_bar, "four-bar", m, 1000.0);
        if (m.method == Integrator::SemiImplicitEuler || m.method == Integrator::LinearlyImplicitEuler) continue;
        check_drift(top.sys, HeavyTop::release, "heavy top", m, 1000.0);
    }
}

TEST_CASE("Constraints per integrator: the drift channel, in the simulator, the trace and a recorder",
          "[kernel][constraints][outputs]")
{
    // RK4 at 5 ms on the four-bar: each step's drift (before the projection)
    // and residuals (after) are in constraint_drift(), in the trace and in the
    // recorder's channels, the same numbers.
    FourBar fb(true);
    Simulator sim(fb.sys);
    sim.trace_steps = true;
    release_four_bar(sim);
    Recorder rec;
    add_drift_channels(rec, sim);
    Real largest = 0.0;
    for (int k = 0; k < 200; ++k) {
        sim.step(0.005);
        rec.sample(sim.time);
        largest = std::max(largest, sim.constraint_drift().position);
    }
    const auto& trace = sim.trace();
    REQUIRE(trace.size() == 200);
    REQUIRE(rec.samples() == 200);
    for (std::size_t k = 0; k < 200; ++k) {
        CHECK(rec.column("constraint_drift")[k] == trace[k].drift);
        CHECK(rec.column("constraint_velocity_drift")[k] == trace[k].velocity_drift);
        CHECK(rec.column("constraint_residual")[k] == trace[k].position_residual);
        CHECK(rec.column("constraint_velocity_residual")[k] == trace[k].velocity_residual);
    }
    const ConstraintDrift& d = sim.constraint_drift();
    CHECK(d.max_position == largest);
    CHECK(d.max_position > 100.0 * sim.constraint_tolerance);   // RK4's error across the constraints, at 5 ms
    CHECK(d.max_position_residual <= sim.constraint_tolerance);

    const std::filesystem::path path = std::filesystem::temp_directory_path() / "mbd_drift_trace.csv";
    sim.write_trace(path.string());
    std::ifstream in(path);
    std::string header;
    std::getline(in, header);
    in.close();
    std::filesystem::remove(path);
    CHECK_THAT(header, ContainsSubstring("time,dt,drift,velocity_drift,position_residual,velocity_residual,"));

    // Left free, nothing is projected: the drift is the state's residual, as
    // a separate solver measures it, and it adds up past the tolerance.
    Simulator free(fb.sys);
    free.enforce_constraints = false;
    release_four_bar(free);
    free.run(1.0, 0.005);
    ConstraintSolver cs(fb.sys.model, fb.sys.constraints);
    Data data(fb.sys.model);
    cs.evaluate(data, free.q, free.v, free.time);
    const VecX gap = cs.J() * free.v - cs.nu();
    const ConstraintDrift& f = free.constraint_drift();
    INFO("free after 1 s: |phi| " << f.position_residual << ", |J v - nu| " << f.velocity_residual);
    CHECK(f.position_residual == cs.phi().norm());
    CHECK(f.velocity_residual == gap.norm());
    CHECK(f.position == f.position_residual);
    CHECK(f.position_residual > 100.0 * free.constraint_tolerance);
}

TEST_CASE("Constraints per integrator: a tolerance that is not positive is refused", "[kernel][constraints]")
{
    FourBar fb(true);
    Simulator sim(fb.sys);
    release_four_bar(sim);
    sim.constraint_tolerance = 0.0;
    CHECK_THROWS_WITH(sim.step(0.005), ContainsSubstring("MBD-K047"));
    sim.constraint_tolerance = std::nan("");
    CHECK_THROWS_WITH(sim.step(0.005), ContainsSubstring("MBD-K047"));
}
