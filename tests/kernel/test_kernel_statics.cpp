// Static equilibrium (plan task 3.5).
//
//   - a mass on a spring settles at m g / k in one Newton step (the problem
//     is linear);
//   - a pendulum with a torsion spring settles where the moments balance;
//   - a four-bar with a spring settles at the minimum of its potential
//     energy along the mechanism's configurations;
//   - a driven joint is frozen where its driver is at t, its acceleration
//     left out;
//   - a body with nothing to hold it, and positions that cannot meet the
//     constraints, are reported.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::FourBar;

namespace {

RigidBodyInertia block(Real mass)
{
    return RigidBodyInertia::from_solid_box(mass, Vec3(0.1, 0.1, 0.1));
}

bool has_message(const std::vector<std::string>& messages, const char* code)
{
    for (const auto& m : messages) {
        if (m.find(code) != std::string::npos) return true;
    }
    return false;
}

} // namespace

TEST_CASE("Statics: a mass on a spring settles at m g / k in one Newton step", "[kernel][statics]")
{
    // A 2 kg block on a vertical slider (gravity along -Z), a 500 N/m spring
    // on the slider's coordinate: a = -g - (k / m) q is linear in q, so one
    // Newton step lands on q = -m g / k = -0.0392 m, once the step limit
    // (0.02 m by default) is raised above it. The central differences carry
    // rounding of about 1e-15 / 2e-6 in G (relative 2e-12 of k / m), so the
    // step lands within about 1e-13 of it.
    System sys;
    const int body = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(),
                                        Transform3(), block(2.0), "block");
    JointCoordinateForceParams spring;
    spring.spring = Curve::linear(500.0);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, spring));

    {
        // With the default limit the step is cut to 0.02 m; the second step,
        // 0.0192 m, is within it and lands.
        Simulator limited(sys);
        const StaticReport r = static_equilibrium(limited);
        CHECK(r.converged);
        CHECK(r.iterations == 2);
        REQUIRE(r.history.size() == 2);
        CHECK(std::abs(limited.q(0) + 2.0 * g_accel / 500.0) < 1e-11);
    }

    Simulator sim(sys);
    StaticOptions options;
    options.max_translation = 1.0;
    const StaticReport r = static_equilibrium(sim, options);
    INFO(r.summary(sys.model));
    CHECK(r.converged);
    CHECK(r.iterations == 1);
    CHECK(std::abs(sim.q(0) + 2.0 * g_accel / 500.0) < 1e-11);
    CHECK(r.residual < 1e-8);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(r.supported == 1);
    CHECK(r.neutral == 0);
    CHECK(r.warnings.empty());
    CHECK(r.notes.empty());
    CHECK(sim.v.norm() == 0.0);
}

TEST_CASE("Statics: a soft spring below a stiff one settles in one Newton step", "[kernel][statics]")
{
    // Two 1 kg blocks on vertical sliders in series (gravity along -Z): the
    // first on a slider from the ground with a spring of k1 = 1e8 N/m on its
    // coordinate, the second on a slider from the first with k2 = 100 N/m.
    // The problem is linear: at rest k1 q1 = -(m1 + m2) g and k2 q2 = -m2 g,
    // so q1 = -2 g / k1 = -1.962e-7 m and q2 = -g / k2 = -0.0981 m, which one
    // Newton step reaches when it sees both directions. Their stiffnesses
    // per unit mass, 1e8 and about 100 s^-2, differ by 1e-6: a step through
    // the normal equations G G^T squares that to 1e-12, below a rank cut of
    // 1e-10, and loses the soft direction, as it lost the bushed HMMWV's
    // chassis on its tyres (task 5.4; the scaled G there ranges over 4.8e-6).
    // Rounding: the perturbed states accelerate at up to k1 h / m1 = 100
    // m/s^2, so G's differences carry about 1e-15 * 100 / 2e-6 = 5e-8, which
    // against the soft entries (about 100 s^-2) is 5e-10 relative: q2 within
    // about 5e-11 m (seen 2.3e-13), q1 far closer.
    System sys;
    const int b1 = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                      block(1.0), "stiff");
    const int b2 = sys.model.add_body(b1, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                      block(1.0), "soft");
    for (const auto& [body, k] : {std::pair{b1, 1e8}, std::pair{b2, 100.0}}) {
        JointCoordinateForceParams spring;
        spring.spring = Curve::linear(k);
        sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, spring));
    }

    Simulator sim(sys);
    StaticOptions options;
    options.max_translation = 1.0;
    const StaticReport r = static_equilibrium(sim, options);
    INFO(r.summary(sys.model));
    CHECK(r.converged);
    CHECK(r.iterations == 1);
    CHECK(r.relaxation_runs == 0);
    CHECK(std::abs(sim.q(0) + 2.0 * g_accel / 1e8) < 1e-15);
    CHECK(std::abs(sim.q(1) + g_accel / 100.0) < 1e-11);
    CHECK(r.supported == 2);
    CHECK(r.neutral == 0);
}

TEST_CASE("Statics: a pendulum with a torsion spring settles where the moments balance",
          "[kernel][statics]")
{
    // An arm of 1 kg with its centre of mass 0.25 m out along its X axis, on a
    // hinge about Z, gravity along -Y; a torsion spring of 20 N m/rad that is
    // unloaded at 0.5 rad. Balance: -k (th - th0) - m g l cos th = 0, solved
    // here by bisection. With the largest acceleration below the tolerance,
    // the angle is within tolerance * I / K of the root, K = k - m g l sin th
    // being the stiffness there.
    const Real m = 1.0, l = 0.25, k = 20.0, th0 = 0.5;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia arm = RigidBodyInertia::from_solid_box(m, Vec3(l, 0.02, 0.02));
    arm.com_B = Vec3(l, 0.0, 0.0);
    const int body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(),
                                        Transform3(), arm, "arm");
    JointCoordinateForceParams spring;
    spring.reference = th0;
    spring.spring = Curve::linear(k);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, spring));
    Simulator sim(sys);

    auto moment = [&](Real th) { return -k * (th - th0) - m * g_accel * l * std::cos(th); };
    Real lo = th0 - 1.0, hi = th0;   // moment > 0 at lo, < 0 at hi
    REQUIRE(moment(lo) > 0.0);
    REQUIRE(moment(hi) < 0.0);
    for (int i = 0; i < 200 && hi - lo > 1e-15; ++i) {
        const Real mid = 0.5 * (lo + hi);
        if (moment(mid) > 0.0) lo = mid;
        else hi = mid;
    }
    const Real th_star = 0.5 * (lo + hi);

    StaticOptions options;
    const StaticReport r = static_equilibrium(sim, options);
    INFO(r.summary(sys.model));
    CHECK(r.converged);
    CHECK(r.iterations <= 6);
    Data data(sys.model);
    const Real inertia = crba(sys.model, data, sim.q)(0, 0);
    const Real stiffness = k - m * g_accel * l * std::sin(th_star);
    CHECK(std::abs(sim.q(0) - th_star) < options.tolerance * inertia / stiffness);
}

TEST_CASE("Statics: a four-bar with a spring settles at the minimum of its potential energy",
          "[kernel][statics]")
{
    // A spring from a point above the linkage to the coupler's middle holds
    // the four-bar up against gravity. Along the mechanism's configurations,
    // parametrized by the crank angle (closed form), the equilibrium is where
    // the potential energy, gravity's and the spring's, is least: found here
    // by a scan and a golden-section search. The search is good to about
    // sqrt(eps) relative near a flat minimum; statics leaves an error of about
    // tolerance / omega^2 with omega^2 of order 10 s^-2; 1e-6 rad covers both.
    FourBar fb;
    const Vec3 anchor(0.35, 0.9, 0.0), on_coupler(0.4, 0.0, 0.0);
    const Real k = 300.0, rest = 0.2;
    fb.sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, fb.coupler, anchor, on_coupler, k, 0.0, rest));
    const Model& model = fb.sys.model;

    Data data(model);
    auto energy = [&](Real th1) {
        const VecX q = FourBar::configuration(th1);
        forward_kinematics(model, data, q);
        const Transform3& X = data.oMi[static_cast<std::size_t>(fb.coupler)];
        const Real L = (X.p + X.q * on_coupler - anchor).norm();
        return potential_energy(model, data) + 0.5 * k * (L - rest) * (L - rest);
    };
    Real best = 0.0, best_energy = energy(0.0);
    for (int i = 1; i < 720; ++i) {
        const Real th = 2.0 * pi * i / 720.0;
        if (energy(th) < best_energy) {
            best = th;
            best_energy = energy(th);
        }
    }
    Real lo = best - 2.0 * pi / 720.0, hi = best + 2.0 * pi / 720.0;
    const Real golden = 0.5 * (std::sqrt(5.0) - 1.0);
    while (hi - lo > 1e-12) {
        const Real x1 = hi - golden * (hi - lo), x2 = lo + golden * (hi - lo);
        if (energy(x1) < energy(x2)) hi = x2;
        else lo = x1;
    }
    const Real th_star = 0.5 * (lo + hi);

    // Start 0.2 rad away, on the same branch.
    Simulator sim(fb.sys);
    sim.q = FourBar::configuration(th_star + 0.2);
    const StaticReport r = static_equilibrium(sim);
    INFO(r.summary(model) << "minimum by search at crank angle " << th_star);
    CHECK(r.converged);
    CHECK(r.iterations < 20);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(r.supported == 1);
    CHECK(std::abs(sim.q(0) - th_star) < 1e-6);
    CHECK(FourBar::gap(sim.q).norm() < 1e-10);
}

TEST_CASE("Statics: a driven joint is frozen where its driver is", "[kernel][statics]")
{
    // A slider driven along Z by s(t) = 0.3 sin 2t carries a 2 kg block on a
    // second slider with a 400 N/m spring; gravity along -Z. At t = 0.7 the
    // driver accelerates (s'' = -1.2 sin 1.4), but statics has no inertia:
    // the block sits at -m g / k on the slider, not at -m (g + s'') / k, which
    // is 5.9e-3 m away.
    System sys;
    const auto slide = std::make_shared<PrismaticJointModel>();
    const int slider = sys.model.add_body(0, slide, Transform3(), Transform3(), block(1.0), "slider");
    const int body = sys.model.add_body(slider, slide, Transform3(), Transform3(), block(2.0), "block");
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, slider,
        TimeFunction([](Real t) { return 0.3 * std::sin(2.0 * t); },
                     [](Real t) { return 0.6 * std::cos(2.0 * t); },
                     [](Real t) { return -1.2 * std::sin(2.0 * t); })));
    JointCoordinateForceParams spring;
    spring.spring = Curve::linear(400.0);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, spring));
    Simulator sim(sys);
    sim.time = 0.7;

    const StaticReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    CHECK(r.converged);
    CHECK(std::abs(sim.q(0) - 0.3 * std::sin(1.4)) < 1e-10);
    CHECK(std::abs(sim.q(1) + 2.0 * g_accel / 400.0) < 1e-10);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(!sim.solver().quasi_static);   // given back as it was
}

TEST_CASE("Statics: a block above its support finds it by relaxation, then Newton", "[kernel][statics]")
{
    // A 2 kg block on a vertical slider, 0.1 m above a one-sided stop at 0 of
    // 2e4 N/m (a joint limit stop pushes and never pulls): above the stop it
    // has a force and no stiffness, so Newton cannot start; relaxation lets
    // it fall onto the stop; Newton then finds q = -m g / k_stop, linear
    // there, to the rounding of the differences (relative 2e-12 of the
    // deflection, as in the spring case).
    const Real m = 2.0, k_stop = 2e4;
    System sys;
    const int body = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(),
                                        Transform3(), block(m), "block");
    JointCoordinateForceParams stop;
    stop.lower_limit = 0.0;
    stop.limit_stiffness = k_stop;
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, stop));
    Simulator sim(sys);
    sim.q(0) = 0.1;

    const StaticReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    CHECK(r.converged);
    CHECK(r.relaxation_runs == 1);
    CHECK(r.relaxation_steps > 0);
    CHECK(has_message(r.notes, "MBD-K073"));
    CHECK(std::abs(sim.q(0) + m * g_accel / k_stop) < 1e-12);
}

TEST_CASE("Statics: a relaxation too long for a stiff element is undone", "[kernel][statics]")
{
    // A 1 kg block on a slider from the ground with a spring of 1e8 N/m, and
    // on it a 2 kg block on a slider of its own, 0.1 m above a one-sided stop
    // of 2e4 N/m. Newton settles the stiff spring under the first block
    // (q1 = -g / k1: the second is falling and weighs nothing on it), then
    // cannot move the falling block, and relaxation starts. When the block
    // lands, its weight sets the stiff spring moving at sqrt(1e8 / 1 kg) =
    // 1e4 rad/s, and the explicit step of 1 ms (w dt = 10, where stability
    // needs below 2) makes that motion grow without bound: the run is undone
    // and the positions are those before it (task 5.4: they used to come
    // back as NaN). With a step of 0.1 ms (w dt = 1) relaxation lands the
    // block and Newton finishes: q1 = -(m1 + m2) g / k1, q2 = -m2 g / k_stop.
    const Real m1 = 1.0, m2 = 2.0, k1 = 1e8, k_stop = 2e4;
    System sys;
    const int stiff = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                         block(m1), "stiff");
    const int falling = sys.model.add_body(stiff, std::make_shared<PrismaticJointModel>(), Transform3(),
                                           Transform3(), block(m2), "falling");
    JointCoordinateForceParams spring;
    spring.spring = Curve::linear(k1);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, stiff, spring));
    JointCoordinateForceParams stop;
    stop.lower_limit = 0.0;
    stop.limit_stiffness = k_stop;
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, falling, stop));

    Simulator sim(sys);
    sim.q(1) = 0.1;
    const StaticReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    CHECK_FALSE(r.converged);
    CHECK(r.relaxation_runs == 1);
    CHECK(r.relaxation_undone == 1);
    CHECK(has_message(r.notes, "undone"));
    CHECK(sim.q(1) == 0.1);
    CHECK(std::abs(sim.q(0) + m1 * g_accel / k1) < 1e-15);

    Simulator again(sys);
    again.q(1) = 0.1;
    StaticOptions options;
    options.relaxation_step = 1e-4;
    const StaticReport s = static_equilibrium(again, options);
    INFO(s.summary(sys.model));
    CHECK(s.converged);
    CHECK(s.relaxation_undone == 0);
    CHECK(std::abs(again.q(0) + (m1 + m2) * g_accel / k1) < 1e-15);
    CHECK(std::abs(again.q(1) + m2 * g_accel / k_stop) < 1e-12);
}

TEST_CASE("Statics: a body with nothing to hold it is reported", "[kernel][statics]")
{
    // A block on a vertical slider under gravity and nothing else: a force
    // with no stiffness. Newton cannot reduce it; relaxation lets the block
    // fall until its budget ends; the report says so.
    System sys;
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       block(1.0), "block");
    Simulator sim(sys);
    StaticOptions options;
    options.max_iterations = 5;
    options.max_relaxation_steps = 200;

    const StaticReport r = static_equilibrium(sim, options);
    INFO(r.summary(sys.model));
    CHECK(!r.converged);
    CHECK(r.relaxation_steps == 200);
    CHECK(std::abs(r.residual - g_accel) < 1e-12);
    CHECK(r.supported == 0);
    CHECK(has_message(r.warnings, "MBD-K070"));
    CHECK(has_message(r.notes, "MBD-K073"));
    CHECK(sim.q(0) < -0.15);   // it fell: g (0.2 s)^2 / 2 = 0.196 m
}

TEST_CASE("Statics: positions that cannot meet the constraints are reported", "[kernel][statics]")
{
    // The tip of a 0.5 m arm held 0.5 m from a point 2 m from the hinge: out of
    // reach.
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia arm = block(1.0);
    arm.com_B = Vec3(0.25, 0.0, 0.0);
    const int body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(),
                                        Transform3(), arm, "arm");
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{0, Transform3::FromTranslation(Vec3(2.0, 0.0, 0.0))},
        Marker{body, Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0))}, 0.5));
    Simulator sim(sys);

    const StaticReport r = static_equilibrium(sim);
    CHECK(!r.converged);
    CHECK(has_message(r.warnings, "MBD-K071"));
    CHECK(!has_message(r.warnings, "MBD-K070"));
    CHECK_THAT(r.summary(sys.model), Catch::Matchers::ContainsSubstring("NOT found"));
}
