// The force library (plan task 3.1, decision D24): tabulated characteristics,
// the spring-damper with preload and stops, and forces on joint coordinates
// (spring, damper, friction, limit stops). Each element's derivatives are
// checked against central differences of its force law, and each element
// against a closed-form motion.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/validate.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

// Central differences with h = 1e-7: truncation h^2 f''' / 6 and round-off
// eps |f| / h are both below 1e-6 of the derivatives for the laws below
// (forces up to 1e4 N, derivatives from 1e2 to 1e6), provided the points stay
// away from kinks (table knots, stop engagement).
constexpr Real h_fd = 1e-7;

bool close(Real analytic, Real numeric)
{
    return std::abs(analytic - numeric) <= 1e-6 * std::max(1.0, std::abs(numeric));
}

/// Time between the first two upward zero crossings of x(t): one period.
Real period_of(const std::vector<Real>& t, const std::vector<Real>& x)
{
    std::vector<Real> up;
    for (std::size_t k = 1; k < x.size() && up.size() < 2; ++k) {
        if (x[k - 1] < 0.0 && x[k] >= 0.0) {
            up.push_back(t[k - 1] + (t[k] - t[k - 1]) * (-x[k - 1]) / (x[k] - x[k - 1]));
        }
    }
    REQUIRE(up.size() == 2);
    return up[1] - up[0];
}

/// A body on a prismatic joint along world axis `axis` (0 = X, 2 = Z).
int slider(System& sys, int axis, Real mass)
{
    const Vec3 dir = axis == 0 ? Vec3::UnitX() : Vec3::UnitZ();
    const Quat R = Quat::FromTwoVectors(Vec3::UnitZ(), dir);
    return sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(R, Vec3::Zero()),
                              Transform3(R, Vec3::Zero()),
                              RigidBodyInertia::from_solid_box(mass, Vec3(0.05, 0.05, 0.05)), "slider");
}

} // namespace

// --- Curves -----------------------------------------------------------------------

TEST_CASE("Force library: tabulated curves pass through their points, monotone, with a continuous slope",
          "[forces][library]")
{
    const Curve line = Curve::linear(3.0, 1.0);
    CHECK(line.value(2.0) == 7.0);
    CHECK(line.slope(-5.0) == 3.0);
    CHECK(Curve().value(4.0) == 0.0);

    // A progressive spring: force against compression, stiffening.
    const std::vector<Real> x{0.0, 0.02, 0.04, 0.06, 0.08};
    const std::vector<Real> y{0.0, 400.0, 900.0, 1600.0, 2600.0};
    const Curve c = Curve::table(x, y);
    for (std::size_t k = 0; k < x.size(); ++k) CHECK(std::abs(c.value(x[k]) - y[k]) < 1e-9);

    // The slope against differences, inside the table (away from its knots)
    // and beyond it, where the curve continues as a line with the end slope.
    for (const Real xs : {-0.03, 0.0113, 0.0297, 0.0461, 0.0702, 0.11}) {
        const Real numeric = (c.value(xs + h_fd) - c.value(xs - h_fd)) / (2.0 * h_fd);
        CHECK(close(c.slope(xs), numeric));
    }
    // The slope is continuous at each knot.
    for (std::size_t k = 1; k + 1 < x.size(); ++k) {
        CHECK(std::abs(c.slope(x[k] - 1e-9) - c.slope(x[k] + 1e-9)) < 1e-6 * std::abs(c.slope(x[k])));
    }
    // Monotone data, monotone curve: no value falls, and each interval stays
    // between its end values (no overshoot).
    Real previous = c.value(x.front());
    for (int i = 1; i <= 800; ++i) {
        const Real xi = 0.08 * i / 800.0;
        const Real yi = c.value(xi);
        CHECK(yi >= previous - 1e-12);
        previous = yi;
    }

    CHECK_THROWS_WITH(Curve::table({0.0, 0.0}, {1.0, 2.0}), ContainsSubstring("MBD-F002"));
    CHECK_THROWS_WITH(Curve::table({0.0}, {1.0}), ContainsSubstring("MBD-F002"));
}

// --- The spring-damper -----------------------------------------------------------------

TEST_CASE("Force library: the spring-damper's derivatives match finite differences",
          "[forces][library]")
{
    SpringDamperParams p;
    p.free_length = 0.35;
    p.spring = Curve::table({-0.1, 0.0, 0.05, 0.1}, {-2500.0, 0.0, 1400.0, 3600.0});
    p.preload = 500.0;
    p.damper = Curve::table({-1.0, -0.1, 0.0, 0.1, 1.0}, {-2000.0, -400.0, 0.0, 900.0, 4000.0});
    p.bump_clearance = 0.06;   // engages below L = 0.29
    p.bump_stop = Curve::table({0.0, 0.01, 0.02}, {0.0, 2000.0, 8000.0});
    p.rebound_clearance = 0.05;   // engages above L = 0.40
    p.rebound_stop = Curve::linear(5e5);
    const SpringDamper sd(0, 1, Vec3::Zero(), Vec3::Zero(), p);

    // From the bump stop to beyond the rebound stop, both damper branches;
    // none of these points is a knot or an engagement point.
    for (const Real L : {0.2213, 0.2731, 0.3077, 0.3311, 0.3622, 0.4117, 0.4403}) {
        for (const Real rate : {-0.83, -0.047, 0.0031, 0.052, 0.61}) {
            const LawValue f = sd.law(L, rate);
            const Real dL = (sd.law(L + h_fd, rate).force - sd.law(L - h_fd, rate).force) / (2.0 * h_fd);
            const Real dr = (sd.law(L, rate + h_fd).force - sd.law(L, rate - h_fd).force) / (2.0 * h_fd);
            INFO("L = " << L << ", rate = " << rate);
            CHECK(close(f.d_position, dL));
            CHECK(close(f.d_rate, dr));
        }
    }

    // The stops act only beyond their clearances.
    CHECK(sd.law(0.30, 0.0).force == p.preload + p.spring.value(p.free_length - 0.30));
    CHECK(sd.law(0.28, 0.0).force > p.preload + p.spring.value(0.07));
    CHECK(sd.law(0.41, 0.0).force < p.preload + p.spring.value(-0.06));
}

TEST_CASE("Force library: with straight lines the spring-damper is the linear one",
          "[forces][library]")
{
    // Same anchors, stiffness, damping and free length: the same forces, at
    // random states of two moving bodies.
    const Real k = 2.5e4, c = 1800.0, L0 = 0.4;
    SpringDamperParams p;
    p.free_length = L0;
    p.spring = Curve::linear(k);
    p.damper = Curve::linear(c);
    const Vec3 a1(0.1, -0.2, 0.3), a2(-0.05, 0.15, 0.0);
    const SpringDamper curved(1, 2, a1, a2, p);
    const LinearSpringDamper linear(1, 2, a1, a2, k, c, L0);

    std::vector<RigidBodyState> states(3);
    for (int trial = 0; trial < 5; ++trial) {
        for (int b = 1; b <= 2; ++b) {
            const Real s = 0.37 * trial + 0.21 * b;
            states[b] = RigidBodyState(Vec3(std::sin(s), 0.3 * b, std::cos(2.0 * s)),
                                       Quat(Eigen::AngleAxisd(s, Vec3(1.0, 2.0, 0.5).normalized())),
                                       Vec3(0.4 * s, -0.2, 0.7), Vec3(0.1, s, -0.3));
        }
        std::vector<RigidBodyForces> f1(3), f2(3);
        curved.apply(states, f1);
        linear.apply(states, f2);
        for (int b = 1; b <= 2; ++b) {
            CHECK((f1[b].f_W - f2[b].f_W).norm() <= 1e-12 * (1.0 + f2[b].f_W.norm()));
            CHECK((f1[b].tau_W - f2[b].tau_W).norm() <= 1e-12 * (1.0 + f2[b].tau_W.norm()));
        }
        // Equal and opposite.
        CHECK((f1[1].f_W + f1[2].f_W).norm() <= 1e-12 * (1.0 + f1[2].f_W.norm()));
    }
}

TEST_CASE("Force library: a mass on a preloaded spring settles at its static deflection",
          "[forces][library]")
{
    // A 20 kg slider on a vertical joint stands on a spring of 2e4 N/m whose
    // lower end is fixed 1 m below the slider's zero, so the spring is at its
    // free length at q = 0. It carries the weight less its 50 N preload:
    // preload - k q = m g, so q = -(m g - preload) / k. Damping of 600 N s/m
    // (ratio 0.47) settles it within a second.
    const Real m = 20.0, k = 2e4, preload = 50.0, L0 = 1.0;
    System sys;
    const int b = slider(sys, 2, m);
    SpringDamperParams p;
    p.free_length = L0;
    p.spring = Curve::linear(k);
    p.preload = preload;
    p.damper = Curve::linear(600.0);
    sys.force_elements.push_back(
        std::make_shared<SpringDamper>(b, 0, Vec3::Zero(), Vec3(0.0, 0.0, -L0), p));
    Simulator sim(sys);
    sim.initialize();
    sim.run(3.0, 1e-3);
    const Real expected = -(m * g_accel - preload) / k;
    CHECK(std::abs(sim.q(0) - expected) < 1e-9);
    CHECK(std::abs(sim.v(0)) < 1e-9);
}

// --- Forces on joint coordinates --------------------------------------------------------

TEST_CASE("Force library: the joint coordinate force's derivatives match finite differences",
          "[forces][library]")
{
    System sys;
    const int b = slider(sys, 0, 1.0);
    JointCoordinateForceParams p;
    p.reference = 0.02;
    p.spring = Curve::table({-0.2, 0.0, 0.1, 0.2}, {-900.0, 0.0, 300.0, 1100.0});
    p.preload = 15.0;
    p.damper = Curve::linear(40.0);
    p.friction = 12.0;
    p.friction_velocity = 0.05;
    p.lower_limit = -0.15;
    p.upper_limit = 0.18;
    p.limit_stiffness = 1e5;
    p.limit_damping = 300.0;
    const JointCoordinateForce jf(sys.model, b, p);

    // Inside the limits, beyond each (rates chosen so the stop pushes), and
    // across the friction's regularised rise.
    for (const Real q : {-0.1731, -0.0412, 0.0533, 0.1377, 0.1902, -0.1611}) {
        for (const Real v : {-0.31, -0.012, 0.004, 0.027, 0.44}) {
            const LawValue f = jf.law(q, v);
            const Real dq = (jf.law(q + h_fd, v).force - jf.law(q - h_fd, v).force) / (2.0 * h_fd);
            const Real dv = (jf.law(q, v + h_fd).force - jf.law(q, v - h_fd).force) / (2.0 * h_fd);
            INFO("q = " << q << ", v = " << v);
            CHECK(close(f.d_position, dq));
            CHECK(close(f.d_rate, dv));
        }
    }
    CHECK_THROWS_WITH(JointCoordinateForce(sys.model, 2, p), ContainsSubstring("MBD-K026"));
    p.friction_velocity = 0.0;
    CHECK_THROWS_WITH(JointCoordinateForce(sys.model, b, p), ContainsSubstring("MBD-K027"));
}

TEST_CASE("Force library: a pendulum with a torsion spring swings at sqrt((k + m g L) / I)",
          "[forces][library]")
{
    // The bob of the joint-physics tests (2 kg, 0.9 m below a hinge about the
    // world Y axis) with a torsion spring of 30 N m / rad in the hinge. A swing
    // of 0.01 rad lengthens the period by about 6e-6 of itself.
    const Real m = 2.0, L = 0.9, k = 30.0;
    RigidBodyInertia bob = RigidBodyInertia::from_solid_box(m, Vec3(0.05, 0.04, 0.03));
    bob.com_B = Vec3(0.0, 0.0, -L);
    const Transform3 z_along_y(Quat(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX())), Vec3::Zero());
    System sys;
    const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), z_along_y, z_along_y,
                                     bob, "pendulum");
    JointCoordinateForceParams p;
    p.spring = Curve::linear(k);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, b, p));

    Simulator sim(sys);
    sim.q(0) = 0.01;
    sim.initialize();
    const Real I_p = bob.I_com_B(1, 1) + m * L * L;
    const Real T = 2.0 * pi * std::sqrt(I_p / (k + m * g_accel * L));
    std::vector<Real> t{sim.time}, x{sim.q(0)};
    for (int s = 0; s < static_cast<int>(2.5 * T / 1e-3); ++s) {
        sim.step(1e-3);
        t.push_back(sim.time);
        x.push_back(sim.q(0));
    }
    CHECK(std::abs(period_of(t, x) - T) < 1e-4 * T);
}

TEST_CASE("Force library: Coulomb friction stops a slider in m v0^2 / (2 F)",
          "[forces][library]")
{
    // 1 kg sliding horizontally at 1 m/s against 10 N of friction stops after
    // 0.05 m in 0.1 s. The regularisation (0.01 m/s) acts as a damper of
    // F / v_f = 1000 N s/m near rest, with time constant m v_f / F = 1 ms: it
    // adds at most about 2 v_f times that, 2e-5 m, to the distance. The step
    // of 0.1 ms keeps that damper well inside RK4's stability.
    const Real m = 1.0, F = 10.0, v0 = 1.0;
    System sys;
    const int b = slider(sys, 0, m);
    JointCoordinateForceParams p;
    p.friction = F;
    p.friction_velocity = 0.01;
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, b, p));
    Simulator sim(sys);
    sim.v(0) = v0;
    sim.initialize();
    sim.run(0.3, 1e-4);
    CHECK(std::abs(sim.q(0) - m * v0 * v0 / (2.0 * F)) < 5e-5);
    CHECK(std::abs(sim.v(0)) < 1e-6);
}

TEST_CASE("Force library: a limit stop holds a falling slider at m g / k past the limit",
          "[forces][library]")
{
    // A 2 kg slider falls 0.1 m onto a stop of 1e5 N/m with 200 N s/m: it
    // bounces (restitution about 0.5), settles within a second, and rests
    // m g / k = 1.96e-4 m past the limit.
    const Real m = 2.0, k = 1e5;
    System sys;
    const int b = slider(sys, 2, m);
    JointCoordinateForceParams p;
    p.lower_limit = -0.1;
    p.limit_stiffness = k;
    p.limit_damping = 200.0;
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, b, p));
    Simulator sim(sys);
    sim.initialize();
    sim.run(2.0, 1e-3);
    CHECK(std::abs(sim.q(0) - (-0.1 - m * g_accel / k)) < 1e-9);
    CHECK(std::abs(sim.v(0)) < 1e-6);
}

TEST_CASE("Force library: joint forces on missing bodies are reported and refused",
          "[forces][library]")
{
    System two;
    slider(two, 0, 1.0);
    const int second = slider(two, 2, 1.0);
    const auto jf = std::make_shared<JointCoordinateForce>(two.model, second,
                                                           JointCoordinateForceParams{});
    System one;
    slider(one, 0, 1.0);
    one.joint_forces.push_back(jf);   // made for body 2 of another model
    const ValidationReport r = validate(one);
    CHECK_FALSE(r.ok());
    CHECK_THAT(r.summary(), ContainsSubstring("MBD-M036"));
    CHECK_THROWS_WITH(Simulator(one), ContainsSubstring("MBD-K043"));
}
