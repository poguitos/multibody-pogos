// The force library's connectors (plan task 3.1, decision D24): the
// rotational spring-damper, the six-axis bushing and the user force. Each is
// checked against finite differences of its law and coordinates, and against
// an independent motion.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/user_force.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/spatial/spatial.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

constexpr Real h_fd = 1e-7;   // see test_force_library.cpp

bool close(Real analytic, Real numeric)
{
    return std::abs(analytic - numeric) <= 1e-6 * std::max(1.0, std::abs(numeric));
}

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

/// Two bodies' world states: body 1 at a general pose, body 2 displaced from
/// it by a given relative pose of the markers, both moving.
std::vector<RigidBodyState> two_bodies(const Vec3& rotation_rel, const Vec3& offset_rel)
{
    std::vector<RigidBodyState> s(3);
    s[1] = RigidBodyState(Vec3(0.3, -0.1, 0.7), Quat(Eigen::AngleAxisd(0.4, Vec3(1, 2, 3).normalized())),
                          Vec3(0.2, 0.5, -0.1), Vec3(0.3, -0.2, 0.6));
    const Quat q2 = s[1].q_WB * exp3(rotation_rel);
    s[2] = RigidBodyState(s[1].p_WB + s[1].q_WB * offset_rel, q2, Vec3(-0.4, 0.1, 0.3),
                          Vec3(-0.1, 0.4, 0.2));
    return s;
}

/// The states after time dt at constant velocities.
std::vector<RigidBodyState> advanced(std::vector<RigidBodyState> s, Real dt)
{
    for (std::size_t b = 1; b < s.size(); ++b) {
        s[b].p_WB += dt * s[b].v_WB;
        s[b].q_WB = exp3(dt * s[b].w_WB) * s[b].q_WB;
    }
    return s;
}

} // namespace

// --- Rotational spring-damper ------------------------------------------------------------

TEST_CASE("Force library: the rotational spring-damper's twist and derivatives",
          "[forces][library]")
{
    RotationalSpringDamperParams p;
    p.reference = 0.1;
    p.spring = Curve::table({-1.0, 0.0, 0.5, 1.0}, {-300.0, 0.0, 120.0, 400.0});
    p.preload = 4.0;
    p.damper = Curve::linear(2.5);
    const RotationalSpringDamper rsd(1, Transform3(), 2, Transform3(), p);

    // The twist of a pure rotation about the markers' Z axis is its angle.
    for (const Real angle : {-2.9, -0.3, 0.0, 0.7, 3.0}) {
        const auto s = two_bodies(Vec3(0.0, 0.0, angle), Vec3::Zero());
        Real twist = 0.0, rate = 0.0;
        rsd.twist(s, twist, rate);
        CHECK(std::abs(twist - angle) < 1e-12);
    }
    for (const Real angle : {-0.73, -0.21, 0.37, 0.88}) {
        for (const Real rate : {-1.3, 0.2, 2.1}) {
            const LawValue f = rsd.law(angle, rate);
            CHECK(close(f.d_position, (rsd.law(angle + h_fd, rate).force
                                       - rsd.law(angle - h_fd, rate).force) / (2.0 * h_fd)));
            CHECK(close(f.d_rate, (rsd.law(angle, rate + h_fd).force
                                   - rsd.law(angle, rate - h_fd).force) / (2.0 * h_fd)));
        }
    }
}

TEST_CASE("Force library: a rotational spring across a hinge moves it as a joint torsion spring does",
          "[forces][library]")
{
    // A body on a hinge about the world Z axis, with its centre of mass off
    // the axis. A torsion spring and damper act once as a rotational
    // spring-damper between the ground and the body, once as a force on the
    // joint coordinate. The twist is then the joint angle, so the two motions
    // must be the same.
    auto run = [](bool rotational) {
        System sys;
        RigidBodyInertia I = RigidBodyInertia::from_solid_box(3.0, Vec3(0.2, 0.05, 0.05));
        I.com_B = Vec3(0.2, 0.0, 0.0);
        const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(),
                                         Transform3(), I, "arm");
        if (rotational) {
            RotationalSpringDamperParams p;
            p.spring = Curve::linear(40.0);
            p.damper = Curve::linear(0.3);
            sys.force_elements.push_back(
                std::make_shared<RotationalSpringDamper>(0, Transform3(), b, Transform3(), p));
        } else {
            JointCoordinateForceParams p;
            p.spring = Curve::linear(40.0);
            p.damper = Curve::linear(0.3);
            sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, b, p));
        }
        Simulator sim(sys);
        sim.q(0) = 0.5;
        sim.initialize();
        std::vector<Real> q;
        for (int s = 0; s < 1000; ++s) {
            sim.step(1e-3);
            q.push_back(sim.q(0));
        }
        return q;
    };
    const std::vector<Real> a = run(true), b = run(false);
    Real worst = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) worst = std::max(worst, std::abs(a[k] - b[k]));
    CHECK(worst < 1e-12);
}

// --- Bushing --------------------------------------------------------------------------------

TEST_CASE("Force library: the bushing's coordinates, rates and derivatives", "[forces][library]")
{
    BushingParams p;
    for (std::size_t i = 0; i < 3; ++i) {
        p.rotational_stiffness[i] = Curve::table({-0.2, 0.0, 0.2}, {-150.0 * (i + 1), 0.0, 90.0 * (i + 1)});
        p.stiffness[i] = Curve::linear(2e5 * (i + 1));
        p.rotational_damping[i] = Curve::linear(3.0);
        p.damping[i] = Curve::table({-1.0, 0.0, 1.0}, {-800.0, 0.0, 1300.0});
    }
    const Transform3 X_1M(Quat(Eigen::AngleAxisd(0.3, Vec3::UnitY())), Vec3(0.1, 0.0, -0.05));
    const Transform3 X_2M(Quat(Eigen::AngleAxisd(-0.2, Vec3::UnitX())), Vec3(0.0, 0.08, 0.0));
    const Bushing bushing(1, X_1M, 2, X_2M, p);

    // Body 2 placed so that the markers' relative pose is known.
    const Vec3 rotation(0.01, 0.02, -0.03), offset(0.004, -0.002, 0.001);
    auto states = two_bodies(Vec3::Zero(), Vec3::Zero());
    {
        const Quat q1M = states[1].q_WB * X_1M.q;
        const Vec3 p1M = states[1].p_WB + states[1].q_WB * X_1M.p;
        const Quat q2M = q1M * exp3(rotation);
        const Vec3 p2M = p1M + q1M * offset;
        states[2].q_WB = q2M * X_2M.q.conjugate();
        states[2].p_WB = p2M - states[2].q_WB * X_2M.p;
    }
    Vec6 x, x_dot;
    bushing.deflection(states, x, x_dot);
    CHECK((x.head<3>() - rotation).norm() < 1e-12);
    CHECK((x.tail<3>() - offset).norm() < 1e-12);

    // The displacement's rate is the derivative of the displacement; the
    // rotational rate is the rotation vector's to first order in the rotation.
    const Real dt = 1e-6;
    Vec6 xp, xm, dummy;
    bushing.deflection(advanced(states, dt), xp, dummy);
    bushing.deflection(advanced(states, -dt), xm, dummy);
    const Vec6 x_dot_fd = (xp - xm) / (2.0 * dt);
    CHECK((x_dot_fd.tail<3>() - x_dot.tail<3>()).norm() < 1e-6 * x_dot.tail<3>().norm());
    CHECK((x_dot_fd.head<3>() - x_dot.head<3>()).norm() < 0.05 * x_dot.head<3>().norm());

    // The law's derivatives, axis by axis.
    const BushingLaw law = bushing.law(x, x_dot);
    for (int i = 0; i < 6; ++i) {
        Vec6 e = Vec6::Zero();
        e(i) = h_fd;
        const Vec6 dpos = (bushing.law(x + e, x_dot).load - bushing.law(x - e, x_dot).load) / (2.0 * h_fd);
        const Vec6 drate = (bushing.law(x, x_dot + e).load - bushing.law(x, x_dot - e).load) / (2.0 * h_fd);
        for (int j = 0; j < 6; ++j) {
            CHECK(close(law.d_position(j, i), dpos(j)));
            CHECK(close(law.d_rate(j, i), drate(j)));
        }
    }
}

TEST_CASE("Force library: a body on a bushing oscillates at sqrt(k / m) and sqrt(k_r / I)",
          "[forces][library]")
{
    // A free body tied to the ground by a bushing at its centre of mass, no
    // gravity: a displacement along X and a rotation about Z each oscillate
    // on their own. The rotational coordinate of a rotation about one axis is
    // exactly its angle, so both are harmonic.
    const Real m = 4.0, k = 1e4, kr = 50.0;
    const RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, Vec3(0.2, 0.1, 0.15));
    auto period = [&](int coordinate, Real start) {
        System sys;
        sys.model.gravity.setZero();
        sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), I, "block");
        BushingParams p;
        for (std::size_t i = 0; i < 3; ++i) {
            p.stiffness[i] = Curve::linear(k);
            p.rotational_stiffness[i] = Curve::linear(kr);
        }
        sys.force_elements.push_back(std::make_shared<Bushing>(0, Transform3(), 1, Transform3(), p));
        Simulator sim(sys);
        if (coordinate == 0) {
            sim.q(0) = start;                                          // along X
        } else {
            sim.q.segment<4>(3) = Quat(Eigen::AngleAxisd(start, Vec3::UnitZ())).coeffs();   // about Z
        }
        sim.initialize();
        std::vector<Real> t{0.0}, x{coordinate == 0 ? sim.q(0) : sim.q(5)};
        for (int s = 0; s < 8000; ++s) {   // 0.8 s: past two upward crossings of both
            sim.step(1e-4);
            t.push_back(sim.time);
            x.push_back(coordinate == 0 ? sim.q(0) : sim.q(5));   // the quaternion's z: sin(angle / 2)
        }
        return period_of(t, x);
    };
    CHECK(std::abs(period(0, 1e-3) - 2.0 * pi * std::sqrt(m / k)) < 1e-6);
    CHECK(std::abs(period(1, 0.01) - 2.0 * pi * std::sqrt(I.I_com_B(2, 2) / kr)) < 1e-6);
}

// --- User force ---------------------------------------------------------------------------------

TEST_CASE("Force library: a user force acts as written", "[forces][library]")
{
    // A constant 6 N along X on a 3 kg free body, no gravity: 2 m/s^2.
    System sys;
    sys.model.gravity.setZero();
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(3.0, Vec3(0.1, 0.1, 0.1)), "block");
    sys.force_elements.push_back(std::make_shared<UserForce>(
        "push", std::vector<BodyIndex>{1},
        [](const std::vector<RigidBodyState>&, std::vector<RigidBodyForces>& forces) {
            forces[1].f_W += Vec3(6.0, 0.0, 0.0);
        }));
    Simulator sim(sys);
    sim.initialize();
    sim.run(1.0, 1e-3);
    CHECK(std::abs(sim.q(0) - 1.0) < 1e-12);    // x = a t^2 / 2
    CHECK(std::abs(sim.v(3) - 2.0) < 1e-12);
    CHECK_THROWS_WITH(UserForce("none", {1}, UserForce::Function{}), ContainsSubstring("MBD-F004"));
}
