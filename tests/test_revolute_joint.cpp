#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/system.hpp"
#include "mbd/simulator.hpp"
#include "mbd/joint.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    // Build a simple pendulum: a point mass on a massless arm of length L,
    // hinged to ground via a revolute joint about world Z.
    //
    // Joint Z axis = world Z. Arm hangs down (along -Y) at q = 0.
    // We model the arm as a thin rod for inertia, with the mass distributed.
    //
    // For a uniform rod of length L and mass m hinged at one end:
    //   I_about_hinge = (1/3) * m * L^2
    //   Period of small oscillation: T = 2*pi * sqrt(2L / (3g))
    //   (since equivalent length L_eff = (2/3)*L for a uniform rod)
    //
    // We instead use a POINT MASS on a massless arm by giving the body
    // an inertia consistent with a point at distance L from the hinge.
    //
    // For a point mass at distance L from the hinge:
    //   - The body's COM is at L from the hinge (parallel-axis theorem)
    //   - Its inertia about the hinge: I_hinge = m*L^2
    //   - Period: T = 2*pi * sqrt(L/g)
    struct PendulumFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
        mbd::Real mass{1.0};
        mbd::Real arm_length{1.0};
    };

    PendulumFixture make_pendulum(mbd::Real mass = 1.0, mbd::Real L = 1.0)
    {
        using namespace mbd;
        PendulumFixture fx;
        fx.mass = mass;
        fx.arm_length = L;

        // We model a point mass at distance L from the hinge.
        // The body's COM is in BODY frame. We choose the body origin to coincide
        // with the hinge point. Then COM offset in body frame is along -Y of body.
        // Body frame at q = 0: aligned with world. At q != 0, rotated about Z.
        //
        // Inertia about COM for a point mass: 0 (but we need to set something
        // small but nonzero for numerical stability). Use a tiny isotropic inertia.
        Mat3 I_com = Mat3::Identity() * Real(1e-6);
        Vec3 com_offset_B(0.0, -L, 0.0); // COM is L below hinge in body frame
        RigidBodyInertia inertia(mass, com_offset_B, I_com);

        fx.body_idx = fx.sys.add_body(inertia, RigidBodyState{}, "pendulum_body",
                                       kGroundIndex);
        fx.sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body_idx));

        return fx;
    }
}

// ============================================================================
// Kinematic tests
// ============================================================================

TEST_CASE("RevoluteJoint: q=0 places body in identity orientation",
          "[revolute][kinematics]")
{
    using namespace mbd;
    auto fx = make_pendulum();

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();
    REQUIRE_THAT(R(0, 0), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(R(1, 1), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(R(2, 2), WithinAbs(1.0, 1e-12));
}

TEST_CASE("RevoluteJoint: q rotates body about Z axis",
          "[revolute][kinematics]")
{
    using namespace mbd;
    auto fx = make_pendulum();

    fx.sys.q.setZero();
    fx.sys.q(0) = pi / 4.0; // 45 degrees about Z
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();

    // Z axis preserved
    REQUIRE_THAT(R(2, 2), WithinAbs(1.0, 1e-9));

    // X axis rotated to (cos(pi/4), sin(pi/4), 0)
    REQUIRE_THAT(R(0, 0), WithinAbs(std::cos(pi / 4.0), 1e-9));
    REQUIRE_THAT(R(1, 0), WithinAbs(std::sin(pi / 4.0), 1e-9));

    // Body COM in world: COM was at body-Y = -L. After rotation by pi/4 about Z,
    // body -Y direction is now (sin(pi/4), -cos(pi/4), 0) in world.
    const Vec3 com_W = fx.sys.states[fx.body_idx].p_WB +
                       R * Vec3(0.0, -fx.arm_length, 0.0);
    REQUIRE_THAT(com_W.x(), WithinAbs(fx.arm_length * std::sin(pi / 4.0), 1e-9));
    REQUIRE_THAT(com_W.y(), WithinAbs(-fx.arm_length * std::cos(pi / 4.0), 1e-9));
}

TEST_CASE("RevoluteJoint: q_dot produces angular velocity about Z",
          "[revolute][kinematics]")
{
    using namespace mbd;
    auto fx = make_pendulum();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 2.0; // 2 rad/s
    fx.sys.compute_kinematics();

    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    REQUIRE_THAT(w.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(w.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(w.z(), WithinAbs(2.0, 1e-9));
}

// ============================================================================
// Force/torque tests
// ============================================================================

TEST_CASE("RevoluteJoint: applied torque about Z gives alpha = tau/I",
          "[revolute][torque]")
{
    using namespace mbd;
    const Real m = 2.0;
    const Real L = 1.5;
    auto fx = make_pendulum(m, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero()); // no gravity for pure torque test
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // For a point mass at distance L from hinge: I_hinge = m*L^2
    const Real I_hinge = m * L * L;
    const Real torque = 5.0;
    const Real alpha_expected = torque / I_hinge;

    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += torque;
    };

    sim.run(0.5, 0.0001);

    // q(t) = 0.5 * alpha * t^2
    const Real q_expected = 0.5 * alpha_expected * 0.5 * 0.5;
    INFO("alpha_expected = " << alpha_expected);
    INFO("q_expected = " << q_expected);
    INFO("q_actual = " << fx.sys.q(0));

    REQUIRE_THAT(fx.sys.q(0), WithinAbs(q_expected, q_expected * 0.02));
}

// ============================================================================
// Pendulum dynamics
// ============================================================================

TEST_CASE("RevoluteJoint: small-angle pendulum has period T = 2*pi*sqrt(L/g)",
          "[revolute][pendulum]")
{
    using namespace mbd;
    const Real L = 1.0;
    auto fx = make_pendulum(1.0, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at small angle 0.05 rad, no initial velocity
    fx.sys.q.setZero();
    fx.sys.q(0) = 0.05;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real T_expected = 2.0 * pi * std::sqrt(L / g_accel);

    INFO("T_expected = " << T_expected);

    // Run for one full period
    sim.run(T_expected, 0.0001);

    // After one period: q should be near 0.05 again, q_dot near 0
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.05, 0.0005));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(0.0, 0.005));
}

TEST_CASE("RevoluteJoint: pendulum energy conserved over multiple swings",
          "[revolute][pendulum][energy]")
{
    using namespace mbd;
    const Real L = 1.0;
    const Real m = 1.0;
    auto fx = make_pendulum(m, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at moderate angle 0.5 rad (still nonlinear but stable)
    fx.sys.q.setZero();
    fx.sys.q(0) = 0.5;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        // KE = 0.5 * I_hinge * omega^2  for a point mass on rigid arm
        const Real I_hinge = m * L * L;
        const Real omega = fx.sys.q_dot(0);
        const Real KE = 0.5 * I_hinge * omega * omega;

        // PE = m*g*h, where h = -L*cos(q) relative to hinge
        const Real h = -L * std::cos(fx.sys.q(0));
        const Real PE = m * g_accel * h;
        return KE + PE;
    };

    const Real E0 = compute_energy();

    // Run for 5 seconds (~5 oscillations of a 1m pendulum)
    sim.run(5.0, 0.0001);

    const Real E1 = compute_energy();

    INFO("E0 = " << E0 << ", E1 = " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.005)); // 0.5% drift over 5s
}

TEST_CASE("RevoluteJoint: pendulum at rest hangs straight down (equilibrium)",
          "[revolute][pendulum][equilibrium]")
{
    using namespace mbd;
    auto fx = make_pendulum();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start exactly at equilibrium (q=0 means COM hangs at -Y, which is down)
    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    // Should remain at rest
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(0.0, 1e-6));
}

// ============================================================================
// Stress: large-angle oscillation
// ============================================================================

TEST_CASE("RevoluteJoint: large-angle pendulum (1 rad) energy conserved",
          "[revolute][pendulum][energy]")
{
    using namespace mbd;
    const Real L = 1.0;
    const Real m = 1.0;
    auto fx = make_pendulum(m, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at 1 rad (~57 degrees), strongly nonlinear
    fx.sys.q.setZero();
    fx.sys.q(0) = 1.0;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        const Real I_hinge = m * L * L;
        const Real omega = fx.sys.q_dot(0);
        const Real KE = 0.5 * I_hinge * omega * omega;
        const Real h = -L * std::cos(fx.sys.q(0));
        const Real PE = m * g_accel * h;
        return KE + PE;
    };

    const Real E0 = compute_energy();

    sim.run(10.0, 0.0001); // 10 seconds, multiple swings

    const Real E1 = compute_energy();

    INFO("Initial energy: " << E0);
    INFO("Final energy: " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.01));
}