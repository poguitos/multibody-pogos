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
    struct SingleBodyFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
        mbd::Real mass{1.0};
        mbd::Vec3 box_extents{0.5, 0.5, 0.5}; // half-extents
    };

    SingleBodyFixture make_single_body(mbd::Real mass = 1.0,
                                       mbd::Vec3 half_extents = mbd::Vec3(0.5, 0.5, 0.5))
    {
        using namespace mbd;
        SingleBodyFixture fx;
        fx.mass = mass;
        fx.box_extents = half_extents;

        auto I = RigidBodyInertia::from_solid_box(mass, half_extents);
        fx.body_idx = fx.sys.add_body(I, RigidBodyState{}, "body", kGroundIndex);
        fx.sys.add_joint(std::make_unique<FreeCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body_idx));
        return fx;
    }
}

// ============================================================================
// Translation tests
// ============================================================================

TEST_CASE("M2: Free fall under gravity matches h = 0.5*g*t^2",
          "[multibody_basics][gravity]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real t = 1.0;
    sim.run(t, 0.001);

    // Expected: y = -0.5 * g * t^2, v_y = -g * t
    const Real y_expected = -0.5 * g_accel * t * t;
    const Real vy_expected = -g_accel * t;

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(y_expected, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.y(), WithinAbs(vy_expected, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-6));
}

TEST_CASE("M2: Constant force gives a = F/m (zero rotation)",
          "[multibody_basics][force]")
{
    using namespace mbd;
    auto fx = make_single_body(2.0); // 2 kg

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real F = 10.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    const Real a_expected = F / fx.mass;
    const Real v_expected = a_expected * 1.0;
    const Real p_expected = 0.5 * a_expected * 1.0 * 1.0;

    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(v_expected, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(p_expected, 0.01));
}

TEST_CASE("M2: Constant force gives a = F/m even when body is yawed",
          "[multibody_basics][force][rotated]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Pre-yaw 90 degrees about Y
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 2.0;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real F = 10.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    // tau(0) is force in WORLD X (after the M1 fix).
    // Expected: body accelerates in world +X with a = F/m = 10 m/s^2.
    // After 1s: v_x = 10, p_x = 5.
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(10.0, 0.1));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(5.0, 0.1));

    // No motion in Y or Z
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 0.05));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 0.05));
}

TEST_CASE("M2: No-force coast preserves velocity (no drift over 10s)",
          "[multibody_basics][stability]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Initial velocity 5 m/s in +X, no forces
    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 5.0;
    fx.sys.compute_kinematics();

    sim.run(10.0, 0.001);

    // After 10s of no force: v unchanged at 5, p = 50
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(5.0, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(50.0, 0.05));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.y(), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.z(), WithinAbs(0.0, 1e-6));
}

TEST_CASE("M2: No-force coast on rotated body preserves world velocity",
          "[multibody_basics][stability][rotated]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Yaw 45 degrees, set q_dot(0) = 5 (world +X velocity)
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 4.0;
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 5.0;
    fx.sys.compute_kinematics();

    sim.run(10.0, 0.001);

    // Body should travel along world +X at 5 m/s. The body's yaw should
    // remain at pi/4 (no torques applied).
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(50.0, 0.5));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 0.1));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 0.1));

    // Yaw should still be approximately pi/4
    REQUIRE_THAT(fx.sys.q(4), WithinAbs(pi / 4.0, 0.05));
}

// ============================================================================
// Rotation tests
// ============================================================================

TEST_CASE("M2: Constant torque about Y gives angular acceleration alpha = tau/I",
          "[multibody_basics][torque]")
{
    using namespace mbd;
    // Use a long thin box to make I_yy distinct
    auto fx = make_single_body(1.0, Vec3(0.1, 0.1, 1.0));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // For a box (2a x 2b x 2c) of mass m: I_yy = (m/3)(a^2 + c^2)
    // half_extents = (0.1, 0.1, 1.0) → I_yy = (1/3)(0.01 + 1.0) = 0.337
    const Real I_yy_expected = (fx.mass / 3.0) *
                               (fx.box_extents.x() * fx.box_extents.x() +
                                fx.box_extents.z() * fx.box_extents.z());

    const Real tau_y = 1.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(4) += tau_y;
    };

    sim.run(0.5, 0.0001);

    // Expected angular acceleration: alpha = tau / I_yy
    // After 0.5s, angle ~ 0.5 * alpha * t^2
    const Real alpha_expected = tau_y / I_yy_expected;
    const Real angle_expected = 0.5 * alpha_expected * 0.5 * 0.5;

    INFO("I_yy_expected = " << I_yy_expected);
    INFO("alpha_expected = " << alpha_expected);
    INFO("Final q(4) = " << fx.sys.q(4));

    REQUIRE_THAT(fx.sys.q(4), WithinAbs(angle_expected, angle_expected * 0.05));
}

TEST_CASE("M2: Spinning body with no torque preserves angular velocity",
          "[multibody_basics][torque][stability]")
{
    using namespace mbd;
    // Cube (symmetric inertia)
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Set initial angular velocity about Y
    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(4) = 1.0; // 1 rad/s about Y
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    // For a cube (symmetric inertia tensor), no precession, omega preserved
    REQUIRE_THAT(fx.sys.q_dot(4), WithinAbs(1.0, 0.01));
}

// ============================================================================
// Energy conservation
// ============================================================================

TEST_CASE("M2: Free fall: KE + PE conserved within RK4 tolerance",
          "[multibody_basics][energy]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Drop from rest at y = 10
    fx.sys.q.setZero();
    fx.sys.q(1) = 10.0;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    auto compute_total_energy = [&]() -> Real {
        const Vec3 v = fx.sys.states[fx.body_idx].v_WB;
        const Real h = fx.sys.states[fx.body_idx].p_WB.y();
        const Real KE = 0.5 * fx.mass * v.squaredNorm();
        const Real PE = fx.mass * g_accel * h;
        return KE + PE;
    };

    const Real E0 = compute_total_energy();

    // Run for 1s (body falls about 5m, never below ground)
    sim.run(1.0, 0.001);

    const Real E1 = compute_total_energy();

    INFO("E0 = " << E0 << ", E1 = " << E1);

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.001)); // 0.1% energy drift
}

TEST_CASE("M2: Spinning body: rotational KE preserved",
          "[multibody_basics][energy]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(4) = 2.0; // 2 rad/s about Y
    fx.sys.compute_kinematics();

    // For a cube (symmetric I), KE_rot = 0.5 * I_yy * omega^2 stays constant
    const Real I_yy = (fx.mass / 3.0) *
                      (fx.box_extents.x() * fx.box_extents.x() +
                       fx.box_extents.z() * fx.box_extents.z());
    const Real KE0 = 0.5 * I_yy * 2.0 * 2.0;

    sim.run(2.0, 0.001);

    // Compute rotational KE from current angular velocity
    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    const Mat3 R_WB = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();
    const Mat3 I_com_W = R_WB * fx.sys.inertias[fx.body_idx].I_com_B * R_WB.transpose();
    const Real KE1 = 0.5 * w.dot(I_com_W * w);

    REQUIRE_THAT(KE1, WithinAbs(KE0, KE0 * 0.005)); // 0.5% drift
}

// ============================================================================
// Long-running stability
// ============================================================================

TEST_CASE("M2: Long-running stability under coupled translation + rotation",
          "[multibody_basics][stability]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Translate at 5 m/s in X, rotate at 0.5 rad/s about Y
    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 5.0;
    fx.sys.q_dot(4) = 0.5;
    fx.sys.compute_kinematics();

    // No external forces
    sim.run(20.0, 0.001);

    // After 20s of no force:
    //   - Translation: 5 * 20 = 100 m in world X
    //   - Rotation: 0.5 * 20 = 10 rad about Y... but rotation vector wraps,
    //     so we just check that the body is still moving sensibly.
    //   - Angular velocity should be preserved at 0.5 rad/s about Y.
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(100.0, 0.5));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 0.1));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 0.1));

    // Angular velocity should be preserved (within RK4 tolerance)
    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    REQUIRE_THAT(w.y(), WithinAbs(0.5, 0.005));
    REQUIRE_THAT(w.x(), WithinAbs(0.0, 0.005));
    REQUIRE_THAT(w.z(), WithinAbs(0.0, 0.005));

    // World velocity in X should be preserved
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(5.0, 0.01));
}