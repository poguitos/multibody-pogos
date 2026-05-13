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
    // Helper to build a single-body system with a FreeCoordJoint to ground.
    // Body is a 1m cube with mass = 1 kg (so inertia ≈ 1/6 about each axis).
    struct SingleBodyFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
        mbd::Real mass{1.0};
    };

    SingleBodyFixture make_single_body()
    {
        using namespace mbd;
        SingleBodyFixture fx;

        auto I = RigidBodyInertia::from_solid_box(fx.mass, Vec3(0.5, 0.5, 0.5));
        fx.body_idx = fx.sys.add_body(I, RigidBodyState{}, "body", kGroundIndex);
        fx.sys.add_joint(std::make_unique<FreeCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body_idx));

        return fx;
    }
}

// ============================================================================
// q indices: which q corresponds to which world coordinate at zero rotation?
// ============================================================================

TEST_CASE("FreeCoordJoint: q(0) sets world X position when not rotated",
          "[free_joint][q]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q(0) = 1.0;
    fx.sys.compute_kinematics();

    INFO("Body world position: " << fx.sys.states[fx.body_idx].p_WB.transpose());

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("FreeCoordJoint: q(1) sets world Y position when not rotated",
          "[free_joint][q]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q(1) = 0.7;
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.7, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("FreeCoordJoint: q(2) sets world Z position when not rotated",
          "[free_joint][q]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q(2) = 0.5;
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.5, 1e-9));
}

// ============================================================================
// q rotation indices: which q corresponds to roll/pitch/yaw?
// ============================================================================

TEST_CASE("FreeCoordJoint: q(3) is rotation about world X (roll)",
          "[free_joint][rotation]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q(3) = 0.5;  // 0.5 rad rotation
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();
    INFO("Rotation matrix:\n" << R);

    // For rotation about X by angle theta:
    //   R[Y axis] should be (0, cos(theta), sin(theta))
    //   R[Z axis] should be (0, -sin(theta), cos(theta))
    REQUIRE_THAT(R(0, 0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(R(1, 1), WithinAbs(std::cos(0.5), 1e-9));
    REQUIRE_THAT(R(2, 2), WithinAbs(std::cos(0.5), 1e-9));
}

TEST_CASE("FreeCoordJoint: q(4) is rotation about world Y (yaw, in our X-fwd Y-up Z-left convention)",
          "[free_joint][rotation]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q(4) = 0.3;
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();
    INFO("Rotation matrix:\n" << R);

    // Rotation about Y by angle theta:
    //   R[X axis] = (cos, 0, -sin)  in standard right-handed
    REQUIRE_THAT(R(0, 0), WithinAbs(std::cos(0.3), 1e-9));
    REQUIRE_THAT(R(1, 1), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(R(2, 2), WithinAbs(std::cos(0.3), 1e-9));
}

TEST_CASE("FreeCoordJoint: q(5) is rotation about world Z",
          "[free_joint][rotation]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q(5) = 0.4;
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();
    INFO("Rotation matrix:\n" << R);

    REQUIRE_THAT(R(0, 0), WithinAbs(std::cos(0.4), 1e-9));
    REQUIRE_THAT(R(2, 2), WithinAbs(1.0, 1e-9));
}

// ============================================================================
// q_dot: linear velocity components — world frame or body frame?
// ============================================================================

TEST_CASE("FreeCoordJoint: q_dot(0) at zero rotation produces +X world velocity",
          "[free_joint][velocity]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 2.0;
    fx.sys.compute_kinematics();

    INFO("Body v_WB: " << fx.sys.states[fx.body_idx].v_WB.transpose());

    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("FreeCoordJoint: q_dot(0) interpretation when body is yawed",
          "[free_joint][velocity]")
{
    using namespace mbd;
    auto fx = make_single_body();

    // Yaw body 90 degrees about Y. Body X axis now points in world -Z.
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 2.0;
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.0;
    fx.sys.compute_kinematics();

    const Vec3 v_W = fx.sys.states[fx.body_idx].v_WB;
    INFO("Body v_WB after yaw: " << v_W.transpose());
    INFO("Body forward in world: " << (fx.sys.states[fx.body_idx].q_WB * Vec3::UnitX()).transpose());

    // If q_dot(0) is "linear velocity in joint/parent frame" (world), we get +X world.
    // If q_dot(0) is "linear velocity in body/joint frame after rotation", we get -Z world.
    // The test will reveal which.
    if (std::abs(v_W.x()) > 0.5) {
        // q_dot(0) is in WORLD frame
        REQUIRE_THAT(v_W.x(), WithinAbs(1.0, 1e-6));
        REQUIRE_THAT(v_W.z(), WithinAbs(0.0, 1e-6));
    } else {
        // q_dot(0) is in BODY frame (after rotation)
        REQUIRE_THAT(v_W.x(), WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(v_W.z(), WithinAbs(-1.0, 1e-6));
    }
}

// ============================================================================
// tau: force interpretation — Newton's 2nd law check
// ============================================================================

TEST_CASE("FreeCoordJoint: tau(0) produces world-X acceleration of F/m at zero rotation",
          "[free_joint][force]")
{
    using namespace mbd;
    auto fx = make_single_body();

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

    // Run 1 second. Under constant force F on mass m, a = F/m, dv = a*t = 10
    sim.run(1.0, 0.001);

    INFO("Final v: " << fx.sys.states[fx.body_idx].v_WB.transpose());
    INFO("Final p: " << fx.sys.states[fx.body_idx].p_WB.transpose());

    // v(1) = a*t = 10 m/s, p(1) = 0.5 * a * t² = 5 m
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(10.0, 0.1));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(5.0, 0.1));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 0.01));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 0.01));
}

TEST_CASE("FreeCoordJoint: tau(0) interpretation when body is yawed",
          "[free_joint][force]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Pre-yaw body 90 degrees about Y. Body X axis now points in world -Z.
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 2.0;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real F = 10.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(1.0, 0.001);

    const Vec3 p_W = fx.sys.states[fx.body_idx].p_WB;
    INFO("Final p: " << p_W.transpose());
    INFO("Body forward in world: " << (fx.sys.states[fx.body_idx].q_WB * Vec3::UnitX()).transpose());

    // If tau(0) is force in WORLD frame: p moves in +X world.
    // If tau(0) is force in BODY/JOINT frame: p moves in body-X = world-Z direction.
    if (std::abs(p_W.x()) > 1.0) {
        REQUIRE_THAT(p_W.x(), WithinAbs(5.0, 0.5));
        // tau(0) is in WORLD frame
    } else {
        REQUIRE_THAT(p_W.z(), WithinAbs(-5.0, 0.5));
        // tau(0) is in BODY frame (negative because body X = -world Z)
    }
}

// ============================================================================
// Magnitude check: F/m gives correct acceleration for Newton's 2nd law
// ============================================================================

TEST_CASE("FreeCoordJoint: F = m*a holds for tau(0) magnitude",
          "[free_joint][force]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Apply 6000 N to a 1 kg body. Expected a = 6000 m/s².
    // After 0.001s: v should be 6 m/s.
    const Real F = 6000.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += F;
    };

    sim.run(0.001, 0.0001);

    const Real expected_v = F / fx.mass * 0.001;
    INFO("Expected v after 1ms: " << expected_v);
    INFO("Actual v after 1ms: " << fx.sys.states[fx.body_idx].v_WB.x());

    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(expected_v, expected_v * 0.05));
}

// ============================================================================
// CRITICAL: Position-velocity consistency under rotation
// ============================================================================
// These tests verify that the FreeCoordJoint kinematics and dynamics agree
// on what q_dot represents.
//
//   Kinematics: q(0:3) is world translation (verified by earlier tests).
//   Therefore q_dot(0:3) should equal the body's world linear velocity.
//
//   Dynamics: motion_subspace says q_dot(0:3) is "linear velocity in joint
//   frame" → v_W = R_WJ * q_dot(0:3). When body is rotated, R_WJ != I, and
//   v_W != q_dot(0:3).
//
//   These two interpretations are mutually inconsistent if R_WJ != I. The
//   tests below check whether they actually disagree in the running code.

TEST_CASE("FreeCoordJoint: kinematic v_WB equals q_dot when body unyawed",
          "[free_joint][consistency]")
{
    using namespace mbd;
    auto fx = make_single_body();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.0;
    fx.sys.compute_kinematics();

    // No rotation → world-X velocity should equal q_dot(0)
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.x(), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("FreeCoordJoint: kinematic v_WB when body yawed - reveals interpretation",
          "[free_joint][consistency]")
{
    using namespace mbd;
    auto fx = make_single_body();

    // Pre-yaw 90 degrees about Y (body forward = world -Z)
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 2.0;
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.0;  // First translation coordinate's rate
    fx.sys.compute_kinematics();

    INFO("Body forward in world: "
         << (fx.sys.states[fx.body_idx].q_WB * Vec3::UnitX()).transpose());
    INFO("v_WB: " << fx.sys.states[fx.body_idx].v_WB.transpose());

    // If q_dot(0) is interpreted as body/joint-frame velocity:
    //   v_W = R_WJ * (1,0,0) = body-X in world = (0, 0, -1)
    // If q_dot(0) is interpreted as world-frame velocity (= dq(0)/dt):
    //   v_W = (1, 0, 0)

    const Vec3 v = fx.sys.states[fx.body_idx].v_WB;
    if (std::abs(v.x() - 1.0) < 0.01) {
        SUCCEED("v_WB matches WORLD interpretation: v(0) = q_dot(0) = 1");
    } else if (std::abs(v.z() + 1.0) < 0.01) {
        FAIL("v_WB matches BODY interpretation: v_W = R_WJ * q_dot, so v.z = -1. "
             "This means q_dot is body-frame, but q itself is world-frame "
             "(per kinematic test). INCONSISTENCY CONFIRMED.");
    } else {
        FAIL("v_WB matches NEITHER interpretation: v = " << v.transpose());
    }
}

TEST_CASE("FreeCoordJoint: q time derivative matches v_WB - integration consistency",
          "[free_joint][consistency]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Pre-yaw 90 degrees, set q_dot(0) = 1, no forces
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 2.0;
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.0;
    fx.sys.compute_kinematics();

    const Real q0_before = fx.sys.q(0);
    const Vec3 p_before = fx.sys.states[fx.body_idx].p_WB;

    // Integrate a small step with NO forces — pure kinematic propagation
    sim.run(0.01, 0.001);

    const Real q0_after = fx.sys.q(0);
    const Vec3 p_after = fx.sys.states[fx.body_idx].p_WB;

    const Real dq0 = q0_after - q0_before;
    const Vec3 dp = p_after - p_before;

    INFO("dq(0) over 0.01s: " << dq0);
    INFO("dp_W over 0.01s: " << dp.transpose());

    // KINEMATIC TRUTH: p_WB = q(0:3), so dp = (dq(0), dq(1), dq(2))
    // If integrators are consistent, dp.x() should equal dq(0).
    REQUIRE_THAT(dp.x(), WithinAbs(dq0, 1e-6));

    // What about the y/z components of dp?
    // If dynamics treats q_dot(0)=1 as world-X: dp = (0.01, 0, 0), dq(0) = 0.01
    // If dynamics treats q_dot(0)=1 as body-X = world-Z (since yawed):
    //   v_W = (0, 0, -1), dp = (0, 0, -0.01)
    //   But what does q(0) integrate to? If q_dot is the rate of q,
    //   then dq(0) = 0.01 even though dp.x() = 0. INCONSISTENT.

    // The smoking gun: dq(0) should equal dp.x() if everything is consistent.
    // If they differ, kinematics integrates q while dynamics moves the body
    // somewhere different, leading to numerical chaos.
}

TEST_CASE("FreeCoordJoint: tau acceleration matches kinematic q acceleration",
          "[free_joint][consistency]")
{
    using namespace mbd;
    auto fx = make_single_body();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Pre-yaw 90 degrees
    fx.sys.q.setZero();
    fx.sys.q(4) = pi / 2.0;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Apply tau(0) = 10 N for 1 second
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += 10.0;
    };

    sim.run(1.0, 0.001);

    INFO("Final q: " << fx.sys.q.transpose());
    INFO("Final q_dot: " << fx.sys.q_dot.transpose());
    INFO("Final p_WB: " << fx.sys.states[fx.body_idx].p_WB.transpose());
    INFO("Final v_WB: " << fx.sys.states[fx.body_idx].v_WB.transpose());

    // Newton's 2nd law: a = F/m = 10 m/s^2
    // After 1s with constant force from rest: v = 10, p = 5
    //
    // If tau(0) is interpreted as world-X force (consistent with q):
    //   q(0) should be ~5, q_dot(0) should be ~10, p_WB = (5, 0, 0)
    //
    // If tau(0) is interpreted as body-X force (inconsistent with q):
    //   The body would accelerate in world -Z direction initially,
    //   but the body would also rotate due to nonlinear coupling,
    //   producing the chaotic result we saw earlier.

    const Real expected_q0 = 5.0;
    const Real expected_v0 = 10.0;

    // Strict consistency check
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(expected_q0, 0.5));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(expected_v0, 1.0));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(expected_q0, 0.5));
}