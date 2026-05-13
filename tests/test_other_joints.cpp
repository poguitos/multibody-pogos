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
    // ====================================================================
    // FixedJoint fixture: body rigidly fixed at a specified offset
    // ====================================================================
    struct FixedJointFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
    };

    FixedJointFixture make_fixed(const mbd::Vec3& offset_W = mbd::Vec3::Zero())
    {
        using namespace mbd;
        FixedJointFixture fx;
        Transform3 X_PJ(Quat::Identity(), offset_W);
        auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.5, 0.5));
        fx.body_idx = fx.sys.add_body(inertia, RigidBodyState{}, "fixed_body",
                                       kGroundIndex);
        fx.sys.add_joint(std::make_unique<FixedJoint>(
            X_PJ, Transform3::Identity(),
            kGroundIndex, fx.body_idx));
        return fx;
    }

    // ====================================================================
    // Spherical pendulum fixture: point mass at L below joint
    // ====================================================================
    struct SphericalPendulumFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
        mbd::Real mass{1.0};
        mbd::Real L{1.0};
    };

    SphericalPendulumFixture make_spherical_pendulum(mbd::Real m = 1.0,
                                                      mbd::Real L = 1.0)
    {
        using namespace mbd;
        SphericalPendulumFixture fx;
        fx.mass = m;
        fx.L = L;

        // Point mass at body offset (0, -L, 0) — like the revolute pendulum
        Mat3 I_com = Mat3::Identity() * Real(1e-6);
        Vec3 com_offset_B(0.0, -L, 0.0);
        RigidBodyInertia inertia(m, com_offset_B, I_com);

        fx.body_idx = fx.sys.add_body(inertia, RigidBodyState{}, "sph_pendulum",
                                       kGroundIndex);
        fx.sys.add_joint(std::make_unique<SphericalCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body_idx));
        return fx;
    }

    // ====================================================================
    // Universal joint fixture: similar to spherical but only 2 DOF
    // ====================================================================
    struct UniversalPendulumFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
        mbd::Real mass{1.0};
        mbd::Real L{1.0};
    };

    UniversalPendulumFixture make_universal_pendulum(mbd::Real m = 1.0,
                                                      mbd::Real L = 1.0)
    {
        using namespace mbd;
        UniversalPendulumFixture fx;
        fx.mass = m;
        fx.L = L;

        Mat3 I_com = Mat3::Identity() * Real(1e-6);
        Vec3 com_offset_B(0.0, -L, 0.0);
        RigidBodyInertia inertia(m, com_offset_B, I_com);

        fx.body_idx = fx.sys.add_body(inertia, RigidBodyState{}, "univ_pendulum",
                                       kGroundIndex);
        fx.sys.add_joint(std::make_unique<UniversalCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body_idx));
        return fx;
    }
}

// ============================================================================
// FixedJoint tests
// ============================================================================

TEST_CASE("FixedJoint: body sits at zero offset by default",
          "[fixed_joint][kinematics]")
{
    using namespace mbd;
    auto fx = make_fixed();

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("FixedJoint: body sits at specified offset",
          "[fixed_joint][kinematics]")
{
    using namespace mbd;
    Vec3 offset(1.0, 2.0, 3.0);
    auto fx = make_fixed(offset);

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(2.0, 1e-12));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(3.0, 1e-12));
}

TEST_CASE("FixedJoint: body has zero DOFs",
          "[fixed_joint][kinematics]")
{
    using namespace mbd;
    auto fx = make_fixed();

    REQUIRE(fx.sys.total_dof == 0);
    REQUIRE(fx.sys.q.size() == 0);
    REQUIRE(fx.sys.q_dot.size() == 0);
}

TEST_CASE("FixedJoint: body does not move under gravity",
          "[fixed_joint][gravity]")
{
    using namespace mbd;
    Vec3 offset(0.5, 1.0, 0.0);
    auto fx = make_fixed(offset);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.run(2.0, 0.001);

    // Body should remain rigidly at the offset — no motion regardless of forces
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.5, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-9));

    REQUIRE_THAT(fx.sys.states[fx.body_idx].v_WB.norm(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].w_WB.norm(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// SphericalCoordJoint kinematic tests
// ============================================================================

TEST_CASE("SphericalJoint: q(0)=alpha rotates about world X",
          "[spherical][kinematics]")
{
    using namespace mbd;
    auto fx = make_spherical_pendulum();

    fx.sys.q.setZero();
    fx.sys.q(0) = 0.5; // r_x
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();

    // Rotation by 0.5 about X
    REQUIRE_THAT(R(0, 0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(R(1, 1), WithinAbs(std::cos(0.5), 1e-9));
    REQUIRE_THAT(R(2, 2), WithinAbs(std::cos(0.5), 1e-9));
    REQUIRE_THAT(R(2, 1), WithinAbs(std::sin(0.5), 1e-9));
}

TEST_CASE("SphericalJoint: q(1)=beta rotates about world Y",
          "[spherical][kinematics]")
{
    using namespace mbd;
    auto fx = make_spherical_pendulum();

    fx.sys.q.setZero();
    fx.sys.q(1) = 0.4; // r_y
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();

    REQUIRE_THAT(R(0, 0), WithinAbs(std::cos(0.4), 1e-9));
    REQUIRE_THAT(R(1, 1), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(R(2, 2), WithinAbs(std::cos(0.4), 1e-9));
}

TEST_CASE("SphericalJoint: q_dot(0:3) at zero rotation gives world angular velocity",
          "[spherical][kinematics]")
{
    using namespace mbd;
    auto fx = make_spherical_pendulum();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.0;
    fx.sys.q_dot(1) = 2.0;
    fx.sys.q_dot(2) = 3.0;
    fx.sys.compute_kinematics();

    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    REQUIRE_THAT(w.x(), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(w.y(), WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(w.z(), WithinAbs(3.0, 1e-9));
}

// ============================================================================
// SphericalCoordJoint dynamics tests
// ============================================================================

TEST_CASE("SphericalJoint: pendulum at rest hangs straight down",
          "[spherical][equilibrium]")
{
    using namespace mbd;
    auto fx = make_spherical_pendulum();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    REQUIRE_THAT(fx.sys.q.norm(), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q_dot.norm(), WithinAbs(0.0, 1e-6));
}

TEST_CASE("SphericalJoint: pendulum swing in XY plane has period 2*pi*sqrt(L/g)",
          "[spherical][pendulum]")
{
    using namespace mbd;
    const Real L = 1.0;
    auto fx = make_spherical_pendulum(1.0, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start at small displacement: rotate by 0.05 rad about Z (planar swing in XY)
    fx.sys.q.setZero();
    fx.sys.q(2) = 0.05;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real T_expected = 2.0 * pi * std::sqrt(L / g_accel);

    sim.run(T_expected, 0.0001);

    // After one period: q(2) should be back to ~0.05, q_dot(2) ~= 0
    REQUIRE_THAT(fx.sys.q(2), WithinAbs(0.05, 0.001));
    REQUIRE_THAT(fx.sys.q_dot(2), WithinAbs(0.0, 0.01));

    // No motion in other DOFs
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(fx.sys.q(1), WithinAbs(0.0, 1e-4));
}

TEST_CASE("SphericalJoint: pendulum energy conserved over multiple swings",
          "[spherical][energy]")
{
    using namespace mbd;
    const Real L = 1.0;
    const Real m = 1.0;
    auto fx = make_spherical_pendulum(m, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Moderate amplitude in XY plane
    fx.sys.q.setZero();
    fx.sys.q(2) = 0.5;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
        // For a point mass at distance L: I_about_hinge = m*L²
        // KE = 0.5 * m * |v_com|^2  (since I_com is tiny)
        const Vec3 v_com = fx.sys.states[fx.body_idx].v_WB
            + w.cross(fx.sys.states[fx.body_idx].q_WB *
                      fx.sys.inertias[fx.body_idx].com_B);
        const Real KE = 0.5 * m * v_com.squaredNorm();

        // PE: COM y-coordinate × m × g
        const Vec3 com_W = fx.sys.states[fx.body_idx].p_WB +
                           fx.sys.states[fx.body_idx].q_WB *
                           fx.sys.inertias[fx.body_idx].com_B;
        const Real PE = m * g_accel * com_W.y();
        return KE + PE;
    };

    const Real E0 = compute_energy();

    sim.run(5.0, 0.0001);

    const Real E1 = compute_energy();
    INFO("E0 = " << E0 << ", E1 = " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.01));
}

TEST_CASE("SphericalJoint: torque-free spin preserves angular velocity (symmetric body)",
          "[spherical][stability]")
{
    using namespace mbd;
    // Symmetric (point-like) inertia means no precession for any spin axis
    auto fx = make_spherical_pendulum(1.0, 0.0); // L=0 makes COM at origin

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(1) = 1.0; // spin about Y at 1 rad/s
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    // For a symmetric (~point) body, angular velocity preserved
    REQUIRE_THAT(fx.sys.q_dot(1), WithinAbs(1.0, 0.01));
    REQUIRE_THAT(fx.sys.q_dot(0), WithinAbs(0.0, 0.01));
    REQUIRE_THAT(fx.sys.q_dot(2), WithinAbs(0.0, 0.01));
}

// ============================================================================
// UniversalCoordJoint tests
// ============================================================================

TEST_CASE("UniversalJoint: q(0) rotates about Z axis",
          "[universal][kinematics]")
{
    using namespace mbd;
    auto fx = make_universal_pendulum();

    fx.sys.q.setZero();
    fx.sys.q(0) = 0.5;
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();

    // Rotation about Z by 0.5
    REQUIRE_THAT(R(2, 2), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(R(0, 0), WithinAbs(std::cos(0.5), 1e-9));
    REQUIRE_THAT(R(1, 1), WithinAbs(std::cos(0.5), 1e-9));
}

TEST_CASE("UniversalJoint: q(1) at q(0)=0 rotates about X axis",
          "[universal][kinematics]")
{
    using namespace mbd;
    auto fx = make_universal_pendulum();

    fx.sys.q.setZero();
    fx.sys.q(1) = 0.3;
    fx.sys.compute_kinematics();

    const Mat3 R = fx.sys.states[fx.body_idx].q_WB.toRotationMatrix();

    // With q(0)=0, the second axis is just world X
    REQUIRE_THAT(R(0, 0), WithinAbs(1.0, 1e-9));
    REQUIRE_THAT(R(1, 1), WithinAbs(std::cos(0.3), 1e-9));
    REQUIRE_THAT(R(2, 2), WithinAbs(std::cos(0.3), 1e-9));
}

TEST_CASE("UniversalJoint: q_dot(0) gives angular velocity about Z",
          "[universal][kinematics]")
{
    using namespace mbd;
    auto fx = make_universal_pendulum();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(0) = 1.5;
    fx.sys.compute_kinematics();

    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    REQUIRE_THAT(w.z(), WithinAbs(1.5, 1e-9));
    REQUIRE_THAT(w.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(w.y(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("UniversalJoint: q_dot(1) at q(0)=0 gives angular velocity about X",
          "[universal][kinematics]")
{
    using namespace mbd;
    auto fx = make_universal_pendulum();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(1) = 2.0;
    fx.sys.compute_kinematics();

    const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
    REQUIRE_THAT(w.x(), WithinAbs(2.0, 1e-9));
    REQUIRE_THAT(w.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(w.z(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("UniversalJoint: pendulum at rest hangs straight down",
          "[universal][equilibrium]")
{
    using namespace mbd;
    auto fx = make_universal_pendulum();

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    sim.run(2.0, 0.001);

    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(fx.sys.q(1), WithinAbs(0.0, 1e-6));
}

TEST_CASE("UniversalJoint: small swing about X has pendulum period",
          "[universal][pendulum]")
{
    using namespace mbd;
    const Real L = 1.0;
    auto fx = make_universal_pendulum(1.0, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // q(1) is the X-axis rotation (when q(0)=0). A small displacement gives
    // pendulum motion in YZ plane.
    fx.sys.q.setZero();
    fx.sys.q(1) = 0.05;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    const Real T_expected = 2.0 * pi * std::sqrt(L / g_accel);

    sim.run(T_expected, 0.0001);

    REQUIRE_THAT(fx.sys.q(1), WithinAbs(0.05, 0.001));
    REQUIRE_THAT(fx.sys.q_dot(1), WithinAbs(0.0, 0.01));
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(0.0, 1e-4));
}

TEST_CASE("UniversalJoint: energy conserved during compound swing",
          "[universal][energy]")
{
    using namespace mbd;
    const Real L = 1.0;
    const Real m = 1.0;
    auto fx = make_universal_pendulum(m, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start with both DOFs displaced
    fx.sys.q.setZero();
    fx.sys.q(0) = 0.2;
    fx.sys.q(1) = 0.3;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        const Vec3 w = fx.sys.states[fx.body_idx].w_WB;
        const Vec3 v_com = fx.sys.states[fx.body_idx].v_WB
            + w.cross(fx.sys.states[fx.body_idx].q_WB *
                      fx.sys.inertias[fx.body_idx].com_B);
        const Real KE = 0.5 * m * v_com.squaredNorm();

        const Vec3 com_W = fx.sys.states[fx.body_idx].p_WB +
                           fx.sys.states[fx.body_idx].q_WB *
                           fx.sys.inertias[fx.body_idx].com_B;
        const Real PE = m * g_accel * com_W.y();
        return KE + PE;
    };

    const Real E0 = compute_energy();
    sim.run(5.0, 0.0001);
    const Real E1 = compute_energy();

    INFO("E0 = " << E0 << ", E1 = " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.01));
}

TEST_CASE("UniversalJoint: torque about Z gives angular acceleration tau/I",
          "[universal][torque]")
{
    using namespace mbd;
    const Real m = 1.0;
    const Real L = 1.0;
    auto fx = make_universal_pendulum(m, L);

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Apply torque about the FIRST DOF (Z axis)
    const Real torque = 1.0;
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(0) += torque;
    };

    // For a point mass at distance L from hinge, rotating about world Z:
    //   I_hinge = m*L^2
    const Real I_hinge = m * L * L;
    const Real alpha_expected = torque / I_hinge;

    sim.run(0.5, 0.0001);

    const Real q_expected = 0.5 * alpha_expected * 0.5 * 0.5;
    REQUIRE_THAT(fx.sys.q(0), WithinAbs(q_expected, q_expected * 0.02));
}