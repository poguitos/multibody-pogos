
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>

#include "mbd/algorithms.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-9;

    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }

    /// Build a single-link pendulum system.
    /// Link: 1m long (half-extents 0.5), mass 1 kg, pivot at origin, axis Z.
    /// Joint frame at parent origin, child joint frame at left end (-0.5, 0, 0).
    mbd::MultibodySystem make_single_pendulum()
    {
        using namespace mbd;

        MultibodySystem sys;

        auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
        sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);

        sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::Identity(),
            Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
            kGroundIndex, 1));

        return sys;
    }

    /// Build a double pendulum (two identical links).
    mbd::MultibodySystem make_double_pendulum()
    {
        using namespace mbd;

        MultibodySystem sys;

        auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
        sys.add_body(inertia, RigidBodyState{}, "link1", kGroundIndex);
        sys.add_body(inertia, RigidBodyState{}, "link2", 1);

        sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::Identity(),
            Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
            kGroundIndex, 1));

        sys.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0)),
            Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
            1, 2));

        return sys;
    }
}

// ============================================================================
// Mass matrix tests
// ============================================================================

TEST_CASE("Mass matrix: single pendulum at q=0", "[algorithms][mass_matrix]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    sys.q(0) = 0.0;
    sys.compute_forward_kinematics();

    MatX M = compute_mass_matrix(sys);

    REQUIRE(M.rows() == 1);
    REQUIRE(M.cols() == 1);

    // M = I_zz_com + m * (L/2)^2
    //   = (1/3)(0.25 + 0.0025) + 1 * 0.25
    //   = 0.084167 + 0.25 = 0.334167
    const Real I_zz = (1.0 / 3.0) * (0.25 + 0.0025);
    const Real M_expected = I_zz + 1.0 * 0.25;

    REQUIRE_THAT(M(0, 0), WithinAbs(M_expected, 1e-10));
}

TEST_CASE("Mass matrix: single pendulum is independent of q (symmetric body)",
          "[algorithms][mass_matrix]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();

    // At q = 0
    sys.q(0) = 0.0;
    sys.compute_forward_kinematics();
    const Real M_at_0 = compute_mass_matrix(sys)(0, 0);

    // At q = pi/3
    sys.q(0) = pi / 3.0;
    sys.compute_forward_kinematics();
    const Real M_at_pi3 = compute_mass_matrix(sys)(0, 0);

    // At q = -1.7
    sys.q(0) = -1.7;
    sys.compute_forward_kinematics();
    const Real M_at_neg = compute_mass_matrix(sys)(0, 0);

    // For a single revolute about a fixed axis, M is constant
    REQUIRE_THAT(M_at_pi3, WithinAbs(M_at_0, 1e-10));
    REQUIRE_THAT(M_at_neg, WithinAbs(M_at_0, 1e-10));
}

TEST_CASE("Mass matrix: double pendulum is 2x2, symmetric, positive definite",
          "[algorithms][mass_matrix]")
{
    using namespace mbd;

    auto sys = make_double_pendulum();
    sys.q << 0.3, -0.5;
    sys.compute_forward_kinematics();

    MatX M = compute_mass_matrix(sys);

    REQUIRE(M.rows() == 2);
    REQUIRE(M.cols() == 2);

    // Symmetric
    REQUIRE_THAT(M(0, 1), WithinAbs(M(1, 0), 1e-12));

    // Positive definite (both eigenvalues > 0)
    Eigen::SelfAdjointEigenSolver<MatX> es(M);
    REQUIRE(es.eigenvalues()(0) > 1e-6);
    REQUIRE(es.eigenvalues()(1) > 1e-6);

    // M(1,1) should equal single-link inertia about its pivot (link2 alone)
    const Real I_zz = (1.0 / 3.0) * (0.25 + 0.0025);
    const Real M22_expected = I_zz + 1.0 * 0.25;
    REQUIRE_THAT(M(1, 1), WithinAbs(M22_expected, 1e-10));
}

TEST_CASE("Mass matrix: double pendulum M(0,0) at q=0",
          "[algorithms][mass_matrix]")
{
    using namespace mbd;

    auto sys = make_double_pendulum();
    sys.q << 0.0, 0.0;
    sys.compute_forward_kinematics();

    MatX M = compute_mass_matrix(sys);

    // M(0,0) = inertia of entire system about joint 1 (at origin, axis Z)
    // Link 1: I_zz_com + m1 * d1^2 = I_zz + 0.25
    // Link 2: I_zz_com + m2 * d2^2 = I_zz + (1.5)^2 = I_zz + 2.25
    // M(0,0) = (I_zz + 0.25) + (I_zz + 2.25)
    const Real I_zz = (1.0 / 3.0) * (0.25 + 0.0025);
    const Real M00_expected = (I_zz + 0.25) + (I_zz + 2.25);

    REQUIRE_THAT(M(0, 0), WithinAbs(M00_expected, 1e-10));
}

// ============================================================================
// Inverse dynamics (RNEA) tests
// ============================================================================

TEST_CASE("RNEA: single pendulum gravity torque at q=0",
          "[algorithms][rnea]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    // Horizontal (q=0): COM at (0.5, 0, 0)
    sys.q(0) = 0.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    VecX tau = inverse_dynamics(sys, VecX::Zero(1), gravity);

    // To hold the pendulum horizontal with zero acceleration:
    // Gravity torque about pivot = m * g * L/2 = 1 * 9.81 * 0.5 = 4.905
    // RNEA returns the torque needed, so tau = +4.905 (counterclockwise)
    REQUIRE_THAT(tau(0), WithinAbs(1.0 * g_accel * 0.5, 1e-9));
}

TEST_CASE("RNEA: single pendulum gravity torque at q=pi/2 (pointing up)",
          "[algorithms][rnea]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    // Pointing up: COM at (0, 0.5, 0). Gravity line through pivot => zero torque.
    sys.q(0) = pi / 2.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    VecX tau = inverse_dynamics(sys, VecX::Zero(1), gravity);

    REQUIRE_THAT(tau(0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("RNEA: single pendulum gravity torque at q=-pi/2 (hanging down)",
          "[algorithms][rnea]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    // Hanging straight down: COM at (0, -0.5, 0). Zero torque (stable eq).
    sys.q(0) = -pi / 2.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    VecX tau = inverse_dynamics(sys, VecX::Zero(1), gravity);

    REQUIRE_THAT(tau(0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("RNEA: tau = M * q_ddot for zero gravity, zero velocity",
          "[algorithms][rnea]")
{
    using namespace mbd;

    auto sys = make_double_pendulum();
    const Vec3 gravity = Vec3::Zero();

    sys.q << 0.3, -0.7;
    sys.q_dot << 0.0, 0.0;
    sys.compute_kinematics();

    MatX M = compute_mass_matrix(sys);

    // With zero gravity and zero velocity, RNEA(q_ddot) should equal M * q_ddot
    VecX q_ddot(2);
    q_ddot << 1.5, -2.3;

    VecX tau_rnea = inverse_dynamics(sys, q_ddot, gravity);
    VecX tau_M    = M * q_ddot;

    REQUIRE_THAT(tau_rnea(0), WithinAbs(tau_M(0), 1e-9));
    REQUIRE_THAT(tau_rnea(1), WithinAbs(tau_M(1), 1e-9));
}

TEST_CASE("RNEA: M * q_ddot + h = tau identity for arbitrary motion",
          "[algorithms][rnea]")
{
    using namespace mbd;

    auto sys = make_double_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    sys.q << 0.5, -0.8;
    sys.q_dot << 1.2, -0.6;
    sys.compute_kinematics();

    MatX M = compute_mass_matrix(sys);

    VecX q_ddot(2);
    q_ddot << 3.0, -1.5;

    // RNEA should give tau = M * q_ddot + h(q, q_dot)
    VecX tau_full = inverse_dynamics(sys, q_ddot, gravity);
    VecX h        = inverse_dynamics(sys, VecX::Zero(2), gravity);

    VecX tau_check = M * q_ddot + h;

    REQUIRE_THAT(tau_full(0), WithinAbs(tau_check(0), 1e-8));
    REQUIRE_THAT(tau_full(1), WithinAbs(tau_check(1), 1e-8));
}

// ============================================================================
// Forward dynamics tests
// ============================================================================

TEST_CASE("Forward dynamics: single pendulum free fall from horizontal",
          "[algorithms][forward_dynamics]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    sys.q(0) = 0.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    VecX q_ddot = forward_dynamics(sys, VecX::Zero(1), gravity);

    // Analytical: q_ddot = -m * g * (L/2) / (I_zz + m * (L/2)^2)
    const Real I_zz = (1.0 / 3.0) * (0.25 + 0.0025);
    const Real M_val = I_zz + 0.25;
    const Real q_ddot_expected = -(g_accel * 0.5) / M_val;

    REQUIRE_THAT(q_ddot(0), WithinAbs(q_ddot_expected, 1e-8));
    REQUIRE(q_ddot(0) < 0.0); // Should accelerate clockwise (gravity pulls COM down)
}

TEST_CASE("Forward dynamics: pendulum at q=-pi/2 (hanging) has zero acceleration",
          "[algorithms][forward_dynamics]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    sys.q(0) = -pi / 2.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    VecX q_ddot = forward_dynamics(sys, VecX::Zero(1), gravity);

    REQUIRE_THAT(q_ddot(0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Forward dynamics: applied torque exactly cancels gravity",
          "[algorithms][forward_dynamics]")
{
    using namespace mbd;

    auto sys = make_single_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    sys.q(0) = 0.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    // The bias torque (gravity) is m * g * L/2 = 4.905
    VecX tau_hold(1);
    tau_hold(0) = 0.0;

    // First get the bias
    VecX h = inverse_dynamics(sys, VecX::Zero(1), gravity);

    // Apply exactly that torque
    VecX q_ddot = forward_dynamics(sys, h, gravity);

    REQUIRE_THAT(q_ddot(0), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Forward dynamics: free fall in zero gravity gives zero acceleration",
          "[algorithms][forward_dynamics]")
{
    using namespace mbd;

    auto sys = make_double_pendulum();
    const Vec3 gravity = Vec3::Zero();

    sys.q << 0.3, -0.8;
    sys.q_dot << 0.0, 0.0;
    sys.compute_kinematics();

    VecX q_ddot = forward_dynamics(sys, VecX::Zero(2), gravity);

    REQUIRE_THAT(q_ddot(0), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(q_ddot(1), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Forward dynamics: inverse of inverse dynamics",
          "[algorithms][forward_dynamics]")
{
    using namespace mbd;

    auto sys = make_double_pendulum();
    const Vec3 gravity(0.0, -g_accel, 0.0);

    sys.q << 0.5, -0.8;
    sys.q_dot << 1.2, -0.6;
    sys.compute_kinematics();

    // Pick an arbitrary q_ddot
    VecX q_ddot_in(2);
    q_ddot_in << 3.0, -1.5;

    // Compute tau from inverse dynamics
    VecX tau = inverse_dynamics(sys, q_ddot_in, gravity);

    // Recover q_ddot from forward dynamics
    VecX q_ddot_out = forward_dynamics(sys, tau, gravity);

    REQUIRE_THAT(q_ddot_out(0), WithinAbs(q_ddot_in(0), 1e-8));
    REQUIRE_THAT(q_ddot_out(1), WithinAbs(q_ddot_in(1), 1e-8));
}

TEST_CASE("Forward dynamics: prismatic joint free fall under gravity",
          "[algorithms][forward_dynamics]")
{
    using namespace mbd;

    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(2.0, Vec3(0.1, 0.1, 0.1));
    sys.add_body(inertia, RigidBodyState{}, "slider", kGroundIndex);

    // Prismatic along Z (default joint axis) — but we want it along Y
    // Rotate joint frame: Rx(-pi/2) maps Z -> +Y
    Mat3 R = Eigen::AngleAxisd(-pi / 2.0, Vec3::UnitX()).toRotationMatrix();
    Transform3 X_J_frame = Transform3::FromRotation(R);

    sys.add_joint(std::make_unique<PrismaticCoordJoint>(
        X_J_frame, X_J_frame, kGroundIndex, 1));

    const Vec3 gravity(0.0, -g_accel, 0.0);
    sys.q(0) = 0.0;
    sys.q_dot(0) = 0.0;
    sys.compute_kinematics();

    VecX q_ddot = forward_dynamics(sys, VecX::Zero(1), gravity);

    // Mass on a vertical rail: q_ddot = -g (sliding down under gravity)
    REQUIRE_THAT(q_ddot(0), WithinAbs(-g_accel, 1e-8));
}