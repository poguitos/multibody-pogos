#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/simulator.hpp"
#include "mbd/constraint.hpp"

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
}

// ============================================================================
// Free body + DistanceConstraint = pendulum
// ============================================================================

TEST_CASE("Constrained dynamics: free body + distance constraint behaves as pendulum",
          "[constrained][pendulum]")
{
    using namespace mbd;

    // --- Constrained system: body on FreeJoint + DistanceConstraint ---
    MultibodySystem sys_c;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    sys_c.add_body(inertia, RigidBodyState{}, "bob", kGroundIndex);

    // FreeJoint: 6 DOF floating body
    sys_c.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    // Distance constraint: ground origin to body origin, length 1.0
    sys_c.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, 1, Vec3::Zero(), Vec3::Zero(), 1.0));

    Simulator sim_c(sys_c);
    sim_c.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim_c.method = IntegrationMethod::RK4;
    sim_c.constraint_alpha = 10.0;
    sim_c.constraint_beta  = 10.0;
    sim_c.initialize();

    // Start at (1, 0, 0), released from rest
    sys_c.q << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys_c.q_dot.setZero();
    sys_c.compute_kinematics();

    // --- Reference: revolute pendulum (tree-based) ---
    MultibodySystem sys_r;
    sys_r.add_body(inertia, RigidBodyState{}, "bob", kGroundIndex);

    // Revolute at origin, axis Z, body at 1m along X
    // For a point mass (COM = body origin), X_CJ = identity
    sys_r.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    Simulator sim_r(sys_r);
    sim_r.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim_r.method = IntegrationMethod::RK4;
    sim_r.initialize();

    // Start horizontal: q = 0 means body at (0, 0, 0)... we need it at (1, 0, 0).
    // For a revolute about Z with X_CJ = identity, body origin = joint origin.
    // We need X_CJ to offset the body. But for a point mass at distance 1,
    // let's just use the tree-based approach differently.
    //
    // Actually: revolute joint at origin with X_CJ = T(-1, 0, 0) places
    // the body at (1, 0, 0) when q=0. Let's rebuild.

    MultibodySystem sys_ref;
    sys_ref.add_body(inertia, RigidBodyState{}, "bob_ref", kGroundIndex);
    sys_ref.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-1.0, 0.0, 0.0)),
        kGroundIndex, 1));

    Simulator sim_ref(sys_ref);
    sim_ref.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim_ref.method = IntegrationMethod::RK4;
    sim_ref.initialize();

    sys_ref.q(0) = 0.0;  // horizontal, body at (1, 0, 0)
    sys_ref.q_dot(0) = 0.0;
    sys_ref.compute_kinematics();

    // Simulate both for 1 second
    const Real dt = 0.001;
    const Real T  = 1.0;
    sim_c.run(T, dt);
    sim_ref.run(T, dt);

    // The constrained body should track the revolute pendulum's position.
    // The motion is in the XY plane for both.
    const Vec3 p_constrained = sys_c.states[1].p_WB;
    const Vec3 p_reference   = sys_ref.states[1].p_WB;

    // They should agree to ~1% (constraint drift limits exact match)
    REQUIRE_THAT(p_constrained.x(), WithinAbs(p_reference.x(), 0.02));
    REQUIRE_THAT(p_constrained.y(), WithinAbs(p_reference.y(), 0.02));
}

// ============================================================================
// Constraint violation stays bounded
// ============================================================================

TEST_CASE("Constrained dynamics: distance constraint violation stays small",
          "[constrained][violation]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    sys.add_body(inertia, RigidBodyState{}, "bob", kGroundIndex);

    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    const Real target_length = 1.0;
    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, 1, Vec3::Zero(), Vec3::Zero(), target_length));

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.constraint_alpha = 10.0;
    sim.constraint_beta  = 10.0;
    sim.set_recording(true);

    // Set initial conditions BEFORE initialize() so the first recorded
    // snapshot has the correct state.
    sys.q << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();

    sim.initialize();

    sim.run(3.0, 0.001);

    // Check constraint violation at end
    Eigen::VectorXd phi;
    sys.constraints[0]->evaluate(sys, phi);
    REQUIRE_THAT(phi(0), WithinAbs(0.0, 0.01));

    // Check violation throughout the simulation by sampling history
    Real max_violation = 0.0;
    for (const auto& rec : sim.history) {
        sys.q     = rec.q;
        sys.q_dot = rec.q_dot;
        sys.compute_kinematics();

        Eigen::VectorXd phi_i;
        sys.constraints[0]->evaluate(sys, phi_i);
        max_violation = std::max(max_violation, std::abs(phi_i(0)));
    }

    // With Baumgarte alpha=10, beta=10, dt=0.001, violation should stay < 1mm
    REQUIRE(max_violation < 0.001);
}

// ============================================================================
// Energy conservation with constraints
// ============================================================================

TEST_CASE("Constrained dynamics: energy is approximately conserved",
          "[constrained][energy]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    sys.add_body(inertia, RigidBodyState{}, "bob", kGroundIndex);

    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, 1, Vec3::Zero(), Vec3::Zero(), 1.0));

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.constraint_alpha = 10.0;
    sim.constraint_beta  = 10.0;
    sim.initialize();

    // Start at 45 degrees in XY plane
    const Real angle = pi / 4.0;
    sys.q << std::cos(angle), std::sin(angle), 0.0, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        sys.compute_kinematics();
        const MatX M = compute_mass_matrix(sys);
        const Real KE = 0.5 * sys.q_dot.transpose() * M * sys.q_dot;
        const Real PE = inertia.mass * g_accel * sys.states[1].p_WB.y();
        return KE + PE;
    };

    const Real E0 = compute_energy();

    sim.run(2.0, 0.001);

    const Real E_final = compute_energy();

    // Baumgarte stabilization adds slight damping, so allow ~1% energy drift
    const Real rel_error = std::abs(E_final - E0) / std::abs(E0);
    REQUIRE(rel_error < 0.01);
}

// ============================================================================
// Constrained forward dynamics matches unconstrained when no constraints
// ============================================================================

TEST_CASE("Constrained dynamics: no constraints falls through to unconstrained",
          "[constrained][fallthrough]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.05, 0.05));
    sys.add_body(inertia, RigidBodyState{}, "link", kGroundIndex);

    sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        Transform3::Identity(),
        Transform3::FromTranslation(Vec3(-0.5, 0.0, 0.0)),
        kGroundIndex, 1));

    // No constraints added

    sys.q(0) = 0.3;
    sys.q_dot(0) = 1.5;
    sys.compute_kinematics();

    const Vec3 grav(0.0, -g_accel, 0.0);
    VecX tau = VecX::Zero(1);

    VecX qdd_unconstrained = forward_dynamics(sys, tau, grav);
    VecX qdd_constrained   = constrained_forward_dynamics(sys, tau, grav);

    REQUIRE_THAT(qdd_constrained(0), WithinAbs(qdd_unconstrained(0), 1e-12));
}

// ============================================================================
// Two-body chain: free body + second body with distance constraint = double pendulum
// ============================================================================

TEST_CASE("Constrained dynamics: two free bodies with distance constraints",
          "[constrained][double]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));

    // Body 1: free floating
    sys.add_body(inertia, RigidBodyState{}, "bob1", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, 1));

    // Body 2: free floating
    sys.add_body(inertia, RigidBodyState{}, "bob2", 1);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        1, 2));

    // Distance constraint: ground to body1, length 1.0
    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, 1, Vec3::Zero(), Vec3::Zero(), 1.0));

    // Distance constraint: body1 to body2, length 1.0
    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        1, 2, Vec3::Zero(), Vec3::Zero(), 1.0));

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.constraint_alpha = 10.0;
    sim.constraint_beta  = 10.0;
    sim.initialize();

    // Body1 at (1, 0, 0), body2 at (2, 0, 0)
    sys.q << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
             2.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    // Simulate
    sim.run(2.0, 0.001);

    // Check both distance constraints
    Eigen::VectorXd phi1, phi2;
    sys.constraints[0]->evaluate(sys, phi1);
    sys.constraints[1]->evaluate(sys, phi2);

    REQUIRE_THAT(phi1(0), WithinAbs(0.0, 0.005));
    REQUIRE_THAT(phi2(0), WithinAbs(0.0, 0.005));

    // System should have moved (not stuck)
    REQUIRE(std::abs(sys.states[1].p_WB.y()) > 0.01);
    REQUIRE(std::abs(sys.states[2].p_WB.y()) > 0.01);
}

// ============================================================================
// Prismatic slider constrained by distance = oscillation
// ============================================================================

TEST_CASE("Constrained dynamics: body Jacobian consistency with mass matrix",
          "[constrained][jacobian]")
{
    using namespace mbd;

    // Build a double pendulum and verify body Jacobian is consistent
    // with the velocity FK.
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

    sys.q << 0.5, -0.8;
    sys.q_dot << 1.2, -0.6;
    sys.compute_kinematics();

    // Body Jacobian at origin should satisfy: v_WB = J_v * q_dot, w_WB = J_omega * q_dot
    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        auto bj = compute_body_jacobian_origin(sys, i);

        Vec3 v_from_jac = bj.J_v * sys.q_dot;
        Vec3 w_from_jac = bj.J_omega * sys.q_dot;

        // Compare with FK results
        require_vec3_near(v_from_jac, sys.states[i].v_WB, 1e-10);
        require_vec3_near(w_from_jac, sys.states[i].w_WB, 1e-10);
    }
}

TEST_CASE("Constrained dynamics: body accelerations are consistent with FD",
          "[constrained][accelerations]")
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

    sys.q << 0.3, -0.7;
    sys.q_dot << 2.0, -1.5;
    sys.compute_kinematics();

    // Compute forward dynamics
    const Vec3 grav(0.0, -g_accel, 0.0);
    VecX q_ddot = forward_dynamics(sys, VecX::Zero(2), grav);

    // Compute body accelerations at this q_ddot
    auto acc = compute_body_accelerations(sys, q_ddot);

    // Verify using finite difference on body velocity:
    // a ≈ (v(t+eps) - v(t-eps)) / (2*eps)
    const Real dt_fd = 1e-6;

    for (BodyIndex i = 1; i < sys.body_count(); ++i) {
        auto bj = compute_body_jacobian_origin(sys, i);

        // v(t+dt) ≈ v(t) + (J_v * q_ddot + ...) * dt  — too complex for FD.
        // Instead verify: a = J_v * q_ddot + J_v_dot * q_dot
        // We already know J_v * q_dot matches v_WB (from Jacobian test).
        // So verify: body_acc = J_v * q_ddot + (a - J_v * q_ddot)
        // where the remainder is the Coriolis/centripetal acceleration.
        //
        // A simpler check: integrate q for a tiny step and compare.

        VecX q_plus  = sys.q + sys.q_dot * dt_fd + 0.5 * q_ddot * dt_fd * dt_fd;
        VecX qd_plus = sys.q_dot + q_ddot * dt_fd;

        VecX q_save  = sys.q;
        VecX qd_save = sys.q_dot;

        sys.q     = q_plus;
        sys.q_dot = qd_plus;
        sys.compute_kinematics();
        Vec3 v_plus = sys.states[i].v_WB;

        VecX q_minus  = q_save - qd_save * dt_fd + 0.5 * q_ddot * dt_fd * dt_fd;
        VecX qd_minus = qd_save - q_ddot * dt_fd;

        sys.q     = q_minus;
        sys.q_dot = qd_minus;
        sys.compute_kinematics();
        Vec3 v_minus = sys.states[i].v_WB;

        Vec3 a_fd = (v_plus - v_minus) / (2.0 * dt_fd);

        // Restore
        sys.q     = q_save;
        sys.q_dot = qd_save;
        sys.compute_kinematics();

        require_vec3_near(acc[i].a, a_fd, 1e-4);
    }
}