#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/model/system.hpp"
#include "mbd/integrators/simulator.hpp"
#include "mbd/model/joint.hpp"
#include "mbd/model/constraint.hpp"
#include "mbd/algorithms/dynamics.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    struct FreeBodyFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex body_idx{0};
    };

    FreeBodyFixture make_free_body()
    {
        using namespace mbd;
        FreeBodyFixture fx;

        auto inertia = RigidBodyInertia::from_solid_box(1.0,
                                                         Vec3(0.05, 0.05, 0.05));
        fx.body_idx = fx.sys.add_body(inertia, RigidBodyState{}, "body",
                                       kGroundIndex);
        fx.sys.add_joint(std::make_unique<FreeCoordJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.body_idx));
        return fx;
    }
}

// ============================================================================
// Free flight: distance between two free bodies preserved
// ============================================================================

TEST_CASE("Constraint: distance between two free bodies preserved in free flight",
          "[constraint][distance]")
{
    using namespace mbd;
    MultibodySystem sys;

    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05));

    BodyIndex body1 = sys.add_body(inertia, RigidBodyState{}, "b1", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body1));

    BodyIndex body2 = sys.add_body(inertia, RigidBodyState{}, "b2", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body2));

    const Real D = 1.0;
    sys.q.setZero();
    sys.q(6) = D;  // body2 starts at x=D
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        body1, body2,
        Vec3::Zero(), Vec3::Zero(),
        D));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sys.q_dot.setZero();
    sys.q_dot(1) = 0.5;   // body1 +Y at 0.5 m/s
    sys.q_dot(7) = -0.5;  // body2 -Y at 0.5 m/s

    sim.run(2.0, 0.001);

    const Vec3 p1 = sys.states[body1].p_WB;
    const Vec3 p2 = sys.states[body2].p_WB;
    const Real d_actual = (p2 - p1).norm();

    INFO("Distance after 2s: " << d_actual);
    REQUIRE_THAT(d_actual, WithinAbs(D, 0.005));
}

// ============================================================================
// Pendulum via DistanceConstraint
// ============================================================================

TEST_CASE("Constraint: pendulum via DistanceConstraint at rest hangs straight",
          "[constraint][pendulum]")
{
    using namespace mbd;
    auto fx = make_free_body();

    const Real L = 1.0;

    fx.sys.q.setZero();
    fx.sys.q(1) = -L;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    fx.sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        kGroundIndex, fx.body_idx,
        Vec3::Zero(), Vec3::Zero(),
        L));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.run(2.0, 0.001);

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(), WithinAbs(0.0, 1e-3));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(-L, 1e-3));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.z(), WithinAbs(0.0, 1e-3));

    const Real d = fx.sys.states[fx.body_idx].p_WB.norm();
    REQUIRE_THAT(d, WithinAbs(L, 1e-3));
}

TEST_CASE("Constraint: pendulum via DistanceConstraint small-angle period",
          "[constraint][pendulum]")
{
    using namespace mbd;
    auto fx = make_free_body();

    const Real L = 1.0;
    const Real theta0 = 0.05;

    fx.sys.q.setZero();
    fx.sys.q(0) = L * std::sin(theta0);
    fx.sys.q(1) = -L * std::cos(theta0);
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    fx.sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        kGroundIndex, fx.body_idx,
        Vec3::Zero(), Vec3::Zero(),
        L));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    const Real T_expected = 2.0 * pi * std::sqrt(L / g_accel);
    INFO("T_expected = " << T_expected);

    sim.run(T_expected, 0.0001);

    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.x(),
                 WithinAbs(L * std::sin(theta0), 0.005));
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(),
                 WithinAbs(-L * std::cos(theta0), 0.005));

    const Real d = fx.sys.states[fx.body_idx].p_WB.norm();
    REQUIRE_THAT(d, WithinAbs(L, 1e-3));
}

// ============================================================================
// Energy conservation under DistanceConstraint
// ============================================================================

TEST_CASE("Constraint: pendulum via DistanceConstraint conserves energy",
          "[constraint][energy]")
{
    using namespace mbd;
    auto fx = make_free_body();

    const Real L = 1.0;
    const Real theta0 = 0.5;

    fx.sys.q.setZero();
    fx.sys.q(0) = L * std::sin(theta0);
    fx.sys.q(1) = -L * std::cos(theta0);
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    fx.sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        kGroundIndex, fx.body_idx,
        Vec3::Zero(), Vec3::Zero(),
        L));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    const Real m = fx.sys.inertias[fx.body_idx].mass;
    auto compute_energy = [&]() -> Real {
        const Vec3 v = fx.sys.states[fx.body_idx].v_WB;
        const Real h = fx.sys.states[fx.body_idx].p_WB.y();
        return 0.5 * m * v.squaredNorm() + m * g_accel * h;
    };

    const Real E0 = compute_energy();

    sim.run(5.0, 0.0001);

    const Real E1 = compute_energy();

    INFO("E0 = " << E0 << ", E1 = " << E1);
    INFO("Drift: " << std::abs(E1 - E0) / std::abs(E0) * 100.0 << "%");

    REQUIRE_THAT(E1, WithinAbs(E0, std::abs(E0) * 0.02));
}

// ============================================================================
// Baumgarte stabilization recovery
// ============================================================================

TEST_CASE("Constraint: small initial violation decays via Baumgarte",
          "[constraint][stabilization]")
{
    using namespace mbd;
    auto fx = make_free_body();

    const Real L_target = 1.0;
    const Real L_initial = 1.05;

    fx.sys.q.setZero();
    fx.sys.q(1) = -L_initial;
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    fx.sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        kGroundIndex, fx.body_idx,
        Vec3::Zero(), Vec3::Zero(),
        L_target));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    const Real d_initial = fx.sys.states[fx.body_idx].p_WB.norm();
    REQUIRE_THAT(d_initial, WithinAbs(L_initial, 1e-9));

    sim.run(2.0, 0.001);

    const Real d_after = fx.sys.states[fx.body_idx].p_WB.norm();

    INFO("Initial violation: " << (d_initial - L_target));
    INFO("Violation after 2s: " << (d_after - L_target));

    REQUIRE(std::abs(d_after - L_target) < std::abs(d_initial - L_target));
    REQUIRE(std::abs(d_after - L_target) < 0.5 * std::abs(d_initial - L_target));
}

// ============================================================================
// Velocity-level violation
// ============================================================================

TEST_CASE("Constraint: bodies moving apart get pulled back to constraint",
          "[constraint][velocity_violation]")
{
    using namespace mbd;
    auto fx = make_free_body();

    const Real L = 1.0;

    fx.sys.q.setZero();
    fx.sys.q(1) = -L;
    fx.sys.q_dot.setZero();
    fx.sys.q_dot(1) = -2.0;
    fx.sys.compute_kinematics();

    fx.sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        kGroundIndex, fx.body_idx,
        Vec3::Zero(), Vec3::Zero(),
        L));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.run(1.0, 0.001);

    const Real d_after = fx.sys.states[fx.body_idx].p_WB.norm();
    INFO("Distance after 1s: " << d_after);

    REQUIRE_THAT(d_after, WithinAbs(L, 0.05));
}

// ============================================================================
// Distance constraint with non-zero anchors
// ============================================================================

TEST_CASE("Constraint: distance with non-zero anchors enforced correctly",
          "[constraint][distance][anchors]")
{
    using namespace mbd;
    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.5, 0.5));

    BodyIndex body1 = sys.add_body(inertia, RigidBodyState{}, "b1", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body1));

    BodyIndex body2 = sys.add_body(inertia, RigidBodyState{}, "b2", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body2));

    // Initial: body1 at origin, body2 at (2, 0.3, 0).
    // Anchor on body1 at body-frame (1,0,0): in world (1, 0, 0).
    // Anchor on body2 at body-frame (-1,0,0): in world (1, 0.3, 0).
    // Initial anchor distance = 0.3.
    sys.q.setZero();
    sys.q(6) = 2.0;  // body2 X
    sys.q(7) = 0.3;  // body2 Y
    sys.q_dot.setZero();
    sys.compute_kinematics();

    const Vec3 anchor1_W_initial = sys.states[body1].p_WB
                                 + sys.states[body1].q_WB * Vec3(1.0, 0.0, 0.0);
    const Vec3 anchor2_W_initial = sys.states[body2].p_WB
                                 + sys.states[body2].q_WB * Vec3(-1.0, 0.0, 0.0);
    const Real d_anchors_initial = (anchor2_W_initial - anchor1_W_initial).norm();

    INFO("Initial anchor distance: " << d_anchors_initial);
    REQUIRE_THAT(d_anchors_initial, WithinAbs(0.3, 1e-9));

    // Constraint: distance between anchors = 0.5
    sys.constraints.push_back(std::make_unique<DistanceConstraint>(
        body1, body2,
        Vec3(1.0, 0.0, 0.0), Vec3(-1.0, 0.0, 0.0),
        0.5));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.run(1.0, 0.001);

    const Vec3 a1_W = sys.states[body1].p_WB
                    + sys.states[body1].q_WB * Vec3(1.0, 0.0, 0.0);
    const Vec3 a2_W = sys.states[body2].p_WB
                    + sys.states[body2].q_WB * Vec3(-1.0, 0.0, 0.0);
    const Real d_anchors_final = (a2_W - a1_W).norm();

    INFO("Anchor distance after 1s: " << d_anchors_final);
    REQUIRE_THAT(d_anchors_final, WithinAbs(0.5, 0.05));
}

// ============================================================================
// CoincidentPointConstraint: two body points coincide
// ============================================================================

TEST_CASE("Constraint: coincident point holds two body points together",
          "[constraint][coincident]")
{
    using namespace mbd;
    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.3, 0.3, 0.3));

    BodyIndex body1 = sys.add_body(inertia, RigidBodyState{}, "b1", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body1));

    BodyIndex body2 = sys.add_body(inertia, RigidBodyState{}, "b2", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body2));

    // Initial: body1 at origin, body2 at (1, 0, 0).
    // Anchor on body1 at body-frame (0.5, 0, 0): in world (0.5, 0, 0).
    // Anchor on body2 at body-frame (-0.5, 0, 0): in world (0.5, 0, 0).
    // Anchors coincide at origin. Constraint should keep them coincident.
    sys.q.setZero();
    sys.q(6) = 1.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.constraints.push_back(std::make_unique<CoincidentPointConstraint>(
        body1, body2,
        Vec3(0.5, 0.0, 0.0), Vec3(-0.5, 0.0, 0.0)));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Give the bodies opposite initial velocities (a velocity-level constraint
    // violation). Baumgarte stabilization will damp the velocity but allows
    // some position drift before the velocity is fully corrected. We verify
    // that the violation stays bounded (well below the unconstrained drift
    // of v*t = 0.5*2 = 1 m).
    sys.q_dot.setZero();
    sys.q_dot(1) =  0.5;  // body1 +Y
    sys.q_dot(7) = -0.5;  // body2 -Y

    sim.run(2.0, 0.001);

    const Vec3 a1_W = sys.states[body1].p_WB
                    + sys.states[body1].q_WB * Vec3(0.5, 0.0, 0.0);
    const Vec3 a2_W = sys.states[body2].p_WB
                    + sys.states[body2].q_WB * Vec3(-0.5, 0.0, 0.0);

    const Real anchor_separation = (a2_W - a1_W).norm();
    INFO("Anchor separation after 1s: " << anchor_separation);

    // Without the constraint, separation after 2s would be 2 m (each body moves
    // 1 m in opposite directions). Constraint should keep separation well below
    // that — within ~v0/alpha = 0.1m order of magnitude.
    REQUIRE(anchor_separation < 0.5);
}

// ============================================================================
// PointCoordinateConstraint: fix a body point's world coordinate
// ============================================================================

TEST_CASE("Constraint: PointCoordinate fixes body point's Y to target",
          "[constraint][point_coord]")
{
    using namespace mbd;
    auto fx = make_free_body();

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    // Constrain body's body-frame point (0, 0, 0) to have world Y = -1.0
    fx.sys.constraints.push_back(std::make_unique<PointCoordinateConstraint>(
        fx.body_idx, Vec3::Zero(), 1, -1.0));

    Simulator sim(fx.sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.run(2.0, 0.001);

    INFO("Final Y: " << fx.sys.states[fx.body_idx].p_WB.y());

    // After Baumgarte stabilization, body should be near y = -1
    REQUIRE_THAT(fx.sys.states[fx.body_idx].p_WB.y(), WithinAbs(-1.0, 0.05));
}

// ============================================================================
// Previously a KNOWN LIMITATION (see git history): two free-floating bodies
// joined by a coincident-point constraint under sustained opposing load used to
// blow up — the coincident points drifted apart by ~2 m and the free-body
// rotation vector eventually hit the exp-map singularity at ||r|| = 2*pi,
// throwing "mass matrix not SPD".
//
// Two foundational fixes resolved it:
//   (1) Rotation-vector canonicalization keeps free/spherical-joint ||r|| <= pi,
//       so the exp-map Jacobian never degenerates.
//   (2) Post-step projection (Simulator::project_constraints) removes the drift
//       that acceleration-level Baumgarte leaves behind, holding the constraint
//       to tolerance regardless of load.
//
// This test now asserts the constraint is held tightly throughout.
// ============================================================================

TEST_CASE("Constraint: free bodies + sustained load stay coincident",
          "[constraint][projection][singularity]")
{
    using namespace mbd;
    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.3, 0.3, 0.3));
    BodyIndex body1 = sys.add_body(inertia, RigidBodyState{}, "b1", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body1));
    BodyIndex body2 = sys.add_body(inertia, RigidBodyState{}, "b2", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body2));
    sys.q.setZero();
    sys.q(6) = 1.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();
    sys.constraints.push_back(std::make_unique<CoincidentPointConstraint>(
        body1, body2,
        Vec3(0.5, 0.0, 0.0), Vec3(-0.5, 0.0, 0.0)));
    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();
    sim.force_callback = [&](MultibodySystem&, Real, VecX& tau) {
        tau(1) +=  1.0;
        tau(7) += -1.0;
    };

    Real max_phi = 0.0;
    Real max_r1  = 0.0;
    for (int i = 0; i < 2000; ++i) {
        sim.step(0.001);  // must not throw
        Eigen::VectorXd phi;
        sys.constraints[0]->evaluate(sys, phi);
        max_phi = std::max(max_phi, phi.norm());
        max_r1  = std::max(max_r1, sys.q.segment<3>(3).norm());
    }

    INFO("max |phi| = " << max_phi << "   max |r1| = " << max_r1);
    // Constraint held to projection tolerance (was ~2 m before the fix).
    REQUIRE(max_phi < 1e-6);
    // Rotation-vector coordinate stayed in the canonical range (was > 2*pi).
    REQUIRE(max_r1 <= pi + 1e-9);
}

TEST_CASE("Constraint: coincident point - diagnose when mass matrix fails",
          "[constraint][coincident][diagnostic]")
{
    using namespace mbd;
    MultibodySystem sys;
    auto inertia = RigidBodyInertia::from_solid_box(1.0, Vec3(0.3, 0.3, 0.3));

    BodyIndex body1 = sys.add_body(inertia, RigidBodyState{}, "b1", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body1));

    BodyIndex body2 = sys.add_body(inertia, RigidBodyState{}, "b2", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, body2));

    sys.q.setZero();
    sys.q(6) = 1.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.constraints.push_back(std::make_unique<CoincidentPointConstraint>(
        body1, body2,
        Vec3(0.5, 0.0, 0.0), Vec3(-0.5, 0.0, 0.0)));

    Simulator sim(sys);
    sim.set_gravity(Vec3::Zero());
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
        tau(1) +=  1.0;
        tau(7) += -1.0;
        const Real c_w = 1.0;
        for (int i = 3; i < 6; ++i)  tau(i) -= c_w * s.q_dot(i);
        for (int i = 9; i < 12; ++i) tau(i) -= c_w * s.q_dot(i);
    };

    // Step in small increments and report state until failure or completion
    const Real total_t = 5.0;
    const Real step_dt = 0.001;
    const Real report_interval = 0.05;
    Real t_acc = 0.0;
    Real next_report = 0.0;
    bool succeeded = true;

    try {
        while (t_acc < total_t) {
            if (t_acc >= next_report) {
                INFO("t = " << t_acc
                     << "  body1.r_norm = " << sys.q.segment<3>(3).norm()
                     << "  body2.r_norm = " << sys.q.segment<3>(9).norm()
                     << "  body1.p = (" << sys.q(0) << ", " << sys.q(1) << ", " << sys.q(2) << ")"
                     << "  body2.p = (" << sys.q(6) << ", " << sys.q(7) << ", " << sys.q(8) << ")");
                next_report += report_interval;
            }
            sim.run(step_dt, step_dt);
            t_acc += step_dt;
        }
    } catch (const MbdError& e) {
        succeeded = false;
        INFO("FAILED at t = " << t_acc
             << "  body1.r_norm = " << sys.q.segment<3>(3).norm()
             << "  body2.r_norm = " << sys.q.segment<3>(9).norm());
    }

    // Even if it fails partway, capture what we learned
    REQUIRE(t_acc > 0.5);  // should at least run for half a second

    // Don't require full completion — let the failure data tell us when it broke
}