#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mbd/constraint.hpp>
#include <mbd/solver.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("DistanceConstraint Jacobian is consistent with Finite Differences",
          "[constraint]")
{
    using namespace mbd;

    MultibodySystem system;

    // Body 1 at origin
    RigidBodyInertia I1 = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    RigidBodyState s1;
    s1.p_WB = Vec3(0, 0, 0);
    BodyIndex b1 = system.add_body(I1, s1);

    // Body 2 at (2,0,0) rotated 90 deg around Z
    RigidBodyInertia I2 = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    RigidBodyState s2;
    s2.p_WB = Vec3(2.0, 0.0, 0.0);
    s2.q_WB = Quat(Eigen::AngleAxisd(pi / 2.0, Vec3::UnitZ()));
    BodyIndex b2 = system.add_body(I2, s2);

    Vec3 anchor1(0.5, 0.0, 0.0);
    Vec3 anchor2(0.0, 0.5, 0.0);
    Real target_dist = 1.0;

    DistanceConstraint constraint(b1, b2, anchor1, anchor2, target_dist);

    Eigen::MatrixXd J1_ana, J2_ana;
    constraint.jacobian(system, J1_ana, J2_ana);

    const Real eps_fd = 1e-7;
    Eigen::VectorXd phi_0;
    constraint.evaluate(system, phi_0);

    // Linear X perturbation on Body 1
    {
        system.states[b1].p_WB.x() += eps_fd;
        Eigen::VectorXd phi_p;
        constraint.evaluate(system, phi_p);
        system.states[b1].p_WB.x() -= eps_fd;

        Real num_deriv = (phi_p(0) - phi_0(0)) / eps_fd;
        REQUIRE_THAT(num_deriv, WithinAbs(J1_ana(0, 0), 1e-5));
    }

    // Angular Z perturbation on Body 1
    {
        Vec3 w_small(0, 0, eps_fd);
        Quat q_orig = system.states[b1].q_WB;

        system.states[b1].q_WB = integrate_quat(q_orig, w_small, 1.0);

        Eigen::VectorXd phi_p;
        constraint.evaluate(system, phi_p);
        system.states[b1].q_WB = q_orig;

        Real num_deriv = (phi_p(0) - phi_0(0)) / eps_fd;
        REQUIRE_THAT(num_deriv, WithinAbs(J1_ana(0, 5), 1e-5));
    }

    // Linear Y perturbation on Body 2
    {
        system.states[b2].p_WB.y() += eps_fd;
        Eigen::VectorXd phi_p;
        constraint.evaluate(system, phi_p);
        system.states[b2].p_WB.y() -= eps_fd;

        Real num_deriv = (phi_p(0) - phi_0(0)) / eps_fd;
        REQUIRE_THAT(num_deriv, WithinAbs(J2_ana(0, 1), 1e-5));
    }
}

TEST_CASE("Solver maintains pendulum length under gravity", "[solver]")
{
    using namespace mbd;

    MultibodySystem system;
    Vec3 gravity(0.0, -9.81, 0.0);

    // Pendulum bob (body 1). Ground is already body 0.
    RigidBodyInertia I_bob = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    RigidBodyState s_bob;
    s_bob.p_WB = Vec3(1.0, 0.0, 0.0);
    BodyIndex b_bob = system.add_body(I_bob, s_bob);

    // Distance constraint: ground origin to bob origin, length 1.0
    system.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, b_bob, Vec3::Zero(), Vec3::Zero(), 1.0
    ));

    // TEST 1: Horizontal (gravity tangential to constraint)
    system.clear_forces();
    solve_constraints(system, gravity);

    REQUIRE_THAT(system.forces[b_bob].f_W.norm(), WithinAbs(0.0, 1e-9));

    // TEST 2: Hanging straight down
    system.states[b_bob].p_WB = Vec3(0.0, -1.0, 0.0);
    system.states[b_bob].v_WB = Vec3::Zero();

    system.clear_forces();
    solve_constraints(system, gravity);

    REQUIRE_THAT(system.forces[b_bob].f_W.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(system.forces[b_bob].f_W.y(), WithinAbs(9.81, 1e-5));
    REQUIRE_THAT(system.forces[b_bob].f_W.z(), WithinAbs(0.0, 1e-9));

    // TEST 3: Centripetal force (bob at bottom, moving at 2 m/s)
    system.states[b_bob].v_WB = Vec3(2.0, 0.0, 0.0);

    system.clear_forces();
    solve_constraints(system, gravity);

    // gravity comp (9.81) + centripetal (mv^2/r = 4.0) = 13.81 N up
    REQUIRE_THAT(system.forces[b_bob].f_W.y(), WithinAbs(13.81, 1e-5));
}

TEST_CASE("RevoluteJoint constrains motion to rotation about axis", "[constraint]")
{
    using namespace mbd;

    MultibodySystem system;
    Vec3 gravity(0.0, -9.81, 0.0);

    // Bar hinged at origin, extending along X
    BodyIndex b_bar = system.add_body(
        RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.1, 0.1)),
        RigidBodyState::at_rest(Vec3(0.5, 0.0, 0.0)));

    // Revolute: ground to bar at origin, axis Z
    system.constraints.push_back(std::make_shared<RevoluteJoint>(
        kGroundIndex, b_bar,
        Vec3::Zero(), Vec3::UnitZ(),
        Vec3(-0.5, 0.0, 0.0), Vec3::UnitZ()
    ));

    system.clear_forces();
    solve_constraints(system, gravity);

    // Reaction force Y positive (holding bar up)
    REQUIRE(system.forces[b_bar].f_W.y() > 0.0);
    // Reaction torque Z negative (pin force at -0.5 from COM creates -Z torque)
    REQUIRE(system.forces[b_bar].tau_W.z() < 0.0);

    // Constraint satisfied
    Eigen::VectorXd phi;
    system.constraints[0]->evaluate(system, phi);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Solver stabilizes position error (Baumgarte)", "[solver]")
{
    using namespace mbd;

    MultibodySystem system;

    // Body at x=1.1 (VIOLATION: target is 1.0)
    RigidBodyState s2 = RigidBodyState::at_rest(Vec3(1.1, 0.0, 0.0));
    BodyIndex b_bob = system.add_body(
        RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), s2);

    system.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, b_bob, Vec3::Zero(), Vec3::Zero(), 1.0
    ));

    SolverConfig config;
    config.alpha = 0.0;
    config.beta  = 10.0;

    system.clear_forces();
    solve_constraints(system, Vec3::Zero(), config);

    // Force pulls bob left: -10.0 N (beta^2 * phi = 100 * 0.1 = 10)
    REQUIRE(system.forces[b_bob].f_W.x() < -9.0);
    REQUIRE_THAT(system.forces[b_bob].f_W.x(), WithinAbs(-10.0, 1e-5));
}

TEST_CASE("MultibodySystem ground body exists at index 0", "[system]")
{
    using namespace mbd;

    MultibodySystem system;

    REQUIRE(system.body_count() == 1);
    REQUIRE(system.is_ground(kGroundIndex));

    REQUIRE(system.states[0].p_WB.isApprox(Vec3::Zero()));
    REQUIRE_THAT(system.states[0].q_WB.angularDistance(Quat::Identity()),
                 WithinAbs(0.0, 1e-12));

    BodyIndex b1 = system.add_body(
        RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)));
    REQUIRE(b1 == 1);
    REQUIRE(system.body_count() == 2);
    REQUIRE_FALSE(system.is_ground(b1));
}