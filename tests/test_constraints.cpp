#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mbd/model/constraint.hpp>
#include <mbd/model/system.hpp>

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

// The tests of the old body-level constraint-force solver (solvers/solver.hpp:
// pendulum and hinge reactions, Baumgarte) moved to the kernel in plan task
// 2.7, as analytic cases: tests/kernel/test_kernel_analytic.cpp.

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