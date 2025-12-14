#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mbd/constraint.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("DistanceConstraint Jacobian is consistent with Finite Differences", "[constraint]")
{
    using namespace mbd;

    MultibodySystem system;

    // Body 1: At origin
    RigidBodyInertia I1 = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    RigidBodyState s1; 
    s1.p_WB = Vec3(0,0,0);
    BodyIndex b1 = system.add_body(I1, s1);

    // Body 2: At (2,0,0) rotated 90 deg around Z
    RigidBodyInertia I2 = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    RigidBodyState s2;
    s2.p_WB = Vec3(2.0, 0.0, 0.0);
    s2.q_WB = Quat(Eigen::AngleAxisd(mbd::pi / 2.0, Vec3::UnitZ()));
    BodyIndex b2 = system.add_body(I2, s2);

    // Constraint: Link local(0.5, 0, 0) on B1 to local(0, 0.5, 0) on B2
    Vec3 anchor1(0.5, 0.0, 0.0);
    Vec3 anchor2(0.0, 0.5, 0.0);
    Real target_dist = 1.0; // Arbitrary, doesn't affect Jacobian deriv
    
    DistanceConstraint constraint(b1, b2, anchor1, anchor2, target_dist);

    // Analytical Jacobian
    Eigen::MatrixXd J1_ana, J2_ana;
    constraint.jacobian(system, J1_ana, J2_ana);

    // Finite Difference Check
    const Real eps = 1e-7;
    
    // Check J1 (Derivative w.r.t Body 1 state changes)
    // We simulate small velocity v * dt
    Eigen::VectorXd phi_0;
    constraint.evaluate(system, phi_0);

    // 1. Linear X perturbation on Body 1
    {
        system.states[b1].p_WB.x() += eps; 
        Eigen::VectorXd phi_p;
        constraint.evaluate(system, phi_p);
        system.states[b1].p_WB.x() -= eps; // reset

        Real num_deriv = (phi_p(0) - phi_0(0)) / eps;
        REQUIRE_THAT(num_deriv, WithinAbs(J1_ana(0, 0), 1e-5));
    }

    // 2. Angular Z perturbation on Body 1
    {
        // Apply small rotation: q_new = q_old * exp(w * dt/2) ...
        // Small rotation approx: w = [0,0,1], angle = eps
        Vec3 w_small(0,0, eps);
        Quat dq = delta_rotation_from_omega(w_small, 1.0); 
        Quat q_orig = system.states[b1].q_WB;
        
        system.states[b1].q_WB = integrate_quat(q_orig, w_small, 1.0);
        
        Eigen::VectorXd phi_p;
        constraint.evaluate(system, phi_p);
        system.states[b1].q_WB = q_orig; // reset

        Real num_deriv = (phi_p(0) - phi_0(0)) / eps;
        // J1 column 5 is Z-angular
        REQUIRE_THAT(num_deriv, WithinAbs(J1_ana(0, 5), 1e-5));
    }

    // Check J2 (Body 2)
    // 3. Linear Y perturbation on Body 2
    {
        system.states[b2].p_WB.y() += eps;
        Eigen::VectorXd phi_p;
        constraint.evaluate(system, phi_p);
        system.states[b2].p_WB.y() -= eps;

        Real num_deriv = (phi_p(0) - phi_0(0)) / eps;
        REQUIRE_THAT(num_deriv, WithinAbs(J2_ana(0, 1), 1e-5));
    }
    
}

#include "mbd/solver.hpp" // Include the new solver

TEST_CASE("Solver maintains pendulum length under gravity", "[solver]")
{
    using namespace mbd;

    MultibodySystem system;
    Vec3 gravity(0.0, -9.81, 0.0);

    // Body 1: "Ground"
    // We use a very large mass to approximate a fixed body.
    // Increased to 1e12 to minimize recoil error in the constraint solver.
    Real m_ground = 1e12;
    RigidBodyInertia I_ground = RigidBodyInertia::from_solid_box(m_ground, Vec3(1,1,1));
    RigidBodyState s_ground;
    BodyIndex b_ground = system.add_body(I_ground, s_ground); // At 0,0,0

    // Body 2: Pendulum bob
    RigidBodyInertia I_bob = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
    RigidBodyState s_bob;
    s_bob.p_WB = Vec3(1.0, 0.0, 0.0); // 1m to the right
    BodyIndex b_bob = system.add_body(I_bob, s_bob);

    // Constraint: Distance 1.0m
    system.constraints.push_back(std::make_shared<DistanceConstraint>(
        b_ground, b_bob, Vec3::Zero(), Vec3::Zero(), 1.0
    ));

    // Helper to reset and solve
    auto reset_and_solve = [&]() {
        system.clear_forces();
        // FIX: Apply counter-gravity force to ground so it doesn't fall
        system.forces[b_ground].f_W -= gravity * m_ground;
        
        solve_constraints(system, gravity);
    };

    // TEST 1: Horizontal (Static release)
    reset_and_solve();
    
    // Gravity is tangential to the constraint (vertical vs horizontal). 
    // Tension should be zero.
    REQUIRE_THAT(system.forces[b_bob].f_W.norm(), WithinAbs(0.0, 1e-9));

    // TEST 2: Pendulum hanging DOWN (Static equilibrium)
    // Bob at (0, -1, 0)
    system.states[b_bob].p_WB = Vec3(0.0, -1.0, 0.0);
    
    reset_and_solve();
    
    // Analysis:
    // Bob wants to fall (-9.81). Ground stays put.
    // Constraint must pull Bob UP with 9.81 N.
    REQUIRE_THAT(system.forces[b_bob].f_W.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(system.forces[b_bob].f_W.y(), WithinAbs(9.81, 1e-5));
    REQUIRE_THAT(system.forces[b_bob].f_W.z(), WithinAbs(0.0, 1e-9));
    
    // TEST 3: Centripetal Force
    // Bob at (0, -1, 0), moving at X=2 m/s
    system.states[b_bob].v_WB = Vec3(2.0, 0.0, 0.0);
    
    reset_and_solve();
    
    // Analysis:
    // Tension = Gravity compensation (9.81) + Centripetal (mv^2/r)
    // Centripetal = 1.0 * 2^2 / 1.0 = 4.0 N
    // Total = 13.81 N UP.
    REQUIRE_THAT(system.forces[b_bob].f_W.y(), WithinAbs(13.81, 1e-5));
}