#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle.hpp"
#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }
}

// ============================================================================
// Static equilibrium
// ============================================================================

TEST_CASE("Vehicle: settles to static equilibrium from perturbed state",
          "[vehicle][static]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start near equilibrium with a 3cm perturbation on chassis height
    set_vehicle_equilibrium(sys, vm);
    sys.q(1) += 0.03;
    sys.compute_kinematics();

    // Simulate with damping for 5 seconds
    sim.run(5.0, 0.001);

    // Chassis should settle to equilibrium height
    REQUIRE_THAT(sys.q(1), WithinAbs(params.chassis_height_eq(), 0.003));

    // Chassis horizontal position unchanged (no lateral/longitudinal forces)
    REQUIRE_THAT(sys.q(0), WithinAbs(0.0, 0.001));
    REQUIRE_THAT(sys.q(2), WithinAbs(0.0, 0.001));

    // No rotations
    REQUIRE_THAT(sys.q(3), WithinAbs(0.0, 0.001));
    REQUIRE_THAT(sys.q(4), WithinAbs(0.0, 0.001));
    REQUIRE_THAT(sys.q(5), WithinAbs(0.0, 0.001));

    // All four suspension travels should be equal and at equilibrium
    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(sys.q(6 + c), WithinAbs(params.q_susp_eq(), 0.003));
    }

    // All velocities near zero
    for (int i = 0; i < sys.total_dof; ++i) {
        REQUIRE_THAT(sys.q_dot(i), WithinAbs(0.0, 0.02));
    }
}

// ============================================================================
// Analytical equilibrium check (no simulation, just force balance)
// ============================================================================

TEST_CASE("Vehicle: equilibrium initial conditions produce near-zero acceleration",
          "[vehicle][equilibrium]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    set_vehicle_equilibrium(sys, vm);

    // Compute forward dynamics — accelerations should be near zero
    sys.clear_forces();
    sys.apply_force_elements();
    VecX tau = project_body_forces_to_joint_space(sys);
    VecX q_ddot = forward_dynamics(sys, tau, Vec3(0.0, -g_accel, 0.0));

    for (int i = 0; i < sys.total_dof; ++i) {
        REQUIRE_THAT(q_ddot(i), WithinAbs(0.0, 0.5));
    }
}

// ============================================================================
// Wheel positions at equilibrium
// ============================================================================

TEST_CASE("Vehicle: wheel world positions are correct at equilibrium",
          "[vehicle][geometry]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    set_vehicle_equilibrium(sys, vm);

    const Real y_wheel = params.wheel_center_height_eq();
    const Real a  = params.front_axle_x;
    const Real b  = params.rear_axle_x;
    const Real ht = params.half_track;

    // FL
    require_vec3_near(sys.states[vm.wheel_bodies[0]].p_WB,
                      Vec3(a, y_wheel, ht), 0.001);
    // FR
    require_vec3_near(sys.states[vm.wheel_bodies[1]].p_WB,
                      Vec3(a, y_wheel, -ht), 0.001);
    // RL
    require_vec3_near(sys.states[vm.wheel_bodies[2]].p_WB,
                      Vec3(-b, y_wheel, ht), 0.001);
    // RR
    require_vec3_near(sys.states[vm.wheel_bodies[3]].p_WB,
                      Vec3(-b, y_wheel, -ht), 0.001);

    // Chassis CG
    REQUIRE_THAT(sys.states[vm.chassis_body].p_WB.y(),
                 WithinAbs(params.chassis_height_eq(), 0.001));
}

// ============================================================================
// Tire loads at equilibrium
// ============================================================================

TEST_CASE("Vehicle: tire vertical loads are correct at equilibrium",
          "[vehicle][loads]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    set_vehicle_equilibrium(sys, vm);

    sys.clear_forces();
    sys.apply_force_elements();

    const Real Fz_expected = params.weight_per_wheel();

    for (int c = 0; c < 4; ++c) {
        Real Fz = vm.tires[c]->get_vertical_force();
        REQUIRE_THAT(Fz, WithinAbs(Fz_expected, 5.0));
    }
}

// ============================================================================
// Undamped energy conservation (bounce)
// ============================================================================

TEST_CASE("Vehicle: undamped bounce conserves energy",
          "[vehicle][energy]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    params.c_susp = 0.0;  // No suspension damping
    params.tire_c_z = 0.0; // No tire damping
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    sys.q(1) += 0.01; // 10mm chassis bounce
    sys.compute_kinematics();

    auto compute_energy = [&]() -> Real {
        sys.compute_kinematics();
        const MatX M = compute_mass_matrix(sys);
        const Real KE = 0.5 * sys.q_dot.transpose() * M * sys.q_dot;

        Real PE_grav = 0.0;
        for (BodyIndex i = 1; i < sys.body_count(); ++i) {
            const auto& st = sys.states[i];
            const auto& in = sys.inertias[i];
            const Mat3 R = st.q_WB.toRotationMatrix();
            const Vec3 com_W = st.p_WB + R * in.com_B;
            PE_grav += in.mass * g_accel * com_W.y();
        }

        // Spring PE: computed from actual world distances
        Real PE_spring = 0.0;
        for (int c = 0; c < 4; ++c) {
            const Vec3 p_mount = sys.states[vm.chassis_body].pose_WB().apply(
                Vec3(c < 2 ? params.front_axle_x : -params.rear_axle_x,
                     0.0,
                     (c % 2 == 0) ? params.half_track : -params.half_track));
            const Vec3 p_wheel = sys.states[vm.wheel_bodies[c]].p_WB;
            const Real dist = (p_wheel - p_mount).norm();
            PE_spring += 0.5 * params.k_susp *
                (dist - params.susp_rest_length) * (dist - params.susp_rest_length);
        }

        // Tire PE
        Real PE_tire = 0.0;
        for (int c = 0; c < 4; ++c) {
            const Real defl = vm.tires[c]->get_deflection();
            PE_tire += 0.5 * params.tire_k_z * defl * defl;
        }

        return KE + PE_grav + PE_spring + PE_tire;
    };

    const Real E0 = compute_energy();

    sim.run(1.0, 0.0005);

    const Real E_final = compute_energy();
    const Real rel_error = std::abs(E_final - E0) / std::abs(E0);

    REQUIRE(rel_error < 5e-3);
}

// ============================================================================
// Symmetry: all four corners behave identically for vertical bounce
// ============================================================================

TEST_CASE("Vehicle: symmetric bounce keeps all corners equal",
          "[vehicle][symmetry]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);
    sys.q(1) += 0.02; // Pure heave perturbation
    sys.compute_kinematics();

    sim.run(0.5, 0.001);

    // All four suspension travels should be identical (symmetric excitation)
    const Real q_FL = sys.q(6);
    REQUIRE_THAT(sys.q(7), WithinAbs(q_FL, 1e-6)); // FR
    REQUIRE_THAT(sys.q(8), WithinAbs(q_FL, 1e-6)); // RL
    REQUIRE_THAT(sys.q(9), WithinAbs(q_FL, 1e-6)); // RR

    // No chassis yaw, roll, or lateral motion
    REQUIRE_THAT(sys.q(2), WithinAbs(0.0, 1e-6)); // tz
    REQUIRE_THAT(sys.q(3), WithinAbs(0.0, 1e-6)); // rx (roll)
    REQUIRE_THAT(sys.q(5), WithinAbs(0.0, 1e-6)); // rz (yaw)
}

// ============================================================================
// Straight-line driving via applied force
// ============================================================================

TEST_CASE("Vehicle: forward force accelerates vehicle in X",
          "[vehicle][driving]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vm);

    // Apply a constant forward force on the chassis via the force callback.
    // This simulates a simplified drive force (not through tires yet).
    const Real F_drive = 5000.0; // 5 kN forward
    sim.force_callback = [&](MultibodySystem& s, Real /*t*/, VecX& tau) {
        // Chassis tx is q(0). The generalized force for translation in the
        // FreeCoordJoint frame is just the force projected onto the joint axes.
        // At small rotations, joint X ≈ world X, so tau(0) ≈ Fx.
        tau(0) += F_drive;
    };

    sim.run(1.0, 0.001);

    // Expected: a ≈ F / m_total = 5000 / 1560 ≈ 3.205 m/s^2
    // After 1s: v ≈ 3.205 m/s, x ≈ 1.603 m
    const Real a_expected = F_drive / params.total_mass();
    const Real v_expected = a_expected * 1.0;
    const Real x_expected = 0.5 * a_expected * 1.0 * 1.0;

    // Allow some tolerance (tires/suspension couple vertical and horizontal)
    REQUIRE_THAT(sys.q_dot(0), WithinAbs(v_expected, v_expected * 0.05));
    REQUIRE_THAT(sys.q(0), WithinAbs(x_expected, x_expected * 0.05));

    // Lateral motion should be zero
    REQUIRE_THAT(sys.q(2), WithinAbs(0.0, 0.01));

    // Vehicle should still be near ground (not flying)
    REQUIRE_THAT(sys.q(1), WithinAbs(params.chassis_height_eq(), 0.02));
}

// ============================================================================
// Total DOF count
// ============================================================================

TEST_CASE("Vehicle: correct DOF count and body count",
          "[vehicle][topology]")
{
    using namespace mbd;

    MultibodySystem sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    // 6 bodies: ground + chassis + 4 wheels
    REQUIRE(sys.body_count() == 6);

    // 10 DOF: 6 (chassis) + 4 (suspension)
    REQUIRE(sys.total_dof == 10);

    // 5 joints
    REQUIRE(sys.joint_count() == 5);

    // 8 force elements: 4 springs + 4 tires
    REQUIRE(sys.force_elements.size() == 8);

    // Body indices
    REQUIRE(vm.chassis_body == 1);
    REQUIRE(vm.wheel_bodies[0] == 2); // FL
    REQUIRE(vm.wheel_bodies[1] == 3); // FR
    REQUIRE(vm.wheel_bodies[2] == 4); // RL
    REQUIRE(vm.wheel_bodies[3] == 5); // RR
}