#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/spatial/spatial.hpp"
#include "mbd/vehicle/vehicle.hpp"

// The simple 5-body vehicle, on the kernel (plan task 2.7). The chassis pose
// is read from its world state: position, and rotation as a rotation vector.

using Catch::Matchers::WithinAbs;

namespace
{
    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }

    const mbd::RigidBodyState& body(const mbd::kernel::Simulator& sim, mbd::BodyIndex i)
    {
        return sim.states()[static_cast<std::size_t>(i)];
    }

    /// Chassis rotation as a rotation vector (roll about X, yaw about Y in
    /// this Y-up layer, pitch about Z for small angles).
    mbd::Vec3 chassis_rotation(const mbd::kernel::Simulator& sim, const mbd::VehicleModel& vm)
    {
        return mbd::log3(body(sim, vm.chassis_body).q_WB);
    }
}

// ============================================================================
// Static equilibrium
// ============================================================================

TEST_CASE("Vehicle: settles to static equilibrium from perturbed state",
          "[vehicle][static]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    // Start near equilibrium with a 3cm perturbation on chassis height
    set_vehicle_equilibrium(sim, vm);
    sim.q(vm.chassis_ty_idx()) += 0.03;
    sim.initialize();

    // Simulate with damping for 5 seconds
    sim.run(5.0, 0.001);

    // Chassis should settle to equilibrium height
    const RigidBodyState& chassis = body(sim, vm.chassis_body);
    REQUIRE_THAT(chassis.p_WB.y(), WithinAbs(params.chassis_height_eq(), 0.003));

    // Chassis horizontal position unchanged (no lateral/longitudinal forces)
    REQUIRE_THAT(chassis.p_WB.x(), WithinAbs(0.0, 0.001));
    REQUIRE_THAT(chassis.p_WB.z(), WithinAbs(0.0, 0.001));

    // No rotations
    require_vec3_near(chassis_rotation(sim, vm), Vec3::Zero(), 0.001);

    // All four suspension travels should be equal and at equilibrium
    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(sim.q(vm.susp_q_idx(static_cast<Corner>(c))), WithinAbs(params.q_susp_eq(), 0.003));
    }

    // All velocities near zero
    for (int i = 0; i < sys.model.nv; ++i) {
        REQUIRE_THAT(sim.v(i), WithinAbs(0.0, 0.02));
    }
}

// ============================================================================
// Analytical equilibrium check (no simulation, just force balance)
// ============================================================================

TEST_CASE("Vehicle: equilibrium initial conditions produce near-zero acceleration",
          "[vehicle][equilibrium]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vm);

    // Forward dynamics with every force applied: accelerations near zero
    const VecX a = sim.acceleration(sim.q, sim.v, sim.time);

    for (int i = 0; i < sys.model.nv; ++i) {
        REQUIRE_THAT(a(i), WithinAbs(0.0, 0.5));
    }
}

// ============================================================================
// Wheel positions at equilibrium
// ============================================================================

TEST_CASE("Vehicle: wheel world positions are correct at equilibrium",
          "[vehicle][geometry]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vm);

    const Real y_wheel = params.wheel_center_height_eq();
    const Real a  = params.front_axle_x;
    const Real b  = params.rear_axle_x;
    const Real ht = params.half_track;

    // FL, FR, RL, RR
    require_vec3_near(body(sim, vm.wheel_bodies[0]).p_WB, Vec3(a, y_wheel, ht), 0.001);
    require_vec3_near(body(sim, vm.wheel_bodies[1]).p_WB, Vec3(a, y_wheel, -ht), 0.001);
    require_vec3_near(body(sim, vm.wheel_bodies[2]).p_WB, Vec3(-b, y_wheel, ht), 0.001);
    require_vec3_near(body(sim, vm.wheel_bodies[3]).p_WB, Vec3(-b, y_wheel, -ht), 0.001);

    // Chassis CG
    REQUIRE_THAT(body(sim, vm.chassis_body).p_WB.y(), WithinAbs(params.chassis_height_eq(), 0.001));
}

// ============================================================================
// Tire loads at equilibrium
// ============================================================================

TEST_CASE("Vehicle: tire vertical loads are correct at equilibrium",
          "[vehicle][loads]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vm);
    sim.acceleration(sim.q, sim.v, sim.time);   // applies the force elements

    const Real Fz_expected = params.weight_per_wheel();

    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(vm.tires[c]->get_vertical_force(), WithinAbs(Fz_expected, 5.0));
    }
}

// ============================================================================
// Undamped energy conservation (bounce)
// ============================================================================

TEST_CASE("Vehicle: undamped bounce conserves energy",
          "[vehicle][energy]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    params.c_susp = 0.0;  // No suspension damping
    params.tire_c_z = 0.0; // No tire damping
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.q(vm.chassis_ty_idx()) += 0.01; // 10mm chassis bounce
    sim.initialize();

    auto compute_energy = [&]() -> Real {
        // Kinetic and gravitational energy of the bodies.
        const Real KE = kernel::kinetic_energy(sys.model, sim.data());
        const Real PE_grav = kernel::potential_energy(sys.model, sim.data());

        // Spring PE: computed from actual world distances
        Real PE_spring = 0.0;
        for (int c = 0; c < 4; ++c) {
            const Vec3 p_mount = body(sim, vm.chassis_body).pose_WB().apply(
                Vec3(c < 2 ? params.front_axle_x : -params.rear_axle_x,
                     0.0,
                     (c % 2 == 0) ? params.half_track : -params.half_track));
            const Vec3 p_wheel = body(sim, vm.wheel_bodies[static_cast<std::size_t>(c)]).p_WB;
            const Real dist = (p_wheel - p_mount).norm();
            PE_spring += 0.5 * params.k_susp *
                (dist - params.susp_rest_length) * (dist - params.susp_rest_length);
        }

        // Tire PE (deflections from the last force evaluation, at this state)
        sim.acceleration(sim.q, sim.v, sim.time);
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

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.q(vm.chassis_ty_idx()) += 0.02; // Pure heave perturbation
    sim.initialize();

    sim.run(0.5, 0.001);

    // All four suspension travels should be identical (symmetric excitation)
    const Real q_FL = sim.q(vm.susp_q_idx(Corner::FL));
    REQUIRE_THAT(sim.q(vm.susp_q_idx(Corner::FR)), WithinAbs(q_FL, 1e-6));
    REQUIRE_THAT(sim.q(vm.susp_q_idx(Corner::RL)), WithinAbs(q_FL, 1e-6));
    REQUIRE_THAT(sim.q(vm.susp_q_idx(Corner::RR)), WithinAbs(q_FL, 1e-6));

    // No chassis yaw, roll, or lateral motion
    const Vec3 rotation = chassis_rotation(sim, vm);
    REQUIRE_THAT(body(sim, vm.chassis_body).p_WB.z(), WithinAbs(0.0, 1e-6));   // lateral
    REQUIRE_THAT(rotation.x(), WithinAbs(0.0, 1e-6));                          // roll
    REQUIRE_THAT(rotation.y(), WithinAbs(0.0, 1e-6));                          // yaw
}

// ============================================================================
// Straight-line driving via applied force
// ============================================================================

TEST_CASE("Vehicle: forward force accelerates vehicle in X",
          "[vehicle][driving]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vm);
    sim.initialize();

    // A constant forward force on the chassis, along its own X axis: the
    // generalized force of the chassis's linear velocity in X.
    // This simulates a simplified drive force (not through tires yet).
    const Real F_drive = 5000.0; // 5 kN forward
    const int v_forward = sys.model.idx_v[vm.chassis_body] + 3;
    sim.force_callback = [&](kernel::Simulator& /*s*/, Real /*t*/, VecX& tau) {
        tau(v_forward) += F_drive;
    };

    sim.run(1.0, 0.001);

    // Expected: a ≈ F / m_total = 5000 / 1560 ≈ 3.205 m/s^2
    // After 1s: v ≈ 3.205 m/s, x ≈ 1.603 m
    const Real a_expected = F_drive / params.total_mass();
    const Real v_expected = a_expected * 1.0;
    const Real x_expected = 0.5 * a_expected * 1.0 * 1.0;

    // Allow some tolerance (tires/suspension couple vertical and horizontal)
    const RigidBodyState& chassis = body(sim, vm.chassis_body);
    REQUIRE_THAT(chassis.v_WB.x(), WithinAbs(v_expected, v_expected * 0.05));
    REQUIRE_THAT(chassis.p_WB.x(), WithinAbs(x_expected, x_expected * 0.05));

    // Lateral motion should be zero
    REQUIRE_THAT(chassis.p_WB.z(), WithinAbs(0.0, 0.01));

    // Vehicle should still be near ground (not flying)
    REQUIRE_THAT(chassis.p_WB.y(), WithinAbs(params.chassis_height_eq(), 0.02));
}

// ============================================================================
// Total DOF count
// ============================================================================

TEST_CASE("Vehicle: correct DOF count and body count",
          "[vehicle][topology]")
{
    using namespace mbd;

    kernel::System sys;
    VehicleParams params;
    auto vm = build_simple_vehicle(sys, params);

    // 6 bodies: ground + chassis + 4 wheels
    REQUIRE(sys.model.nbodies() == 6);

    // 10 DOF: 6 (chassis) + 4 (suspension); 11 coordinates with the quaternion
    REQUIRE(sys.model.nv == 10);
    REQUIRE(sys.model.nq == 11);

    // 5 joints, one per body but ground
    REQUIRE(sys.model.nbodies() - 1 == 5);

    // 8 force elements: 4 springs + 4 tires
    REQUIRE(sys.force_elements.size() == 8);

    // Body indices
    REQUIRE(vm.chassis_body == 1);
    REQUIRE(vm.wheel_bodies[0] == 2); // FL
    REQUIRE(vm.wheel_bodies[1] == 3); // FR
    REQUIRE(vm.wheel_bodies[2] == 4); // RL
    REQUIRE(vm.wheel_bodies[3] == 5); // RR

    // Suspension coordinates follow the chassis's seven
    REQUIRE(vm.susp_q_idx(Corner::FL) == 7);
    REQUIRE(vm.susp_v_idx(Corner::FL) == 6);
}
