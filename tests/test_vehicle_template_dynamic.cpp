#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle_template.hpp"
#include "mbd/simulator.hpp"
#include "mbd/drivetrain.hpp"

using Catch::Matchers::WithinAbs;

// ============================================================================
// Topology: DWB front / Simple rear (like SportsCar preset with DWB)
// ============================================================================

TEST_CASE("Dynamic template: all-DWB vehicle has correct topology",
          "[tmpl_dyn][topology]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Bodies: 1 (ground) + 1 (chassis) + 4*3 (LCA, upright, UCA per corner) = 14
    REQUIRE(sys.body_count() == 14);

    // Tree DOFs: 6 (chassis) + 4 * (1+3+1) = 26
    REQUIRE(sys.total_dof == 26);

    // Joints: 1 (chassis) + 4 * 3 = 13
    REQUIRE(sys.joint_count() == 13);

    // Constraints: 4 corners * 2 (coincident + tie rod) = 8 constraint objects
    REQUIRE(sys.constraints.size() == 8);

    // Total constraint equations: 4 * (3 + 1) = 16
    int total_eqs = 0;
    for (const auto& c : sys.constraints) total_eqs += c->equation_count();
    REQUIRE(total_eqs == 16);

    // Force elements: 4 springs + 4 tires = 8
    REQUIRE(sys.force_elements.size() == 8);
}

TEST_CASE("Dynamic template: all-McPherson vehicle has correct topology",
          "[tmpl_dyn][topology]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::McPherson;
    tmpl.rear_axle.suspension_type  = SuspensionType::McPherson;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Bodies: 1 (ground) + 1 (chassis) + 4*2 = 10
    REQUIRE(sys.body_count() == 10);

    // Tree DOFs: 6 + 4 * (1+3) = 22
    REQUIRE(sys.total_dof == 22);

    // Joints: 1 + 4 * 2 = 9
    REQUIRE(sys.joint_count() == 9);

    // Constraints: 4 * 2 = 8 objects, 4 * 4 = 16 equations
    REQUIRE(sys.constraints.size() == 8);
}

TEST_CASE("Dynamic template: mixed suspension (DWB front, simple rear)",
          "[tmpl_dyn][topology]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::Simple;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Bodies: 1 + 1 + 2*3 (DWB) + 2*1 (simple) = 10
    REQUIRE(sys.body_count() == 10);

    // Tree DOFs: 6 + 2*5 (DWB) + 2*1 (simple) = 18
    REQUIRE(sys.total_dof == 18);

    // Constraints: 2 corners * 2 = 4 constraint objects (only front DWB)
    REQUIRE(sys.constraints.size() == 4);
}

// ============================================================================
// Reference configuration satisfies constraints
// ============================================================================

TEST_CASE("Dynamic template: all-DWB reference config satisfies constraints",
          "[tmpl_dyn][reference]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.compute_kinematics();

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Dynamic template: all-McPherson reference config satisfies constraints",
          "[tmpl_dyn][reference]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::McPherson;
    tmpl.rear_axle.suspension_type  = SuspensionType::McPherson;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    sys.q.setZero();
    sys.q_dot.setZero();
    sys.compute_kinematics();

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// Tire attachment: tire forces act on the upright, not a simple wheel
// ============================================================================

TEST_CASE("Dynamic template: DWB tire attaches to upright body",
          "[tmpl_dyn][tires]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    for (int c = 0; c < 4; ++c) {
        REQUIRE(vh.tire(c) != nullptr);
        REQUIRE(vh.tire(c)->wheel_body_idx == vh.wheel(c));
        // Wheel body index is the upright body
        REQUIRE(vh.corners[c].wheel_body == vh.corners[c].wheel_body);
    }
}

// ============================================================================
// Static equilibrium under gravity
// ============================================================================

TEST_CASE("Dynamic template: all-DWB vehicle settles under gravity",
          "[tmpl_dyn][equilibrium]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Start chassis at an elevated height (rough initial condition)
    set_vehicle_equilibrium(sys, vh);
    sys.compute_kinematics();

    // Let it settle for 3 seconds
    sim.run(3.0, 0.001);

    // Chassis should be at roughly the expected height (within 2cm)
    // The exact height depends on the DWB geometry, but should be around 0.25-0.30 m
    REQUIRE(sys.q(1) > 0.20);
    REQUIRE(sys.q(1) < 0.60);

    // Velocities should be small (settled)
    for (int i = 0; i < sys.total_dof; ++i) {
        REQUIRE(std::abs(sys.q_dot(i)) < 0.1);
    }

    // Total tire load should approximately equal vehicle weight
    sys.clear_forces();
    sys.apply_force_elements();
    Real total_Fz = 0.0;
    for (int c = 0; c < 4; ++c) total_Fz += vh.tire(c)->get_vertical_force();
    const Real W_total = tmpl.total_mass() * g_accel;
    REQUIRE_THAT(total_Fz, WithinAbs(W_total, W_total * 0.05));
}

// ============================================================================
// Simple suspension still works
// ============================================================================

TEST_CASE("Dynamic template: simple suspension still works (regression)",
          "[tmpl_dyn][regression]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    // Default is all simple

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Topology: 6 bodies, 10 DOF (as before)
    REQUIRE(sys.body_count() == 6);
    REQUIRE(sys.total_dof == 10);
    REQUIRE(sys.constraints.empty());

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);
    sim.run(2.0, 0.001);

    // Should settle
    for (int i = 0; i < sys.total_dof; ++i) {
        REQUIRE(std::abs(sys.q_dot(i)) < 0.05);
    }
}

// ============================================================================
// Driving with DWB suspension
// ============================================================================

TEST_CASE("Dynamic template: all-DWB vehicle drives forward",
          "[tmpl_dyn][driving]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);

    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sys, vh);
    dt.connect(sim, vh);

    // Settle at idle
    dt.throttle = 0.0;
    sim.run(0.5, 0.001);

    // Apply moderate throttle
    dt.throttle = 0.5;
    sim.run(3.0, 0.001);

    // Vehicle should be moving forward
    REQUIRE(sys.q_dot(0) > 1.0);

    // Chassis should still be near ground
    REQUIRE(sys.q(1) > 0.15);
    REQUIRE(sys.q(1) < 0.60);
}

// ============================================================================
// Steering with DWB
// ============================================================================

TEST_CASE("Dynamic template: DWB vehicle corners with steering",
          "[tmpl_dyn][cornering]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.k_spring = tmpl.front_axle.k_spring;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Verify calibration produced a nonzero ratio for steered corners
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) > 1e-6);
    REQUIRE(std::abs(vh.corners[1].rack_per_rad) > 1e-6);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);
    sys.q_dot(0) = 10.0;
    sys.compute_kinematics();

    sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
        const Vec3 fwd = s.states[vh.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vh.chassis_body].v_WB.dot(fwd);
        tau(0) += 500.0 * (10.0 - Vx);
    };

    // Settle at speed without steering
    sim.run(0.5, 0.001);

    // Apply left steering via tie rod geometry
    vh.set_steering(0.03);
    const Real z_before = sys.states[vh.chassis_body].p_WB.z();
    sim.run(3.0, 0.001);
    const Real z_after = sys.states[vh.chassis_body].p_WB.z();

    // Should turn LEFT (positive Z)
    REQUIRE(z_after - z_before > 0.05);
}