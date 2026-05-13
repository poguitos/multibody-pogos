#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle_template.hpp"
#include "mbd/simulator.hpp"
#include "mbd/drivetrain.hpp"

using Catch::Matchers::WithinAbs;

// ============================================================================
// Topology tests
// ============================================================================

TEST_CASE("Template: DefaultSedan has correct topology", "[template][topology]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());

    // ground + chassis + 4 wheels = 6 bodies
    REQUIRE(sys.body_count() == 6);

    // 6 (chassis) + 4 (prismatic) = 10 DOF
    REQUIRE(sys.total_dof == 10);

    // 5 joints (1 free + 4 prismatic)
    REQUIRE(sys.joint_count() == 5);

    // 8 force elements (4 springs + 4 tires)
    REQUIRE(sys.force_elements.size() == 8);

    // Chassis body is 1
    REQUIRE(vh.chassis_body == 1);
}

TEST_CASE("Template: SportsCar preset builds successfully", "[template][topology]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, VehicleTemplate::SportsCar());

    // SportsCar uses DWB front + DWB rear: 1 (ground) + 1 (chassis) + 4*3 = 14 bodies
    REQUIRE(sys.body_count() == 14);
    // DOFs: 6 (chassis) + 4*(1+3+1) = 26
    REQUIRE(sys.total_dof == 26);
}

TEST_CASE("Template: FWDHatchback preset builds successfully", "[template][topology]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, VehicleTemplate::FWDHatchback());

    // FWDHatchback: McPherson front (2 bodies each) + Simple rear (1 body each)
    // Bodies: 1 (ground) + 1 (chassis) + 2*2 (MC front) + 2*1 (simple rear) = 8
    REQUIRE(sys.body_count() == 8);
    // DOFs: 6 (chassis) + 2*(1+3) (MC front) + 2*1 (simple rear) = 16
    REQUIRE(sys.total_dof == 16);
}

// ============================================================================
// Equilibrium tests
// ============================================================================

TEST_CASE("Template: DefaultSedan reaches static equilibrium", "[template][equilibrium]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);
    sys.q(1) += 0.02; // Small perturbation
    sys.compute_kinematics();

    sim.run(5.0, 0.001);

    // Velocities should be near zero
    for (int i = 0; i < sys.total_dof; ++i) {
        REQUIRE_THAT(sys.q_dot(i), WithinAbs(0.0, 0.02));
    }
}

TEST_CASE("Template: equilibrium tire loads are correct", "[template][equilibrium]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    set_vehicle_equilibrium(sys, vh);

    sys.clear_forces();
    sys.apply_force_elements();

    // Front/rear loads depend on CG position (front_axle_x vs rear_axle_x)
    const Real L = tmpl.wheelbase();
    const Real W_total = tmpl.total_mass() * g_accel;
    const Real W_front_per = W_total * tmpl.rear_axle_x / L * 0.5;
    const Real W_rear_per  = W_total * tmpl.front_axle_x / L * 0.5;

    // Allow 5% tolerance — equilibrium is approximate with different F/R springs
    REQUIRE_THAT(vh.tire(0)->get_vertical_force(), WithinAbs(W_front_per, W_front_per * 0.05));
    REQUIRE_THAT(vh.tire(1)->get_vertical_force(), WithinAbs(W_front_per, W_front_per * 0.05));
    REQUIRE_THAT(vh.tire(2)->get_vertical_force(), WithinAbs(W_rear_per,  W_rear_per * 0.05));
    REQUIRE_THAT(vh.tire(3)->get_vertical_force(), WithinAbs(W_rear_per,  W_rear_per * 0.05));

    // Total should equal weight within 1%
    Real total_Fz = 0.0;
    for (int c = 0; c < 4; ++c) total_Fz += vh.tire(c)->get_vertical_force();
    REQUIRE_THAT(total_Fz, WithinAbs(W_total, W_total * 0.01));
}

// ============================================================================
// All four tires have correct accessors
// ============================================================================

TEST_CASE("Template: tire accessors work for all corners", "[template][tires]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());

    for (int c = 0; c < 4; ++c) {
        REQUIRE(vh.tire(c) != nullptr);
        REQUIRE(vh.wheel(c) >= 2); // After chassis
    }

    // All wheel body indices should be distinct
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            REQUIRE(vh.wheel(i) != vh.wheel(j));
        }
    }
}

// ============================================================================
// Steering via vehicle handle
// ============================================================================

TEST_CASE("Template: steering applies to front axle only (sedan)", "[template][steering]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());

    vh.set_steering(0.1);

    // Front tires should have nonzero steer angles
    REQUIRE(vh.tire(0)->steer_angle > 0.0);
    REQUIRE(vh.tire(1)->steer_angle > 0.0);
    REQUIRE(vh.tire(0)->steer_angle > vh.tire(1)->steer_angle); // Inner > outer

    // Rear tires should have zero steer
    REQUIRE_THAT(vh.tire(2)->steer_angle, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(vh.tire(3)->steer_angle, WithinAbs(0.0, 1e-12));

    vh.clear_steering();
    for (int c = 0; c < 4; ++c) {
        REQUIRE_THAT(vh.tire(c)->steer_angle, WithinAbs(0.0, 1e-12));
    }
}

// ============================================================================
// Parameter changes affect behavior
// ============================================================================

TEST_CASE("Template: heavier car has lower tire load frequency", "[template][params]")
{
    using namespace mbd;

    // Light car
    auto tmpl_light = VehicleTemplate::DefaultSedan();
    tmpl_light.chassis.mass = 1000.0;

    MultibodySystem sys_light;
    auto vh_light = build_vehicle(sys_light, tmpl_light);

    // Heavy car (same springs)
    auto tmpl_heavy = VehicleTemplate::DefaultSedan();
    tmpl_heavy.chassis.mass = 2000.0;

    MultibodySystem sys_heavy;
    auto vh_heavy = build_vehicle(sys_heavy, tmpl_heavy);

    // Natural frequency: omega = sqrt(k/m)
    // Heavier car should have lower frequency
    const Real f_light = std::sqrt(tmpl_light.front_axle.k_spring /
                                   tmpl_light.total_mass());
    const Real f_heavy = std::sqrt(tmpl_heavy.front_axle.k_spring /
                                   tmpl_heavy.total_mass());

    REQUIRE(f_heavy < f_light);
}

TEST_CASE("Template: stiffer springs give higher frequency", "[template][params]")
{
    using namespace mbd;

    auto tmpl_soft = VehicleTemplate::DefaultSedan();
    tmpl_soft.front_axle.k_spring = 20000.0;
    tmpl_soft.rear_axle.k_spring  = 20000.0;

    auto tmpl_stiff = VehicleTemplate::DefaultSedan();
    tmpl_stiff.front_axle.k_spring = 40000.0;
    tmpl_stiff.rear_axle.k_spring  = 40000.0;

    const Real f_soft  = std::sqrt(tmpl_soft.front_axle.k_spring /
                                   tmpl_soft.total_mass());
    const Real f_stiff = std::sqrt(tmpl_stiff.front_axle.k_spring /
                                   tmpl_stiff.total_mass());

    REQUIRE(f_stiff > f_soft);
}

// ============================================================================
// Driving with drivetrain integration
// ============================================================================

TEST_CASE("Template: vehicle with drivetrain accelerates",
          "[template][drivetrain]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
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

    // Settle
    dt.throttle = 0.0;
    sim.run(0.3, 0.001);

    // Accelerate
    dt.throttle = 0.5;
    sim.run(3.0, 0.001);

    REQUIRE(sys.q_dot(0) > 3.0);
}

// ============================================================================
// Cornering with template-built vehicle
// ============================================================================

TEST_CASE("Template: vehicle corners with steering", "[template][cornering]")
{
    using namespace mbd;

    // Use uniform spring rates to avoid pitch-induced load imbalance
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.rear_axle.k_spring = tmpl.front_axle.k_spring;
    tmpl.rear_axle.c_damper = tmpl.front_axle.c_damper;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);
    sys.q_dot(0) = 10.0;
    sys.compute_kinematics();

    // Gentle speed controller
    sim.force_callback = [&](MultibodySystem& s, Real, VecX& tau) {
        const Vec3 fwd = s.states[vh.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = s.states[vh.chassis_body].v_WB.dot(fwd);
        tau(0) += 500.0 * (10.0 - Vx);
    };

    // Settle
    sim.run(0.5, 0.001);

    // Steer left and drive
    vh.set_steering(0.03);
    const Real z_before = sys.states[vh.chassis_body].p_WB.z();
    sim.run(3.0, 0.001);
    const Real z_after = sys.states[vh.chassis_body].p_WB.z();

    // Should turn left (positive Z)
    REQUIRE(z_after - z_before > 0.05);
}

// ============================================================================
// Kinematic analysis from template
// ============================================================================

TEST_CASE("Template: DWB kinematic analysis from template",
          "[template][kinematics]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::SportsCar();

    MultibodySystem sys;
    auto [dwb, bump_idx] = build_dwb_for_analysis(sys, tmpl, 0); // FL corner

    set_dwb_reference(sys, dwb);

    auto result = sweep_bump_travel(
        sys, bump_idx, dwb.upright_body,
        dwb.params.wheel_center.y(),
        -0.03, 0.03, 11);

    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
    }

    // Should produce measurable camber change
    const Real camber_range = std::abs(result.points.back().camber -
                                       result.points.front().camber);
    REQUIRE(camber_range > 0.001);
}

// ============================================================================
// Template presets produce different vehicles
// ============================================================================

TEST_CASE("Template: different presets have different masses",
          "[template][presets]")
{
    using namespace mbd;

    auto sedan = VehicleTemplate::DefaultSedan();
    auto sports = VehicleTemplate::SportsCar();
    auto hatch = VehicleTemplate::FWDHatchback();

    REQUIRE(sedan.total_mass() != sports.total_mass());
    REQUIRE(sports.total_mass() != hatch.total_mass());
}

TEST_CASE("Template: different presets have different drivetrains",
          "[template][presets]")
{
    using namespace mbd;

    auto sedan = VehicleTemplate::DefaultSedan();
    auto hatch = VehicleTemplate::FWDHatchback();

    REQUIRE(sedan.drivetrain.layout == DriveLayout::RWD);
    REQUIRE(hatch.drivetrain.layout == DriveLayout::FWD);
}

// ============================================================================
// Symmetry
// ============================================================================

TEST_CASE("Template: symmetric bounce keeps corners equal",
          "[template][symmetry]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    set_vehicle_equilibrium(sys, vh);
    sys.q(1) += 0.015; // Pure heave
    sys.compute_kinematics();

    sim.run(0.5, 0.001);

    // Left/right symmetry: FL=FR and RL=RR
    // Front/rear may differ due to different spring rates
    REQUIRE_THAT(sys.q(7), WithinAbs(sys.q(6), 1e-5)); // FR == FL
    REQUIRE_THAT(sys.q(9), WithinAbs(sys.q(8), 1e-5)); // RR == RL
}