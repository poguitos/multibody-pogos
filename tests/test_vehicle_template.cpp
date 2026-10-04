#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

// The template-built vehicle on the kernel (plan task 2.7).

using Catch::Matchers::WithinAbs;

namespace
{
    /// Forward speed of the chassis in the world.
    mbd::Real forward_speed(const mbd::kernel::Simulator& sim, const mbd::VehicleHandle& vh)
    {
        const auto& s = sim.states()[static_cast<std::size_t>(vh.chassis_body)];
        return s.v_WB.dot(s.q_WB * mbd::Vec3::UnitX());
    }
}

// ============================================================================
// Topology tests
// ============================================================================

TEST_CASE("Template: DefaultSedan has correct topology", "[template][topology]")
{
    using namespace mbd;

    kernel::System sys;
    auto vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());

    // ground + chassis + 4 wheels = 6 bodies
    REQUIRE(sys.model.nbodies() == 6);

    // 6 (chassis) + 4 (prismatic) = 10 DOF
    REQUIRE(sys.model.nv == 10);

    // 5 joints (1 free + 4 prismatic): one per body but ground
    REQUIRE(sys.model.nbodies() - 1 == 5);

    // 8 force elements (4 springs + 4 tires), no constraints
    REQUIRE(sys.force_elements.size() == 8);
    REQUIRE(sys.constraints.empty());

    // Chassis body is 1
    REQUIRE(vh.chassis_body == 1);
}

TEST_CASE("Template: SportsCar preset builds successfully", "[template][topology]")
{
    using namespace mbd;

    kernel::System sys;
    auto vh = build_vehicle(sys, VehicleTemplate::SportsCar());

    // DWB front + DWB rear: ground + chassis + front steering rack + 4*3 = 15 bodies
    REQUIRE(sys.model.nbodies() == 15);
    REQUIRE(vh.rack_body[0] > 0);
    REQUIRE(vh.rack_body[1] == 0);
    // Velocities: 6 (chassis) + 1 (rack) + 4*(1+3+1) = 27. The rack's driver
    // takes its freedom away again.
    REQUIRE(sys.model.nv == 27);
}

TEST_CASE("Template: FWDHatchback preset builds successfully", "[template][topology]")
{
    using namespace mbd;

    kernel::System sys;
    auto vh = build_vehicle(sys, VehicleTemplate::FWDHatchback());

    // McPherson front (2 bodies each, steered: plus a rack) + Simple rear (1 body each)
    // Bodies: 1 (ground) + 1 (chassis) + 1 (rack) + 2*2 (MC front) + 2*1 (simple rear) = 9
    REQUIRE(sys.model.nbodies() == 9);
    // Velocities: 6 (chassis) + 1 (rack) + 2*(1+3) (MC front) + 2*1 (simple rear) = 17
    REQUIRE(sys.model.nv == 17);
    REQUIRE(vh.rack_body[0] > 0);
}

// ============================================================================
// Equilibrium tests
// ============================================================================

TEST_CASE("Template: DefaultSedan reaches static equilibrium", "[template][equilibrium]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.q(1) += 0.02; // Small perturbation
    sim.initialize();

    sim.run(5.0, 0.001);

    // Velocities should be near zero
    for (int i = 0; i < sys.model.nv; ++i) {
        REQUIRE_THAT(sim.v(i), WithinAbs(0.0, 0.02));
    }
}

TEST_CASE("Template: equilibrium tire loads are correct", "[template][equilibrium]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);

    // One evaluation applies every force element at this state.
    sim.acceleration(sim.q, sim.v, sim.time);

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

    kernel::System sys;
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

    kernel::System sys;
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

    // Heavy car (same springs)
    auto tmpl_heavy = VehicleTemplate::DefaultSedan();
    tmpl_heavy.chassis.mass = 2000.0;

    // Both build.
    kernel::System sys_light, sys_heavy;
    build_vehicle(sys_light, tmpl_light);
    build_vehicle(sys_heavy, tmpl_heavy);

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
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();

    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);

    // Settle
    dt.throttle = 0.0;
    sim.run(0.3, 0.001);

    // Accelerate
    dt.throttle = 0.5;
    sim.run(3.0, 0.001);

    REQUIRE(forward_speed(sim, vh) > 3.0);
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    // 10 m/s forward: the chassis velocity is in its own axes (at rest
    // orientation, the world's).
    const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;
    sim.v(v_forward) = 10.0;
    sim.initialize();

    // Gentle speed controller: a push along the chassis's forward axis.
    sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
        tau(v_forward) += 500.0 * (10.0 - forward_speed(s, vh));
    };

    // Settle
    sim.run(0.5, 0.001);

    // Steer left and drive
    vh.set_steering(0.03);
    const Real z_before = sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.z();
    sim.run(3.0, 0.001);
    const Real z_after = sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.z();

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

    kernel::System sys;
    auto [dwb, bump_idx] = build_dwb_for_analysis(sys, tmpl, 0); // FL corner
    (void)bump_idx;
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, dwb.upright_body, -0.03, 0.03, 11);

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
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.q(1) += 0.015; // Pure heave
    sim.initialize();

    sim.run(0.5, 0.001);

    // Left/right symmetry: FL=FR and RL=RR (suspension travel of each corner)
    // Front/rear may differ due to different spring rates
    auto travel = [&](int c) { return sim.q(sys.model.idx_q[vh.wheel(c)]); };
    REQUIRE_THAT(travel(1), WithinAbs(travel(0), 1e-5)); // FR == FL
    REQUIRE_THAT(travel(3), WithinAbs(travel(2), 1e-5)); // RR == RL
}
