#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

// Template-built vehicles with linkage suspensions, on the kernel (plan task
// 2.7). A steered axle with linkage suspension carries a steering rack: one
// more body and velocity, held by one more constraint equation (its driver).

using Catch::Matchers::WithinAbs;

namespace
{
    int equation_count(const mbd::kernel::System& sys)
    {
        int n = 0;
        for (const auto& c : sys.constraints) n += c->size();
        return n;
    }

    mbd::Real forward_speed(const mbd::kernel::Simulator& sim, const mbd::VehicleHandle& vh)
    {
        const auto& s = sim.states()[static_cast<std::size_t>(vh.chassis_body)];
        return s.v_WB.dot(s.q_WB * mbd::Vec3::UnitX());
    }

    mbd::Real total_tire_load(mbd::kernel::Simulator& sim, const mbd::VehicleHandle& vh)
    {
        sim.acceleration(sim.q, sim.v, sim.time);   // applies the force elements
        mbd::Real total = 0.0;
        for (int c = 0; c < 4; ++c) total += vh.tire(c)->get_vertical_force();
        return total;
    }
}

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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    // Bodies: 1 (ground) + 1 (chassis) + 1 (front rack) + 4*3 (LCA, upright, UCA) = 15
    REQUIRE(sys.model.nbodies() == 15);

    // Velocities: 6 (chassis) + 1 (rack) + 4 * (1+3+1) = 27
    REQUIRE(sys.model.nv == 27);

    // Constraints: 4 corners * 2 (upper ball joint + tie rod) + the rack driver = 9
    REQUIRE(sys.constraints.size() == 9);

    // Equations: 4 * (3 + 1) + 1 = 17, leaving 27 - 17 = 10 freedoms
    REQUIRE(equation_count(sys) == 17);

    // Force elements: 4 springs + 4 tires = 8
    REQUIRE(sys.force_elements.size() == 8);
    REQUIRE(vh.rack_body[0] > 0);
}

TEST_CASE("Dynamic template: all-McPherson vehicle has correct topology",
          "[tmpl_dyn][topology]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::McPherson;
    tmpl.rear_axle.suspension_type  = SuspensionType::McPherson;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    // Bodies: 1 (ground) + 1 (chassis) + 1 (rack) + 4*2 = 11
    REQUIRE(sys.model.nbodies() == 11);

    // Velocities: 6 + 1 + 4 * (1+3) = 23
    REQUIRE(sys.model.nv == 23);

    // Constraints: 4 * 2 + 1 = 9; equations 4 * (2 strut line + 1 tie rod) + 1 = 13
    REQUIRE(sys.constraints.size() == 9);
    REQUIRE(equation_count(sys) == 13);
}

TEST_CASE("Dynamic template: mixed suspension (DWB front, simple rear)",
          "[tmpl_dyn][topology]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::Simple;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    // Bodies: 1 + 1 + 1 (rack) + 2*3 (DWB) + 2*1 (simple) = 11
    REQUIRE(sys.model.nbodies() == 11);

    // Velocities: 6 + 1 + 2*5 (DWB) + 2*1 (simple) = 19
    REQUIRE(sys.model.nv == 19);

    // Constraints: 2 corners * 2 (only front DWB) + the rack driver = 5
    REQUIRE(sys.constraints.size() == 5);
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

    kernel::System sys;
    build_vehicle(sys, tmpl);

    Kinematics k(sys);   // the neutral configuration
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-9));
}

TEST_CASE("Dynamic template: all-McPherson reference config satisfies constraints",
          "[tmpl_dyn][reference]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::McPherson;
    tmpl.rear_axle.suspension_type  = SuspensionType::McPherson;

    kernel::System sys;
    build_vehicle(sys, tmpl);

    Kinematics k(sys);
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-9));
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    for (int c = 0; c < 4; ++c) {
        REQUIRE(vh.tire(c) != nullptr);
        REQUIRE(vh.tire(c)->wheel_body_idx == vh.wheel(c));
        REQUIRE(sys.model.name[vh.wheel(c)] == "dyn_upright");
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);   // rough initial condition
    sim.initialize();

    // Let it settle for 3 seconds
    sim.run(3.0, 0.001);

    // Chassis should be at roughly the expected height
    // The exact height depends on the DWB geometry, but should be around 0.25-0.30 m
    REQUIRE(sim.q(1) > 0.20);
    REQUIRE(sim.q(1) < 0.60);

    // Velocities should be small (settled)
    for (int i = 0; i < sys.model.nv; ++i) {
        REQUIRE(std::abs(sim.v(i)) < 0.1);
    }

    // Total tire load should approximately equal vehicle weight
    const Real W_total = tmpl.total_mass() * g_accel;
    REQUIRE_THAT(total_tire_load(sim, vh), WithinAbs(W_total, W_total * 0.05));
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    // Topology: 6 bodies, 10 DOF (as before)
    REQUIRE(sys.model.nbodies() == 6);
    REQUIRE(sys.model.nv == 10);
    REQUIRE(sys.constraints.empty());

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    sim.run(2.0, 0.001);

    // Should settle
    for (int i = 0; i < sys.model.nv; ++i) {
        REQUIRE(std::abs(sim.v(i)) < 0.05);
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();

    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);

    // Settle at idle
    dt.throttle = 0.0;
    sim.run(0.5, 0.001);

    // Apply moderate throttle
    dt.throttle = 0.5;
    sim.run(3.0, 0.001);

    // Vehicle should be moving forward
    REQUIRE(forward_speed(sim, vh) > 1.0);

    // Chassis should still be near ground
    REQUIRE(sim.q(1) > 0.15);
    REQUIRE(sim.q(1) < 0.60);
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    // Verify calibration produced a nonzero ratio for steered corners
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) > 1e-6);
    REQUIRE(std::abs(vh.corners[1].rack_per_rad) > 1e-6);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;
    sim.v(v_forward) = 10.0;
    sim.initialize();

    sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
        tau(v_forward) += 500.0 * (10.0 - forward_speed(s, vh));
    };

    // Settle at speed without steering
    sim.run(0.5, 0.001);

    // Apply left steering through the rack and the tie rods
    vh.set_steering(0.03);
    const Real z_before = sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.z();
    sim.run(3.0, 0.001);
    const Real z_after = sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.z();

    // Should turn LEFT (positive Z)
    REQUIRE(z_after - z_before > 0.05);
}
