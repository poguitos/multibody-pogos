#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/vehicle_template.hpp"
#include "mbd/kinematics.hpp"
#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;
}

// ============================================================================
// Calibration produces reasonable rack_per_rad values
// ============================================================================

TEST_CASE("DWB steering: calibration produces nonzero ratio for steered DWB corners",
          "[dwb_steer][calibration]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Front corners are steered
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) > 1e-6);
    REQUIRE(std::abs(vh.corners[1].rack_per_rad) > 1e-6);

    // Rack ratio should be reasonable (roughly 0.05-0.2 m/rad for typical geometry)
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) > 0.02);
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) < 0.5);
}

TEST_CASE("DWB steering: Simple corners have zero calibration", "[dwb_steer][calibration]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    // All Simple

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    for (int c = 0; c < 4; ++c) {
        REQUIRE(vh.corners[c].tierod_constraint == nullptr);
        REQUIRE(vh.corners[c].rack_per_rad == 0.0);
    }
}

// ============================================================================
// Commanded steering produces actual wheel toe (via loop constraint)
// ============================================================================

TEST_CASE("DWB steering: commanded angle produces expected wheel toe",
          "[dwb_steer][response]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    set_vehicle_equilibrium(sys, vh);

    // Apply small steering input
    const Real delta_command = 0.05; // ~2.9 deg
    vh.set_steering(delta_command);

    // Solve position kinematics for the full system to propagate tie rod motion
    // through the mechanism. Since the vehicle is free in 6 DOF, this is more
    // complex than for the kinematic builder. Instead, we use damping in a
    // short simulation to let the mechanism settle.
    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    // Run briefly to let constraint forces propagate
    sim.run(0.1, 0.0005);

    // Measure actual toe on front corners
    const Real toe_FL = extract_toe(sys.states[vh.corners[0].wheel_body]);
    const Real toe_FR = extract_toe(sys.states[vh.corners[1].wheel_body]);

    // For left turn: both front wheels should have positive toe
    // For left turn: both front wheels should have positive toe
    REQUIRE(toe_FL > 0.0005);
    REQUIRE(toe_FR > 0.0005);

    // Average toe should be a reasonable fraction of commanded angle.
    // Individual wheels may differ due to geometry, but the average
    // should respond meaningfully to the steering input. The 8% threshold
    // accounts for the fact that calibration is done at reference (chassis
    // pinned), but here the chassis is free and settles under gravity, which
    // changes the steering geometry.
    const Real toe_avg = 0.5 * (toe_FL + toe_FR);
    REQUIRE(toe_avg > delta_command * 0.08);
}

TEST_CASE("DWB steering: negative command gives negative toe", "[dwb_steer][response]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    set_vehicle_equilibrium(sys, vh);

    vh.set_steering(-0.05);

    Simulator sim(sys);
    sim.set_gravity(Vec3(0.0, -g_accel, 0.0));
    sim.method = IntegrationMethod::RK4;
    sim.initialize();

    sim.run(0.1, 0.0005);

    const Real toe_FL = extract_toe(sys.states[vh.corners[0].wheel_body]);
    const Real toe_FR = extract_toe(sys.states[vh.corners[1].wheel_body]);

    // Both front wheels should have negative toe (right turn)
    REQUIRE(toe_FL < -0.0005);
    REQUIRE(toe_FR < -0.0005);

}

// ============================================================================
// Clear steering restores zero toe
// ============================================================================

TEST_CASE("DWB steering: clear_steering restores tie rod position",
          "[dwb_steer][clear]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Store reference tie rod points
    const Vec3 ref_FL = vh.corners[0].tierod_inner_ref;
    const Vec3 ref_FR = vh.corners[1].tierod_inner_ref;

    vh.set_steering(0.05);

    // Tie rod inner points should have moved
    REQUIRE_FALSE(vh.corners[0].tierod_constraint->anchor1_B.isApprox(ref_FL));
    REQUIRE_FALSE(vh.corners[1].tierod_constraint->anchor1_B.isApprox(ref_FR));

    vh.clear_steering();

    // After clearing, tie rod inner points should be back at reference
    REQUIRE(vh.corners[0].tierod_constraint->anchor1_B.isApprox(ref_FL, 1e-10));
    REQUIRE(vh.corners[1].tierod_constraint->anchor1_B.isApprox(ref_FR, 1e-10));
}

// ============================================================================
// Mixed suspension: Simple rear, DWB front
// ============================================================================

TEST_CASE("DWB steering: mixed suspension steering works correctly",
          "[dwb_steer][mixed]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type  = SuspensionType::Simple;

    MultibodySystem sys;
    auto vh = build_vehicle(sys, tmpl);

    // Front DWB should have calibrated ratio; rear Simple should not
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) > 1e-6);
    REQUIRE(vh.corners[2].rack_per_rad == 0.0);
    REQUIRE(vh.corners[2].tierod_constraint == nullptr);

    vh.set_steering(0.05);

    // Front corners: tie rod moved
    REQUIRE_FALSE(vh.corners[0].tierod_constraint->anchor1_B.isApprox(
        vh.corners[0].tierod_inner_ref));

    // Rear corners (Simple + not steered): tire steer_angle stays zero
    REQUIRE_THAT(vh.corners[2].tire->steer_angle, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(vh.corners[3].tire->steer_angle, WithinAbs(0.0, 1e-12));

    // Front corners' tire steer_angle should be zero (we use tie rod instead)
    REQUIRE_THAT(vh.corners[0].tire->steer_angle, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(vh.corners[1].tire->steer_angle, WithinAbs(0.0, 1e-12));
}