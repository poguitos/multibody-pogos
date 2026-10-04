#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/analysis/position_kinematics.hpp"

// Steering of linkage suspensions through the steering rack, on the kernel
// (plan task 2.7): set_steering moves the rack, its driver carries the tie
// rods, and the linkage turns the wheels.

using Catch::Matchers::WithinAbs;

namespace
{
    mbd::VehicleTemplate all_dwb_sedan()
    {
        auto tmpl = mbd::VehicleTemplate::DefaultSedan();
        tmpl.front_axle.suspension_type = mbd::SuspensionType::DoubleWishbone;
        tmpl.rear_axle.suspension_type  = mbd::SuspensionType::DoubleWishbone;
        return tmpl;
    }

    /// Toe of the two front wheels after steering by delta and settling 0.1 s.
    std::pair<mbd::Real, mbd::Real> front_toe_after_steering(mbd::Real delta)
    {
        using namespace mbd;
        kernel::System sys;
        auto vh = build_vehicle(sys, all_dwb_sedan());

        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);
        vh.set_steering(delta);
        sim.initialize();   // the projection moves the rack and the linkage

        sim.run(0.1, 0.0005);

        return {extract_toe(sim.states()[static_cast<std::size_t>(vh.corners[0].wheel_body)]),
                extract_toe(sim.states()[static_cast<std::size_t>(vh.corners[1].wheel_body)])};
    }
}

// ============================================================================
// Calibration produces reasonable rack_per_rad values
// ============================================================================

TEST_CASE("DWB steering: calibration produces nonzero ratio for steered DWB corners",
          "[dwb_steer][calibration]")
{
    using namespace mbd;

    kernel::System sys;
    auto vh = build_vehicle(sys, all_dwb_sedan());

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

    kernel::System sys;
    auto vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());   // all Simple

    for (int c = 0; c < 4; ++c) {
        REQUIRE_FALSE(vh.corners[c].on_rack);
        REQUIRE(vh.corners[c].rack_per_rad == 0.0);
    }
    REQUIRE(vh.rack[0] == nullptr);
    REQUIRE(vh.rack[1] == nullptr);
}

// ============================================================================
// Commanded steering produces actual wheel toe (via loop constraint)
// ============================================================================

TEST_CASE("DWB steering: commanded angle produces expected wheel toe",
          "[dwb_steer][response]")
{
    using namespace mbd;

    // Small steering input: ~2.9 deg
    const Real delta_command = 0.05;
    const auto [toe_FL, toe_FR] = front_toe_after_steering(delta_command);

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

    const auto [toe_FL, toe_FR] = front_toe_after_steering(-0.05);

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

    kernel::System sys;
    auto vh = build_vehicle(sys, all_dwb_sedan());
    REQUIRE(vh.rack[0] != nullptr);

    vh.set_steering(0.05);

    // The rack, which carries the tie rods' inner points, has moved left:
    // its travel is the command times the mean calibrated ratio.
    const Real ratio = 0.5 * (std::abs(vh.corners[0].rack_per_rad) + std::abs(vh.corners[1].rack_per_rad));
    REQUIRE_THAT(vh.rack[0]->travel, WithinAbs(0.05 * ratio, 1e-15));
    REQUIRE(vh.rack[0]->travel > 0.0);

    vh.clear_steering();

    // After clearing, the rack is back at the reference
    REQUIRE(vh.rack[0]->travel == 0.0);
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

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    // Front DWB should have calibrated ratio; rear Simple should not
    REQUIRE(std::abs(vh.corners[0].rack_per_rad) > 1e-6);
    REQUIRE(vh.corners[2].rack_per_rad == 0.0);
    REQUIRE_FALSE(vh.corners[2].on_rack);

    vh.set_steering(0.05);

    // Front corners: the rack moved
    REQUIRE(vh.rack[0]->travel > 0.0);

    // Rear corners (Simple + not steered): tire steer_angle stays zero
    REQUIRE_THAT(vh.corners[2].tire->steer_angle, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(vh.corners[3].tire->steer_angle, WithinAbs(0.0, 1e-12));

    // Front corners' tire steer_angle should be zero (we use tie rod instead)
    REQUIRE_THAT(vh.corners[0].tire->steer_angle, WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(vh.corners[1].tire->steer_angle, WithinAbs(0.0, 1e-12));
}

TEST_CASE("DWB steering: the rack's toe matches its calibration on a pinned chassis",
          "[dwb_steer][calibration]")
{
    // With the chassis pinned, moving the rack by delta * ratio must turn the
    // left front wheel by close to delta: the calibration is linear for
    // small steps, so a 0.02 rad command lands within 10 % of it.
    using namespace mbd;

    const auto tmpl = all_dwb_sedan();
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::System pinned;
    const BodyIndex chassis = pinned.model.add_body(
        0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
        Transform3::Identity(), RigidBodyInertia::from_solid_box(1000.0, tmpl.chassis.half_extents),
        "chassis");
    auto rack = std::make_shared<SteeringRack>();
    const BodyIndex rack_body = detail::add_steering_rack(pinned, chassis, rack, "rack");
    const Vec3 wheel_center(tmpl.front_axle_x, 0.0, tmpl.front_axle.half_track);
    const auto p = detail::make_dwb_params_for_corner(wheel_center, tmpl.front_axle.dwb, false,
                                                      tmpl.front_axle.arm_mass,
                                                      tmpl.front_axle.upright_mass);
    const auto dwb = build_double_wishbone_corner_dynamic(pinned, chassis, p, rack_body);

    Kinematics k(pinned);
    REQUIRE(k.solve());
    const Real toe0 = extract_toe(k.state(dwb.upright_body));
    rack->travel = 0.02 * vh.corners[0].rack_per_rad;
    REQUIRE(k.solve());
    const Real dtoe = extract_toe(k.state(dwb.upright_body)) - toe0;
    REQUIRE_THAT(dtoe, WithinAbs(0.02, 0.002));
}
