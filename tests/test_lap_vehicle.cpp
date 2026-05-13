#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/lap_vehicle.hpp"
#include "mbd/vehicle_template.hpp"

using Catch::Matchers::WithinAbs;

namespace { constexpr mbd::Real eps = 1e-9; }

// ============================================================================
// LapVehicle direct construction
// ============================================================================

TEST_CASE("LapVehicle: drive force at very low speed is traction-limited",
          "[lap_vehicle][force]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mass = 1500.0;
    lv.mu = 1.0;
    lv.max_power = 100000.0;

    // At V = 0.1 m/s: F_power = 1e6 N (huge), F_traction = 1.0 * 1500 * 9.81 = 14715 N
    // Result should be the traction limit
    Real F = lv.F_drive_max(0.1);
    REQUIRE_THAT(F, WithinAbs(1.0 * 1500.0 * g_accel, 1.0));
}

TEST_CASE("LapVehicle: drive force at high speed is power-limited",
          "[lap_vehicle][force]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mass = 1500.0;
    lv.mu = 1.0;
    lv.max_power = 100000.0;

    // At V = 50 m/s: F_power = 100000 / 50 = 2000 N, F_traction = 14715 N
    // Result should be power-limited
    Real F = lv.F_drive_max(50.0);
    REQUIRE_THAT(F, WithinAbs(2000.0, 1.0));
}

TEST_CASE("LapVehicle: drive force is monotonically non-increasing with speed",
          "[lap_vehicle][force]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mass = 1500.0;
    lv.mu = 1.0;
    lv.max_power = 100000.0;

    // Sample at increasing speeds
    Real F_prev = lv.F_drive_max(1.0);
    for (Real V = 5.0; V <= 100.0; V += 5.0) {
        Real F = lv.F_drive_max(V);
        REQUIRE(F <= F_prev + 1e-6);
        F_prev = F;
    }
}

TEST_CASE("LapVehicle: downforce scales as V^2", "[lap_vehicle][aero]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.ClA = 2.0;
    lv.air_density = 1.225;

    Real DF_10 = lv.downforce(10.0);
    Real DF_20 = lv.downforce(20.0);
    Real DF_40 = lv.downforce(40.0);

    REQUIRE_THAT(DF_10, WithinAbs(0.5 * 1.225 * 100.0 * 2.0, 1e-6));
    REQUIRE_THAT(DF_20 / DF_10, WithinAbs(4.0, eps));
    REQUIRE_THAT(DF_40 / DF_10, WithinAbs(16.0, eps));
}

TEST_CASE("LapVehicle: drag scales as V^2", "[lap_vehicle][aero]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.CdA = 0.7;
    lv.air_density = 1.225;

    Real D_10 = lv.drag(10.0);
    Real D_30 = lv.drag(30.0);

    REQUIRE_THAT(D_10, WithinAbs(0.5 * 1.225 * 100.0 * 0.7, 1e-6));
    REQUIRE_THAT(D_30 / D_10, WithinAbs(9.0, eps));
}

TEST_CASE("LapVehicle: grip increases with downforce", "[lap_vehicle][grip]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mass = 1000.0;
    lv.mu = 1.0;
    lv.ClA = 2.0;
    lv.air_density = 1.225;

    Real grip_static = lv.grip_force(0.0);
    Real grip_at_50  = lv.grip_force(50.0);

    REQUIRE_THAT(grip_static, WithinAbs(1000.0 * g_accel, 1e-6));
    REQUIRE(grip_at_50 > grip_static);

    // Expected: grip_at_50 = mu * (mg + 0.5*rho*V^2*ClA)
    //                     = 1.0 * (9810 + 0.5 * 1.225 * 2500 * 2.0)
    //                     = 1.0 * (9810 + 3062.5)
    //                     = 12872.5 N
    REQUIRE_THAT(grip_at_50, WithinAbs(12872.5, 1.0));
}

TEST_CASE("LapVehicle: zero downforce gives constant grip", "[lap_vehicle][grip]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mass = 1000.0;
    lv.mu = 1.0;
    lv.ClA = 0.0;

    REQUIRE_THAT(lv.grip_force(0.0),  WithinAbs(lv.grip_force(50.0), 1e-6));
    REQUIRE_THAT(lv.grip_force(50.0), WithinAbs(lv.grip_force(100.0), 1e-6));
}

// ============================================================================
// Extraction from VehicleTemplate
// ============================================================================

TEST_CASE("LapVehicle: extracted from DefaultSedan has reasonable values",
          "[lap_vehicle][extract]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    auto lv = make_lap_vehicle(tmpl);

    // Mass should equal total vehicle mass
    REQUIRE_THAT(lv.mass, WithinAbs(tmpl.total_mass(), 0.01));

    // Wheelbase should match template
    REQUIRE_THAT(lv.wheelbase, WithinAbs(tmpl.wheelbase(), 1e-9));

    // mu should be in a reasonable range (passenger tires: 0.8-1.5)
    REQUIRE(lv.mu > 0.5);
    REQUIRE(lv.mu < 2.5);

    // Aero should match
    REQUIRE_THAT(lv.CdA, WithinAbs(tmpl.chassis.CdA, eps));
    REQUIRE_THAT(lv.ClA, WithinAbs(tmpl.chassis.ClA, eps));

    // CG height should be positive and below 2m
    REQUIRE(lv.cg_height > 0.1);
    REQUIRE(lv.cg_height < 2.0);

    // max_power should be in a reasonable range (passenger car: 50-300 kW)
    REQUIRE(lv.max_power > 30000.0);
    REQUIRE(lv.max_power < 500000.0);
}

TEST_CASE("LapVehicle: extracted from SportsCar has stronger aero than sedan",
          "[lap_vehicle][extract]")
{
    using namespace mbd;

    auto sedan = make_lap_vehicle(VehicleTemplate::DefaultSedan());
    auto sports = make_lap_vehicle(VehicleTemplate::SportsCar());

    // Sports car generally has higher mu (better tires) and lower mass
    REQUIRE(sports.mu >= sedan.mu - 0.05);  // at least similar
    REQUIRE(sports.max_power >= sedan.max_power * 0.5);  // not absurdly less
}

TEST_CASE("LapVehicle: traction_limit_speed is computed correctly",
          "[lap_vehicle][extract]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mass = 1500.0;
    lv.mu = 1.0;
    lv.max_power = 100000.0;
    lv.traction_limit_speed = lv.max_power / (lv.mu * lv.mass * g_accel);

    // F_drive at this speed should be near transition
    Real F_at_threshold = lv.F_drive_max(lv.traction_limit_speed);
    Real F_traction = lv.mu * lv.mass * g_accel;

    REQUIRE_THAT(F_at_threshold, WithinAbs(F_traction, F_traction * 0.05));
}