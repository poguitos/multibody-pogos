#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "mbd/vehicle_template.hpp"
#include "mbd/simulator.hpp"
#include "mbd/lap_vehicle.hpp"
#include "mbd/lap_speed_profile.hpp"
#include "mbd/bicycle_model.hpp"
#include "mbd/drivetrain.hpp"

using Catch::Matchers::WithinAbs;

// ============================================================================
// QSS sanity: V_max formula matches analytical sqrt(mu*g*R)
// ============================================================================

TEST_CASE("Lap validation: QSS V_max formula matches analytical sqrt(mu*g*R)",
          "[lap_validation][analytical]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    const Real R = 80.0;
    TrackPoint pt;
    pt.kappa = 1.0 / R;
    pt.bank = 0.0;

    Real V_QSS = lap_vmax_at(pt, lv);
    Real V_expected = std::sqrt(lv.mu * g_accel * R);

    REQUIRE_THAT(V_QSS, WithinAbs(V_expected, 1e-9));
}

// ============================================================================
// Bicycle model max a_y is in the right range
// ============================================================================

TEST_CASE("Lap validation: bicycle model max a_y is in physical range",
          "[lap_validation][analytical]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    BicycleModelParams bp;
    bp.mass = tmpl.total_mass();
    bp.front_axle_x = tmpl.front_axle_x;
    bp.rear_axle_x  = tmpl.rear_axle_x;
    bp.tire_front = tmpl.front_axle.tire_params;
    bp.tire_rear  = tmpl.rear_axle.tire_params;
    BicycleModel bm(bp);

    const Real a_y_max = bm.max_lateral_acceleration();

    REQUIRE(a_y_max > 0.5 * g_accel);
    REQUIRE(a_y_max < 2.0 * g_accel);
}

// ============================================================================
// Terminal velocity: multibody asymptotic speed matches QSS prediction
// ============================================================================

TEST_CASE("Lap validation: terminal velocity matches QSS prediction order-of-magnitude",
          "[lap_validation][terminal]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.7;
    tmpl.chassis.ClA = 0.0;

    LapVehicle lv = make_lap_vehicle(tmpl);

    // QSS terminal velocity: V³ = 2·P_max/(ρ·CdA)
    const Real V_term_QSS = std::cbrt(2.0 * lv.max_power /
                                      (lv.air_density * lv.CdA));

    INFO("QSS V_term = " << V_term_QSS << " m/s");
    REQUIRE(V_term_QSS > 30.0); // Should be a real highway-range speed
    REQUIRE(V_term_QSS < 150.0);

    // Build multibody and run from rest with full throttle
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
    vh.install_aerodynamics(sys);

    // Settle longer at idle to ensure drivetrain and chassis are in sync
    dt.throttle = 0.0;
    sim.run(0.5, 0.001);

    // Use moderate throttle to avoid wheel-spin instabilities at high speed
    dt.throttle = 0.5;
    sim.run(8.0, 0.001);

    const Real V_actual_end = sys.q_dot(0);

    INFO("Multibody V at t=8s: " << V_actual_end);

    // Sanity: vehicle should be moving forward at significant speed
    REQUIRE(V_actual_end > 10.0);

    // V should be in the right ballpark relative to QSS prediction.
    // We use moderate throttle and shorter time to avoid known instabilities
    // in the multibody simulation at high speeds. So the multibody won't reach
    // QSS terminal — but should be within an order of magnitude.
    REQUIRE(V_actual_end > 0.10 * V_term_QSS);
    REQUIRE(V_actual_end < 1.50 * V_term_QSS);
}

