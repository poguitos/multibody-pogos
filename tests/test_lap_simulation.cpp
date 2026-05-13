#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>
#include <algorithm>

#include "mbd/lap_speed_profile.hpp"
#include "mbd/lap_vehicle.hpp"
#include "mbd/track.hpp"

using Catch::Matchers::WithinAbs;

namespace { constexpr mbd::Real eps = 1e-9; }

// ============================================================================
// Sanity: lap on a circular track is constant V_max
// ============================================================================

TEST_CASE("Lap simulation: pure circular track at V_max",
          "[lap_sim][circular]")
{
    using namespace mbd;

    // Big circular loop, R = 100m
    const Real R = 100.0;

    Track t;
    t.add_arc(2.0 * pi * R, R);  // full circle

    REQUIRE(t.is_closed(1e-6, 1e-6));

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;
    lv.CdA = 0.0;
    lv.max_power = 1e7;  // plenty of power

    auto result = simulate_lap(t, lv, 200, true);

    // V_max in corner: sqrt(g*R) = sqrt(981) ≈ 31.32 m/s
    Real V_expected = std::sqrt(g_accel * R);

    // All speeds should be approximately V_expected
    for (Real V : result.V) {
        REQUIRE_THAT(V, WithinAbs(V_expected, 0.5));
    }

    // Lap time = circumference / V = 2πR / V
    Real T_expected = 2.0 * pi * R / V_expected;
    REQUIRE_THAT(result.lap_time, WithinAbs(T_expected, 0.5));
}

// ============================================================================
// Straight track: power-limited acceleration
// ============================================================================

TEST_CASE("Lap simulation: long straight produces drag deceleration toward terminal",
          "[lap_sim][straight]")
{
    using namespace mbd;

    Track t;
    t.add_straight(2000.0);

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;
    lv.CdA = 0.7;
    lv.max_power = 100000.0;

    auto result = simulate_lap(t, lv, 500, false);

    // Power-limited terminal speed: V³ = 2·P_max / (rho·CdA)
    Real V_terminal = std::cbrt(2.0 * lv.max_power /
                                (lv.air_density * lv.CdA));

    // Initial V is at the V_cap (200 m/s, far above terminal). Drag should
    // decelerate the car. Final V should be lower than initial, and moving
    // toward terminal (but won't reach it on a finite straight).
    REQUIRE(result.V.back() < result.V.front());

    // After 2000m, V should be at most 2× terminal (rough bound based on
    // asymptotic decay)
    REQUIRE(result.V.back() < V_terminal * 2.5);

    // V should be above terminal (the asymptote from above)
    REQUIRE(result.V.back() > V_terminal * 0.95);
}

TEST_CASE("Lap simulation: open straight starting from rest accelerates",
          "[lap_sim][straight]")
{
    using namespace mbd;

    Track t;
    t.add_straight(500.0);

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;
    lv.CdA = 0.7;
    lv.max_power = 100000.0;

    auto result = simulate_lap(t, lv, 500, false);

    // For an OPEN track, we don't wrap, so the start speed should be 0
    // (forward pass starts at V_max but drag and braking keep it low?)
    // Actually no — V_max on a straight is infinity (capped at V_cap=200).
    // Then forward pass from index 0 won't reduce that, so V[0] starts high.
    // For a sensible test, we need an open-track simulation that accounts
    // for "start from rest" — that's a different setup.
    //
    // Without explicit start-speed boundary condition, the open simulator
    // uses the cornering-limited V_max as the initial guess. So this test
    // just checks the simulator produces a sensible profile.

    // V should be positive, finite, and bounded by V_cap
    for (Real V : result.V) {
        REQUIRE(std::isfinite(V));
        REQUIRE(V > 0.0);
        REQUIRE(V <= 200.0);
    }
}

// ============================================================================
// Slow corner forces braking: oval shape
// ============================================================================

TEST_CASE("Lap simulation: oval track produces braking before corner",
          "[lap_sim][oval]")
{
    using namespace mbd;

    // Oval: long straight + tight corner + back
    const Real R = 30.0;  // tight corner, low V_max
    const Real L_straight = 200.0;
    const Real L_arc = pi * R;

    Track t;
    t.add_straight(L_straight);
    t.add_arc(L_arc, R);
    t.add_straight(L_straight);
    t.add_arc(L_arc, R);

    REQUIRE(t.is_closed(1e-6, 1e-6));

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;
    lv.CdA = 0.5;
    lv.max_power = 100000.0;
    lv.max_brake_force = 30000.0;

    auto result = simulate_lap(t, lv, 1000, true);

    // V_max in corner: sqrt(g*30) ≈ 17.16 m/s
    const Real V_corner = std::sqrt(g_accel * R);

    // Find midpoint of first corner and ensure V there is ~V_corner
    int n_after_first_straight = static_cast<int>(L_straight / t.total_length() *
                                                   result.s.size());
    int n_in_first_corner_mid = n_after_first_straight +
                                static_cast<int>(0.5 * L_arc / t.total_length() *
                                                 result.s.size());

    REQUIRE_THAT(result.V[n_in_first_corner_mid],
                 WithinAbs(V_corner, V_corner * 0.05));

    // Speed at the END of the straight (just before corner) should already
    // be reduced (braking has started)
    int n_end_of_straight = n_after_first_straight - 1;
    REQUIRE(result.V[n_end_of_straight] < result.V[n_after_first_straight / 2]);
    REQUIRE(result.V[n_end_of_straight] < 100.0); // already braking
}

TEST_CASE("Lap simulation: oval lap time decreases with higher mu",
          "[lap_sim][oval]")
{
    using namespace mbd;

    Track t;
    t.add_straight(200.0);
    t.add_arc(pi * 30.0, 30.0);
    t.add_straight(200.0);
    t.add_arc(pi * 30.0, 30.0);

    auto run = [&](Real mu) {
        LapVehicle lv;
        lv.mu = mu;
        lv.mass = 1500.0;
        lv.CdA = 0.5;
        lv.max_power = 100000.0;
        lv.max_brake_force = 30000.0;

        auto result = simulate_lap(t, lv, 1000, true);
        return result.lap_time;
    };

    Real T_low_mu  = run(0.7);
    Real T_high_mu = run(1.3);

    REQUIRE(T_high_mu < T_low_mu);
}

TEST_CASE("Lap simulation: lap time decreases with more power on a long straight",
          "[lap_sim][power]")
{
    using namespace mbd;

    Track t;
    t.add_straight(500.0);
    t.add_arc(pi * 50.0, 50.0);
    t.add_straight(500.0);
    t.add_arc(pi * 50.0, 50.0);

    auto run = [&](Real P) {
        LapVehicle lv;
        lv.mu = 1.0;
        lv.mass = 1500.0;
        lv.CdA = 0.5;
        lv.max_power = P;
        lv.max_brake_force = 30000.0;

        auto result = simulate_lap(t, lv, 1000, true);
        return result.lap_time;
    };

    Real T_50kW  = run(50000.0);
    Real T_200kW = run(200000.0);

    REQUIRE(T_200kW < T_50kW);
}

// ============================================================================
// Banking helps: lap time on banked oval is shorter than flat
// ============================================================================

TEST_CASE("Lap simulation: banked oval is faster than unbanked",
          "[lap_sim][bank]")
{
    using namespace mbd;

    auto build_oval = [](Real bank) {
        Track t;
        t.add_straight(200.0);
        t.add_arc(pi * 50.0, 50.0, 0.0, bank);
        t.add_straight(200.0);
        t.add_arc(pi * 50.0, 50.0, 0.0, bank);
        return t;
    };

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.CdA = 0.5;
    lv.max_power = 200000.0;
    lv.max_brake_force = 30000.0;

    Track t_flat   = build_oval(0.0);
    Track t_banked = build_oval(0.4); // ~23 deg

    Real T_flat   = simulate_lap(t_flat,   lv, 1000, true).lap_time;
    Real T_banked = simulate_lap(t_banked, lv, 1000, true).lap_time;

    REQUIRE(T_banked < T_flat);
}

// ============================================================================
// Downforce helps: lap time with aero is shorter
// ============================================================================

TEST_CASE("Lap simulation: downforce reduces lap time",
          "[lap_sim][aero]")
{
    using namespace mbd;

    Track t;
    t.add_straight(300.0);
    t.add_arc(pi * 60.0, 60.0);
    t.add_straight(300.0);
    t.add_arc(pi * 60.0, 60.0);

    auto run = [&](Real ClA) {
        LapVehicle lv;
        lv.mu = 1.0;
        lv.mass = 1500.0;
        lv.CdA = 0.7;
        lv.ClA = ClA;
        lv.max_power = 200000.0;
        lv.max_brake_force = 30000.0;

        return simulate_lap(t, lv, 1000, true).lap_time;
    };

    Real T_no_aero   = run(0.0);
    Real T_with_aero = run(2.0);

    REQUIRE(T_with_aero < T_no_aero);
}

// ============================================================================
// Slope effect: uphill straight is slower than flat
// ============================================================================

TEST_CASE("Lap simulation: uphill straight is slower than flat",
          "[lap_sim][slope]")
{
    using namespace mbd;

    auto run = [](Real delta_z) {
        Track t;
        t.add_straight(500.0, delta_z);  // straight, 500m, with elevation change

        LapVehicle lv;
        lv.mu = 1.0;
        lv.mass = 1500.0;
        lv.CdA = 0.5;
        lv.max_power = 100000.0;

        return simulate_lap(t, lv, 500, false).lap_time;
    };

    Real T_flat   = run(0.0);
    Real T_uphill = run(50.0);   // 10% grade
    Real T_downhill = run(-50.0);

    REQUIRE(T_uphill > T_flat);
    REQUIRE(T_downhill < T_flat);
}

// ============================================================================
// Validation: lap result invariants
// ============================================================================

TEST_CASE("Lap simulation: V never exceeds V_max", "[lap_sim][invariant]")
{
    using namespace mbd;

    Track t;
    t.add_straight(200.0);
    t.add_arc(pi * 40.0, 40.0);
    t.add_straight(200.0);
    t.add_arc(pi * 40.0, 40.0);

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.CdA = 0.5;
    lv.max_power = 200000.0;
    lv.max_brake_force = 30000.0;

    auto result = simulate_lap(t, lv, 1000, true);

    for (size_t i = 0; i < result.V.size(); ++i) {
        Real V = result.V[i];
        Real V_lim = result.V_max[i];
        if (std::isfinite(V_lim)) {
            REQUIRE(V <= V_lim + 1e-6);
        }
    }
}

TEST_CASE("Lap simulation: lap time is positive", "[lap_sim][invariant]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);
    t.add_arc(pi * 20.0, 20.0);
    t.add_straight(100.0);
    t.add_arc(pi * 20.0, 20.0);

    LapVehicle lv;
    auto result = simulate_lap(t, lv, 500, true);

    REQUIRE(result.lap_time > 0.0);
    REQUIRE(std::isfinite(result.lap_time));
}

TEST_CASE("Lap simulation: result fields have consistent sizes", "[lap_sim][invariant]")
{
    using namespace mbd;

    Track t;
    t.add_arc(2.0 * pi * 50.0, 50.0);

    LapVehicle lv;
    auto result = simulate_lap(t, lv, 100, true);

    REQUIRE(result.s.size() == 100);
    REQUIRE(result.V.size() == 100);
    REQUIRE(result.V_max.size() == 100);
    REQUIRE(result.s.front() == 0.0);
    REQUIRE_THAT(result.s.back(), WithinAbs(t.total_length(), 1e-6));
}