#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <limits>

#include "mbd/lap_speed_profile.hpp"
#include "mbd/lap_vehicle.hpp"
#include "mbd/track.hpp"

using Catch::Matchers::WithinAbs;

namespace { constexpr mbd::Real eps = 1e-9; }

// ============================================================================
// Single-point V_max
// ============================================================================

TEST_CASE("Speed profile: V_max on a straight is infinite", "[speed_profile][limit]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    TrackPoint pt;
    pt.kappa = 0.0;
    pt.bank = 0.0;

    Real V = lap_vmax_at(pt, lv);
    REQUIRE(std::isinf(V));
}

TEST_CASE("Speed profile: V_max on a flat unbanked corner = sqrt(mu*g/|kappa|)",
          "[speed_profile][corner]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    TrackPoint pt;
    pt.kappa = 0.02;  // 50m radius
    pt.bank = 0.0;

    Real V = lap_vmax_at(pt, lv);
    Real V_expected = std::sqrt(lv.mu * g_accel / std::abs(pt.kappa));
    REQUIRE_THAT(V, WithinAbs(V_expected, 1e-9));
}

TEST_CASE("Speed profile: V_max scales with sqrt(radius)",
          "[speed_profile][corner]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    auto vmax_at_R = [&](Real R) {
        TrackPoint pt;
        pt.kappa = 1.0 / R;
        pt.bank = 0.0;
        return lap_vmax_at(pt, lv);
    };

    Real V_50  = vmax_at_R(50.0);
    Real V_200 = vmax_at_R(200.0);

    // V scales as sqrt(R), so V_200 / V_50 = 2
    REQUIRE_THAT(V_200 / V_50, WithinAbs(2.0, 1e-6));
}

TEST_CASE("Speed profile: higher mu allows higher V_max",
          "[speed_profile][corner]")
{
    using namespace mbd;

    LapVehicle lv1, lv2;
    lv1.mu = 1.0;
    lv2.mu = 1.5;

    TrackPoint pt;
    pt.kappa = 0.02;
    pt.bank = 0.0;

    Real V1 = lap_vmax_at(pt, lv1);
    Real V2 = lap_vmax_at(pt, lv2);

    REQUIRE(V2 > V1);
    // V scales as sqrt(mu), so V2/V1 = sqrt(1.5/1.0) = sqrt(1.5)
    REQUIRE_THAT(V2 / V1, WithinAbs(std::sqrt(1.5), 1e-6));
}

TEST_CASE("Speed profile: positive banking increases V_max in matched turn",
          "[speed_profile][bank]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    TrackPoint flat;
    flat.kappa = 0.02;
    flat.bank = 0.0;

    TrackPoint banked;
    banked.kappa = 0.02;     // left turn
    banked.bank = 0.3;       // positive bank (helps in left turn)

    Real V_flat   = lap_vmax_at(flat, lv);
    Real V_banked = lap_vmax_at(banked, lv);

    REQUIRE(V_banked > V_flat);
}

TEST_CASE("Speed profile: negative (off-camber) banking decreases V_max",
          "[speed_profile][bank]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;

    TrackPoint flat;
    flat.kappa = 0.02;
    flat.bank = 0.0;

    TrackPoint off_camber;
    off_camber.kappa = 0.02;  // left turn
    off_camber.bank = -0.2;   // off-camber

    Real V_flat = lap_vmax_at(flat, lv);
    Real V_off  = lap_vmax_at(off_camber, lv);

    REQUIRE(V_off < V_flat);
}

TEST_CASE("Speed profile: bank direction matters (right turn with positive bank)",
          "[speed_profile][bank]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;

    // Same |kappa|, opposite signs, with same positive bank
    TrackPoint left_turn;
    left_turn.kappa = 0.02;
    left_turn.bank = 0.2;

    TrackPoint right_turn;
    right_turn.kappa = -0.02; // right turn
    right_turn.bank = 0.2;    // bank that helped left turn now hurts right turn

    Real V_left  = lap_vmax_at(left_turn, lv);
    Real V_right = lap_vmax_at(right_turn, lv);

    REQUIRE(V_left > V_right);
}

TEST_CASE("Speed profile: ideal banking matches no-friction speed",
          "[speed_profile][bank]")
{
    using namespace mbd;

    // Even with mu=0, a banked corner can sustain a speed where bank alone
    // provides centripetal force: V² = g·tan(phi)/|kappa|

    LapVehicle lv;
    lv.mu = 0.0;     // no friction
    lv.ClA = 0.0;
    lv.mass = 1500.0;

    const Real R   = 50.0;
    const Real phi = 0.3;

    TrackPoint pt;
    pt.kappa = 1.0 / R;  // left turn
    pt.bank  = phi;

    // At V_balance = sqrt(g·tan(phi)·R), no friction needed → that's V_max
    Real V_expected = std::sqrt(g_accel * std::tan(phi) * R);
    Real V          = lap_vmax_at(pt, lv);
    REQUIRE_THAT(V, WithinAbs(V_expected, 1e-6));
}

TEST_CASE("Speed profile: downforce increases V_max", "[speed_profile][aero]")
{
    using namespace mbd;

    LapVehicle lv1;
    lv1.mu = 1.0;
    lv1.mass = 1500.0;
    lv1.ClA = 0.0;

    LapVehicle lv2 = lv1;
    lv2.ClA = 2.0;

    TrackPoint pt;
    pt.kappa = 0.02;
    pt.bank = 0.0;

    Real V_no_aero   = lap_vmax_at(pt, lv1);
    Real V_with_aero = lap_vmax_at(pt, lv2);

    REQUIRE(V_with_aero > V_no_aero);
}

// ============================================================================
// Sampled profile
// ============================================================================

TEST_CASE("Speed profile: sample on simple straight + corner track",
          "[speed_profile][sample]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);
    t.add_arc(pi / 2.0 * 50.0, 50.0);  // quarter-circle at R=50

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    auto profile = sample_vmax_profile(t, lv, 51);

    REQUIRE(profile.s.size() == 51);
    REQUIRE(profile.V_max.size() == 51);

    // First sample is on the straight: V_max should be infinity
    REQUIRE(std::isinf(profile.V_max.front()));

    // Last sample is in the corner: V_max should be finite
    REQUIRE(std::isfinite(profile.V_max.back()));

    // Expected V in corner: sqrt(g/0.02) ≈ 22.15 m/s
    Real V_corner_expected = std::sqrt(g_accel * 50.0);
    REQUIRE_THAT(profile.V_max.back(), WithinAbs(V_corner_expected, 0.1));
}

TEST_CASE("Speed profile: sample on closed track produces finite values everywhere except straights",
          "[speed_profile][sample]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);
    t.add_arc(pi * 25.0, 25.0);  // half-circle
    t.add_straight(100.0);
    t.add_arc(pi * 25.0, 25.0);

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    auto profile = sample_vmax_profile(t, lv, 101);

    int n_finite = 0;
    int n_inf = 0;
    for (Real V : profile.V_max) {
        if (std::isinf(V)) ++n_inf;
        else if (std::isfinite(V)) ++n_finite;
    }

    REQUIRE(n_inf > 0);     // some straight samples
    REQUIRE(n_finite > 0);  // some corner samples
}

// ============================================================================
// Validation against bicycle model
// ============================================================================

TEST_CASE("Speed profile: V_max matches max_lateral_acceleration / kappa",
          "[speed_profile][validation]")
{
    using namespace mbd;

    LapVehicle lv;
    lv.mu = 1.0;
    lv.mass = 1500.0;
    lv.ClA = 0.0;

    const Real kappa = 0.05;
    TrackPoint pt;
    pt.kappa = kappa;
    pt.bank = 0.0;

    Real V = lap_vmax_at(pt, lv);
    Real a_lat = V * V * kappa;
    Real a_lat_expected = lv.mu * g_accel;

    REQUIRE_THAT(a_lat, WithinAbs(a_lat_expected, 1e-6));
}