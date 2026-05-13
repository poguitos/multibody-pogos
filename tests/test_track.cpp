#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <vector>

#include "mbd/track.hpp"

using Catch::Matchers::WithinAbs;

namespace { constexpr mbd::Real eps = 1e-9; }

// ============================================================================
// Straight tracks
// ============================================================================

TEST_CASE("Track: single straight segment", "[track][straight]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    REQUIRE_THAT(t.total_length(), WithinAbs(100.0, eps));
    REQUIRE(t.segment_count() == 1);

    auto p0 = t.query(0.0);
    REQUIRE_THAT(p0.x, WithinAbs(0.0, eps));
    REQUIRE_THAT(p0.y, WithinAbs(0.0, eps));
    REQUIRE_THAT(p0.psi, WithinAbs(0.0, eps));
    REQUIRE_THAT(p0.kappa, WithinAbs(0.0, eps));

    auto p_mid = t.query(50.0);
    REQUIRE_THAT(p_mid.x, WithinAbs(50.0, eps));
    REQUIRE_THAT(p_mid.y, WithinAbs(0.0, eps));
    REQUIRE_THAT(p_mid.psi, WithinAbs(0.0, eps));

    auto p_end = t.query(100.0);
    REQUIRE_THAT(p_end.x, WithinAbs(100.0, eps));
    REQUIRE_THAT(p_end.y, WithinAbs(0.0, eps));
}

TEST_CASE("Track: chained straights are continuous", "[track][straight]")
{
    using namespace mbd;

    Track t;
    t.add_straight(50.0);
    t.add_straight(50.0);

    REQUIRE_THAT(t.total_length(), WithinAbs(100.0, eps));
    REQUIRE(t.segment_count() == 2);

    auto p_join = t.query(50.0);
    REQUIRE_THAT(p_join.x, WithinAbs(50.0, eps));

    auto p_end = t.query(100.0);
    REQUIRE_THAT(p_end.x, WithinAbs(100.0, eps));
}

// ============================================================================
// Arc segments
// ============================================================================

TEST_CASE("Track: quarter-circle arc (left turn)", "[track][arc]")
{
    using namespace mbd;

    // Quarter-circle of radius 50, length = pi/2 * 50, left turn (positive kappa)
    const Real R = 50.0;
    const Real L = pi / 2.0 * R;

    Track t;
    t.add_arc(L, R);

    REQUIRE_THAT(t.total_length(), WithinAbs(L, eps));

    // At s=0: position (0,0), heading 0
    auto p0 = t.query(0.0);
    REQUIRE_THAT(p0.x, WithinAbs(0.0, eps));
    REQUIRE_THAT(p0.y, WithinAbs(0.0, eps));
    REQUIRE_THAT(p0.psi, WithinAbs(0.0, eps));
    REQUIRE_THAT(p0.kappa, WithinAbs(1.0 / R, eps));

    // At halfway: heading = pi/4, position halfway around the arc
    auto p_mid = t.query(L / 2.0);
    REQUIRE_THAT(p_mid.psi, WithinAbs(pi / 4.0, eps));

    // At end: position (R, R) — turning left from origin heading +X
    // Center is at (0, R). Final point is at (R, R).
    auto p_end = t.query(L);
    REQUIRE_THAT(p_end.x, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.y, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.psi, WithinAbs(pi / 2.0, eps));
}

TEST_CASE("Track: quarter-circle arc (right turn)", "[track][arc]")
{
    using namespace mbd;

    const Real R = 50.0;
    const Real L = pi / 2.0 * R;

    Track t;
    t.add_arc(L, -R);  // negative = right turn

    auto p_end = t.query(L);
    // Right turn: ends at (R, -R), heading -pi/2
    REQUIRE_THAT(p_end.x, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.y, WithinAbs(-R, 1e-6));
    REQUIRE_THAT(p_end.psi, WithinAbs(-pi / 2.0, eps));
}

TEST_CASE("Track: arc curvature equals 1/radius with sign", "[track][arc]")
{
    using namespace mbd;

    Track t;
    t.add_arc(10.0, 25.0);    // left
    t.add_arc(10.0, -50.0);   // right

    auto p_first = t.query(5.0);
    REQUIRE_THAT(p_first.kappa, WithinAbs(1.0 / 25.0, eps));

    auto p_second = t.query(15.0);
    REQUIRE_THAT(p_second.kappa, WithinAbs(-1.0 / 50.0, eps));
}

TEST_CASE("Track: add_arc_by_angle produces correct geometry", "[track][arc]")
{
    using namespace mbd;

    const Real R = 30.0;
    const Real sweep = pi / 3.0;  // 60 degrees, left turn

    Track t;
    t.add_arc_by_angle(sweep, R);

    REQUIRE_THAT(t.total_length(), WithinAbs(sweep * R, eps));

    auto p_end = t.query(t.total_length());
    REQUIRE_THAT(p_end.psi, WithinAbs(sweep, 1e-9));
}

// ============================================================================
// Combined tracks
// ============================================================================

TEST_CASE("Track: straight then arc maintains continuity",
          "[track][combined]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);
    t.add_arc(pi / 2.0 * 30.0, 30.0);  // quarter-circle radius 30 left

    // At end of straight
    auto p_join = t.query(100.0);
    REQUIRE_THAT(p_join.x, WithinAbs(100.0, eps));
    REQUIRE_THAT(p_join.y, WithinAbs(0.0, eps));
    REQUIRE_THAT(p_join.psi, WithinAbs(0.0, eps));

    // After the arc: should be at (130, 30), heading pi/2
    auto p_end = t.query(t.total_length());
    REQUIRE_THAT(p_end.x, WithinAbs(130.0, 1e-6));
    REQUIRE_THAT(p_end.y, WithinAbs(30.0, 1e-6));
    REQUIRE_THAT(p_end.psi, WithinAbs(pi / 2.0, eps));
}

TEST_CASE("Track: closed track (oval) detected as closed",
          "[track][closed]")
{
    using namespace mbd;

    // An oval: straight, half-circle, straight, half-circle
    const Real L_straight = 100.0;
    const Real R = 25.0;
    const Real L_half_arc = pi * R;

    Track t;
    t.add_straight(L_straight);
    t.add_arc(L_half_arc, R);   // half-circle left
    t.add_straight(L_straight);
    t.add_arc(L_half_arc, R);   // close the loop

    REQUIRE_THAT(t.total_length(),
                 WithinAbs(2.0 * L_straight + 2.0 * L_half_arc, eps));

    REQUIRE(t.is_closed(1e-6, 1e-6));
}

TEST_CASE("Track: open track (single straight) is not closed",
          "[track][closed]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    REQUIRE_FALSE(t.is_closed());
}

// ============================================================================
// Out-of-range queries clamp
// ============================================================================

TEST_CASE("Track: query below 0 returns start", "[track][query]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    auto p = t.query(-10.0);
    REQUIRE_THAT(p.x, WithinAbs(0.0, eps));
    REQUIRE_THAT(p.s, WithinAbs(0.0, eps));
}

TEST_CASE("Track: query above total length returns end", "[track][query]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    auto p = t.query(150.0);
    REQUIRE_THAT(p.x, WithinAbs(100.0, eps));
    REQUIRE_THAT(p.s, WithinAbs(100.0, eps));
}

// ============================================================================
// Wrap_s for closed tracks
// ============================================================================

TEST_CASE("Track: wrap_s wraps positive overshoot", "[track][wrap]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    REQUIRE_THAT(t.wrap_s(0.0), WithinAbs(0.0, eps));
    REQUIRE_THAT(t.wrap_s(50.0), WithinAbs(50.0, eps));
    REQUIRE_THAT(t.wrap_s(100.0), WithinAbs(0.0, eps));
    REQUIRE_THAT(t.wrap_s(150.0), WithinAbs(50.0, eps));
    REQUIRE_THAT(t.wrap_s(250.0), WithinAbs(50.0, eps));
}

TEST_CASE("Track: wrap_s wraps negative", "[track][wrap]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    REQUIRE_THAT(t.wrap_s(-50.0), WithinAbs(50.0, eps));
    REQUIRE_THAT(t.wrap_s(-150.0), WithinAbs(50.0, eps));
}

// ============================================================================
// Polyline construction
// ============================================================================

TEST_CASE("Track: from_polyline reproduces straight-line geometry",
          "[track][polyline]")
{
    using namespace mbd;

    std::vector<Vec2> pts = {
        {0.0, 0.0}, {10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}
    };

    Track t = Track::from_polyline(pts);

    REQUIRE_THAT(t.total_length(), WithinAbs(30.0, eps));
    REQUIRE(t.segment_count() == 3);

    auto p_mid = t.query(15.0);
    REQUIRE_THAT(p_mid.x, WithinAbs(15.0, eps));
    REQUIRE_THAT(p_mid.y, WithinAbs(0.0, eps));
}

TEST_CASE("Track: from_polyline computes curvature on circular polyline",
          "[track][polyline]")
{
    using namespace mbd;

    // Sample a circle of radius 20 at 36 points
    const Real R = 20.0;
    const int N = 36;
    std::vector<Vec2> pts;
    for (int i = 0; i <= N; ++i) {
        const Real theta = 2.0 * pi * i / N;
        pts.emplace_back(R * std::cos(theta), R * std::sin(theta));
    }

    Track t = Track::from_polyline(pts);

    // Total length should be approximately the circle perimeter
    const Real expected_perim = 2.0 * pi * R;
    REQUIRE_THAT(t.total_length(), WithinAbs(expected_perim, expected_perim * 0.01));

    // Curvature should be approximately 1/R with the right sign.
    // Counterclockwise (positive sweep direction in our point list) => positive curvature
    // Sample a few segments away from the endpoints
    for (std::size_t i = 5; i + 5 < t.segment_count(); ++i) {
        const auto& seg = t.segment(i);
        REQUIRE_THAT(seg.kappa, WithinAbs(1.0 / R, 0.05 / R));
    }
}

// ============================================================================
// Validation
// ============================================================================

TEST_CASE("Track: zero-length segments are rejected", "[track][validation]")
{
    using namespace mbd;

    Track t;
    REQUIRE_THROWS_AS(t.add_straight(0.0), MbdError);
    REQUIRE_THROWS_AS(t.add_straight(-1.0), MbdError);
    REQUIRE_THROWS_AS(t.add_arc(0.0, 10.0), MbdError);
    REQUIRE_THROWS_AS(t.add_arc(10.0, 0.0), MbdError);
}

// ============================================================================
// Elevation
// ============================================================================

TEST_CASE("Track: default elevation is zero", "[track][elevation]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    auto p = t.query(50.0);
    REQUIRE_THAT(p.z, WithinAbs(0.0, eps));
    REQUIRE_THAT(p.slope, WithinAbs(0.0, eps));
}

TEST_CASE("Track: straight with positive elevation gain", "[track][elevation]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0, 10.0);  // +10m elevation over 100m

    auto p_start = t.query(0.0);
    auto p_mid   = t.query(50.0);
    auto p_end   = t.query(100.0);

    REQUIRE_THAT(p_start.z, WithinAbs(0.0, eps));
    REQUIRE_THAT(p_mid.z,   WithinAbs(5.0, eps));
    REQUIRE_THAT(p_end.z,   WithinAbs(10.0, eps));

    // Slope = 10 / 100 = 0.1 throughout
    REQUIRE_THAT(p_start.slope, WithinAbs(0.1, eps));
    REQUIRE_THAT(p_mid.slope,   WithinAbs(0.1, eps));
    REQUIRE_THAT(p_end.slope,   WithinAbs(0.1, eps));
}

TEST_CASE("Track: straight with negative elevation (downhill)", "[track][elevation]")
{
    using namespace mbd;

    Track t;
    t.add_straight(50.0, -5.0);

    auto p_end = t.query(50.0);
    REQUIRE_THAT(p_end.z, WithinAbs(-5.0, eps));
    REQUIRE_THAT(p_end.slope, WithinAbs(-0.1, eps));
}

TEST_CASE("Track: chained segments accumulate elevation", "[track][elevation]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0, 10.0);   // climb 10m
    t.add_straight(50.0, -5.0);    // descend 5m
    t.add_straight(50.0);          // level

    auto p_after_first  = t.query(100.0);
    auto p_after_second = t.query(150.0);
    auto p_end          = t.query(200.0);

    REQUIRE_THAT(p_after_first.z,  WithinAbs(10.0, eps));
    REQUIRE_THAT(p_after_second.z, WithinAbs(5.0, eps));
    REQUIRE_THAT(p_end.z,          WithinAbs(5.0, eps));
}

TEST_CASE("Track: arc with elevation gain", "[track][elevation]")
{
    using namespace mbd;

    const Real R = 30.0;
    const Real L = pi / 2.0 * R;

    Track t;
    t.add_arc(L, R, 5.0);  // quarter-circle climbing 5m

    // 2D geometry should be unchanged
    auto p_end = t.query(L);
    REQUIRE_THAT(p_end.x, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.y, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.psi, WithinAbs(pi / 2.0, eps));

    // Elevation linearly increases
    REQUIRE_THAT(p_end.z, WithinAbs(5.0, eps));

    auto p_mid = t.query(L / 2.0);
    REQUIRE_THAT(p_mid.z, WithinAbs(2.5, eps));
}

TEST_CASE("Track: closed track with non-zero elevation difference is not closed",
          "[track][closed][elevation]")
{
    using namespace mbd;

    // Same oval as before, but with +1m elevation on one straight (net imbalance)
    const Real L_straight = 100.0;
    const Real R = 25.0;
    const Real L_half_arc = pi * R;

    Track t;
    t.add_straight(L_straight, 1.0);   // climb 1m
    t.add_arc(L_half_arc, R);
    t.add_straight(L_straight);
    t.add_arc(L_half_arc, R);

    REQUIRE_FALSE(t.is_closed(1e-3, 1e-6));
}

TEST_CASE("Track: closed track with balanced elevation profile is closed",
          "[track][closed][elevation]")
{
    using namespace mbd;

    // Climb on first straight, descend on second straight: net zero
    const Real L_straight = 100.0;
    const Real R = 25.0;
    const Real L_half_arc = pi * R;

    Track t;
    t.add_straight(L_straight, 5.0);    // climb 5m
    t.add_arc(L_half_arc, R);
    t.add_straight(L_straight, -5.0);   // descend 5m
    t.add_arc(L_half_arc, R);

    REQUIRE(t.is_closed(1e-6, 1e-6));
}

// ============================================================================
// Banking
// ============================================================================

TEST_CASE("Track: default banking is zero", "[track][bank]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0);

    auto p = t.query(50.0);
    REQUIRE_THAT(p.bank, WithinAbs(0.0, eps));
}

TEST_CASE("Track: straight with banking", "[track][bank]")
{
    using namespace mbd;

    Track t;
    t.add_straight(100.0, 0.0, 0.1);  // 100m, level, 0.1 rad bank (~5.7°)

    auto p_start = t.query(0.0);
    auto p_mid   = t.query(50.0);
    auto p_end   = t.query(100.0);

    // Banking is constant within a segment
    REQUIRE_THAT(p_start.bank, WithinAbs(0.1, eps));
    REQUIRE_THAT(p_mid.bank,   WithinAbs(0.1, eps));
    REQUIRE_THAT(p_end.bank,   WithinAbs(0.1, eps));

    // 2D geometry unaffected
    REQUIRE_THAT(p_end.x, WithinAbs(100.0, eps));
    REQUIRE_THAT(p_end.y, WithinAbs(0.0, eps));
}

TEST_CASE("Track: arc with banking", "[track][bank]")
{
    using namespace mbd;

    const Real R = 30.0;
    const Real L = pi / 2.0 * R;

    Track t;
    t.add_arc(L, R, 0.0, 0.3);  // quarter-circle, level, 0.3 rad bank (~17°)

    auto p_mid = t.query(L / 2.0);
    REQUIRE_THAT(p_mid.bank, WithinAbs(0.3, eps));

    auto p_end = t.query(L);
    REQUIRE_THAT(p_end.x, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.y, WithinAbs(R, 1e-6));
    REQUIRE_THAT(p_end.bank, WithinAbs(0.3, eps));
}

TEST_CASE("Track: banking transitions via chained segments", "[track][bank]")
{
    using namespace mbd;

    // Approach a banked corner via a transition straight
    Track t;
    t.add_straight(50.0, 0.0, 0.0);   // flat approach
    t.add_straight(20.0, 0.0, 0.15);  // banking transition (sharp jump in this model)
    t.add_arc(50.0, 30.0, 0.0, 0.3);  // banked corner

    REQUIRE_THAT(t.query(25.0).bank, WithinAbs(0.0, eps));
    REQUIRE_THAT(t.query(60.0).bank, WithinAbs(0.15, eps));
    REQUIRE_THAT(t.query(80.0).bank, WithinAbs(0.3, eps));
}

TEST_CASE("Track: banked oval combining elevation and banking",
          "[track][bank][elevation]")
{
    using namespace mbd;

    // Daytona-style: long banked arcs with steep banking
    const Real R = 50.0;
    const Real L_arc = pi * R;
    const Real bank_corner = 0.5;  // ~28.6° (close to Daytona's 31°)

    Track t;
    t.add_straight(100.0);
    t.add_arc(L_arc, R, 0.0, bank_corner);
    t.add_straight(100.0);
    t.add_arc(L_arc, R, 0.0, bank_corner);

    REQUIRE(t.is_closed(1e-6, 1e-6));

    // Sample inside one of the banked corners
    auto p_in_corner = t.query(100.0 + L_arc / 2.0);
    REQUIRE_THAT(p_in_corner.bank, WithinAbs(bank_corner, eps));
    REQUIRE_THAT(p_in_corner.kappa, WithinAbs(1.0 / R, eps));
}

TEST_CASE("Track: negative banking (off-camber)", "[track][bank]")
{
    using namespace mbd;

    Track t;
    t.add_arc(50.0, 30.0, 0.0, -0.1);  // off-camber left turn

    auto p = t.query(25.0);
    REQUIRE_THAT(p.bank, WithinAbs(-0.1, eps));
    // Curvature still positive (left turn)
    REQUIRE(p.kappa > 0.0);
}

// ============================================================================
// Clothoid segments
// ============================================================================

TEST_CASE("Track: clothoid with zero curvature change is a straight line",
          "[track][clothoid]")
{
    using namespace mbd;

    Track t;
    t.add_clothoid(100.0, 0.0, 0.0);

    auto p_end = t.query(100.0);
    REQUIRE_THAT(p_end.x, WithinAbs(100.0, 1e-6));
    REQUIRE_THAT(p_end.y, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(p_end.psi, WithinAbs(0.0, 1e-6));

    // Curvature should be 0 throughout
    auto p_mid = t.query(50.0);
    REQUIRE_THAT(p_mid.kappa, WithinAbs(0.0, eps));
}

TEST_CASE("Track: clothoid with constant non-zero curvature matches arc",
          "[track][clothoid]")
{
    using namespace mbd;

    const Real R = 50.0;
    const Real kappa = 1.0 / R;
    const Real L = pi / 2.0 * R;  // quarter-circle length

    Track t_clothoid;
    t_clothoid.add_clothoid(L, kappa, kappa);

    Track t_arc;
    t_arc.add_arc(L, R);

    auto p_clo = t_clothoid.query(L);
    auto p_arc = t_arc.query(L);

    // Both should reach (R, R) with heading pi/2
    REQUIRE_THAT(p_clo.x, WithinAbs(p_arc.x, 1e-4));
    REQUIRE_THAT(p_clo.y, WithinAbs(p_arc.y, 1e-4));
    REQUIRE_THAT(p_clo.psi, WithinAbs(p_arc.psi, 1e-9));
}

TEST_CASE("Track: clothoid curvature varies linearly", "[track][clothoid]")
{
    using namespace mbd;

    Track t;
    t.add_clothoid(100.0, 0.0, 0.02);  // ramps from straight to 1/50 m radius

    auto p_start = t.query(0.0);
    auto p_quarter = t.query(25.0);
    auto p_half = t.query(50.0);
    auto p_end = t.query(100.0);

    REQUIRE_THAT(p_start.kappa,   WithinAbs(0.0,    eps));
    REQUIRE_THAT(p_quarter.kappa, WithinAbs(0.005,  eps));
    REQUIRE_THAT(p_half.kappa,    WithinAbs(0.01,   eps));
    REQUIRE_THAT(p_end.kappa,     WithinAbs(0.02,   eps));
}

TEST_CASE("Track: clothoid heading change matches integral of curvature",
          "[track][clothoid]")
{
    using namespace mbd;

    // Heading change = integral of kappa(s) ds = 0.5*(k0 + k1)*L
    const Real L = 100.0;
    const Real k0 = 0.001;
    const Real k1 = 0.020;
    const Real expected_dpsi = 0.5 * (k0 + k1) * L;

    Track t;
    t.add_clothoid(L, k0, k1);

    auto p_end = t.query(L);
    REQUIRE_THAT(p_end.psi, WithinAbs(expected_dpsi, 1e-9));
}

TEST_CASE("Track: clothoid is continuous with adjacent segments",
          "[track][clothoid]")
{
    using namespace mbd;

    // Build a corner-entry: straight -> clothoid -> arc
    const Real R = 40.0;
    const Real kappa = 1.0 / R;
    const Real L_clothoid = 30.0;

    Track t;
    t.add_straight(50.0);
    t.add_clothoid(L_clothoid, 0.0, kappa);
    t.add_arc(pi / 4.0 * R, R);  // 45-degree arc afterward

    // At end of straight: position (50,0), heading 0
    auto p_after_straight = t.query(50.0);
    REQUIRE_THAT(p_after_straight.x, WithinAbs(50.0, 1e-6));
    REQUIRE_THAT(p_after_straight.y, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(p_after_straight.psi, WithinAbs(0.0, 1e-6));

    // At start of clothoid (same s, queried from inside clothoid): same point
    auto p_into_clothoid = t.query(50.001);
    REQUIRE_THAT(p_into_clothoid.x, WithinAbs(50.001, 1e-3));
    REQUIRE_THAT(p_into_clothoid.y, WithinAbs(0.0, 1e-3));

    // At end of clothoid: heading = 0.5 * kappa * L_clothoid (for ramp from 0)
    const Real psi_after_clothoid = 0.5 * kappa * L_clothoid;
    auto p_after_clothoid = t.query(50.0 + L_clothoid);
    REQUIRE_THAT(p_after_clothoid.psi, WithinAbs(psi_after_clothoid, 1e-6));
    REQUIRE_THAT(p_after_clothoid.kappa, WithinAbs(kappa, eps));
}

TEST_CASE("Track: clothoid with negative curvature transition (right turn entry)",
          "[track][clothoid]")
{
    using namespace mbd;

    Track t;
    t.add_clothoid(50.0, 0.0, -0.02);

    auto p_end = t.query(50.0);
    REQUIRE_THAT(p_end.kappa, WithinAbs(-0.02, eps));

    // Heading change: 0.5 * (0 + -0.02) * 50 = -0.5 rad
    REQUIRE_THAT(p_end.psi, WithinAbs(-0.5, 1e-9));

    // y should be negative (turning right)
    REQUIRE(p_end.y < -0.01);
}

TEST_CASE("Track: clothoid carries elevation and banking",
          "[track][clothoid][elevation][bank]")
{
    using namespace mbd;

    Track t;
    t.add_clothoid(50.0, 0.0, 0.01, 5.0, 0.1);

    auto p_mid = t.query(25.0);
    REQUIRE_THAT(p_mid.z, WithinAbs(2.5, eps));
    REQUIRE_THAT(p_mid.bank, WithinAbs(0.1, eps));
    REQUIRE_THAT(p_mid.kappa, WithinAbs(0.005, eps));
}

TEST_CASE("Track: zero-length clothoid is rejected", "[track][clothoid][validation]")
{
    using namespace mbd;

    Track t;
    REQUIRE_THROWS_AS(t.add_clothoid(0.0, 0.0, 0.01), MbdError);
    REQUIRE_THROWS_AS(t.add_clothoid(-1.0, 0.0, 0.01), MbdError);
}