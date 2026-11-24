// The three-dimensional surface of a lap track as a road (plan task 7.2,
// D61): points placed at a known arc length and lateral distance from the
// centreline are projected back to them, the surface's height there is the
// centreline's elevation less the distance times the banking's tangent, and
// its normal is that of the heights.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

#include "mbd/analysis/track_surface.hpp"

using namespace mbd;
using Catch::Matchers::WithinAbs;

namespace {

/// The normal of the surface's heights by central differences of step
/// 1e-5 m: the projection's rounding (about 1e-15 m in height) divided by the
/// step stays near 1e-10, and the truncation, h^2 times the third derivative
/// of heights curved on the scale of the turns' radii, far below.
Vec3 normal_by_differences(const Road& road, Real x, Real y)
{
    const Real h = 1e-5;
    const Real dx = (road.at(x + h, y).height - road.at(x - h, y).height) / (2.0 * h);
    const Real dy = (road.at(x, y + h).height - road.at(x, y - h).height) / (2.0 * h);
    return Vec3(-dx, -dy, 1.0).normalized();
}

} // namespace

TEST_CASE("Track surface: a sloped, banked straight", "[track][road]")
{
    // A straight along +X from the origin, 100 m rising 5 m (slope 0.05),
    // banked 0.1 rad (the left side lower). At s = 30 and 2 m to the left,
    // the point is (30, 2); the surface there is 0.05 * 30 - 2 tan(0.1) high,
    // and the plane h = 0.05 x - y tan(0.1) has the normal
    // (-0.05, tan(0.1), 1), normalised.
    Track t;
    t.add_straight(100.0, 5.0, 0.1);
    const TrackSurfaceRoad road(t, 0.9);
    Real s = 0.0, d = 0.0;
    road.project(30.0, 2.0, s, d);
    CHECK_THAT(s, WithinAbs(30.0, 1e-12));
    CHECK_THAT(d, WithinAbs(2.0, 1e-12));
    const RoadPoint r = road.at(30.0, 2.0);
    CHECK_THAT(r.height, WithinAbs(1.5 - 2.0 * std::tan(0.1), 1e-12));
    CHECK((r.normal - Vec3(-0.05, std::tan(0.1), 1.0).normalized()).norm() < 1e-14);
    CHECK(r.friction == 0.9);
}

TEST_CASE("Track surface: a banked left turn, its outside higher", "[track][road]")
{
    // A left turn of radius 50 m from the origin heading +X: its centre is at
    // (0, 50), and at arc length s the centreline is at angle th = s / 50,
    // (50 sin th, 50 - 50 cos th), its left normal (-sin th, cos th). A point
    // 3 m to the right (d = -3, outside the turn) on a banking of 0.08 rad
    // (the inside lower) is 3 tan(0.08) high.
    Track t;
    t.add_arc_by_angle(pi / 2.0, 50.0, 0.0, 0.08);
    const TrackSurfaceRoad road(t);
    for (const Real s_true : {5.0, 20.0, 60.0}) {
        for (const Real d_true : {-3.0, 0.0, 4.0}) {
            const Real th = s_true / 50.0;
            const Real x = 50.0 * std::sin(th) - d_true * std::sin(th);
            const Real y = 50.0 - 50.0 * std::cos(th) + d_true * std::cos(th);
            Real s = 0.0, d = 0.0;
            road.project(x, y, s, d);
            CHECK_THAT(s, WithinAbs(s_true, 1e-10));
            CHECK_THAT(d, WithinAbs(d_true, 1e-10));
            const RoadPoint r = road.at(x, y);
            CHECK_THAT(r.height, WithinAbs(-d_true * std::tan(0.08), 1e-10));
            INFO("s " << s_true << ", d " << d_true << ": normal " << r.normal.transpose() << ", from heights "
                      << normal_by_differences(road, x, y).transpose());
            CHECK((r.normal - normal_by_differences(road, x, y)).norm() < 1e-8);
        }
    }
}

TEST_CASE("Track surface: elevation along a turn, and a closed track queried across its start",
          "[track][road]")
{
    // A climbing right turn: 40 m of a radius 30 m right turn rising 2 m.
    // Its elevation at s is 2 s / 40; on the centreline the height is that.
    Track climb;
    climb.add_straight(10.0);
    climb.add_arc(40.0, -30.0, 2.0);
    const TrackSurfaceRoad climbing(climb);
    for (const Real s_true : {15.0, 30.0, 45.0}) {
        const TrackPoint c = climb.query(s_true);
        const RoadPoint r = climbing.at(c.x, c.y);
        CHECK_THAT(r.height, WithinAbs(2.0 * (s_true - 10.0) / 40.0, 1e-10));
        CHECK((r.normal - normal_by_differences(climbing, c.x, c.y)).norm() < 1e-8);
    }

    // An oval: 200 m straights and 60 m half circles, closed, starting at the
    // origin along +X; its last half circle, centred at (0, 60), ends there.
    // The point (1, 2), on the first straight, projects to s = 1, d = 2. The
    // point (-1, 2) is on that last turn: (-1, -58) from its centre, so it
    // projects 60 atan(1 / 58) before the end, and lies 60 - sqrt(1 + 58^2)
    // inside the turn (to the left).
    Track oval;
    oval.add_straight(200.0);
    oval.add_arc_by_angle(pi, 60.0);
    oval.add_straight(200.0);
    oval.add_arc_by_angle(pi, 60.0);
    REQUIRE(oval.is_closed(1e-9, 1e-9));
    const TrackSurfaceRoad road(oval);
    const Real L = oval.total_length();
    Real s = 0.0, d = 0.0;
    road.project(1.0, 2.0, s, d);
    CHECK_THAT(s, WithinAbs(1.0, 1e-9));
    CHECK_THAT(d, WithinAbs(2.0, 1e-9));
    road.project(-1.0, 2.0, s, d);
    CHECK_THAT(s, WithinAbs(L - 60.0 * std::atan(1.0 / 58.0), 1e-9));
    CHECK_THAT(d, WithinAbs(60.0 - std::sqrt(1.0 + 58.0 * 58.0), 1e-9));
}

TEST_CASE("Track surface: a track with no segment is refused (MBD-F050)", "[track][road]")
{
    try {
        TrackSurfaceRoad road{Track{}};
        FAIL("no error");
    } catch (const MbdError& e) {
        CHECK(std::string(e.what()).rfind("MBD-F050", 0) == 0);
    }
}
