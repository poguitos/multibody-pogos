#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mbd/math.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("skew(v) reproduces the cross product", "[math]")
{
    mbd::Vec3 v(1.0, 2.0, 3.0);
    mbd::Vec3 w(-4.0, 5.0, -6.0);

    mbd::Vec3 vw_cross = v.cross(w);
    mbd::Vec3 vw_skew  = mbd::skew(v) * w;

    constexpr double tol = 1e-12;
    REQUIRE_THAT(vw_cross.x(), WithinAbs(vw_skew.x(), tol));
    REQUIRE_THAT(vw_cross.y(), WithinAbs(vw_skew.y(), tol));
    REQUIRE_THAT(vw_cross.z(), WithinAbs(vw_skew.z(), tol));
}

TEST_CASE("integrate_quat preserves unit norm for small rotation", "[math]")
{
    mbd::Quat q0 = mbd::Quat::Identity();

    // Rotate a bit around z
    mbd::Vec3 omega(0.0, 0.0, 1.0); // rad/s
    mbd::Real dt = 0.01;            // s

    mbd::Quat q1 = mbd::integrate_quat(q0, omega, dt);

    double norm = q1.norm();
    REQUIRE_THAT(norm, WithinAbs(1.0, 1e-12));
}
