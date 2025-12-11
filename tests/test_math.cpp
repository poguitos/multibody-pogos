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

TEST_CASE("skew matrix reproduces cross product", "[math]")
{
    constexpr double tol = 1e-12;

    const mbd::Vec3 v(1.2, -0.5, 2.3);
    const mbd::Vec3 w(-0.7, 0.9, 1.1);

    const mbd::Mat3 S = mbd::skew(v);

    const mbd::Vec3 vw_cross = v.cross(w);   // reference: v × w
    const mbd::Vec3 vw_skew  = S * w;        // test: [v]× w

    REQUIRE_THAT(vw_skew.x(), WithinAbs(vw_cross.x(), tol));
    REQUIRE_THAT(vw_skew.y(), WithinAbs(vw_cross.y(), tol));
    REQUIRE_THAT(vw_skew.z(), WithinAbs(vw_cross.z(), tol));
}

TEST_CASE("integrate_quat integrates small constant yaw rate", "[math]")
{
    constexpr double dt       = 0.01;
    constexpr double yaw_rate = 1.0;      // rad/s
    constexpr double tol      = 1e-12;

    const mbd::Vec3 omega(0.0, 0.0, yaw_rate); // body angular velocity
    const mbd::Quat q0 = mbd::Quat::Identity();

    const mbd::Quat q1 = mbd::integrate_quat(q0, omega, dt);

    // For small dt, rotation angle is |omega| * dt around z:
    const double half_angle = 0.5 * yaw_rate * dt;

    // Expected quaternion for pure yaw about +Z:
    const double w_expected = std::cos(half_angle);
    const double z_expected = std::sin(half_angle);

    REQUIRE_THAT(q1.w(), WithinAbs(w_expected, tol));
    REQUIRE_THAT(q1.x(), WithinAbs(0.0,       tol));
    REQUIRE_THAT(q1.y(), WithinAbs(0.0,       tol));
    REQUIRE_THAT(q1.z(), WithinAbs(z_expected, tol));

    // Ensure we stay on S^3
    REQUIRE_THAT(q1.norm(), WithinAbs(1.0, 1e-12));
}


