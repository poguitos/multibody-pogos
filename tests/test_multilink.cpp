#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include <fstream>
#include <string>

#include "mbd/vehicle/suspension/multilink.hpp"

// Kinematics of the multilink corner, on the kernel (plan task 2.7). k.t is
// the bump travel prescribed by the corner's driver.

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;
}

TEST_CASE("Multilink: reference configuration satisfies all constraints",
          "[multilink][reference]")
{
    using namespace mbd;

    kernel::System sys;
    build_multilink_corner(sys);
    Kinematics k(sys);

    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Multilink: upright at wheel center at reference", "[multilink][reference]")
{
    using namespace mbd;

    kernel::System sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    Kinematics k(sys);

    REQUIRE_THAT(k.state(ml.upright_body).p_WB.x(), WithinAbs(p.wheel_center.x(), 1e-10));
    REQUIRE_THAT(k.state(ml.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y(), 1e-10));
    REQUIRE_THAT(k.state(ml.upright_body).p_WB.z(), WithinAbs(p.wheel_center.z(), 1e-10));
}

TEST_CASE("Multilink: camber and toe zero at reference", "[multilink][reference]")
{
    using namespace mbd;

    kernel::System sys;
    auto ml = build_multilink_corner(sys);
    Kinematics k(sys);

    REQUIRE_THAT(extract_camber(k.state(ml.upright_body)), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(k.state(ml.upright_body)), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Multilink: 6 DOF and 6 constraints", "[multilink][topology]")
{
    using namespace mbd;

    kernel::System sys;
    build_multilink_corner(sys);

    // A free upright: six velocities (seven coordinates with the quaternion).
    REQUIRE(sys.model.nv == 6);
    REQUIRE(sys.model.nq == 7);
    REQUIRE(sys.constraints.size() == 6);
}

TEST_CASE("Multilink: Newton-Raphson converges for 20mm bump", "[multilink][solver]")
{
    using namespace mbd;

    kernel::System sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    Kinematics k(sys);

    k.t = 0.02;
    REQUIRE(k.solve());

    REQUIRE_THAT(k.state(ml.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y() + 0.02, 1e-8));
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-8));
}

TEST_CASE("Multilink: converges for 20mm droop", "[multilink][solver]")
{
    using namespace mbd;

    kernel::System sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    Kinematics k(sys);

    k.t = -0.02;
    REQUIRE(k.solve());

    REQUIRE_THAT(k.state(ml.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y() - 0.02, 1e-8));
}

TEST_CASE("Multilink: kinematic sweep converges over full range",
          "[multilink][sweep]")
{
    using namespace mbd;

    kernel::System sys;
    auto ml = build_multilink_corner(sys);
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, ml.upright_body, -0.04, 0.04, 21);

    REQUIRE(result.points.size() == 21);

    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
    }

    // Mid-sweep should be near zero camber and toe
    const auto& mid = result.points[10];
    REQUIRE_THAT(mid.bump, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(mid.camber, WithinAbs(0.0, 0.1 * deg));
    REQUIRE_THAT(mid.toe, WithinAbs(0.0, 0.1 * deg));
}

TEST_CASE("Multilink: camber variation is bounded",
          "[multilink][camber]")
{
    using namespace mbd;

    kernel::System sys;
    auto ml = build_multilink_corner(sys);
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, ml.upright_body, -0.03, 0.03, 11);

    // Camber should stay within ±5 deg over ±30mm travel
    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
        REQUIRE(std::abs(pt.camber) < 5.0 * deg);
    }
}

TEST_CASE("Multilink: toe variation is bounded",
          "[multilink][toe]")
{
    using namespace mbd;

    kernel::System sys;
    auto ml = build_multilink_corner(sys);
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, ml.upright_body, -0.03, 0.03, 11);

    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
        REQUIRE(std::abs(pt.toe) < 3.0 * deg);
    }
}

TEST_CASE("Multilink: different link lengths produce different camber behavior",
          "[multilink][design]")
{
    using namespace mbd;

    // Config A: upper links shorter than lower (DWB-like, negative camber gain)
    MultilinkParams p_a;
    // Upper links span: sqrt((0.10-0.15)^2 + (0.40-0.42)^2 + (0.68-0.32)^2) ≈ 0.36
    // Lower links span: sqrt((0.20-0.20)^2 + (0.15-0.18)^2 + (0.70-0.25)^2) ≈ 0.45
    // Upper < Lower: should give negative camber gain

    kernel::System sys_a;
    auto ml_a = build_multilink_corner(sys_a, p_a);
    Kinematics k_a(sys_a);
    auto result_a = sweep_bump_travel(k_a, ml_a.upright_body, -0.03, 0.03, 11);

    // Config B: make upper links longer by moving inner points outward
    MultilinkParams p_b = p_a;
    p_b.inner[2] = Vec3( 0.15, 0.42, 0.50);  // upper front inner moved outboard
    p_b.inner[3] = Vec3(-0.15, 0.42, 0.50);  // upper rear inner moved outboard

    kernel::System sys_b;
    auto ml_b = build_multilink_corner(sys_b, p_b);
    Kinematics k_b(sys_b);
    auto result_b = sweep_bump_travel(k_b, ml_b.upright_body, -0.03, 0.03, 11);

    // Both should converge
    for (const auto& pt : result_a.points) { REQUIRE(pt.converged); }
    for (const auto& pt : result_b.points) { REQUIRE(pt.converged); }

    // They should have different camber gain
    const Real diff = std::abs(result_a.camber_gain() - result_b.camber_gain());
    REQUIRE(diff > 0.01); // Measurably different
}

TEST_CASE("Multilink: CSV export", "[multilink][csv]")
{
    using namespace mbd;

    kernel::System sys;
    auto ml = build_multilink_corner(sys);
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, ml.upright_body, -0.02, 0.02, 5);

    result.export_csv("test_multilink_sweep.csv");

    std::ifstream file("test_multilink_sweep.csv");
    REQUIRE(file.good());

    std::string header;
    std::getline(file, header);
    REQUIRE(header.find("bump_mm") != std::string::npos);

    int lines = 0;
    std::string line;
    while (std::getline(file, line)) { ++lines; }
    REQUIRE(lines == 5);
}
