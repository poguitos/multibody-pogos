#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/multilink.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;
}

TEST_CASE("Multilink: reference configuration satisfies all constraints",
          "[multilink][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto ml = build_multilink_corner(sys);
    set_multilink_reference(sys, ml);

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Multilink: upright at wheel center at reference", "[multilink][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    set_multilink_reference(sys, ml);

    REQUIRE_THAT(sys.states[ml.upright_body].p_WB.x(),
                 WithinAbs(p.wheel_center.x(), 1e-10));
    REQUIRE_THAT(sys.states[ml.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y(), 1e-10));
    REQUIRE_THAT(sys.states[ml.upright_body].p_WB.z(),
                 WithinAbs(p.wheel_center.z(), 1e-10));
}

TEST_CASE("Multilink: camber and toe zero at reference", "[multilink][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto ml = build_multilink_corner(sys);
    set_multilink_reference(sys, ml);

    REQUIRE_THAT(extract_camber(sys.states[ml.upright_body]), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(sys.states[ml.upright_body]), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Multilink: 6 DOF and 6 constraints", "[multilink][topology]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto ml = build_multilink_corner(sys);

    REQUIRE(sys.total_dof == 6);
    REQUIRE(sys.constraints.size() == 6);
}

TEST_CASE("Multilink: Newton-Raphson converges for 20mm bump", "[multilink][solver]")
{
    using namespace mbd;

    MultibodySystem sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    set_multilink_reference(sys, ml);

    auto* h = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[ml.bump_constraint_idx].get());
    h->target = p.wheel_center.y() + 0.02;

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    REQUIRE_THAT(sys.states[ml.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() + 0.02, 1e-8));

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-8));
}

TEST_CASE("Multilink: converges for 20mm droop", "[multilink][solver]")
{
    using namespace mbd;

    MultibodySystem sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    set_multilink_reference(sys, ml);

    auto* h = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[ml.bump_constraint_idx].get());
    h->target = p.wheel_center.y() - 0.02;

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    REQUIRE_THAT(sys.states[ml.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() - 0.02, 1e-8));
}

TEST_CASE("Multilink: kinematic sweep converges over full range",
          "[multilink][sweep]")
{
    using namespace mbd;

    MultibodySystem sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    set_multilink_reference(sys, ml);

    auto result = sweep_bump_travel(
        sys, ml.bump_constraint_idx, ml.upright_body,
        p.wheel_center.y(), -0.04, 0.04, 21);

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

    MultibodySystem sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    set_multilink_reference(sys, ml);

    auto result = sweep_bump_travel(
        sys, ml.bump_constraint_idx, ml.upright_body,
        p.wheel_center.y(), -0.03, 0.03, 11);

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

    MultibodySystem sys;
    MultilinkParams p;
    auto ml = build_multilink_corner(sys, p);
    set_multilink_reference(sys, ml);

    auto result = sweep_bump_travel(
        sys, ml.bump_constraint_idx, ml.upright_body,
        p.wheel_center.y(), -0.03, 0.03, 11);

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

    MultibodySystem sys_a;
    auto ml_a = build_multilink_corner(sys_a, p_a);
    set_multilink_reference(sys_a, ml_a);

    auto result_a = sweep_bump_travel(
        sys_a, ml_a.bump_constraint_idx, ml_a.upright_body,
        p_a.wheel_center.y(), -0.03, 0.03, 11);

    // Config B: make upper links longer by moving inner points outward
    MultilinkParams p_b = p_a;
    p_b.inner[2] = Vec3( 0.15, 0.42, 0.50);  // upper front inner moved outboard
    p_b.inner[3] = Vec3(-0.15, 0.42, 0.50);  // upper rear inner moved outboard

    MultibodySystem sys_b;
    auto ml_b = build_multilink_corner(sys_b, p_b);
    set_multilink_reference(sys_b, ml_b);

    auto result_b = sweep_bump_travel(
        sys_b, ml_b.bump_constraint_idx, ml_b.upright_body,
        p_b.wheel_center.y(), -0.03, 0.03, 11);

    // Both should converge
    for (const auto& pt : result_a.points) { REQUIRE(pt.converged); }
    for (const auto& pt : result_b.points) { REQUIRE(pt.converged); }

    // They should have different camber gain
    Real gain_a = result_a.camber_gain();
    Real gain_b = result_b.camber_gain();

    const Real diff = std::abs(gain_a - gain_b);
    REQUIRE(diff > 0.01); // Measurably different
}

TEST_CASE("Multilink: CSV export", "[multilink][csv]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto ml = build_multilink_corner(sys);
    set_multilink_reference(sys, ml);

    auto result = sweep_bump_travel(
        sys, ml.bump_constraint_idx, ml.upright_body,
        ml.params.wheel_center.y(), -0.02, 0.02, 5);

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