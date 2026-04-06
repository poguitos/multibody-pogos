#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/double_wishbone.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-9;
    constexpr mbd::Real deg = mbd::pi / 180.0;
}

// ============================================================================
// Reference configuration
// ============================================================================

TEST_CASE("DWB: reference configuration satisfies all constraints",
          "[dwb][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto dwb = build_double_wishbone_corner(sys);
    set_dwb_reference(sys, dwb);

    VecX phi = evaluate_all_constraints(sys);

    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB: body positions at reference configuration",
          "[dwb][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    // LCA body origin at pivot
    REQUIRE_THAT(sys.states[dwb.lca_body].p_WB.y(),
                 WithinAbs(p.lca_pivot.y(), 1e-10));
    REQUIRE_THAT(sys.states[dwb.lca_body].p_WB.z(),
                 WithinAbs(p.lca_pivot.z(), 1e-10));

    // UCA body origin at pivot
    REQUIRE_THAT(sys.states[dwb.uca_body].p_WB.y(),
                 WithinAbs(p.uca_pivot.y(), 1e-10));
    REQUIRE_THAT(sys.states[dwb.uca_body].p_WB.z(),
                 WithinAbs(p.uca_pivot.z(), 1e-10));

    // Upright at wheel center
    REQUIRE_THAT(sys.states[dwb.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y(), 1e-10));
    REQUIRE_THAT(sys.states[dwb.upright_body].p_WB.z(),
                 WithinAbs(p.wheel_center.z(), 1e-10));
}

TEST_CASE("DWB: camber and toe are zero at reference",
          "[dwb][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto dwb = build_double_wishbone_corner(sys);
    set_dwb_reference(sys, dwb);

    Real camber = extract_camber(sys.states[dwb.upright_body]);
    Real toe    = extract_toe(sys.states[dwb.upright_body]);

    REQUIRE_THAT(camber, WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(toe, WithinAbs(0.0, 1e-10));
}

// ============================================================================
// Position solver
// ============================================================================

TEST_CASE("DWB: Newton-Raphson converges from reference",
          "[dwb][solver]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto dwb = build_double_wishbone_corner(sys);
    set_dwb_reference(sys, dwb);

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB: Newton-Raphson converges for 20mm bump",
          "[dwb][solver]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    // Prescribe 20mm bump (wheel up)
    auto* height_con = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[dwb.bump_constraint_idx].get());
    height_con->target = p.wheel_center.y() + 0.02;

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    // Wheel center should be at target height
    REQUIRE_THAT(sys.states[dwb.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() + 0.02, 1e-8));

    // All constraints satisfied
    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-8));
}

TEST_CASE("DWB: Newton-Raphson converges for 20mm droop",
          "[dwb][solver]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    auto* height_con = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[dwb.bump_constraint_idx].get());
    height_con->target = p.wheel_center.y() - 0.02;

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    REQUIRE_THAT(sys.states[dwb.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() - 0.02, 1e-8));
}

// ============================================================================
// Camber gain (the fundamental double-wishbone property)
// ============================================================================

TEST_CASE("DWB: negative camber gain in bump (unequal-length arms)",
          "[dwb][camber]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    // Verify UCA is shorter than LCA (necessary for negative camber gain)
    const Real lca_span = (p.lca_outer - p.lca_pivot).norm();
    const Real uca_span = (p.uca_outer - p.uca_pivot).norm();
    REQUIRE(uca_span < lca_span);

    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    // --- Bump: 30mm compression (wheel up = Y increases in chassis frame) ---
    auto* height_con = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[dwb.bump_constraint_idx].get());
    height_con->target = p.wheel_center.y() + 0.03;

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    Real camber_bump = extract_camber(sys.states[dwb.upright_body]);

    // Negative camber in bump: top of wheel tilts inward (-Z for left wheel)
    REQUIRE(camber_bump < -0.1 * deg);

    // --- Droop: 30mm extension (wheel down = Y decreases in chassis frame) ---
    set_dwb_reference(sys, dwb);
    height_con->target = p.wheel_center.y() - 0.03;

    ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    Real camber_droop = extract_camber(sys.states[dwb.upright_body]);

    // Positive camber in droop
    REQUIRE(camber_droop > 0.1 * deg);

    // Camber change should be roughly antisymmetric
    REQUIRE(std::abs(camber_bump + camber_droop) < std::abs(camber_bump));
}

// ============================================================================
// Full kinematic sweep
// ============================================================================

TEST_CASE("DWB: kinematic sweep produces valid results",
          "[dwb][sweep]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    // Sweep from -40mm droop to +40mm bump
    auto result = sweep_bump_travel(
        sys, dwb.bump_constraint_idx, dwb.upright_body,
        p.wheel_center.y(),
        -0.04, 0.04,
        21);

    REQUIRE(result.points.size() == 21);

    // All points should converge
    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
    }

    // At zero bump (middle point), camber should be near zero
    const auto& mid = result.points[10];
    REQUIRE_THAT(mid.bump, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(mid.camber, WithinAbs(0.0, 0.01 * deg));

    // Camber gain should be negative (key design property)
    REQUIRE(result.camber_gain() < 0.0);

    // Camber should be monotonically decreasing with increasing bump
    for (size_t i = 1; i < result.points.size(); ++i) {
        REQUIRE(result.points[i].camber <= result.points[i - 1].camber + 0.01 * deg);
    }
}

TEST_CASE("DWB: toe change is small (well-designed tie rod)",
          "[dwb][toe]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    auto result = sweep_bump_travel(
        sys, dwb.bump_constraint_idx, dwb.upright_body,
        p.wheel_center.y(),
        -0.03, 0.03,
        11);

    // Toe change should be small over ±30mm travel (well-designed geometry)
    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
        REQUIRE(std::abs(pt.toe) < 2.0 * deg);
    }
}

// ============================================================================
// Geometric extraction
// ============================================================================

TEST_CASE("Camber extraction: identity rotation gives zero camber",
          "[kinematics][camber]")
{
    mbd::RigidBodyState s;
    s.q_WB = mbd::Quat::Identity();

    REQUIRE_THAT(mbd::extract_camber(s), WithinAbs(0.0, eps));
}

TEST_CASE("Camber extraction: 5 deg tilt gives correct camber",
          "[kinematics][camber]")
{
    using namespace mbd;

    // Tilt wheel 5 degrees: rotate body frame about X by 5 deg
    // This tilts body Y toward +Z (positive camber for left wheel)
    RigidBodyState s;
    s.q_WB = Quat(Eigen::AngleAxisd(5.0 * deg, Vec3::UnitX()));

    Real camber = extract_camber(s);
    REQUIRE_THAT(camber, WithinAbs(5.0 * deg, 0.01 * deg));
}

TEST_CASE("Toe extraction: identity rotation gives zero toe",
          "[kinematics][toe]")
{
    mbd::RigidBodyState s;
    s.q_WB = mbd::Quat::Identity();

    REQUIRE_THAT(mbd::extract_toe(s), WithinAbs(0.0, eps));
}

TEST_CASE("Toe extraction: 2 deg yaw gives correct toe",
          "[kinematics][toe]")
{
    using namespace mbd;

    // Ry(-2 deg) rotates body X toward +Z, giving positive toe.
    // (Ry(+2 deg) would rotate body X toward -Z, giving negative toe.)
    RigidBodyState s;
    s.q_WB = Quat(Eigen::AngleAxisd(-2.0 * deg, Vec3::UnitY()));

    Real toe = extract_toe(s);
    REQUIRE_THAT(toe, WithinAbs(2.0 * deg, 0.01 * deg));
}

// ============================================================================
// CSV export (smoke test — just verifies no crash)
// ============================================================================

TEST_CASE("DWB: CSV export writes a file", "[dwb][csv]")
{
    using namespace mbd;

    MultibodySystem sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    set_dwb_reference(sys, dwb);

    auto result = sweep_bump_travel(
        sys, dwb.bump_constraint_idx, dwb.upright_body,
        p.wheel_center.y(),
        -0.02, 0.02,
        5);

    // Export to a temporary file
    result.export_csv("test_dwb_sweep.csv");

    // Verify file exists and has content
    std::ifstream file("test_dwb_sweep.csv");
    REQUIRE(file.good());

    std::string header;
    std::getline(file, header);
    REQUIRE(header.find("bump_mm") != std::string::npos);

    int line_count = 0;
    std::string line;
    while (std::getline(file, line)) {
        ++line_count;
    }
    REQUIRE(line_count == 5);
}