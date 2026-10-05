#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/vehicle/suspension/double_wishbone.hpp"

// Kinematics of the double-wishbone corner, on the kernel (plan task 2.7).
// The corner's bump driver holds the wheel-centre height at its nominal value
// plus k.t, so setting k.t prescribes the bump travel.

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

    kernel::System sys;
    build_double_wishbone_corner(sys);
    Kinematics k(sys);

    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB: body positions at reference configuration",
          "[dwb][reference]")
{
    using namespace mbd;

    kernel::System sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    // LCA body origin at pivot
    REQUIRE_THAT(k.state(dwb.lca_body).p_WB.y(), WithinAbs(p.lca_pivot.y(), 1e-10));
    REQUIRE_THAT(k.state(dwb.lca_body).p_WB.z(), WithinAbs(p.lca_pivot.z(), 1e-10));

    // UCA body origin at pivot
    REQUIRE_THAT(k.state(dwb.uca_body).p_WB.y(), WithinAbs(p.uca_pivot.y(), 1e-10));
    REQUIRE_THAT(k.state(dwb.uca_body).p_WB.z(), WithinAbs(p.uca_pivot.z(), 1e-10));

    // Upright at wheel center
    REQUIRE_THAT(k.state(dwb.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y(), 1e-10));
    REQUIRE_THAT(k.state(dwb.upright_body).p_WB.z(), WithinAbs(p.wheel_center.z(), 1e-10));
}

TEST_CASE("DWB: camber and toe are zero at reference",
          "[dwb][reference]")
{
    using namespace mbd;

    kernel::System sys;
    auto dwb = build_double_wishbone_corner(sys);
    Kinematics k(sys);

    REQUIRE_THAT(extract_camber(k.state(dwb.upright_body)), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(k.state(dwb.upright_body)), WithinAbs(0.0, 1e-10));
}

// ============================================================================
// Position solver
// ============================================================================

TEST_CASE("DWB: Newton-Raphson converges from reference",
          "[dwb][solver]")
{
    using namespace mbd;

    kernel::System sys;
    build_double_wishbone_corner(sys);
    Kinematics k(sys);

    REQUIRE(k.solve());
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB: Newton-Raphson converges for 20mm bump",
          "[dwb][solver]")
{
    using namespace mbd;

    kernel::System sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    // Prescribe 20mm bump (wheel up)
    k.t = 0.02;
    REQUIRE(k.solve());

    // Wheel center at target height, all constraints satisfied
    REQUIRE_THAT(k.state(dwb.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y() + 0.02, 1e-8));
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-8));
}

TEST_CASE("DWB: Newton-Raphson converges for 20mm droop",
          "[dwb][solver]")
{
    using namespace mbd;

    kernel::System sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    k.t = -0.02;
    REQUIRE(k.solve());

    REQUIRE_THAT(k.state(dwb.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y() - 0.02, 1e-8));
}

// ============================================================================
// Camber gain (the fundamental double-wishbone property)
// ============================================================================

TEST_CASE("DWB: negative camber gain in bump (unequal-length arms)",
          "[dwb][camber]")
{
    using namespace mbd;

    kernel::System sys;
    DoubleWishboneParams p;
    // Verify UCA is shorter than LCA (necessary for negative camber gain)
    const Real lca_span = (p.lca_outer - p.lca_pivot).norm();
    const Real uca_span = (p.uca_outer - p.uca_pivot).norm();
    REQUIRE(uca_span < lca_span);

    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    // --- Bump: 30mm compression (wheel up = Y increases in chassis frame) ---
    k.t = 0.03;
    REQUIRE(k.solve());
    const Real camber_bump = extract_camber(k.state(dwb.upright_body));

    // Negative camber in bump: top of wheel tilts inward (-Z for left wheel)
    REQUIRE(camber_bump < -0.1 * deg);

    // --- Droop: 30mm extension, from the reference ---
    k.q = sys.model.neutral_configuration();
    k.t = -0.03;
    REQUIRE(k.solve());
    const Real camber_droop = extract_camber(k.state(dwb.upright_body));

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

    kernel::System sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    // Sweep from -40mm droop to +40mm bump
    auto result = sweep_bump_travel(k, dwb.upright_body, -0.04, 0.04, 21);

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

    kernel::System sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, dwb.upright_body, -0.03, 0.03, 11);

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

    kernel::System sys;
    DoubleWishboneParams p;
    auto dwb = build_double_wishbone_corner(sys, p);
    Kinematics k(sys);

    auto result = sweep_bump_travel(k, dwb.upright_body, -0.02, 0.02, 5);

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

// ============================================================================
// Kinematic analysis along a prescribed motion (plan task 3.6)
// ============================================================================

TEST_CASE("Kinematic analysis: a sweep gives the configurations of solving with a fixed target",
          "[dwb][kinematics]")
{
    // Before the kernel, a bump sweep edited the bump constraint's target
    // between solves. Now the target is nominal + t and the sweep varies t.
    // The old way is rebuilt here, a target held in a variable and edited
    // before each solve, and from the same starting point it must reach the
    // same configuration: the position equations are the same, so the
    // iterations are too.
    using namespace mbd;
    const DoubleWishboneParams p;

    kernel::System swept;
    const auto dwb = build_double_wishbone_corner(swept, p);
    Kinematics k(swept);

    kernel::System edited;
    build_double_wishbone_corner(edited, p);
    Real target = p.wheel_center.y();
    edited.constraints[dwb.bump_constraint_idx] = std::make_shared<kernel::Dot2>(
        kernel::Marker{0, Transform3::Identity()}, 1,
        kernel::Marker{dwb.upright_body, Transform3::Identity()},
        kernel::TimeFunction([&target](Real) { return target; },
                             [](Real) { return 0.0; },
                             [](Real) { return 0.0; }));
    Kinematics old_way(edited);

    VecX q_start = k.q;
    Real worst = 0.0;
    for (int i = 0; i <= 10; ++i) {
        const Real bump = -0.05 + 0.01 * i;
        k.t = bump;
        k.q = q_start;
        REQUIRE(k.solve());

        target = p.wheel_center.y() + bump;
        old_way.q = q_start;
        REQUIRE(old_way.solve());

        worst = std::max(worst, (k.q - old_way.q).cwiseAbs().maxCoeff());
        q_start = k.q;
    }
    CHECK(worst < 1e-12);
}

TEST_CASE("Kinematic analysis: velocities and accelerations along the bump motion",
          "[dwb][kinematics]")
{
    // With t the bump travel, the wheel centre rises at exactly 1 per unit of
    // t, with no acceleration. The joint velocities must also be the
    // derivatives of the configurations with respect to t: central
    // differences with h = 1e-4 have a truncation error of order
    // h^2 = 1e-8 times the third derivative, and the solves are converged to
    // 1e-14, which adds 1e-14 / (2 h) = 5e-11.
    using namespace mbd;
    kernel::System sys;
    const auto dwb = build_double_wishbone_corner(sys);
    Kinematics k(sys);

    const Real t0 = 0.02, h = 1e-4;
    k.t = t0;
    REQUIRE(k.solve(100, 1e-14));
    const VecX q_mid = k.q;
    const VecX v = k.velocities();
    const VecX a = k.accelerations();

    kernel::Data data(sys.model);
    kernel::forward_kinematics(sys.model, data, q_mid, v, a);
    const Vec6 wheel_velocity = kernel::body_velocity_world(data, dwb.upright_body);
    const Vec6 wheel_acceleration = kernel::body_acceleration_world(data, dwb.upright_body);
    CHECK(std::abs(wheel_velocity(4) - 1.0) < 1e-9);    // world Y, linear part
    CHECK(std::abs(wheel_acceleration(4)) < 1e-8);

    k.t = t0 + h;
    REQUIRE(k.solve(100, 1e-14));
    const VecX q_plus = k.q;
    k.t = t0 - h;
    k.q = q_mid;
    REQUIRE(k.solve(100, 1e-14));
    const VecX q_minus = k.q;
    VecX dq;
    kernel::difference(sys.model, q_minus, q_plus, dq);
    INFO("velocities " << v.transpose() << ", differences " << (dq / (2.0 * h)).transpose());
    CHECK((dq / (2.0 * h) - v).norm() < 1e-6 * v.norm());
}
