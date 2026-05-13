#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/double_wishbone.hpp"
#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;

    /// Build a "fixed chassis" test fixture: a FixedJoint from ground,
    /// then a DWB corner dynamically attached to it.
    struct DwbFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex chassis_body{0};
        mbd::DoubleWishboneCorner dwb;
        int chassis_q_idx_start{-1};
    };

    DwbFixture make_fixed_chassis_dwb(const mbd::DoubleWishboneParams& p)
    {
        using namespace mbd;
        DwbFixture fx;

        // Chassis as a FIXED body (0 DOF) via FixedJoint from ground.
        // This rigidly pins the chassis at identity pose, so the DWB corner
        // mechanism has exactly 1 net DOF (bump travel), identical to the
        // ground-parented kinematic builder.
        auto I_chassis = RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8));
        fx.chassis_body = fx.sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
        fx.sys.add_joint(std::make_unique<FixedJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.chassis_body));
        fx.chassis_q_idx_start = -1; // no q for chassis

        fx.dwb = build_double_wishbone_corner_dynamic(fx.sys, fx.chassis_body, p);

        return fx;
    }
}

// ============================================================================
// Reference configuration: constraints satisfied with chassis at identity
// ============================================================================

TEST_CASE("DWB dynamic: reference configuration satisfies all constraints",
          "[dwb_dyn][reference]")
{
    using namespace mbd;

    DoubleWishboneParams p;
    auto fx = make_fixed_chassis_dwb(p);

    // Chassis at identity, all suspension q = 0
    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    VecX phi = evaluate_all_constraints(fx.sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB dynamic: body positions at reference", "[dwb_dyn][reference]")
{
    using namespace mbd;

    DoubleWishboneParams p;
    auto fx = make_fixed_chassis_dwb(p);

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    // Chassis at origin
    REQUIRE_THAT(fx.sys.states[fx.chassis_body].p_WB.norm(), WithinAbs(0.0, 1e-10));

    // LCA origin at p.lca_pivot
    const auto& lca_state = fx.sys.states[fx.dwb.lca_body];
    REQUIRE_THAT((lca_state.p_WB - p.lca_pivot).norm(), WithinAbs(0.0, 1e-10));

    // UCA origin at p.uca_pivot
    const auto& uca_state = fx.sys.states[fx.dwb.uca_body];
    REQUIRE_THAT((uca_state.p_WB - p.uca_pivot).norm(), WithinAbs(0.0, 1e-10));

    // Upright origin at p.wheel_center
    const auto& upr_state = fx.sys.states[fx.dwb.upright_body];
    REQUIRE_THAT((upr_state.p_WB - p.wheel_center).norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB dynamic: camber and toe zero at reference",
          "[dwb_dyn][reference]")
{
    using namespace mbd;

    DoubleWishboneParams p;
    auto fx = make_fixed_chassis_dwb(p);

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    const auto& upr = fx.sys.states[fx.dwb.upright_body];
    REQUIRE_THAT(extract_camber(upr), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(upr), WithinAbs(0.0, 1e-10));
}

// ============================================================================
// Chassis translation: corner moves rigidly with chassis
// ============================================================================

TEST_CASE("DWB dynamic: chassis translation carries the corner",
          "[dwb_dyn][chassis_motion]")
{
    using namespace mbd;

    // For this test, use a FreeCoordJoint chassis so we can translate it
    DoubleWishboneParams p;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, chassis));

    auto dwb = build_double_wishbone_corner_dynamic(sys, chassis, p);

    // Translate chassis by (0.1, 0.2, 0.3)
    sys.q.setZero();
    sys.q(0) = 0.1;
    sys.q(1) = 0.2;
    sys.q(2) = 0.3;
    sys.compute_kinematics();

    const Vec3 offset(0.1, 0.2, 0.3);

    REQUIRE_THAT((sys.states[dwb.lca_body].p_WB - (p.lca_pivot + offset)).norm(),
                 WithinAbs(0.0, 1e-9));
    REQUIRE_THAT((sys.states[dwb.uca_body].p_WB - (p.uca_pivot + offset)).norm(),
                 WithinAbs(0.0, 1e-9));
    REQUIRE_THAT((sys.states[dwb.upright_body].p_WB - (p.wheel_center + offset)).norm(),
                 WithinAbs(0.0, 1e-9));

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// Single DOF: with chassis fixed, the mechanism has 1 DOF (bump travel)
// ============================================================================

TEST_CASE("DWB dynamic: mechanism has correct DOF counts",
          "[dwb_dyn][dof]")
{
    using namespace mbd;

    DoubleWishboneParams p;
    auto fx = make_fixed_chassis_dwb(p);

    // Tree DOFs: 0 (fixed chassis) + 1 (LCA rev) + 3 (spherical) + 1 (UCA rev) = 5
    REQUIRE(fx.sys.total_dof == 5);

    // Constraints: 3 (coincident) + 1 (tie rod) = 4 loop equations
    REQUIRE(fx.sys.constraints.size() == 2);
    int total_eqs = 0;
    for (const auto& c : fx.sys.constraints) total_eqs += c->equation_count();
    REQUIRE(total_eqs == 4);

    // Net DOF = 5 - 4 = 1 (bump travel)
}

// ============================================================================
// Bump travel: prescribing wheel Y moves the mechanism consistently
// ============================================================================

TEST_CASE("DWB dynamic: prescribed wheel Y triggers consistent motion",
          "[dwb_dyn][bump]")
{
    using namespace mbd;

    DoubleWishboneParams p;
    auto fx = make_fixed_chassis_dwb(p);

    // Bump prescription on wheel center Y
    const Real bump = 0.02;
    const size_t bump_idx = fx.sys.constraints.size();
    fx.sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        fx.dwb.upright_body, Vec3::Zero(), 1, p.wheel_center.y() + bump));

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    bool ok = solve_position_kinematics(fx.sys, 100, 1e-8);
    REQUIRE(ok);

    // Wheel at prescribed Y
    REQUIRE_THAT(fx.sys.states[fx.dwb.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() + bump, 1e-6));

    // Camber changed
    const Real camber = extract_camber(fx.sys.states[fx.dwb.upright_body]);
    REQUIRE(std::abs(camber) > 0.1 * deg);

    // All constraints satisfied
    VecX phi = evaluate_all_constraints(fx.sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-6));
}

// ============================================================================
// Comparison with kinematic builder: same geometry should give same sweep
// ============================================================================
TEST_CASE("DWB dynamic: sweep matches ground-parented kinematic builder",
          "[dwb_dyn][comparison]")
{
    using namespace mbd;

    DoubleWishboneParams p;

    // --- Dynamic version (parented to fixed chassis) ---
    auto fx = make_fixed_chassis_dwb(p);

    const size_t bump_idx = fx.sys.constraints.size();
    fx.sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        fx.dwb.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    auto dyn_sweep = sweep_bump_travel(
        fx.sys, bump_idx, fx.dwb.upright_body,
        p.wheel_center.y(), -0.02, 0.02, 11);

    for (const auto& pt : dyn_sweep.points) {
        REQUIRE(pt.converged);
    }

    // --- Kinematic version (parented to ground) ---
    MultibodySystem sys_kin;
    auto dwb_kin = build_double_wishbone_corner(sys_kin, p);
    set_dwb_reference(sys_kin, dwb_kin);

    auto kin_sweep = sweep_bump_travel(
        sys_kin, dwb_kin.bump_constraint_idx, dwb_kin.upright_body,
        p.wheel_center.y(), -0.02, 0.02, 11);

    for (const auto& pt : kin_sweep.points) {
        REQUIRE(pt.converged);
    }

    // Camber curves should match closely
    for (size_t i = 0; i < dyn_sweep.points.size(); ++i) {
        REQUIRE_THAT(dyn_sweep.points[i].camber,
                     WithinAbs(kin_sweep.points[i].camber, 0.05 * deg));
    }
}