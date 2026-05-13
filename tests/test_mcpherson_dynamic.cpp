#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/mcpherson.hpp"
#include "mbd/simulator.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;

    struct MCFixture {
        mbd::MultibodySystem sys;
        mbd::BodyIndex chassis_body{0};
        mbd::McPhersonCorner mc;
    };

    MCFixture make_fixed_chassis_mcpherson(const mbd::McPhersonParams& p)
    {
        using namespace mbd;
        MCFixture fx;

        auto I_chassis = RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8));
        fx.chassis_body = fx.sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
        fx.sys.add_joint(std::make_unique<FixedJoint>(
            Transform3::Identity(), Transform3::Identity(),
            kGroundIndex, fx.chassis_body));

        fx.mc = build_mcpherson_corner_dynamic(fx.sys, fx.chassis_body, p);

        return fx;
    }
}

// ============================================================================
// Reference configuration
// ============================================================================

TEST_CASE("McPherson dynamic: reference configuration satisfies all constraints",
          "[mc_dyn][reference]")
{
    using namespace mbd;

    McPhersonParams p;
    auto fx = make_fixed_chassis_mcpherson(p);

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    VecX phi = evaluate_all_constraints(fx.sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("McPherson dynamic: body positions at reference",
          "[mc_dyn][reference]")
{
    using namespace mbd;

    McPhersonParams p;
    auto fx = make_fixed_chassis_mcpherson(p);

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    // LCA origin at p.lca_pivot
    REQUIRE_THAT((fx.sys.states[fx.mc.lca_body].p_WB - p.lca_pivot).norm(),
                 WithinAbs(0.0, 1e-10));

    // Upright origin at p.wheel_center
    REQUIRE_THAT((fx.sys.states[fx.mc.upright_body].p_WB - p.wheel_center).norm(),
                 WithinAbs(0.0, 1e-10));
}

TEST_CASE("McPherson dynamic: camber and toe zero at reference",
          "[mc_dyn][reference]")
{
    using namespace mbd;

    McPhersonParams p;
    auto fx = make_fixed_chassis_mcpherson(p);

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    const auto& upr = fx.sys.states[fx.mc.upright_body];
    REQUIRE_THAT(extract_camber(upr), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(upr), WithinAbs(0.0, 1e-10));
}

// ============================================================================
// Chassis translation carries the corner
// ============================================================================

TEST_CASE("McPherson dynamic: chassis translation carries the corner",
          "[mc_dyn][chassis_motion]")
{
    using namespace mbd;

    McPhersonParams p;

    MultibodySystem sys;
    auto I_chassis = RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8));
    BodyIndex chassis = sys.add_body(I_chassis, RigidBodyState{}, "chassis", kGroundIndex);
    sys.add_joint(std::make_unique<FreeCoordJoint>(
        Transform3::Identity(), Transform3::Identity(),
        kGroundIndex, chassis));

    auto mc = build_mcpherson_corner_dynamic(sys, chassis, p);

    sys.q.setZero();
    sys.q(0) = 0.1;
    sys.q(1) = 0.2;
    sys.q(2) = 0.3;
    sys.compute_kinematics();

    const Vec3 offset(0.1, 0.2, 0.3);

    REQUIRE_THAT((sys.states[mc.lca_body].p_WB - (p.lca_pivot + offset)).norm(),
                 WithinAbs(0.0, 1e-9));
    REQUIRE_THAT((sys.states[mc.upright_body].p_WB - (p.wheel_center + offset)).norm(),
                 WithinAbs(0.0, 1e-9));

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// DOF count
// ============================================================================

TEST_CASE("McPherson dynamic: correct DOF counts",
          "[mc_dyn][dof]")
{
    using namespace mbd;

    McPhersonParams p;
    auto fx = make_fixed_chassis_mcpherson(p);

    // Tree DOFs: 0 (fixed chassis) + 1 (LCA rev) + 3 (spherical) = 4
    REQUIRE(fx.sys.total_dof == 4);

    // Constraints: 3 (strut line) + 1 (tie rod) = 4 equations (rank 3)
    REQUIRE(fx.sys.constraints.size() == 2);
    int total_eqs = 0;
    for (const auto& c : fx.sys.constraints) total_eqs += c->equation_count();
    REQUIRE(total_eqs == 4);

    // Net DOF = 4 - 3 = 1 (strut line is rank 2, so effectively 3 eqs + 1 tie rod)
}

// ============================================================================
// Bump travel
// ============================================================================

TEST_CASE("McPherson dynamic: prescribed wheel Y triggers consistent motion",
          "[mc_dyn][bump]")
{
    using namespace mbd;

    McPhersonParams p;
    auto fx = make_fixed_chassis_mcpherson(p);

    const Real bump = 0.02;
    const size_t bump_idx = fx.sys.constraints.size();
    fx.sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        fx.mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y() + bump));

    fx.sys.q.setZero();
    fx.sys.q_dot.setZero();
    fx.sys.compute_kinematics();

    bool ok = solve_position_kinematics(fx.sys, 100, 1e-8);
    REQUIRE(ok);

    REQUIRE_THAT(fx.sys.states[fx.mc.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() + bump, 1e-6));

    const Real camber = extract_camber(fx.sys.states[fx.mc.upright_body]);
    REQUIRE(std::abs(camber) > 0.05 * deg);

    VecX phi = evaluate_all_constraints(fx.sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-6));
}

// ============================================================================
// Comparison with kinematic builder
// ============================================================================

TEST_CASE("McPherson dynamic: sweep matches kinematic builder",
          "[mc_dyn][comparison]")
{
    using namespace mbd;

    McPhersonParams p;

    // Dynamic version
    auto fx = make_fixed_chassis_mcpherson(p);

    const size_t bump_idx = fx.sys.constraints.size();
    fx.sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        fx.mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    auto dyn_sweep = sweep_bump_travel(
        fx.sys, bump_idx, fx.mc.upright_body,
        p.wheel_center.y(), -0.02, 0.02, 11);

    for (const auto& pt : dyn_sweep.points) {
        REQUIRE(pt.converged);
    }

    // Kinematic version
    MultibodySystem sys_kin;
    auto mc_kin = build_mcpherson_corner(sys_kin, p);
    set_mcpherson_reference(sys_kin, mc_kin);

    auto kin_sweep = sweep_bump_travel(
        sys_kin, mc_kin.bump_constraint_idx, mc_kin.upright_body,
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

// ============================================================================
// Negative camber gain in bump
// ============================================================================

TEST_CASE("McPherson dynamic: negative camber gain in bump",
          "[mc_dyn][camber]")
{
    using namespace mbd;

    McPhersonParams p;
    auto fx = make_fixed_chassis_mcpherson(p);

    const size_t bump_idx = fx.sys.constraints.size();
    fx.sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        fx.mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));

    fx.sys.q.setZero();
    fx.sys.compute_kinematics();

    auto sweep = sweep_bump_travel(
        fx.sys, bump_idx, fx.mc.upright_body,
        p.wheel_center.y(), -0.03, 0.03, 11);

    for (const auto& pt : sweep.points) {
        REQUIRE(pt.converged);
    }

    REQUIRE(sweep.camber_gain() < 0.0);
}