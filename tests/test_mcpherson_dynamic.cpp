#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <cmath>

#include "mbd/vehicle/suspension/mcpherson.hpp"

// The dynamic McPherson corner (parented to a chassis), on the kernel (plan
// task 2.7).

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;

    struct MCFixture {
        mbd::kernel::System sys;
        mbd::BodyIndex chassis_body{0};
        mbd::McPhersonCorner mc;
    };

    /// A McPherson corner on a chassis pinned to the ground at identity.
    void make_fixed_chassis_mcpherson(MCFixture& fx, const mbd::McPhersonParams& p)
    {
        using namespace mbd;
        fx.chassis_body = fx.sys.model.add_body(
            0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
            Transform3::Identity(),
            RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8)), "chassis");
        fx.mc = build_mcpherson_corner_dynamic(fx.sys, fx.chassis_body, p);
    }
}

// ============================================================================
// Reference configuration
// ============================================================================

TEST_CASE("McPherson dynamic: reference configuration satisfies all constraints",
          "[mc_dyn][reference]")
{
    using namespace mbd;

    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, McPhersonParams{});

    Kinematics k(fx.sys);
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("McPherson dynamic: body positions at reference",
          "[mc_dyn][reference]")
{
    using namespace mbd;

    McPhersonParams p;
    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, p);
    Kinematics k(fx.sys);

    // LCA origin at p.lca_pivot, upright origin at p.wheel_center
    REQUIRE_THAT((k.state(fx.mc.lca_body).p_WB - p.lca_pivot).norm(), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT((k.state(fx.mc.upright_body).p_WB - p.wheel_center).norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("McPherson dynamic: camber and toe zero at reference",
          "[mc_dyn][reference]")
{
    using namespace mbd;

    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, McPhersonParams{});
    Kinematics k(fx.sys);

    REQUIRE_THAT(extract_camber(k.state(fx.mc.upright_body)), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(k.state(fx.mc.upright_body)), WithinAbs(0.0, 1e-10));
}

// ============================================================================
// Chassis translation carries the corner
// ============================================================================

TEST_CASE("McPherson dynamic: chassis translation carries the corner",
          "[mc_dyn][chassis_motion]")
{
    using namespace mbd;

    McPhersonParams p;

    kernel::System sys;
    const BodyIndex chassis = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
        Transform3::Identity(), RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8)),
        "chassis");
    auto mc = build_mcpherson_corner_dynamic(sys, chassis, p);

    Kinematics k(sys);
    const Vec3 offset(0.1, 0.2, 0.3);
    k.q.segment<3>(sys.model.idx_q[chassis]) = offset;
    k.update();

    REQUIRE_THAT((k.state(mc.lca_body).p_WB - (p.lca_pivot + offset)).norm(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT((k.state(mc.upright_body).p_WB - (p.wheel_center + offset)).norm(), WithinAbs(0.0, 1e-9));

    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// DOF count
// ============================================================================

TEST_CASE("McPherson dynamic: correct DOF counts",
          "[mc_dyn][dof]")
{
    using namespace mbd;

    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, McPhersonParams{});

    // Tree DOFs: 0 (fixed chassis) + 1 (LCA rev) + 3 (spherical) = 4
    REQUIRE(fx.sys.model.nv == 4);

    // Constraints: 2 (top mount on the strut line) + 1 (tie rod) = 3 equations.
    // A point held on a line loses its two translations across the line.
    REQUIRE(fx.sys.constraints.size() == 2);
    int total_eqs = 0;
    for (const auto& c : fx.sys.constraints) total_eqs += c->size();
    REQUIRE(total_eqs == 3);

    // The equations are independent, so net DOF = 4 - 3 = 1: suspension travel.
    kernel::Data data(fx.sys.model);
    kernel::ConstraintSolver solver(fx.sys.model, fx.sys.constraints);
    solver.evaluate(data, fx.sys.model.neutral_configuration(), VecX::Zero(fx.sys.model.nv), 0.0);
    Eigen::JacobiSVD<MatX> svd(solver.J());
    REQUIRE(svd.rank() == 3);
}

// ============================================================================
// Bump travel
// ============================================================================

TEST_CASE("McPherson dynamic: prescribed wheel Y triggers consistent motion",
          "[mc_dyn][bump]")
{
    using namespace mbd;

    McPhersonParams p;
    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, p);
    fx.sys.constraints.push_back(point_height_driver(fx.mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));

    Kinematics k(fx.sys);
    const Real bump = 0.02;
    k.t = bump;
    REQUIRE(k.solve(100, 1e-8));

    REQUIRE_THAT(k.state(fx.mc.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y() + bump, 1e-6));
    REQUIRE(std::abs(extract_camber(k.state(fx.mc.upright_body))) > 0.05 * deg);
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-6));
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
    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, p);
    fx.sys.constraints.push_back(point_height_driver(fx.mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    Kinematics k_dyn(fx.sys);
    auto dyn_sweep = sweep_bump_travel(k_dyn, fx.mc.upright_body, -0.02, 0.02, 11);

    for (const auto& pt : dyn_sweep.points) {
        REQUIRE(pt.converged);
    }

    // Kinematic version
    kernel::System sys_kin;
    auto mc_kin = build_mcpherson_corner(sys_kin, p);
    Kinematics k_kin(sys_kin);
    auto kin_sweep = sweep_bump_travel(k_kin, mc_kin.upright_body, -0.02, 0.02, 11);

    for (const auto& pt : kin_sweep.points) {
        REQUIRE(pt.converged);
    }

    // The same mechanism: the camber curves agree to the solver's tolerance.
    for (size_t i = 0; i < dyn_sweep.points.size(); ++i) {
        REQUIRE_THAT(dyn_sweep.points[i].camber, WithinAbs(kin_sweep.points[i].camber, 1e-8));
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
    MCFixture fx;
    make_fixed_chassis_mcpherson(fx, p);
    fx.sys.constraints.push_back(point_height_driver(fx.mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    Kinematics k(fx.sys);

    auto sweep = sweep_bump_travel(k, fx.mc.upright_body, -0.03, 0.03, 11);

    for (const auto& pt : sweep.points) {
        REQUIRE(pt.converged);
    }

    REQUIRE(sweep.camber_gain() < 0.0);
}
