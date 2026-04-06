#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/mcpherson.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;
}

TEST_CASE("McPherson: reference configuration satisfies all constraints",
          "[mcpherson][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto mc = build_mcpherson_corner(sys);
    set_mcpherson_reference(sys, mc);

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("McPherson: body positions at reference", "[mcpherson][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    McPhersonParams p;
    auto mc = build_mcpherson_corner(sys, p);
    set_mcpherson_reference(sys, mc);

    REQUIRE_THAT(sys.states[mc.lca_body].p_WB.y(),
                 WithinAbs(p.lca_pivot.y(), 1e-10));
    REQUIRE_THAT(sys.states[mc.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y(), 1e-10));
    REQUIRE_THAT(sys.states[mc.upright_body].p_WB.z(),
                 WithinAbs(p.wheel_center.z(), 1e-10));
}

TEST_CASE("McPherson: camber and toe zero at reference", "[mcpherson][reference]")
{
    using namespace mbd;

    MultibodySystem sys;
    auto mc = build_mcpherson_corner(sys);
    set_mcpherson_reference(sys, mc);

    REQUIRE_THAT(extract_camber(sys.states[mc.upright_body]), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(sys.states[mc.upright_body]), WithinAbs(0.0, 1e-10));
}

TEST_CASE("McPherson: Newton-Raphson converges for 25mm bump", "[mcpherson][solver]")
{
    using namespace mbd;

    MultibodySystem sys;
    McPhersonParams p;
    auto mc = build_mcpherson_corner(sys, p);
    set_mcpherson_reference(sys, mc);

    auto* h = dynamic_cast<PointCoordinateConstraint*>(
        sys.constraints[mc.bump_constraint_idx].get());
    h->target = p.wheel_center.y() + 0.025;

    bool ok = solve_position_kinematics(sys);
    REQUIRE(ok);

    REQUIRE_THAT(sys.states[mc.upright_body].p_WB.y(),
                 WithinAbs(p.wheel_center.y() + 0.025, 1e-8));

    VecX phi = evaluate_all_constraints(sys);
    REQUIRE_THAT(phi.norm(), WithinAbs(0.0, 1e-8));
}

TEST_CASE("McPherson: negative camber gain in bump", "[mcpherson][camber]")
{
    using namespace mbd;

    MultibodySystem sys;
    McPhersonParams p;
    auto mc = build_mcpherson_corner(sys, p);
    set_mcpherson_reference(sys, mc);

    auto result = sweep_bump_travel(
        sys, mc.bump_constraint_idx, mc.upright_body,
        p.wheel_center.y(), -0.03, 0.03, 11);

    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
    }

    // McPherson should produce negative camber gain
    REQUIRE(result.camber_gain() < 0.0);

    // Middle point should have near-zero camber
    REQUIRE_THAT(result.points[5].camber, WithinAbs(0.0, 0.1 * deg));
}

TEST_CASE("McPherson: full kinematic sweep", "[mcpherson][sweep]")
{
    using namespace mbd;

    MultibodySystem sys;
    McPhersonParams p;
    auto mc = build_mcpherson_corner(sys, p);
    set_mcpherson_reference(sys, mc);

    auto result = sweep_bump_travel(
        sys, mc.bump_constraint_idx, mc.upright_body,
        p.wheel_center.y(), -0.04, 0.04, 21);

    REQUIRE(result.points.size() == 21);

    for (const auto& pt : result.points) {
        REQUIRE(pt.converged);
    }

    // Camber should be monotonically decreasing with increasing bump
    for (size_t i = 1; i < result.points.size(); ++i) {
        REQUIRE(result.points[i].camber <= result.points[i - 1].camber + 0.05 * deg);
    }

    // Toe change should be small for a well-designed tie rod
    for (const auto& pt : result.points) {
        REQUIRE(std::abs(pt.toe) < 3.0 * deg);
    }
}