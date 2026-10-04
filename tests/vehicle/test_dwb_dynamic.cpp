#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include "mbd/vehicle/suspension/double_wishbone.hpp"

// The dynamic double-wishbone corner (parented to a chassis), on the kernel
// (plan task 2.7).

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;

    /// A "fixed chassis" test fixture: a fixed joint from ground, then a DWB
    /// corner dynamically attached to it. The chassis is rigidly pinned at
    /// identity, so the corner has exactly 1 net DOF (bump travel), like the
    /// ground-parented kinematic builder.
    struct DwbFixture {
        mbd::kernel::System sys;
        mbd::BodyIndex chassis_body{0};
        mbd::DoubleWishboneCorner dwb;
    };

    void make_fixed_chassis_dwb(DwbFixture& fx, const mbd::DoubleWishboneParams& p)
    {
        using namespace mbd;
        fx.chassis_body = fx.sys.model.add_body(
            0, std::make_shared<kernel::FixedJointModel>(), Transform3::Identity(),
            Transform3::Identity(),
            RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8)), "chassis");
        fx.dwb = build_double_wishbone_corner_dynamic(fx.sys, fx.chassis_body, p);
    }
}

// ============================================================================
// Reference configuration: constraints satisfied with chassis at identity
// ============================================================================

TEST_CASE("DWB dynamic: reference configuration satisfies all constraints",
          "[dwb_dyn][reference]")
{
    using namespace mbd;

    DwbFixture fx;
    make_fixed_chassis_dwb(fx, DoubleWishboneParams{});

    Kinematics k(fx.sys);   // chassis at identity, suspension at neutral
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB dynamic: body positions at reference", "[dwb_dyn][reference]")
{
    using namespace mbd;

    DoubleWishboneParams p;
    DwbFixture fx;
    make_fixed_chassis_dwb(fx, p);
    Kinematics k(fx.sys);

    // Chassis at origin
    REQUIRE_THAT(k.state(fx.chassis_body).p_WB.norm(), WithinAbs(0.0, 1e-10));

    // LCA origin at p.lca_pivot, UCA at p.uca_pivot, upright at the wheel centre
    REQUIRE_THAT((k.state(fx.dwb.lca_body).p_WB - p.lca_pivot).norm(), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT((k.state(fx.dwb.uca_body).p_WB - p.uca_pivot).norm(), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT((k.state(fx.dwb.upright_body).p_WB - p.wheel_center).norm(), WithinAbs(0.0, 1e-10));
}

TEST_CASE("DWB dynamic: camber and toe zero at reference",
          "[dwb_dyn][reference]")
{
    using namespace mbd;

    DwbFixture fx;
    make_fixed_chassis_dwb(fx, DoubleWishboneParams{});
    Kinematics k(fx.sys);

    REQUIRE_THAT(extract_camber(k.state(fx.dwb.upright_body)), WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(extract_toe(k.state(fx.dwb.upright_body)), WithinAbs(0.0, 1e-10));
}

// ============================================================================
// Chassis translation: corner moves rigidly with chassis
// ============================================================================

TEST_CASE("DWB dynamic: chassis translation carries the corner",
          "[dwb_dyn][chassis_motion]")
{
    using namespace mbd;

    // For this test, use a free chassis so we can translate it
    DoubleWishboneParams p;

    kernel::System sys;
    const BodyIndex chassis = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
        Transform3::Identity(), RigidBodyInertia::from_solid_box(10000.0, Vec3(1.5, 0.3, 0.8)),
        "chassis");
    auto dwb = build_double_wishbone_corner_dynamic(sys, chassis, p);

    // Translate chassis by (0.1, 0.2, 0.3): the free joint's translation
    Kinematics k(sys);
    const Vec3 offset(0.1, 0.2, 0.3);
    k.q.segment<3>(sys.model.idx_q[chassis]) = offset;
    k.update();

    REQUIRE_THAT((k.state(dwb.lca_body).p_WB - (p.lca_pivot + offset)).norm(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT((k.state(dwb.uca_body).p_WB - (p.uca_pivot + offset)).norm(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT((k.state(dwb.upright_body).p_WB - (p.wheel_center + offset)).norm(), WithinAbs(0.0, 1e-9));

    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// Single DOF: with chassis fixed, the mechanism has 1 DOF (bump travel)
// ============================================================================

TEST_CASE("DWB dynamic: mechanism has correct DOF counts",
          "[dwb_dyn][dof]")
{
    using namespace mbd;

    DwbFixture fx;
    make_fixed_chassis_dwb(fx, DoubleWishboneParams{});

    // Tree DOFs: 0 (fixed chassis) + 1 (LCA rev) + 3 (spherical) + 1 (UCA rev) = 5
    REQUIRE(fx.sys.model.nv == 5);

    // Constraints: 3 (upper ball joint) + 1 (tie rod) = 4 loop equations
    REQUIRE(fx.sys.constraints.size() == 2);
    int total_eqs = 0;
    for (const auto& c : fx.sys.constraints) total_eqs += c->size();
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
    DwbFixture fx;
    make_fixed_chassis_dwb(fx, p);

    // Bump prescription on wheel center Y: nominal + t
    fx.sys.constraints.push_back(point_height_driver(fx.dwb.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));

    Kinematics k(fx.sys);
    const Real bump = 0.02;
    k.t = bump;
    REQUIRE(k.solve(100, 1e-8));

    // Wheel at prescribed Y
    REQUIRE_THAT(k.state(fx.dwb.upright_body).p_WB.y(), WithinAbs(p.wheel_center.y() + bump, 1e-6));

    // Camber changed
    REQUIRE(std::abs(extract_camber(k.state(fx.dwb.upright_body))) > 0.1 * deg);

    // All constraints satisfied
    REQUIRE_THAT(k.phi().norm(), WithinAbs(0.0, 1e-6));
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
    DwbFixture fx;
    make_fixed_chassis_dwb(fx, p);
    fx.sys.constraints.push_back(point_height_driver(fx.dwb.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    Kinematics k_dyn(fx.sys);
    auto dyn_sweep = sweep_bump_travel(k_dyn, fx.dwb.upright_body, -0.02, 0.02, 11);

    for (const auto& pt : dyn_sweep.points) {
        REQUIRE(pt.converged);
    }

    // --- Kinematic version (parented to ground) ---
    kernel::System sys_kin;
    auto dwb_kin = build_double_wishbone_corner(sys_kin, p);
    Kinematics k_kin(sys_kin);
    auto kin_sweep = sweep_bump_travel(k_kin, dwb_kin.upright_body, -0.02, 0.02, 11);

    for (const auto& pt : kin_sweep.points) {
        REQUIRE(pt.converged);
    }

    // The same mechanism: the camber curves agree to the solver's tolerance.
    for (size_t i = 0; i < dyn_sweep.points.size(); ++i) {
        REQUIRE_THAT(dyn_sweep.points[i].camber,
                     WithinAbs(kin_sweep.points[i].camber, 1e-8));
    }
}
