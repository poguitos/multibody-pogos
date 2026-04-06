#pragma once

// McPherson strut suspension corner builder.
//
// The McPherson strut replaces the upper control arm of a DWB with a
// strut (combined spring-damper and structural link). The strut axis,
// rigidly attached to the upright, must pass through the fixed top mount.
//
// Tree: ground → LCA (revolute) → upright (spherical at lower ball joint)
// Constraints: StrutLineConstraint + tie rod distance + bump prescription

#include "mbd/system.hpp"
#include "mbd/kinematics.hpp"
#include "mbd/double_wishbone.hpp" // for detail::rotation_align_z_to

namespace mbd {

// ============================================================================
// Parameters
// ============================================================================

struct McPhersonParams {
    // Lower control arm
    Vec3 lca_pivot{0.0, 0.20, 0.25};
    Vec3 lca_outer{0.0, 0.15, 0.65};

    // Strut
    Vec3 strut_top_mount{0.0, 0.65, 0.55};
    Vec3 strut_lower{0.0, 0.38, 0.72};

    // Tie rod
    Vec3 tierod_inner{0.10, 0.22, 0.25};
    Vec3 tierod_outer{0.10, 0.20, 0.68};

    // Wheel
    Vec3 wheel_center{0.0, 0.25, 0.75};

    // LCA pivot axis
    Vec3 arm_axis{Vec3::UnitX()};

    // Body masses (for inertia, not critical for kinematics)
    Real arm_mass{1.0};
    Real upright_mass{5.0};
};

// ============================================================================
// Handle
// ============================================================================

struct McPhersonCorner {
    BodyIndex lca_body{0};
    BodyIndex upright_body{0};

    int lca_joint_idx{-1};
    int spherical_joint_idx{-1};

    size_t strut_constraint_idx{0};
    size_t tierod_constraint_idx{0};
    size_t bump_constraint_idx{0};

    McPhersonParams params;
};

// ============================================================================
// Builder
// ============================================================================

inline McPhersonCorner build_mcpherson_corner(
    MultibodySystem& sys,
    const McPhersonParams& p = McPhersonParams{})
{
    McPhersonCorner mc;
    mc.params = p;

    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);

    // --- LCA body (origin at pivot) ---
    auto I_arm = RigidBodyInertia::from_solid_box(
        p.arm_mass, Vec3(0.02, 0.02, 0.2));
    mc.lca_body = sys.add_body(I_arm, RigidBodyState{}, "MC_LCA", kGroundIndex);

    Transform3 X_PJ_lca(R_arm, p.lca_pivot);
    Transform3 X_CJ_lca = Transform3::FromRotation(R_arm);
    mc.lca_joint_idx = sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        X_PJ_lca, X_CJ_lca, kGroundIndex, mc.lca_body));

    // --- Upright body (origin at wheel center) ---
    auto I_upright = RigidBodyInertia::from_solid_box(
        p.upright_mass, Vec3(0.05, 0.1, 0.05));
    mc.upright_body = sys.add_body(I_upright, RigidBodyState{}, "MC_upright", mc.lca_body);

    Transform3 X_PJ_sph = Transform3::FromTranslation(p.lca_outer - p.lca_pivot);
    Transform3 X_CJ_sph = Transform3::FromTranslation(p.lca_outer - p.wheel_center);
    mc.spherical_joint_idx = sys.add_joint(std::make_unique<SphericalCoordJoint>(
        X_PJ_sph, X_CJ_sph, mc.lca_body, mc.upright_body));

    // --- Strut line constraint ---
    // Strut axis in upright body frame: from strut_lower toward strut_top_mount.
    // At reference configuration (R_upright = I, p_upright = wheel_center),
    // the world-frame direction equals the body-frame direction.
    const Vec3 strut_lower_B = p.strut_lower - p.wheel_center;
    const Vec3 strut_axis_B  = (p.strut_top_mount - p.strut_lower).normalized();

    mc.strut_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<StrutLineConstraint>(
        mc.upright_body, p.strut_top_mount, strut_lower_B, strut_axis_B));

    // --- Tie rod distance constraint ---
    mc.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, mc.upright_body,
        p.tierod_inner,
        p.tierod_outer - p.wheel_center,
        (p.tierod_outer - p.tierod_inner).norm()));

    // --- Bump prescription ---
    mc.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        mc.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));

    return mc;
}

inline void set_mcpherson_reference(MultibodySystem& sys, const McPhersonCorner& /*mc*/)
{
    sys.q.setZero();
    sys.q_dot.setZero();
    sys.compute_forward_kinematics();
}

} // namespace mbd