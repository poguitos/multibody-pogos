#pragma once

// Double-wishbone suspension corner builder.
//
// Builds a planar-ish double-wishbone corner using the hybrid tree + constraint
// formulation. The mechanism has 0 net DOF when bump travel is prescribed.

#include "mbd/system.hpp"
#include "mbd/kinematics.hpp"

namespace mbd {

// ============================================================================
// Hardpoint parameters
// ============================================================================

struct DoubleWishboneParams {
    // All coordinates in the parent (chassis/ground) frame.
    // X = forward, Y = up, Z = left.

    // Lower control arm
    Vec3 lca_pivot{0.0, 0.20, 0.30};     ///< Inner pivot point
    Vec3 lca_outer{0.0, 0.15, 0.70};     ///< Outer ball joint (lower)

    // Upper control arm
    Vec3 uca_pivot{0.0, 0.38, 0.35};     ///< Inner pivot point
    Vec3 uca_outer{0.0, 0.35, 0.68};     ///< Outer ball joint (upper)

    // Tie rod
    Vec3 tierod_inner{0.10, 0.22, 0.28}; ///< Chassis/rack attachment
    Vec3 tierod_outer{0.10, 0.20, 0.70}; ///< Upright attachment

    // Wheel center
    Vec3 wheel_center{0.0, 0.25, 0.78};

    // Arm pivot axis (typically along vehicle X for a planar front-view mechanism)
    Vec3 arm_axis{Vec3::UnitX()};

    // Body masses (small; only needed for inertia construction, not kinematic analysis)
    Real arm_mass{1.0};
    Real upright_mass{5.0};
};

// ============================================================================
// Handle for the built mechanism
// ============================================================================

struct DoubleWishboneCorner {
    BodyIndex lca_body{0};
    BodyIndex upright_body{0};
    BodyIndex uca_body{0};

    int lca_joint_idx{-1};
    int spherical_joint_idx{-1};
    int uca_joint_idx{-1};

    size_t coincident_constraint_idx{0};
    size_t tierod_constraint_idx{0};
    size_t bump_constraint_idx{0};

    DoubleWishboneParams params;
};

// ============================================================================
// Builder helper: rotation matrix that aligns Z with a target axis
// ============================================================================

namespace detail {

inline Mat3 rotation_align_z_to(const Vec3& target)
{
    Vec3 z = target.normalized();
    Vec3 x;
    if (std::abs(z.dot(Vec3::UnitY())) < 0.9) {
        x = z.cross(Vec3::UnitY()).normalized();
    } else {
        x = z.cross(Vec3::UnitX()).normalized();
    }
    Vec3 y = z.cross(x).normalized();

    Mat3 R;
    R.col(0) = x;
    R.col(1) = y;
    R.col(2) = z;
    return R;
}

} // namespace detail

// ============================================================================
// Builder function
// ============================================================================

/// Build a double-wishbone suspension corner on an existing MultibodySystem.
///
/// All bodies are parented to ground (kGroundIndex). The mechanism is fully
/// constrained when the bump prescription constraint is active (0 net DOF).
///
/// Returns a DoubleWishboneCorner handle with all indices.
inline DoubleWishboneCorner build_double_wishbone_corner(
    MultibodySystem& sys,
    const DoubleWishboneParams& p = DoubleWishboneParams{})
{
    DoubleWishboneCorner dwb;
    dwb.params = p;

    // Rotation that aligns joint Z with the arm pivot axis
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);

    // --- LCA body (body origin at pivot point) ---
    auto I_arm = RigidBodyInertia::from_solid_box(
        p.arm_mass, Vec3(0.02, 0.02, 0.2));
    dwb.lca_body = sys.add_body(I_arm, RigidBodyState{}, "LCA", kGroundIndex);

    // RevoluteCoordJoint: ground → LCA, pivot at lca_pivot, axis = arm_axis
    Transform3 X_PJ_lca(R_arm, p.lca_pivot);
    Transform3 X_CJ_lca = Transform3::FromRotation(R_arm);
    auto lca_joint = std::make_unique<RevoluteCoordJoint>(
        X_PJ_lca, X_CJ_lca, kGroundIndex, dwb.lca_body);
    dwb.lca_joint_idx = sys.add_joint(std::move(lca_joint));

    // --- Upright body (body origin at wheel center) ---
    auto I_upright = RigidBodyInertia::from_solid_box(
        p.upright_mass, Vec3(0.05, 0.1, 0.05));
    dwb.upright_body = sys.add_body(I_upright, RigidBodyState{}, "upright", dwb.lca_body);

    // SphericalCoordJoint: LCA → upright, joint at lca_outer (lower ball joint)
    Transform3 X_PJ_sph = Transform3::FromTranslation(p.lca_outer - p.lca_pivot);
    Transform3 X_CJ_sph = Transform3::FromTranslation(p.lca_outer - p.wheel_center);
    auto sph_joint = std::make_unique<SphericalCoordJoint>(
        X_PJ_sph, X_CJ_sph, dwb.lca_body, dwb.upright_body);
    dwb.spherical_joint_idx = sys.add_joint(std::move(sph_joint));

    // --- UCA body (body origin at pivot point) ---
    dwb.uca_body = sys.add_body(I_arm, RigidBodyState{}, "UCA", kGroundIndex);

    // RevoluteCoordJoint: ground → UCA, pivot at uca_pivot, axis = arm_axis
    Transform3 X_PJ_uca(R_arm, p.uca_pivot);
    Transform3 X_CJ_uca = Transform3::FromRotation(R_arm);
    auto uca_joint = std::make_unique<RevoluteCoordJoint>(
        X_PJ_uca, X_CJ_uca, kGroundIndex, dwb.uca_body);
    dwb.uca_joint_idx = sys.add_joint(std::move(uca_joint));

    // --- Loop closure: UCA outer ball joint coincides with upright ball joint ---
    // Point on UCA body = uca_outer - uca_pivot (in UCA body frame)
    // Point on upright body = uca_outer - wheel_center (in upright body frame)
    dwb.coincident_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<CoincidentPointConstraint>(
        dwb.uca_body, dwb.upright_body,
        p.uca_outer - p.uca_pivot,
        p.uca_outer - p.wheel_center));

    // --- Tie rod: distance constraint ---
    // Inner (on ground) at tierod_inner, outer (on upright) at tierod_outer - wheel_center
    dwb.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        kGroundIndex, dwb.upright_body,
        p.tierod_inner,
        p.tierod_outer - p.wheel_center,
        (p.tierod_outer - p.tierod_inner).norm()));

    // --- Bump prescription: wheel center Y = target ---
    dwb.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<PointCoordinateConstraint>(
        dwb.upright_body,
        Vec3::Zero(), // wheel center is upright origin
        1,            // Y axis
        p.wheel_center.y()));  // nominal height

    return dwb;
}

/// Set the double-wishbone to its reference (zero-bump) configuration.
inline void set_dwb_reference(MultibodySystem& sys, const DoubleWishboneCorner& /*dwb*/)
{
    sys.q.setZero();
    sys.q_dot.setZero();
    sys.compute_forward_kinematics();
}

} // namespace mbd