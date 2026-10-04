#pragma once

// Double-wishbone suspension corner builder.
//
// Builds a planar-ish double-wishbone corner using the hybrid tree + constraint
// formulation. The mechanism has 0 net DOF when bump travel is prescribed.

#include "mbd/model/system.hpp"
#include "mbd/analysis/position_kinematics.hpp"

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

// ============================================================================
// Dynamic DWB corner: parent is chassis (not ground)
// ============================================================================

/// Build a DWB corner for dynamic simulation, parented to a moving chassis body.
///
/// Topology:
///   chassis → LCA (revolute) → upright (spherical at LCA outer)
///   chassis → UCA (revolute)
/// Loop constraints:
///   CoincidentPoint: UCA outer point ≡ upright's upper ball joint point
///   DistanceConstraint: tie rod inner (on chassis) to outer (on upright)
///
/// The hardpoints in \p p are interpreted as coordinates in the CHASSIS body
/// frame at the reference pose (chassis at identity, wheel at p.wheel_center).
///
/// \param sys              The multibody system (chassis body must already exist).
/// \param chassis_body     Index of the chassis body.
/// \param p                Hardpoint parameters in chassis frame.
inline DoubleWishboneCorner build_double_wishbone_corner_dynamic(
    MultibodySystem& sys,
    BodyIndex chassis_body,
    const DoubleWishboneParams& p)
{
    DoubleWishboneCorner dwb;
    dwb.params = p;

    // Rotation aligning joint Z with the arm pivot axis
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);

    // --- LCA body (origin at lca_pivot in chassis frame) ---
    auto I_arm = RigidBodyInertia::from_solid_box(
        p.arm_mass, Vec3(0.02, 0.02, 0.2));
    dwb.lca_body = sys.add_body(I_arm, RigidBodyState{}, "dyn_LCA", chassis_body);

    Transform3 X_PJ_lca(R_arm, p.lca_pivot);
    Transform3 X_CJ_lca = Transform3::FromRotation(R_arm);
    dwb.lca_joint_idx = sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        X_PJ_lca, X_CJ_lca, chassis_body, dwb.lca_body));

    // --- Upright body (origin at wheel_center in chassis frame) ---
    auto I_upright = RigidBodyInertia::from_solid_box(
        p.upright_mass, Vec3(0.05, 0.1, 0.05));
    dwb.upright_body = sys.add_body(I_upright, RigidBodyState{}, "dyn_upright", dwb.lca_body);

    // Spherical joint at lca_outer.
    // With X_CJ_lca = FromRotation(R_arm), the LCA body frame has identity
    // orientation at reference (same as chassis), so chassis-frame vectors
    // equal LCA-body-frame vectors.
    Transform3 X_PJ_sph = Transform3::FromTranslation(p.lca_outer - p.lca_pivot);
    Transform3 X_CJ_sph = Transform3::FromTranslation(p.lca_outer - p.wheel_center);
    dwb.spherical_joint_idx = sys.add_joint(std::make_unique<SphericalCoordJoint>(
        X_PJ_sph, X_CJ_sph, dwb.lca_body, dwb.upright_body));

    // --- UCA body (origin at uca_pivot in chassis frame) ---
    dwb.uca_body = sys.add_body(I_arm, RigidBodyState{}, "dyn_UCA", chassis_body);

    Transform3 X_PJ_uca(R_arm, p.uca_pivot);
    Transform3 X_CJ_uca = Transform3::FromRotation(R_arm);
    dwb.uca_joint_idx = sys.add_joint(std::make_unique<RevoluteCoordJoint>(
        X_PJ_uca, X_CJ_uca, chassis_body, dwb.uca_body));

    // --- Loop closure: UCA outer ≡ upright's upper ball joint ---
    // UCA body frame has identity orientation at reference (same reasoning
    // as LCA), so chassis-frame vectors equal UCA-body-frame vectors.
    const Vec3 uca_outer_in_uca = p.uca_outer - p.uca_pivot;
    const Vec3 uca_outer_in_upright = p.uca_outer - p.wheel_center;
    
    dwb.coincident_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<CoincidentPointConstraint>(
        dwb.uca_body, dwb.upright_body,
        uca_outer_in_uca, uca_outer_in_upright));

    // --- Tie rod distance constraint ---
    // Inner point on chassis body: p.tierod_inner (already in chassis frame)
    // Outer point on upright body: p.tierod_outer - p.wheel_center
    const Real tierod_length = (p.tierod_outer - p.tierod_inner).norm();
    dwb.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<DistanceConstraint>(
        chassis_body, dwb.upright_body,
        p.tierod_inner,
        p.tierod_outer - p.wheel_center,
        tierod_length));

    // Note: NO bump prescription constraint (that's for kinematic analysis)
    dwb.bump_constraint_idx = 0; // unused in dynamic mode

    return dwb;
}

/// Set the double-wishbone to its reference (zero-bump) configuration.
inline void set_dwb_reference(MultibodySystem& sys, const DoubleWishboneCorner& /*dwb*/)
{
    sys.q.setZero();
    sys.q_dot.setZero();
    sys.compute_forward_kinematics();
}

// ============================================================================
// On the kernel (plan task 2.7)
// ============================================================================
//
// Same topology and frames as above. The kinematic corner's bump driver holds
// the wheel-centre height at wheel_center.y() + t (see Kinematics). The
// reference configuration is the model's neutral one.

/// Kinematic DWB corner on ground, with its bump driver.
inline DoubleWishboneCorner build_double_wishbone_corner(
    kernel::System& sys,
    const DoubleWishboneParams& p = DoubleWishboneParams{})
{
    using namespace kernel;
    DoubleWishboneCorner dwb;
    dwb.params = p;
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);
    const auto I_arm = RigidBodyInertia::from_solid_box(p.arm_mass, Vec3(0.02, 0.02, 0.2));
    const auto I_upright = RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.1, 0.05));
    const auto revolute = std::make_shared<RevoluteJointModel>();

    dwb.lca_body = sys.model.add_body(0, revolute, Transform3(R_arm, p.lca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "LCA");
    dwb.upright_body = sys.model.add_body(
        dwb.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I_upright, "upright");
    dwb.uca_body = sys.model.add_body(0, revolute, Transform3(R_arm, p.uca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "UCA");
    dwb.lca_joint_idx = dwb.lca_body;
    dwb.spherical_joint_idx = dwb.upright_body;
    dwb.uca_joint_idx = dwb.uca_body;

    // Upper ball joint: the UCA's outer point on the upright.
    dwb.coincident_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{dwb.uca_body, Transform3::FromTranslation(p.uca_outer - p.uca_pivot)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.uca_outer - p.wheel_center)}));

    // Tie rod.
    dwb.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{0, Transform3::FromTranslation(p.tierod_inner)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.tierod_outer - p.wheel_center)},
        (p.tierod_outer - p.tierod_inner).norm()));

    // Bump prescription: wheel-centre height = wheel_center.y() + t.
    dwb.bump_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(point_height_driver(dwb.upright_body, Vec3::Zero(), 1, p.wheel_center.y()));
    return dwb;
}

/// DWB corner for dynamic simulation, on the chassis (see the legacy version
/// above for the topology). The tie rod's inner point is on `tierod_body`,
/// the chassis by default or a steering rack whose frame coincides with the
/// chassis frame at zero travel; its coordinates are the same in both.
inline DoubleWishboneCorner build_double_wishbone_corner_dynamic(
    kernel::System& sys,
    BodyIndex chassis_body,
    const DoubleWishboneParams& p,
    BodyIndex tierod_body = -1)
{
    using namespace kernel;
    if (tierod_body < 0) tierod_body = chassis_body;
    DoubleWishboneCorner dwb;
    dwb.params = p;
    const Mat3 R_arm = detail::rotation_align_z_to(p.arm_axis);
    const auto I_arm = RigidBodyInertia::from_solid_box(p.arm_mass, Vec3(0.02, 0.02, 0.2));
    const auto I_upright = RigidBodyInertia::from_solid_box(p.upright_mass, Vec3(0.05, 0.1, 0.05));
    const auto revolute = std::make_shared<RevoluteJointModel>();

    // Arm frames: origin at the pivot, axes of the chassis at the reference.
    dwb.lca_body = sys.model.add_body(chassis_body, revolute, Transform3(R_arm, p.lca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "dyn_LCA");
    dwb.upright_body = sys.model.add_body(
        dwb.lca_body, std::make_shared<SphericalJointModel>(),
        Transform3::FromTranslation(p.lca_outer - p.lca_pivot),
        Transform3::FromTranslation(p.lca_outer - p.wheel_center), I_upright, "dyn_upright");
    dwb.uca_body = sys.model.add_body(chassis_body, revolute, Transform3(R_arm, p.uca_pivot),
                                      Transform3::FromRotation(R_arm), I_arm, "dyn_UCA");
    dwb.lca_joint_idx = dwb.lca_body;
    dwb.spherical_joint_idx = dwb.upright_body;
    dwb.uca_joint_idx = dwb.uca_body;

    dwb.coincident_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{dwb.uca_body, Transform3::FromTranslation(p.uca_outer - p.uca_pivot)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.uca_outer - p.wheel_center)}));

    dwb.tierod_constraint_idx = sys.constraints.size();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{tierod_body, Transform3::FromTranslation(p.tierod_inner)},
        Marker{dwb.upright_body, Transform3::FromTranslation(p.tierod_outer - p.wheel_center)},
        (p.tierod_outer - p.tierod_inner).norm()));

    dwb.bump_constraint_idx = 0;   // no bump driver in a dynamic corner
    return dwb;
}

} // namespace mbd