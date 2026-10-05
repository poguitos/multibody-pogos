#pragma once

// Double-wishbone suspension corner builder.
//
// Builds a planar-ish double-wishbone corner using the hybrid tree + constraint
// formulation. The mechanism has 0 net DOF when bump travel is prescribed.

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

Mat3 rotation_align_z_to(const Vec3& target);

} // namespace detail

// ============================================================================
// Builders
// ============================================================================
//
// Tree: LCA on a revolute at its pivot, the upright on a spherical joint at
// the lower ball joint, the UCA on a revolute at its pivot. Loop closures:
// the upper ball joint (the UCA's outer point on the upright) and the tie rod
// (a distance). That leaves one degree of freedom, the bump travel. Arm
// frames have their origin at the pivot and the parent's axes at the
// reference; the upright's origin is the wheel centre. The reference is the
// model's neutral configuration.
//
// The kinematic corner hangs from ground and adds a bump driver that holds
// the wheel-centre height at wheel_center.y() + t (see Kinematics). The
// dynamic corner hangs from a chassis body and has no driver.

/// Kinematic DWB corner on ground, with its bump driver.
DoubleWishboneCorner build_double_wishbone_corner(
    kernel::System& sys,
    const DoubleWishboneParams& p = DoubleWishboneParams{});

/// DWB corner for dynamic simulation, on the chassis (the topology is the
/// kinematic corner's, without the driver). The tie rod's inner point is on
/// `tierod_body`, the chassis by default or a steering rack whose frame
/// coincides with the chassis frame at zero travel; its coordinates are the
/// same in both.
DoubleWishboneCorner build_double_wishbone_corner_dynamic(
    kernel::System& sys,
    BodyIndex chassis_body,
    const DoubleWishboneParams& p,
    BodyIndex tierod_body = -1);

} // namespace mbd
