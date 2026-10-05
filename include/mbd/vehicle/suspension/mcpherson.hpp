#pragma once

// McPherson strut suspension corner builder.
//
// The McPherson strut replaces the upper control arm of a DWB with a
// strut (combined spring-damper and structural link). The strut axis,
// rigidly attached to the upright, must pass through the fixed top mount.
//
// Tree: ground → LCA (revolute) → upright (spherical at lower ball joint)
// Constraints: the top mount on the strut axis (2 equations), the tie rod
// (1) and, in the kinematic corner, the bump driver (1): 4 tree DOFs less
// 4 equations leaves none.

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/vehicle/suspension/double_wishbone.hpp" // for detail::rotation_align_z_to

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
// Builders
// ============================================================================
//
// Tree: LCA on a revolute at its pivot, the upright on a spherical joint at
// the lower ball joint. Loop closures: the strut's top mount held on the
// strut axis, a line fixed in the upright (2 equations), and the tie rod
// (1). The kinematic corner hangs from ground and adds a bump driver
// (wheel-centre height = wheel_center.y() + t); the dynamic corner hangs from
// a chassis body, which carries the top mount.

/// Kinematic McPherson corner on ground, with its bump driver: wheel-centre
/// height = wheel_center.y() + t (see Kinematics).
McPhersonCorner build_mcpherson_corner(
    kernel::System& sys,
    const McPhersonParams& p = McPhersonParams{});

/// McPherson corner for dynamic simulation, on the chassis. The strut's top
/// mount is on the chassis; the tie rod's inner point on `tierod_body` (the
/// chassis by default, or a steering rack: see the DWB version).
McPhersonCorner build_mcpherson_corner_dynamic(
    kernel::System& sys,
    BodyIndex chassis_body,
    const McPhersonParams& p,
    BodyIndex tierod_body = -1);

} // namespace mbd
