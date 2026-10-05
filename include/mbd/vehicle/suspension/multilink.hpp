#pragma once

// Multi-link (5-link) suspension corner builder.
//
// The upright is a free body constrained by 5 distance constraints
// (one per link) plus a bump prescription. Each link connects a ground
// point to an upright point. No intermediate bodies are needed.
//
// This is the simplest possible formulation: 6 DOF - 6 constraints = 0 net DOF.

#include "mbd/analysis/position_kinematics.hpp"

#include <array>

namespace mbd {

// ============================================================================
// Parameters
// ============================================================================

struct MultilinkParams {
    /// Link endpoints: inner[i] is on ground, outer[i] is in world frame
    /// at the reference configuration. The builder converts to body frame.
    ///
    /// Geometry is designed for a left-rear corner with links spanning
    /// multiple directions for a well-conditioned constraint Jacobian:
    ///   Link 0: lower trailing (primarily fore-aft)
    ///   Link 1: lower lateral  (primarily lateral)
    ///   Link 2: upper diagonal (fore-aft + lateral)
    ///   Link 3: upper lateral  (primarily lateral, higher)
    ///   Link 4: toe link       (fore-aft + lateral, mid height)
    std::array<Vec3, 5> inner{{
        Vec3(-0.35, 0.15, 0.35),   // lower trailing: far aft, fairly inboard
        Vec3( 0.00, 0.15, 0.25),   // lower lateral: centered, inboard
        Vec3(-0.25, 0.42, 0.40),   // upper diagonal: aft, mid-lateral
        Vec3( 0.05, 0.42, 0.30),   // upper lateral: slightly forward, inboard
        Vec3( 0.20, 0.22, 0.30)    // toe link: forward, inboard
    }};

    std::array<Vec3, 5> outer{{
        Vec3(-0.05, 0.15, 0.72),   // lower trailing outer
        Vec3( 0.00, 0.12, 0.72),   // lower lateral outer
        Vec3(-0.05, 0.40, 0.70),   // upper diagonal outer
        Vec3( 0.05, 0.40, 0.70),   // upper lateral outer
        Vec3( 0.10, 0.20, 0.72)    // toe link outer
    }};

    std::array<std::string, 5> names{{
        "lower_trailing", "lower_lateral", "upper_diagonal", "upper_lateral", "toe_link"
    }};

    Vec3 wheel_center{0.0, 0.28, 0.75};

    Real upright_mass{8.0};
};
// ============================================================================
// Handle
// ============================================================================

struct MultilinkCorner {
    BodyIndex upright_body{0};
    int free_joint_idx{-1};
    std::array<size_t, 5> link_constraint_indices{};
    size_t bump_constraint_idx{0};
    MultilinkParams params;
};

// ============================================================================
// Builder
// ============================================================================

/// Kinematic multilink corner: a free upright held by five links, with its
/// bump driver: wheel-centre height = wheel_center.y() + t (see Kinematics).
MultilinkCorner build_multilink_corner(
    kernel::System& sys,
    const MultilinkParams& p = MultilinkParams{});

} // namespace mbd
