#pragma once

// Masses, centres of mass and inertias of a suspension's parts from their
// geometry (plan task 7.6, finding F9, decision D67): a part's mass is given,
// and where it lies follows from the points it joins.
//
//   - A link or an arm is a uniform rod between two points (an arm: its
//     chassis pivot and its outer joint), of radius `rod_radius`: its centre
//     of mass midway, m L^2 / 12 about any axis across it, m r^2 / 2 about
//     its own.
//   - An upright or a hub carrier is its mass shared equally among the
//     points it joins (ball joints, the tie rod's end, the wheel centre),
//     each share a solid sphere of radius `lump_radius`: its centre of mass
//     the points' centroid, its inertia theirs about it.
//
// All in a body's frame (points given in it). The radii only keep every
// inertia positive; the arms' and uprights' own shapes are not known.

#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/model/rigid_body.hpp"

namespace mbd {

inline constexpr Real rod_radius = 0.015;    ///< [m]
inline constexpr Real lump_radius = 0.03;    ///< [m]

/// A uniform rod of `mass` from a to b (points in the body's frame).
RigidBodyInertia rod_inertia(Real mass, const Vec3& a, const Vec3& b);

/// `mass` shared equally among `points` (in the body's frame), each share a
/// solid sphere of radius lump_radius.
RigidBodyInertia lumped_inertia(Real mass, const std::vector<Vec3>& points);

/// Two parts of one body as one: the masses added, the centre of mass their
/// weighted mean, the inertias moved to it (parallel axes) and added.
RigidBodyInertia combine(const RigidBodyInertia& a, const RigidBodyInertia& b);

} // namespace mbd
