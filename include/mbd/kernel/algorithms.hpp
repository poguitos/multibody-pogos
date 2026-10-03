#pragma once

// Recursive algorithms of the kinematics kernel (plan task 2.4).
//
// All work on body-frame spatial vectors (spatial/spatial.hpp) in one pass
// over the tree from the root to the leaves, and for the dynamics a second
// pass back. References: R. Featherstone, "Rigid Body Dynamics Algorithms"
// (2008): RNEA, CRBA and ABA in chapters 5 to 7.
//
// Results are written to Data and returned by reference; they stay valid
// until the next call with the same Data.

#include "mbd/kernel/model.hpp"

namespace mbd::kernel {

// --- Kinematics --------------------------------------------------------------

/// Body placements (data.liMi, data.oMi) and motion subspaces (data.S).
void forward_kinematics(const Model& model, Data& data, const VecX& q);

/// The above, plus body velocities (data.v) and velocity-product
/// accelerations (data.c).
void forward_kinematics(const Model& model, Data& data, const VecX& q, const VecX& v);

/// The above, plus body accelerations (data.a) for joint accelerations `a`.
/// These are physical accelerations: gravity is not included.
void forward_kinematics(const Model& model, Data& data,
                        const VecX& q, const VecX& v, const VecX& a);

/// Velocity of body i: [angular velocity; velocity of the body origin], both
/// in world axes. After forward_kinematics with velocities.
Vec6 body_velocity_world(const Data& data, int i);

/// Acceleration of body i: [angular acceleration; acceleration of the body
/// origin], both in world axes. After forward_kinematics with accelerations.
Vec6 body_acceleration_world(const Data& data, int i);

/// Jacobian of body i: J * v = body_velocity_world(i). 6 x nv, in world axes.
/// After forward_kinematics (positions are enough).
void body_jacobian_world(const Model& model, const Data& data, int i, MatX& J);

// --- Dynamics ------------------------------------------------------------------

/// Inverse dynamics, recursive Newton-Euler: the generalized forces that
/// produce accelerations `a` at (q, v), under gravity. Returns data.tau; also
/// leaves the kinematics, the body accelerations and the joint forces data.f
/// in data.
const VecX& rnea(const Model& model, Data& data,
                 const VecX& q, const VecX& v, const VecX& a);

/// Mass matrix, composite-rigid-body algorithm. Returns data.M (symmetric,
/// both triangles filled).
const MatX& crba(const Model& model, Data& data, const VecX& q);

/// Forward dynamics, articulated-body algorithm: the accelerations produced
/// by generalized forces `tau` at (q, v), under gravity. Returns data.ddq;
/// also leaves the kinematics and the body accelerations in data.
const VecX& aba(const Model& model, Data& data,
                const VecX& q, const VecX& v, const VecX& tau);

// --- Whole-system quantities, after forward_kinematics with velocities ------

/// [angular momentum about the world origin; linear momentum], world axes.
Vec6 momentum_world(const Model& model, const Data& data);

/// Kinetic energy.
Real kinetic_energy(const Model& model, const Data& data);

/// Gravitational potential energy (zero at the world origin).
Real potential_energy(const Model& model, const Data& data);

/// Centre of mass of all bodies, in the world. After forward_kinematics.
Vec3 center_of_mass(const Model& model, const Data& data);

// --- Configuration space -----------------------------------------------------

/// q_dot = G(q) * v for the whole system. A Runge-Kutta scheme combines the
/// q_dot of its stages like any other derivative and calls normalize() once,
/// at the end of the step; normalizing the stages would lower its order.
void q_dot(const Model& model, const VecX& q, const VecX& v, VecX& qd);

/// Restore unit quaternions and any other invariant of the coordinates.
void normalize(const Model& model, VecX& q);

/// q_out = q (+) v dt: every joint moved at its constant velocity for a time
/// dt, through the exponential map, so quaternions stay unit. q_out may be q.
void integrate(const Model& model, const VecX& q, const VecX& v, Real dt, VecX& q_out);

/// dv = q1 (-) q0, so that integrate(q0, dv, 1) gives q1.
void difference(const Model& model, const VecX& q0, const VecX& q1, VecX& dv);

} // namespace mbd::kernel
