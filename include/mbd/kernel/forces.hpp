#pragma once

// Applied forces on the kinematics kernel.
//
// The force elements (forces/force_element.hpp: springs, tyres, aerodynamics)
// read world body states and add world forces, each at a body origin with its
// moment about that origin (RigidBodyForces). These functions connect them to
// the kernel: body states from Data, and the generalized forces of those
// body forces.

#include <vector>

#include "mbd/kernel/model.hpp"
#include "mbd/model/rigid_body.hpp"

namespace mbd::kernel {

/// World state of body i: origin, orientation, velocity of the origin and
/// angular velocity. Needs forward_kinematics with velocities.
RigidBodyState body_state(const Data& data, int i);

/// The states of all bodies, ground (index 0) included.
void body_states(const Model& model, const Data& data, std::vector<RigidBodyState>& states);

/// The generalized forces of forces on the bodies: tau = sum over i of
/// J_i^T [tau_W; f_W], with J_i the world Jacobian of body i at its origin.
/// Computed in one pass from the leaves, using data.f as working storage.
/// Needs forward_kinematics (positions are enough). tau is resized to nv.
void generalized_forces(const Model& model, Data& data,
                        const std::vector<RigidBodyForces>& forces, VecX& tau);

} // namespace mbd::kernel
