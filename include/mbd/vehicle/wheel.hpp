#pragma once

// The wheel as a body (plan task 7.3, D4, D62).
//
// A wheel turns on a revolute joint, its spin joint, about the Y axis of the
// body that carries it (the upright of a linkage corner, the slider of a
// simple corner): ISO 8855's lateral axis, so that a forward roll is a
// positive rotation. Its spin is part of the multibody state: its angular
// momentum gives the gyroscopic moments, and the torques on it (drive,
// brake, the tyre's) react on the bodies that apply them.

#include "mbd/core/core.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/model/rigid_body.hpp"

#include <string>

namespace mbd {

/// A wheel's inertia: a uniform thin disc of the wheel's mass and the tyre's
/// free radius, centred on the body's origin, its axis the body's Y:
/// m R^2 / 2 about the axle and m R^2 / 4 about any diameter.
RigidBodyInertia wheel_inertia(Real mass, Real radius);

/// Add a wheel of `mass` and `radius` (wheel_inertia) to `model`: a body on a
/// revolute joint about the Y axis of `carrier`, its origin at `centre` (in
/// the carrier's frame) and its axes the carrier's at a joint angle of zero.
/// Returns the wheel's body index.
BodyIndex add_wheel(kernel::Model& model, BodyIndex carrier, const Vec3& centre, Real mass, Real radius,
                    const std::string& name);

/// Set the wheel `wheel`, which `tire` rolls on, spinning at its free-rolling
/// rate in the simulator's present state (FullTireForce::free_rolling_spin):
/// its spin joint's rate is changed by the difference, and the simulator's
/// kinematics refreshed. Give a moving vehicle rolling wheels with this, so
/// that its tyres do not start by skidding.
void set_free_rolling(kernel::Simulator& sim, BodyIndex wheel, const FullTireForce& tire);

} // namespace mbd
