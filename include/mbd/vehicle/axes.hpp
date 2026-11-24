#pragma once

// The vehicle layer's axes (plan task 7.1, D3): ISO 8855. The world and the
// vehicle have X forward, Y to the left and Z up; gravity acts along -Z. A
// wheel's frame has X along its heading, Y along its spin axis to the left,
// Z up. Steering, yaw and lateral acceleration are positive to the left;
// docs/conventions.md gives every sign.
//
// The kernel assumes no axes: it takes gravity as a vector. The vehicle
// builders set it from iso_gravity(). The tyres take the ground from a road
// (mbd/forces/road.hpp, task 7.2), whose heights are along Z and whose normal
// it gives; the aerodynamics, which measure ride height above flat ground,
// are given ground_up(gravity). "Up" is decided by these and nowhere else.

#include "mbd/core/core.hpp"

namespace mbd {

/// Gravity in ISO 8855 world axes [m/s^2].
inline Vec3 iso_gravity() { return Vec3(0.0, 0.0, -g_accel); }

/// The normal of flat ground under `gravity`: the unit vector against it,
/// ISO's +Z when gravity is zero.
inline Vec3 ground_up(const Vec3& gravity)
{
    const Real g = gravity.norm();
    return g > Real(0.0) ? Vec3(-gravity / g) : Vec3(Vec3::UnitZ());
}

} // namespace mbd
