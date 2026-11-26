#pragma once

// Trailing-arm suspension corner (plan task 7.6, decision D67).
//
// One arm, hinged to the chassis about an axis, carries the wheel's hub: a
// trailing arm when the axis is lateral, a semi-trailing arm when it is
// turned in plan view. One degree of freedom and no loop: the wheel centre
// moves on a circle about the axis. The arm is the corner's upright (it takes
// the spring, the brake's reaction and the wheel's spin joint); its frame has
// its origin at the wheel centre and the chassis's axes at the reference.

#include "mbd/kernel/simulator.hpp"

namespace mbd {

struct TrailingArmParams {
    // Chassis coordinates (ISO 8855: X forward, Y left, Z up): a left corner.
    Vec3 pivot{0.45, 0.40, 0.30};   ///< A point on the arm's hinge axis
    Vec3 axis{Vec3::UnitY()};       ///< The hinge axis's direction
    Vec3 wheel_center{0.0, 0.75, 0.28};
    Real arm_mass{5.0};             ///< The arm, a rod from the pivot to the wheel centre [kg]
    Real hub_mass{15.0};            ///< The hub carrier at the wheel centre [kg]
};

struct TrailingArmCorner {
    BodyIndex arm_body{0};
    TrailingArmParams params;
};

/// The arm on a revolute joint on `chassis_body` (body 0 for an arm on the
/// ground), at the reference with the chassis's axes and its origin at the
/// wheel centre. Throws MBD-V002 for an axis that is zero or a pivot on the
/// wheel centre.
TrailingArmCorner build_trailing_arm_corner(kernel::System& sys, BodyIndex chassis_body, const TrailingArmParams& p);

} // namespace mbd
