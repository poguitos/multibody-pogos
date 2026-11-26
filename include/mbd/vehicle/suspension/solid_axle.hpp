#pragma once

// Solid-axle suspension (plan task 7.6, decision D67): one rigid axle carries
// both wheels of an axle, located by four links: two lower trailing links, a
// central upper link and a Panhard rod (a three-link axle). The axle is a
// free body on the chassis and each link a distance between a point of the
// chassis and one of the axle: six coordinates less four equations leave
// two degrees of freedom, heave and roll. The Panhard rod sets the axle's
// roll centre: the axle rolls about a point on the rod's line.

#include <array>

#include "mbd/kernel/simulator.hpp"

namespace mbd {

struct SolidAxleParams {
    // Chassis coordinates (ISO 8855: X forward, Y left, Z up).
    Vec3 axle_center{-1.35, 0.0, 0.0};   ///< Midway between the wheel centres
    Real half_track{0.8};                ///< The wheel centres at axle_center +- half_track Y
    /// Lower trailing links, left then right: the chassis's end and the axle's.
    std::array<Vec3, 2> lower_chassis{{Vec3(-0.85, 0.45, -0.05), Vec3(-0.85, -0.45, -0.05)}};
    std::array<Vec3, 2> lower_axle{{Vec3(-1.35, 0.45, -0.10), Vec3(-1.35, -0.45, -0.10)}};
    /// The central upper link.
    Vec3 upper_chassis{-1.00, 0.0, 0.15};
    Vec3 upper_axle{-1.35, 0.0, 0.12};
    /// The Panhard rod, across the car.
    Vec3 panhard_chassis{-1.35, -0.55, 0.05};
    Vec3 panhard_axle{-1.35, 0.55, 0.05};
    Real axle_mass{60.0};   ///< The axle with its hubs and brakes, a rod between the wheel centres [kg]
};

struct SolidAxle {
    BodyIndex axle_body{0};
    std::array<std::size_t, 4> link_constraint_indices{};   ///< Lower left, lower right, upper, Panhard
    SolidAxleParams params;
};

/// The axle on a free joint on `chassis_body` (body 0 for an axle on the
/// ground), its frame at the axle's centre with the chassis's axes at the
/// reference, and its four links. Throws MBD-V002 for a half track that is
/// not positive or a link of zero length.
SolidAxle build_solid_axle(kernel::System& sys, BodyIndex chassis_body, const SolidAxleParams& p);

} // namespace mbd
