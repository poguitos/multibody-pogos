#pragma once

// A vehicle's mass properties (plan task 7.11, finding F9, decision D72).
//
// The chassis body is made from its own mass, centre of mass and rotational
// inertia (ChassisConfig: com, inertia, or the solid box of half_extents) and
// the masses it carries, a driver, fuel, payload (lumped_masses), added by
// parallel axes. mass_report rolls up the built car as it stands: its total
// mass, its centre of mass and that point's height above the road, its
// inertia about it, and the sprung and unsprung mass at each corner.
//
// Sprung and unsprung, as SAE J670 defines them: the unsprung mass is what the
// tyres carry directly, the sprung mass what the suspension carries. A part
// joined to both, such as an arm, is shared between them. Each corner's
// unsprung mass is found by virtual work (CornerHandle::unsprung_mass): the
// weight the corner's own parts put on its tyre, the chassis held and the
// spring taken away. The sprung mass is the rest, its weight acting where the
// whole car's acts once each corner's unsprung weight is taken off at its
// contact; each corner's share of it follows by the lever rule along the
// road, first between the axles, then side to side on each axle. Four
// supports leave that split open (the springs decide how weight goes across
// the diagonals); the lever rule on each axle is the convention used here.

#include <array>
#include <iosfwd>
#include <string>

#include "mbd/vehicle/vehicle_template.hpp"

namespace mbd {

/// The chassis body's inertia, in the chassis frame: its own mass at `com`
/// with its `inertia` (or the solid box of half_extents, about `com`), and
/// each lumped mass at its position with its own inertia, as one body
/// (combine, mbd/vehicle/suspension/mass.hpp). Throws MBD-V008 for a chassis
/// mass that is not positive, a lumped mass that is negative, a value that is
/// not finite, or a rotational inertia that no real body has (inertia_fault).
RigidBodyInertia chassis_inertia(const ChassisConfig& chassis);

/// A built vehicle's mass properties, as it stands in a simulator.
struct MassReport {
    std::string name;                      ///< The template's
    Real total_mass{0.0};                  ///< Every body's [kg]
    Vec3 centre_of_mass{Vec3::Zero()};     ///< In the chassis frame [m]
    /// The centre of mass's height above the road beneath it, along the
    /// road's normal there [m].
    Real cg_height{0.0};
    /// The rotational inertia about the centre of mass, in the chassis's axes
    /// [kg m^2]: every body's own, turned into those axes, and its mass's
    /// about the centre of mass (parallel axes). Products of inertia carry
    /// their minus signs (I_xz = -sum(m x z)).
    Mat3 inertia{Mat3::Zero()};
    Real roll_inertia() const { return inertia(0, 0); }    ///< About the chassis's X through the centre of mass
    Real pitch_inertia() const { return inertia(1, 1); }   ///< About its Y
    Real yaw_inertia() const { return inertia(2, 2); }     ///< About its Z

    /// Each tyre's contact with the road (wheel_contact), in the chassis
    /// frame [m]: FL, FR, RL, RR.
    std::array<Vec3, 4> contact{};
    std::array<Real, 4> unsprung{};        ///< Each corner's unsprung mass (CornerHandle::unsprung_mass) [kg]
    Real sprung_mass{0.0};                 ///< total_mass less the four unsprung masses [kg]
    /// The sprung mass each corner carries [kg]. Positions are measured along
    /// the road beneath the centre of mass: forward (the chassis's X laid on
    /// the road) and to the left. The sprung mass's weight acts where the
    /// whole car's first moment less each unsprung mass's at its contact puts
    /// it; each axle's share of it is by the lever rule between the axles
    /// (each at its contacts' mean), and the left's and the right's by the
    /// lever rule between that axle's two contacts.
    std::array<Real, 4> sprung{};

    /// A corner's sprung and unsprung mass: its tyre's load at rest over g,
    /// with the lever rule's split [kg].
    Real corner_mass(int c) const
    {
        return sprung[static_cast<std::size_t>(c)] + unsprung[static_cast<std::size_t>(c)];
    }
};

/// The mass properties of the vehicle `vh` built in `sim`'s system, in the
/// simulator's present state: after set_vehicle_equilibrium, at its design
/// position at rest on its road. Throws MBD-V008 if a corner has no tyre.
MassReport mass_report(const kernel::Simulator& sim, const VehicleHandle& vh);

/// The report as a table.
std::ostream& operator<<(std::ostream& os, const MassReport& report);

namespace detail {

/// The share of `mass`, whose weight acts at x, carried at x_a when it rests
/// on supports at x_a and x_b (the lever rule): mass (x - x_b) / (x_a - x_b).
inline Real lever_share(Real mass, Real x, Real x_a, Real x_b) { return mass * (x - x_b) / (x_a - x_b); }

} // namespace detail

} // namespace mbd
