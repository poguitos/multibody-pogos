#pragma once

// Aerodynamic forces on the vehicle chassis.
//
// Implements drag, downforce, and (optionally) ride-height-dependent
// downforce. Forces are applied at a configurable center of pressure
// on the chassis body.

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"

#include <cmath>

namespace mbd {

// ============================================================================
// Aero parameters
// ============================================================================

struct AeroParams {
    Real CdA{0.7};      ///< Drag area [m^2]. Typical sedan: 0.6-0.8. Sports: 0.4-0.6. Race: 0.8-1.5
    Real ClA{0.0};      ///< Downforce area [m^2]. Positive = downforce. Sedan: ~0. Race: 1-4.
    Real air_density{1.225};  ///< [kg/m^3] at sea level

    /// Center of pressure offset from chassis body origin (CG), in chassis frame.
    Vec3 cop_offset_chassis{Vec3::Zero()};

    /// Ride-height-dependent downforce.
    /// Effective ClA = ClA + dClA_dh * (h_ref - h) when h < h_ref, else just ClA.
    /// Set dClA_dh = 0 to disable.
    Real h_ref{0.10};       ///< Reference ride height [m]
    Real dClA_dh{0.0};      ///< Sensitivity [m^2 / m]. Set 0 to disable.
};

// ============================================================================
// Aerodynamic force element
// ============================================================================

class AerodynamicForce : public ForceElement {
public:
    BodyIndex chassis_idx;
    AeroParams params;

    AerodynamicForce(BodyIndex chassis, const AeroParams& p);

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;
};

} // namespace mbd
