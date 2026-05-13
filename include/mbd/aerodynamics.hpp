#pragma once

// Aerodynamic forces on the vehicle chassis.
//
// Implements drag, downforce, and (optionally) ride-height-dependent
// downforce. Forces are applied at a configurable center of pressure
// on the chassis body.

#include "mbd/core.hpp"
#include "mbd/force_element.hpp"

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

    AerodynamicForce(BodyIndex chassis, const AeroParams& p)
        : chassis_idx(chassis), params(p)
    {
        MBD_THROW_IF(p.CdA < 0.0, "AerodynamicForce: CdA must be non-negative");
        MBD_THROW_IF(p.ClA < 0.0, "AerodynamicForce: ClA must be non-negative");
        MBD_THROW_IF(p.air_density <= 0.0, "AerodynamicForce: air_density must be positive");
    }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override
    {
        const auto& chassis = states[chassis_idx];

        // Horizontal velocity of chassis CG (zero out vertical component)
        Vec3 v_horiz = chassis.v_WB;
        v_horiz.y() = 0.0;
        const Real V = v_horiz.norm();

        if (V < Real(1e-6)) return; // No aero at zero speed

        // Dynamic pressure
        const Real q_dyn = Real(0.5) * params.air_density * V * V;

        // --- Drag ---
        // F_drag = -q * CdA * v_unit (along -velocity)
        const Vec3 v_unit = v_horiz / V;
        const Vec3 F_drag = -q_dyn * params.CdA * v_unit;

        // --- Downforce ---
        // Effective ClA based on ride height
        const Real h = chassis.p_WB.y();
        Real ClA_eff = params.ClA;
        if (params.dClA_dh != Real(0.0) && h < params.h_ref) {
            ClA_eff += params.dClA_dh * (params.h_ref - h);
        }
        ClA_eff = std::max(ClA_eff, Real(0.0));

        const Vec3 F_lift = Vec3(0.0, -q_dyn * ClA_eff, 0.0); // -Y direction

        // --- Total force at CoP ---
        const Vec3 F_total = F_drag + F_lift;

        // CoP world position (relative to chassis CG)
        const Vec3 r_cop_W = chassis.q_WB * params.cop_offset_chassis;

        // Apply force at CoP and corresponding moment about chassis CG
        forces[chassis_idx].f_W   += F_total;
        forces[chassis_idx].tau_W += r_cop_W.cross(F_total);
    }
};

} // namespace mbd