#pragma once

// Simplified vehicle model for QSS (quasi-steady-state) lap simulation.
//
// Captures the minimum parameters needed: mass, CG height, friction
// coefficient, aerodynamic coefficients, and drive/brake force vs speed.
// Derived from a VehicleTemplate via make_lap_vehicle().

#include "mbd/core/core.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

#include <algorithm>
#include <cmath>

namespace mbd {

// ============================================================================
// LapVehicle
// ============================================================================

struct LapVehicle {
    Real mass{1500.0};            ///< Total vehicle mass [kg]
    Real cg_height{0.5};          ///< CG height above ground [m]
    Real wheelbase{2.6};          ///< Wheelbase [m]
    Real track{1.6};              ///< Average track width [m]
    Real mu{1.2};                 ///< Combined friction coefficient
    Real CdA{0.7};                ///< Drag area [m^2]
    Real ClA{0.0};                ///< Downforce area [m^2]
    Real air_density{1.225};      ///< [kg/m^3]
    Real max_power{120000.0};     ///< Peak engine power [W]
    Real max_brake_force{30000.0};///< Peak braking force [N]
    Real traction_limit_speed{5.0}; ///< Below this V, drive force is traction-limited [m/s]

    /// Maximum drive force at given horizontal speed.
    /// Below traction_limit_speed: full traction (mu * m * g).
    /// Above: power-limited (P_max / V).
    Real F_drive_max(Real V) const;

    /// Maximum braking force (constant by default; could be made V-dependent).
    Real F_brake_max(Real /*V*/) const
    {
        return max_brake_force;
    }

    /// Aerodynamic downforce at given speed [N].
    Real downforce(Real V) const
    {
        return Real(0.5) * air_density * V * V * ClA;
    }

    /// Aerodynamic drag force at given speed [N].
    Real drag(Real V) const
    {
        return Real(0.5) * air_density * V * V * CdA;
    }

    /// Available grip force [N] = friction × normal load (weight + downforce).
    /// Used for the friction-circle limit.
    Real grip_force(Real V) const
    {
        return mu * (mass * g_accel + downforce(V));
    }
};

// ============================================================================
// Extraction from VehicleTemplate
// ============================================================================

/// Build a LapVehicle from a VehicleTemplate, extracting:
/// - mass: chassis + 4 wheels (approximate)
/// - cg_height: chassis ride height + half of half_extents.y (rough estimate)
/// - wheelbase: from front/rear axle X positions
/// - track: from average half_track
/// - mu: mean of Pacejka peak mu_x and mu_y
/// - CdA, ClA: from chassis aero config
/// - max_power: from drivetrain max engine torque × max engine speed
/// - max_brake_force: total brake torque of the four wheels / tire_radius
LapVehicle make_lap_vehicle(const VehicleTemplate& tmpl);
} // namespace mbd
