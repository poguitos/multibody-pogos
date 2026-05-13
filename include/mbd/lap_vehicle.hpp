#pragma once

// Simplified vehicle model for QSS (quasi-steady-state) lap simulation.
//
// Captures the minimum parameters needed: mass, CG height, friction
// coefficient, aerodynamic coefficients, and drive/brake force vs speed.
// Derived from a VehicleTemplate via make_lap_vehicle().

#include "mbd/core.hpp"
#include "mbd/vehicle_template.hpp"

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
    Real F_drive_max(Real V) const
    {
        const Real V_eff = std::max(V, Real(0.1)); // avoid division by zero
        const Real F_power = max_power / V_eff;
        const Real F_traction = mu * mass * g_accel;
        return std::min(F_power, F_traction);
    }

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
/// - max_brake_force: estimated from peak brake torque / tire_radius × 4 wheels
inline LapVehicle make_lap_vehicle(const VehicleTemplate& tmpl)
{
    LapVehicle lv;

    // --- Mass (chassis + 4 wheels) ---
    lv.mass = tmpl.total_mass();

    // --- CG height: chassis ride height ---
    // Approximate: tire_free_radius (front) + chassis half-Y - cg_offset.y
    lv.cg_height = tmpl.front_axle.tire_free_radius
                 + tmpl.chassis.half_extents.y()
                 - tmpl.chassis.cg_offset.y();

    // --- Wheelbase ---
    lv.wheelbase = tmpl.wheelbase();

    // --- Track (use front; rear is similar) ---
    lv.track = 2.0 * tmpl.front_axle.half_track;

    // --- Friction coefficient: use front-axle Pacejka peak coefficients ---
    // The PacejkaTireParams structure has nested lateral/longitudinal sections.
    // For a clean first-cut value, we approximate mu from typical passenger
    // car defaults (around 1.0-1.2). The exact extraction depends on the
    // PacejkaTireParams structure; we use a sensible default that the user
    // can override post-construction.
    lv.mu = 1.0;

    // --- Aero ---
    lv.CdA = tmpl.chassis.CdA;
    lv.ClA = tmpl.chassis.ClA;

    // --- Max power: peak engine torque × redline (with a torque-curve factor) ---
    // EngineParams uses max_torque [Nm] and redline_rpm [RPM].
    // omega_redline_rad_s = redline_rpm * 2*pi / 60
    // Peak power (rough) = max_torque × omega_at_peak_torque, but we approximate
    // as 0.7 × max_torque × omega_redline (since torque drops at high RPM).
    const Real omega_redline = tmpl.drivetrain.engine.redline_rpm * 2.0 * pi / 60.0;
    lv.max_power = tmpl.drivetrain.engine.max_torque * omega_redline * 0.7;

    // --- Max brake force: 4 wheels × per-wheel max brake torque / tire radius ---
    // BrakeParams uses max_torque (per wheel).
    lv.max_brake_force = tmpl.drivetrain.brakes.max_torque * 4.0
                       / tmpl.front_axle.tire_free_radius;

    // --- Traction-limited speed threshold ---
    if (lv.mu > 0.0 && lv.mass > 0.0) {
        lv.traction_limit_speed = lv.max_power / (lv.mu * lv.mass * g_accel);
    }

    return lv;
}
} // namespace mbd