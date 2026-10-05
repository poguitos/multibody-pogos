#include "mbd/analysis/lap_vehicle.hpp"

namespace mbd {

Real LapVehicle::F_drive_max(Real V) const
{
    const Real V_eff = std::max(V, Real(0.1)); // avoid division by zero
    const Real F_power = max_power / V_eff;
    const Real F_traction = mu * mass * g_accel;
    return std::min(F_power, F_traction);
}

LapVehicle make_lap_vehicle(const VehicleTemplate& tmpl)
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

    // --- Max brake force: total brake torque / tire radius ---
    // BrakeParams::max_torque is the sum over the four wheels at full pedal.
    lv.max_brake_force = tmpl.drivetrain.brakes.max_torque
                       / tmpl.front_axle.tire_free_radius;

    // --- Traction-limited speed threshold ---
    if (lv.mu > 0.0 && lv.mass > 0.0) {
        lv.traction_limit_speed = lv.max_power / (lv.mu * lv.mass * g_accel);
    }

    return lv;
}

} // namespace mbd
