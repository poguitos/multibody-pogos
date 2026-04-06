#pragma once

// Drivetrain model: engine, gearbox, differential, brakes, wheel spin dynamics.
//
// The drivetrain manages wheel angular velocities and couples to the chassis
// purely through tire slip forces. It does NOT add forces to the MBS directly.

#include "mbd/core.hpp"
#include "mbd/vehicle.hpp"
#include "mbd/simulator.hpp"

#include <array>
#include <algorithm>
#include <cmath>

namespace mbd {

// ============================================================================
// Drive layout
// ============================================================================

enum class DriveLayout { RWD, FWD, AWD };

// ============================================================================
// Engine parameters
// ============================================================================

struct EngineParams {
    Real max_torque{400.0};            ///< Peak torque [Nm]
    Real idle_rpm{1000.0};
    Real peak_torque_rpm{4500.0};      ///< RPM at peak torque
    Real redline_rpm{7000.0};
    Real idle_torque_fraction{0.4};    ///< Fraction of max_torque at idle
    Real redline_torque_fraction{0.7}; ///< Fraction of max_torque at redline
    Real inertia{0.15};               ///< Crankshaft + flywheel [kg*m^2]
};

// ============================================================================
// Gearbox parameters
// ============================================================================

struct GearboxParams {
    std::vector<Real> ratios{{3.5, 2.3, 1.7, 1.3, 1.0, 0.8}};
    Real final_drive{3.5};
    Real efficiency{0.92};
    Real shift_up_rpm{6500.0};
    Real shift_down_rpm{2000.0};
};

// ============================================================================
// Brake parameters
// ============================================================================

struct BrakeParams {
    Real max_torque{3000.0};     ///< Maximum brake torque per wheel [Nm]
    Real front_bias{0.65};       ///< Front brake proportion (0..1)
};

// ============================================================================
// Full drivetrain parameters
// ============================================================================

struct DrivetrainParams {
    EngineParams engine;
    GearboxParams gearbox;
    BrakeParams brakes;
    DriveLayout layout{DriveLayout::RWD};
    Real front_torque_split{0.4}; ///< For AWD: fraction of torque to front (0..1)
};

// ============================================================================
// Drivetrain class
// ============================================================================

class Drivetrain {
public:
    DrivetrainParams params;

    // --- Control inputs (set by user each step) ---
    Real throttle{0.0};    ///< 0..1
    Real brake{0.0};       ///< 0..1

    // --- State ---
    std::array<Real, 4> wheel_omega{{0.0, 0.0, 0.0, 0.0}};
    int current_gear{1};   ///< 1-indexed

    // --- Cached telemetry ---
    Real engine_rpm{0.0};
    Real engine_torque_out{0.0};
    std::array<Real, 4> drive_torque{{0.0, 0.0, 0.0, 0.0}};
    std::array<Real, 4> brake_torque_out{{0.0, 0.0, 0.0, 0.0}};

    explicit Drivetrain(const DrivetrainParams& p = DrivetrainParams{})
        : params(p)
    {}

    // --- Engine torque curve ---

    /// Normalized engine torque curve (0..1), piecewise linear.
    static Real torque_curve(Real rpm, const EngineParams& ep)
    {
        if (rpm < ep.idle_rpm) return ep.idle_torque_fraction;
        if (rpm > ep.redline_rpm) return Real(0.0); // Rev limiter

        if (rpm <= ep.peak_torque_rpm) {
            const Real t = (rpm - ep.idle_rpm) / (ep.peak_torque_rpm - ep.idle_rpm);
            return ep.idle_torque_fraction + t * (Real(1.0) - ep.idle_torque_fraction);
        } else {
            const Real t = (rpm - ep.peak_torque_rpm) / (ep.redline_rpm - ep.peak_torque_rpm);
            return Real(1.0) + t * (ep.redline_torque_fraction - Real(1.0));
        }
    }

    /// Compute engine torque at given RPM and throttle.
    static Real compute_engine_torque(Real rpm, Real throttle_input,
                                      const EngineParams& ep)
    {
        const Real clamped_throttle = std::clamp(throttle_input, Real(0.0), Real(1.0));
        return ep.max_torque * clamped_throttle * torque_curve(rpm, ep);
    }

    // --- Gear ratio helpers ---

    Real total_ratio() const
    {
        int idx = std::clamp(current_gear, 1, static_cast<int>(params.gearbox.ratios.size())) - 1;
        return params.gearbox.ratios[idx] * params.gearbox.final_drive;
    }

    int num_gears() const { return static_cast<int>(params.gearbox.ratios.size()); }

    /// Compute engine RPM from wheel angular velocity.
    Real omega_to_rpm(Real omega_wheel_avg) const
    {
        const Real omega_engine = std::abs(omega_wheel_avg) * total_ratio();
        return omega_engine * Real(60.0) / (Real(2.0) * pi);
    }

    // --- Initialization ---

    /// Set wheel omegas to match the current vehicle speed (free-rolling).
    void initialize(const MultibodySystem& sys, const VehicleModel& vm)
    {
        const Vec3 fwd_W = sys.states[vm.chassis_body].q_WB * Vec3::UnitX();
        const Real Vx = sys.states[vm.chassis_body].v_WB.dot(fwd_W);

        const Real R_eff = vm.params.tire_free_radius * Real(0.97);
        const Real omega_init = std::max(Vx, Real(0.0)) / R_eff;

        for (int c = 0; c < 4; ++c) {
            wheel_omega[c] = omega_init;
        }

        // Select appropriate gear for current speed
        if (Vx < Real(0.5)) {
            current_gear = 1;
        } else {
            const Real rpm_target = (params.engine.peak_torque_rpm +
                                     params.engine.idle_rpm) * Real(0.5);
            for (int g = num_gears(); g >= 1; --g) {
                current_gear = g;
                if (omega_to_rpm(omega_init) >= rpm_target) break;
            }
        }

        engine_rpm = std::max(omega_to_rpm(omega_init), params.engine.idle_rpm);
    }

    // --- Per-stage callback: set tire omegas ---

    void apply_to_tires(const VehicleModel& vm) const
    {
        for (int c = 0; c < 4; ++c) {
            if (is_driven(c)) {
                vm.tires[c]->omega_wheel = wheel_omega[c];
                vm.tires[c]->auto_free_roll = false;
            } else {
                vm.tires[c]->auto_free_roll = true;
            }
        }
    }

    // --- Post-step: compute torques, integrate wheel spin, auto-shift ---

    void step(Real dt, const VehicleModel& vm)
    {
        const auto& ep = params.engine;
        const auto& gp = params.gearbox;
        const auto& bp = params.brakes;

        // --- Average driven wheel omega for RPM computation ---
        Real omega_driven_avg = Real(0.0);
        int n_driven = 0;
        for (int c = 0; c < 4; ++c) {
            if (is_driven(c)) {
                omega_driven_avg += wheel_omega[c];
                ++n_driven;
            }
        }
        if (n_driven > 0) omega_driven_avg /= n_driven;

        // --- Engine RPM ---
        engine_rpm = omega_to_rpm(omega_driven_avg);
        engine_rpm = std::max(engine_rpm, ep.idle_rpm);

        // --- Auto shift ---
        if (engine_rpm > gp.shift_up_rpm && current_gear < num_gears()) {
            current_gear++;
            engine_rpm = omega_to_rpm(omega_driven_avg);
        } else if (engine_rpm < gp.shift_down_rpm && current_gear > 1) {
            current_gear--;
            engine_rpm = omega_to_rpm(omega_driven_avg);
        }
        engine_rpm = std::max(engine_rpm, ep.idle_rpm);

        // --- Engine torque ---
        engine_torque_out = compute_engine_torque(engine_rpm, throttle, ep);

        // --- Drive torque distribution ---
        const Real T_at_wheels = engine_torque_out * total_ratio() * gp.efficiency;

        // Distribute to driven wheels
        drive_torque.fill(0.0);
        switch (params.layout) {
            case DriveLayout::RWD:
                drive_torque[2] = T_at_wheels * Real(0.5); // RL
                drive_torque[3] = T_at_wheels * Real(0.5); // RR
                break;
            case DriveLayout::FWD:
                drive_torque[0] = T_at_wheels * Real(0.5); // FL
                drive_torque[1] = T_at_wheels * Real(0.5); // FR
                break;
            case DriveLayout::AWD: {
                const Real T_front = T_at_wheels * params.front_torque_split;
                const Real T_rear  = T_at_wheels * (Real(1.0) - params.front_torque_split);
                drive_torque[0] = T_front * Real(0.5);
                drive_torque[1] = T_front * Real(0.5);
                drive_torque[2] = T_rear  * Real(0.5);
                drive_torque[3] = T_rear  * Real(0.5);
                break;
            }
        }

        // --- Brake torque ---
        const Real clamped_brake = std::clamp(brake, Real(0.0), Real(1.0));
        const Real T_brake_total = clamped_brake * bp.max_torque;
        const Real T_brake_front = T_brake_total * bp.front_bias;
        const Real T_brake_rear  = T_brake_total * (Real(1.0) - bp.front_bias);

        brake_torque_out[0] = T_brake_front;
        brake_torque_out[1] = T_brake_front;
        brake_torque_out[2] = T_brake_rear;
        brake_torque_out[3] = T_brake_rear;

        // --- Integrate wheel spin ODE per wheel ---
        for (int c = 0; c < 4; ++c) {
            const Real R_eff = vm.params.tire_free_radius -
                               Real(0.5) * vm.tires[c]->get_deflection();
            const Real Fx = vm.tires[c]->get_Fx();

            const Real I_eff = compute_effective_inertia(c);

            // Net torque: drive - brake (opposing rotation) - tire reaction
            Real T_brake_applied = brake_torque_out[c];
            if (wheel_omega[c] > Real(0.0)) {
                T_brake_applied = -T_brake_applied;
            } else if (wheel_omega[c] < Real(0.0)) {
                // Keep positive to decelerate negative rotation
            } else {
                T_brake_applied = Real(0.0);
            }

            const Real T_net = drive_torque[c] + T_brake_applied - Fx * R_eff;
            Real omega_new = wheel_omega[c] + (T_net / I_eff) * dt;

            // Prevent brake from reversing wheel direction
            if (clamped_brake > Real(0.01)) {
                if (wheel_omega[c] >= Real(0.0) && omega_new < Real(0.0)) {
                    omega_new = Real(0.0);
                } else if (wheel_omega[c] <= Real(0.0) && omega_new > Real(0.0)) {
                    omega_new = Real(0.0);
                }
            }

            // Prevent negative omega (wheels don't spin backwards in normal driving)
            if (omega_new < Real(0.0) && drive_torque[c] >= Real(0.0)) {
                omega_new = Real(0.0);
            }

            wheel_omega[c] = omega_new;
        }
    }

    // --- Helpers ---

    bool is_driven(int corner) const
    {
        switch (params.layout) {
            case DriveLayout::RWD: return corner >= 2;
            case DriveLayout::FWD: return corner < 2;
            case DriveLayout::AWD: return true;
        }
        return false;
    }

    Real compute_effective_inertia(int corner) const
    {
        // Wheel rotational inertia: I = 0.5 * m * R^2 (solid cylinder approx)
        const Real I_wheel = Real(0.5) * Real(40.0) * Real(0.35) * Real(0.35);

        if (!is_driven(corner)) return I_wheel;

        // Add reflected engine inertia through gear train
        int n_driven = 0;
        for (int c = 0; c < 4; ++c) {
            if (is_driven(c)) ++n_driven;
        }

        const Real ratio = total_ratio();
        const Real I_reflected = params.engine.inertia * ratio * ratio /
                                 static_cast<Real>(n_driven);

        return I_wheel + I_reflected;
    }

    // --- Convenience: connect to simulator ---

    /// Register this drivetrain's callbacks on a Simulator.
    /// Call after simulator.initialize().
    void connect(Simulator& sim, const VehicleModel& vm)
    {
        sim.pre_force_callback = [this, &vm](MultibodySystem& /*sys*/, Real /*t*/) {
            apply_to_tires(vm);
        };

        sim.post_step_callback = [this, &vm](MultibodySystem& /*sys*/, Real dt) {
            step(dt, vm);
        };
    }
};

} // namespace mbd