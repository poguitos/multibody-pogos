#pragma once

// Drivetrain model: engine, gearbox, differential, brakes, wheel spin dynamics.
//
// Wheel spin is a state of this class, one value per wheel, advanced once per
// simulation step. It is not yet a degree of freedom of the multibody model
// (see documentation/Master_plan.md, task 7.3). Every wheel has its own spin,
// driven or not, and every tyre computes its slip from it, so that brake and
// drive torques reach the road through the tyre on all four corners.

#include "mbd/core/core.hpp"
#include "mbd/vehicle/drivetrain_params.hpp"
#include "mbd/vehicle/vehicle.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/kernel/simulator.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

namespace mbd {

// ============================================================================
// Drivetrain class
// ============================================================================

class Drivetrain {
public:
    /// The four tyres in corner order: FL, FR, RL, RR.
    using TireSet = std::array<FullTireForce*, 4>;

    DrivetrainParams params;

    // --- Control inputs (set by user each step) ---
    Real throttle{0.0};    ///< 0..1
    Real brake{0.0};       ///< 0..1

    // --- State ---
    std::array<Real, 4> wheel_omega{{0.0, 0.0, 0.0, 0.0}};
    int current_gear{1};   ///< 1-indexed

    // --- Per-wheel data (FL, FR, RL, RR), set by initialize() ---

    /// Spin inertia of wheel and tyre about the axle [kg m^2].
    /// Default: a 40 kg, 0.35 m wheel as a uniform disc.
    std::array<Real, 4> wheel_inertia{{2.45, 2.45, 2.45, 2.45}};

    // --- Cached telemetry ---
    Real engine_rpm{0.0};
    Real engine_torque_out{0.0};
    std::array<Real, 4> drive_torque{{0.0, 0.0, 0.0, 0.0}};

    /// Brake torque applied at each wheel [Nm], as a magnitude.
    std::array<Real, 4> brake_torque_out{{0.0, 0.0, 0.0, 0.0}};

    explicit Drivetrain(const DrivetrainParams& p = DrivetrainParams{})
        : params(p)
    {}

    /// Spin inertia of a wheel approximated as a uniform disc: m * R^2 / 2.
    static Real disc_inertia(Real mass, Real radius)
    {
        return Real(0.5) * mass * radius * radius;
    }

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

    /// Set wheel inertias from the vehicle, wheel omegas to match the current
    /// vehicle speed (free-rolling), and the gear for that speed.
    void initialize(const kernel::Simulator& sim, const VehicleModel& vm)
    {
        const Real R = vm.params.tire_free_radius;
        const Real I = disc_inertia(vm.params.wheel_mass, R);
        wheel_inertia = {{I, I, I, I}};
        initialize_from_speed(sim.states(), vm.chassis_body, {{R, R, R, R}});
    }

    /// Initialize from a VehicleHandle (template-built vehicle).
    void initialize(const kernel::Simulator& sim, const VehicleHandle& vh)
    {
        const auto& front = vh.tmpl.front_axle;
        const auto& rear  = vh.tmpl.rear_axle;
        const Real R_f = front.tire_free_radius;
        const Real R_r = rear.tire_free_radius;
        const Real I_f = disc_inertia(front.wheel_mass, R_f);
        const Real I_r = disc_inertia(rear.wheel_mass, R_r);
        wheel_inertia = {{I_f, I_f, I_r, I_r}};
        initialize_from_speed(sim.states(), vh.chassis_body, {{R_f, R_f, R_r, R_r}});
    }

    // --- Per-stage callback: hand the wheel spins to the tyres ---

    void apply_to_tires(const TireSet& tires) const
    {
        for (int c = 0; c < 4; ++c) {
            tires[c]->omega_wheel = wheel_omega[c];
            tires[c]->auto_free_roll = false;
        }
    }

    void apply_to_tires(const VehicleModel& vm) const
    {
        apply_to_tires(vm.tires);
    }

    // --- Post-step: compute torques, integrate wheel spin, auto-shift ---

    void step(Real dt, const TireSet& tires)
    {
        const auto& ep = params.engine;
        const auto& gp = params.gearbox;
        const auto& bp = params.brakes;

        // --- Average driven wheel omega for RPM computation ---
        const Real omega_driven_avg = driven_wheel_speed();

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
        // bp.max_torque is the sum over the four wheels at full pedal.
        const Real pedal = std::clamp(brake, Real(0.0), Real(1.0));
        const Real T_front_wheel = pedal * bp.max_torque * bp.front_bias * Real(0.5);
        const Real T_rear_wheel  = pedal * bp.max_torque * (Real(1.0) - bp.front_bias) * Real(0.5);

        brake_torque_out[0] = T_front_wheel;
        brake_torque_out[1] = T_front_wheel;
        brake_torque_out[2] = T_rear_wheel;
        brake_torque_out[3] = T_rear_wheel;

        // --- Integrate wheel spin ODE per wheel ---
        for (int c = 0; c < 4; ++c) {
            wheel_omega[c] = advance_wheel(c, dt, *tires[c]);
        }
    }

    void step(Real dt, const VehicleModel& vm)
    {
        step(dt, vm.tires);
    }

    void step(Real dt, const VehicleHandle& vh)
    {
        step(dt, tires_of(vh));
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

    /// Inertia that a torque at wheel `corner` has to accelerate: the wheel
    /// itself and, for a driven wheel, its share of the engine inertia seen
    /// through the gear train.
    Real compute_effective_inertia(int corner) const
    {
        const Real I_wheel = wheel_inertia[corner];

        if (!is_driven(corner)) return I_wheel;

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
    void connect(kernel::Simulator& sim, const TireSet& tires)
    {
        sim.pre_force_callback = [this, tires](kernel::Simulator& /*sim*/, Real /*t*/) {
            apply_to_tires(tires);
        };

        sim.post_step_callback = [this, tires](kernel::Simulator& /*sim*/, Real dt) {
            step(dt, tires);
        };
    }

    void connect(kernel::Simulator& sim, const VehicleModel& vm)
    {
        connect(sim, vm.tires);
    }

    /// Connect to a simulator using a VehicleHandle.
    void connect(kernel::Simulator& sim, const VehicleHandle& vh)
    {
        connect(sim, tires_of(vh));
    }

private:
    static TireSet tires_of(const VehicleHandle& vh)
    {
        return {{vh.corners[0].tire, vh.corners[1].tire,
                 vh.corners[2].tire, vh.corners[3].tire}};
    }

    /// Mean spin of the driven wheels.
    Real driven_wheel_speed() const
    {
        Real sum = Real(0.0);
        int n_driven = 0;
        for (int c = 0; c < 4; ++c) {
            if (is_driven(c)) {
                sum += wheel_omega[c];
                ++n_driven;
            }
        }
        return (n_driven > 0) ? sum / n_driven : Real(0.0);
    }

    void initialize_from_speed(const std::vector<RigidBodyState>& states, BodyIndex chassis,
                               const std::array<Real, 4>& free_radius)
    {
        const auto& chassis_state = states[static_cast<std::size_t>(chassis)];
        const Vec3 fwd_W = chassis_state.q_WB * Vec3::UnitX();
        const Real Vx = chassis_state.v_WB.dot(fwd_W);

        // Free rolling on a tyre that is about 3 % smaller under load.
        for (int c = 0; c < 4; ++c) {
            wheel_omega[c] = Vx / (free_radius[c] * Real(0.97));
        }

        // Select appropriate gear for current speed
        const Real omega_init = driven_wheel_speed();
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

    /// One step of the spin equation of wheel c,
    ///
    ///     I * d(omega)/dt = T_drive + T_brake - Fx * R,
    ///
    /// with Fx the longitudinal tyre force (positive = traction) and R the
    /// rolling radius. Returns the new spin.
    Real advance_wheel(int c, Real dt, const FullTireForce& tire) const
    {
        const Real R = tire.get_rolling_radius();
        const Real I = compute_effective_inertia(c);

        // Torque on the wheel apart from the brake.
        const Real T_free = drive_torque[c] - tire.get_Fx() * R;

        // The tyre force rises steeply with wheel spin: at low speed
        // d(Fx * R)/d(omega) is several thousand N m s/rad, far too stiff for
        // an explicit 1 ms step on a wheel of a few kg m^2. That slope is
        // therefore taken implicitly (linearly implicit Euler):
        //
        //     (I + dt * k) * (omega_new - omega) = dt * T(omega),
        //     k = d(Fx * R)/d(omega) >= 0.
        const Real k = tire.longitudinal_force_slope() * R;
        const Real I_step = I + dt * k;

        const Real omega   = wheel_omega[c];
        const Real T_brake = brake_torque_out[c];

        if (T_brake <= Real(0.0)) {
            return omega + dt * T_free / I_step;
        }

        // The brake is dry friction. It holds a stationary wheel as long as
        // the other torques do not exceed it ...
        if (omega == Real(0.0)) {
            if (std::abs(T_free) <= T_brake) return Real(0.0);
            const Real sign = (T_free > Real(0.0)) ? Real(1.0) : Real(-1.0);
            return dt * (T_free - sign * T_brake) / I_step;
        }

        // ... and opposes the rotation of a turning one.
        const Real direction = (omega > Real(0.0)) ? Real(1.0) : Real(-1.0);
        const Real omega_new = omega + dt * (T_free - direction * T_brake) / I_step;

        // Friction cannot reverse the rotation: a wheel brought to rest within
        // this step stays at rest, and the next step decides whether the brake
        // holds it.
        if (omega_new * direction <= Real(0.0)) return Real(0.0);
        return omega_new;
    }
};

} // namespace mbd
