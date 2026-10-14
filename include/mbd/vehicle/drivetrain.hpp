#pragma once

// Drivetrain model: engine, gearbox, differential, brakes, wheel spin dynamics.
//
// The four wheel spins are the drivetrain's states (plan task 4.1): once
// connected, the simulator integrates them with the multibody state, by the
// same method and in the same stages, from
//
//     I * d(omega)/dt = T_drive - T_brake sign(omega) - Fx * R,
//
// with I the wheel's inertia (and, for a driven wheel, its share of the
// engine's through the gears), Fx the tyre's longitudinal force at that
// evaluation and R its rolling radius. The wheels are not yet bodies of the
// multibody model (task 7.3). Every wheel has its own spin, driven or not, and
// every tyre computes its slip from it, so brake and drive torques reach the
// road on all four corners.
//
// The brake is dry friction. On a turning wheel it opposes the rotation, in
// the direction the wheel turned at the start of the step, kept through the
// step so that the rates stay smooth within it (task 4.2: an error-controlled
// step would otherwise shrink without end where the sign flips); a wheel that
// comes to rest within a step while the brake can hold it (the other torques
// below the brake torque) is locked at zero, and released when they exceed
// it or the pedal is let go. Locking, the brake's direction and gear shifts
// are decided at the end of each step (end_step).
//
// At low speed the tyre's slip stiffness makes the spin equation stiff:
// below the tyre's V_low an undriven 40 kg wheel's spin relaxes to free
// rolling at 2800 to 3100 per second (measured on the simple vehicle at 0.5
// m/s: RK4 holds it at 0.9 ms, not at 1 ms, its limit being 2.785 / dt).
// Beyond that limit RK4's spins do not blow up but chatter, bounded by the
// tyre's saturation (a few per cent of the speed at 1 ms). Use shorter
// steps, or Integrator::GeneralizedAlpha (task 4.3), stable at any step.

#include "mbd/core/core.hpp"
#include "mbd/vehicle/drivetrain_params.hpp"
#include "mbd/vehicle/vehicle.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

namespace mbd {

// ============================================================================
// Drivetrain class
// ============================================================================

class Drivetrain : public kernel::StateElement {
public:
    /// The four tyres in corner order: FL, FR, RL, RR.
    using TireSet = std::array<FullTireForce*, 4>;

    DrivetrainParams params;

    // --- Control inputs (set by user each step) ---
    Real throttle{0.0};    ///< 0..1
    Real brake{0.0};       ///< 0..1

    // --- State ---
    /// Wheel spins [rad/s]. Set by initialize(); once connected, the
    /// simulator holds the spins in its x, and this copy follows them at the
    /// end of every step.
    std::array<Real, 4> wheel_omega{{0.0, 0.0, 0.0, 0.0}};
    int current_gear{1};   ///< 1-indexed
    /// Wheels held at rest by their brakes.
    std::array<bool, 4> wheel_locked{{false, false, false, false}};

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
    static Real torque_curve(Real rpm, const EngineParams& ep);

    /// Compute engine torque at given RPM and throttle.
    static Real compute_engine_torque(Real rpm, Real throttle_input,
                                      const EngineParams& ep);

    // --- Gear ratio helpers ---

    Real total_ratio() const;

    int num_gears() const { return static_cast<int>(params.gearbox.ratios.size()); }

    /// Compute engine RPM from wheel angular velocity.
    Real omega_to_rpm(Real omega_wheel_avg) const;

    // --- Initialization ---

    /// Set wheel inertias from the vehicle, wheel omegas to match the current
    /// vehicle speed (free-rolling), and the gear for that speed.
    void initialize(const kernel::Simulator& sim, const VehicleModel& vm);

    /// Initialize from a VehicleHandle (template-built vehicle).
    void initialize(const kernel::Simulator& sim, const VehicleHandle& vh);

    // --- As a state element of the simulator (task 4.1) ---

    const char* name() const override { return "drivetrain"; }
    int size() const override { return 4; }
    /// The spins from wheel_omega.
    void initial_states(kernel::VecRef x) const override;
    /// Hands the spins of this evaluation to the tyres.
    void begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x) override;
    /// Engine, drive and brake torques at these spins, and the spins' rates.
    void rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot) override;
    /// Locks and releases braked wheels, shifts gear, updates wheel_omega.
    void end_step(const kernel::Simulator& sim, Real t, kernel::VecRef x) override;

    // --- Helpers ---

    bool is_driven(int corner) const;

    /// Inertia that a torque at wheel `corner` has to accelerate: the wheel
    /// itself and, for a driven wheel, its share of the engine inertia seen
    /// through the gear train.
    Real compute_effective_inertia(int corner) const;

    // --- Convenience: connect to simulator ---

    /// Attach this drivetrain to a simulator as a state element driving the
    /// given tyres: its spins start at wheel_omega (call initialize() first).
    /// The drivetrain must outlive the simulator.
    void connect(kernel::Simulator& sim, const TireSet& tires);

    void connect(kernel::Simulator& sim, const VehicleModel& vm);

    /// Connect to a simulator using a VehicleHandle.
    void connect(kernel::Simulator& sim, const VehicleHandle& vh);

private:
    static TireSet tires_of(const VehicleHandle& vh);

    /// Mean of the driven wheels' spins.
    Real driven_wheel_speed(const std::array<Real, 4>& omega) const;
    Real driven_wheel_speed() const { return driven_wheel_speed(wheel_omega); }

    void initialize_from_speed(const std::vector<RigidBodyState>& states, BodyIndex chassis,
                               const std::array<Real, 4>& free_radius);

    /// Engine speed and torque, drive and brake torques at these spins, in
    /// the telemetry members (no gear change).
    void update_torques(const std::array<Real, 4>& omega);

    /// The torque on wheel c apart from the brake, at the tyre's last
    /// evaluation: T_drive - Fx * R.
    Real free_torque(int c) const;

    TireSet tires_{{nullptr, nullptr, nullptr, nullptr}};
    std::array<Real, 4> omega_last_step_{{0.0, 0.0, 0.0, 0.0}};
    Real last_step_time_{0.0};
    /// The sign the brake torque opposes during the current step (0 for a
    /// locked wheel), set at the end of the step before.
    std::array<Real, 4> brake_direction_{{0.0, 0.0, 0.0, 0.0}};
};

} // namespace mbd
