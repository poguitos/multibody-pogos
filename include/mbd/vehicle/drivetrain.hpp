#pragma once

// Drivetrain model: engine, gearbox, differential, brakes.
//
// The wheels are bodies on spin joints (plan task 7.3, D62, mbd/vehicle/
// wheel.hpp): their spins are part of the multibody state, and the
// drivetrain applies torques to them. A driven wheel takes its share of the
// engine's torque through the gears, the reaction on the chassis (an inboard
// differential, half-shafts); a braked wheel takes its brake's torque, the
// reaction on its upright (outboard brakes). The tyre's longitudinal force
// turns the wheel back through its arm to the road, so
//
//     I * d(omega)/dt = T_drive + T_brake + (the tyre's moment about the axle)
//
// arises in the multibody equations, with I the wheel body's inertia about its
// axle. Every wheel has its own spin, driven or not, and every tyre computes
// its slip from it, so brake and drive torques reach the road on all four
// corners.
//
// The engine's own inertia, which the spins carried through the gears while
// they were the drivetrain's states (tasks 1.7 and 4.1), is not in the model
// until task 7.8 gives the engine a speed of its own (D62):
// params.engine.inertia is not read.
//
// The brake is dry friction. On a turning wheel it opposes the rotation
// relative to the upright, in the direction the wheel turned at the start of
// the step, kept through the step so that the rates stay smooth within it
// (task 4.2: an error-controlled step would otherwise shrink without end
// where the sign flips); a wheel that comes to rest within a step while the
// brake can hold it (the other torques below the brake torque) is locked, and
// released when they exceed it or the pedal is let go. A locked wheel turns
// with its upright: at the end of each step its spin joint's rate is set to
// zero (the drivetrain keeps the simulator it was connected to for this), and
// within the step its brake gives the other torques' opposite, with a
// damping that takes the spin the upright's own turning leaves to rest
// (lock_time), never more than the brake's torque. Locking, the brake's
// direction and gear shifts are decided at the end of each step (end_step).
//
// At low speed the tyre's slip stiffness makes the spin stiff: below the
// tyre's V_low an undriven 40 kg wheel's spin relaxes to free rolling at about
// 3000 per second (task 4.1's measurement; RK4's limit is 2.785 / dt). Beyond
// that limit RK4's spins do not blow up but chatter, bounded by the tyre's
// saturation. Use shorter steps, or Integrator::GeneralizedAlpha (task 4.3) or
// LinearlyImplicitEuler (task 4.5), stable at any step.

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
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
    /// What the drivetrain acts on at one corner.
    struct Corner {
        FullTireForce* tire{nullptr};  ///< The tyre on the wheel
        BodyIndex wheel{0};            ///< The wheel, on its spin joint
        BodyIndex upright{0};          ///< The body it turns on: the brake reacts on it
    };
    /// The four corners in order: FL, FR, RL, RR.
    using Corners = std::array<Corner, 4>;

    DrivetrainParams params;

    // --- Control inputs (set by user each step) ---
    Real throttle{0.0};    ///< 0..1
    Real brake{0.0};       ///< 0..1

    // --- State ---
    /// Each wheel's spin relative to its upright [rad/s], at the end of the
    /// last step (telemetry: the spins are the multibody state's).
    std::array<Real, 4> wheel_omega{{0.0, 0.0, 0.0, 0.0}};
    int current_gear{1};   ///< 1-indexed
    /// Wheels held at rest by their brakes.
    std::array<bool, 4> wheel_locked{{false, false, false, false}};

    /// How fast a locked wheel's brake takes the spin its upright's turning
    /// leaves within a step to rest [s]: its damping is the wheel's inertia
    /// over this time.
    Real lock_time{0.005};

    // --- Per-wheel data (FL, FR, RL, RR), set by connect() ---
    /// Spin inertia of each wheel body about its axle [kg m^2], from the model.
    std::array<Real, 4> wheel_inertia{{0.0, 0.0, 0.0, 0.0}};

    // --- Cached telemetry ---
    Real engine_rpm{0.0};
    Real engine_torque_out{0.0};
    std::array<Real, 4> drive_torque{{0.0, 0.0, 0.0, 0.0}};
    /// Brake torque available at each wheel [Nm], as a magnitude.
    std::array<Real, 4> brake_torque_out{{0.0, 0.0, 0.0, 0.0}};
    /// The brake torque applied to each wheel at the last evaluation [Nm],
    /// along the wheel's spin axis (against a forward roll: negative).
    std::array<Real, 4> brake_torque_applied{{0.0, 0.0, 0.0, 0.0}};

    explicit Drivetrain(const DrivetrainParams& p = DrivetrainParams{})
        : params(p)
    {}

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

    /// Set the wheels spinning at free rolling for the vehicle's present
    /// speed (roll_wheels), and the gear for that speed.
    void initialize(kernel::Simulator& sim, const VehicleModel& vm);

    /// Initialize from a VehicleHandle (template-built vehicle).
    void initialize(kernel::Simulator& sim, const VehicleHandle& vh);

    // --- As a state element of the simulator: no states, its end_step ---

    const char* name() const override { return "drivetrain"; }
    int size() const override { return 0; }
    void initial_states(kernel::VecRef) const override {}
    void begin_evaluation(const kernel::Simulator&, Real, kernel::ConstVecRef) override {}
    void rates(const kernel::Simulator&, Real, kernel::ConstVecRef, kernel::VecRef) override {}

    /// Locks and releases braked wheels (a locked wheel's spin joint set to
    /// rest in the simulator connected to), shifts gear, updates wheel_omega.
    void end_step(const kernel::Simulator& sim, Real t, kernel::VecRef x) override;

    /// The torques at the bodies' states: engine, drive and brake torques
    /// (telemetry updated), applied to the wheels, the chassis and the
    /// uprights. Called by the force element connect() adds to the system.
    void apply_torques(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces);

    // --- Helpers ---

    bool is_driven(int corner) const;

    // --- Convenience: connect to simulator ---

    /// Attach this drivetrain to a simulator: a force element that applies
    /// its torques goes into the simulator's system, and the drivetrain is
    /// attached as a state element for its end_step. `chassis` takes the
    /// drive torques' reactions. Call initialize() first. The drivetrain
    /// must outlive the simulator and its system.
    void connect(kernel::Simulator& sim, BodyIndex chassis, const Corners& corners);
    void connect(kernel::Simulator& sim, const VehicleModel& vm);
    /// Connect to a simulator using a VehicleHandle.
    void connect(kernel::Simulator& sim, const VehicleHandle& vh);

private:
    /// Mean of the driven wheels' spins.
    Real driven_wheel_speed(const std::array<Real, 4>& omega) const;
    Real driven_wheel_speed() const { return driven_wheel_speed(wheel_omega); }

    static Corners corners_of(const VehicleModel& vm);
    static Corners corners_of(const VehicleHandle& vh);

    /// The bodies acted on, and the wheels' inertias from the model.
    void set_corners(const kernel::Simulator& sim, BodyIndex chassis, const Corners& corners);

    /// Initialize: roll the wheels already done, the spins read and the gear
    /// chosen for them.
    void initialize_from_spins(const kernel::Simulator& sim);

    /// The gear for these spins: first below 0.5 m/s at the driven wheels,
    /// otherwise the highest that keeps the engine at or above the middle of
    /// idle and peak torque speeds.
    void choose_gear(const std::array<Real, 4>& omega);

    /// Engine speed and torque, drive and brake torques at these spins, in
    /// the telemetry members (no gear change).
    void update_torques(const std::array<Real, 4>& omega);

    /// Each wheel's spin relative to its upright, in `states`.
    std::array<Real, 4> spins(const std::vector<RigidBodyState>& states) const;

    /// The torque on wheel c apart from the brake, at the tyre's last
    /// evaluation: the drive torque and the tyre's moment about the axle.
    Real free_torque(int c) const;

    BodyIndex chassis_{0};
    Corners corners_{};
    kernel::Simulator* sim_{nullptr};   ///< The simulator connected to: end_step sets locked spins to rest
    std::array<Real, 4> omega_last_step_{{0.0, 0.0, 0.0, 0.0}};
    Real last_step_time_{0.0};
    /// The sign the brake torque opposes during the current step (0 for a
    /// locked wheel), set at the end of the step before.
    std::array<Real, 4> brake_direction_{{0.0, 0.0, 0.0, 0.0}};
};

} // namespace mbd
