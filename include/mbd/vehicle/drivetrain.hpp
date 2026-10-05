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

    // --- Per-stage callback: hand the wheel spins to the tyres ---

    void apply_to_tires(const TireSet& tires) const;

    void apply_to_tires(const VehicleModel& vm) const;

    // --- Post-step: compute torques, integrate wheel spin, auto-shift ---

    void step(Real dt, const TireSet& tires);

    void step(Real dt, const VehicleModel& vm);

    void step(Real dt, const VehicleHandle& vh);

    // --- Helpers ---

    bool is_driven(int corner) const;

    /// Inertia that a torque at wheel `corner` has to accelerate: the wheel
    /// itself and, for a driven wheel, its share of the engine inertia seen
    /// through the gear train.
    Real compute_effective_inertia(int corner) const;

    // --- Convenience: connect to simulator ---

    /// Register this drivetrain's callbacks on a Simulator.
    /// Call after simulator.initialize().
    void connect(kernel::Simulator& sim, const TireSet& tires);

    void connect(kernel::Simulator& sim, const VehicleModel& vm);

    /// Connect to a simulator using a VehicleHandle.
    void connect(kernel::Simulator& sim, const VehicleHandle& vh);

private:
    static TireSet tires_of(const VehicleHandle& vh);

    /// Mean spin of the driven wheels.
    Real driven_wheel_speed() const;

    void initialize_from_speed(const std::vector<RigidBodyState>& states, BodyIndex chassis,
                               const std::array<Real, 4>& free_radius);

    /// One step of the spin equation of wheel c,
    ///
    ///     I * d(omega)/dt = T_drive + T_brake - Fx * R,
    ///
    /// with Fx the longitudinal tyre force (positive = traction) and R the
    /// rolling radius. Returns the new spin.
    Real advance_wheel(int c, Real dt, const FullTireForce& tire) const;
};

} // namespace mbd
