#pragma once

// Bicycle model (single-track model) for steady-state cornering analysis.
//
// Collapses left/right tires into single equivalent axle forces.
// Computes understeer gradient, characteristic speed, yaw rate gain,
// and the full nonlinear δ vs a_y cornering diagram.
//
// Sign convention:
//   Positive steering angle δ = left turn
//   Positive lateral acceleration a_y = leftward (centripetal in left turn)
//   Positive slip angle α = generates positive Fy

#include "mbd/core/core.hpp"
#include "mbd/forces/pacejka.hpp"
#include "mbd/vehicle/vehicle.hpp"

#include <vector>
#include <cmath>
#include <algorithm>

namespace mbd {

// ============================================================================
// Bicycle model parameters
// ============================================================================

struct BicycleModelParams {
    Real mass{1560.0};
    Real front_axle_x{1.35};  ///< Distance from CG to front axle [m] (a)
    Real rear_axle_x{1.35};   ///< Distance from CG to rear axle [m] (b)
    PacejkaTireParams tire_front{PacejkaTireParams::DefaultPassengerCar()};
    PacejkaTireParams tire_rear{PacejkaTireParams::DefaultPassengerCar()};

    Real wheelbase() const { return front_axle_x + rear_axle_x; }

    /// Construct from VehicleParams.
    static BicycleModelParams FromVehicle(const VehicleParams& vp);
};

// ============================================================================
// Bicycle model analysis class
// ============================================================================

class BicycleModel {
public:
    BicycleModelParams params;
    PacejkaTire tire_front;
    PacejkaTire tire_rear;

    explicit BicycleModel(const BicycleModelParams& p = BicycleModelParams{})
        : params(p)
        , tire_front(p.tire_front)
        , tire_rear(p.tire_rear)
    {}

    // --- Static loads ---

    /// Front axle load [N].
    Real front_axle_load() const
    {
        return params.mass * g_accel * params.rear_axle_x / params.wheelbase();
    }

    /// Rear axle load [N].
    Real rear_axle_load() const
    {
        return params.mass * g_accel * params.front_axle_x / params.wheelbase();
    }

    /// Front per-tire load [N].
    Real front_tire_load() const { return front_axle_load() * Real(0.5); }

    /// Rear per-tire load [N].
    Real rear_tire_load() const { return rear_axle_load() * Real(0.5); }

    // --- Cornering stiffness ---

    /// Front axle cornering stiffness [N/rad] (2 tires).
    Real front_axle_cornering_stiffness() const
    {
        return Real(2.0) * tire_front.cornering_stiffness(front_tire_load());
    }

    /// Rear axle cornering stiffness [N/rad] (2 tires).
    Real rear_axle_cornering_stiffness() const
    {
        return Real(2.0) * tire_rear.cornering_stiffness(rear_tire_load());
    }

    // --- Linear understeer gradient ---

    /// Understeer gradient [rad / (m/s^2)].
    /// Positive = understeer, negative = oversteer, zero = neutral.
    Real understeer_gradient() const;

    /// Characteristic speed [m/s] (only meaningful for understeer, K_us > 0).
    /// At this speed, yaw rate gain is half the low-speed value.
    Real characteristic_speed() const;

    /// Low-speed (kinematic) yaw rate gain [1/m].
    /// At zero speed: d(r)/d(delta) = V/L.
    /// At speed V: d(r)/d(delta) = V / (L + K_us * V^2).
    Real yaw_rate_gain(Real V) const;

    // --- Linear steady-state steering angle ---

    /// Required steering angle at speed V and lateral acceleration a_y [rad].
    /// Linear model: delta = (L/V^2 + K_us) * a_y.
    Real linear_steering_angle(Real V, Real a_y) const;

    // --- Nonlinear steady-state (Pacejka-based) ---

    /// Invert Pacejka lateral force to find slip angle.
    /// Finds alpha such that 2 * Fy(alpha, Fz_per_tire) = F_axle_required.
    /// Returns alpha in radians. Returns NaN if the required force exceeds the tire limit.
    static Real invert_axle_force(const PacejkaTire& tire,
                                  Real F_axle_required,
                                  Real Fz_per_tire,
                                  Real alpha_guess = 0.0);

    /// Compute the nonlinear steady-state steering angle for given speed and
    /// lateral acceleration.
    /// Returns NaN if the required forces exceed the tire limit.
    Real nonlinear_steering_angle(Real V, Real a_y) const;

    // --- Cornering diagram ---

    struct CorneringPoint {
        Real a_y{0.0};        ///< Lateral acceleration [m/s^2]
        Real delta_linear{0.0};    ///< Steering angle, linear model [rad]
        Real delta_nonlinear{0.0}; ///< Steering angle, Pacejka model [rad]
        Real alpha_f{0.0};    ///< Front slip angle [rad]
        Real alpha_r{0.0};    ///< Rear slip angle [rad]
        bool valid{true};
    };

    /// Compute the steady-state cornering diagram: δ vs a_y at a given speed.
    /// Sweeps a_y from 0 to a_y_max in n_steps.
    std::vector<CorneringPoint> cornering_diagram(
        Real V,
        Real a_y_max = 0.0,
        int n_steps = 41) const;

    /// Maximum lateral acceleration [m/s^2] before either axle saturates.
    Real max_lateral_acceleration() const;
};

} // namespace mbd
