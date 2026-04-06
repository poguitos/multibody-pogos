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

#include "mbd/core.hpp"
#include "mbd/pacejka.hpp"
#include "mbd/vehicle.hpp"

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
    static BicycleModelParams FromVehicle(const VehicleParams& vp)
    {
        BicycleModelParams bp;
        bp.mass = vp.total_mass();
        bp.front_axle_x = vp.front_axle_x;
        bp.rear_axle_x  = vp.rear_axle_x;
        bp.tire_front = vp.tire_params;
        bp.tire_rear  = vp.tire_params;
        return bp;
    }
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
    Real understeer_gradient() const
    {
        const Real m = params.mass;
        const Real L = params.wheelbase();
        const Real a = params.front_axle_x;
        const Real b = params.rear_axle_x;
        const Real C_f = front_axle_cornering_stiffness();
        const Real C_r = rear_axle_cornering_stiffness();

        return (m / L) * (b / C_f - a / C_r);
    }

    /// Characteristic speed [m/s] (only meaningful for understeer, K_us > 0).
    /// At this speed, yaw rate gain is half the low-speed value.
    Real characteristic_speed() const
    {
        const Real K = understeer_gradient();
        if (K <= Real(0.0)) return std::numeric_limits<Real>::infinity();
        return std::sqrt(params.wheelbase() / K);
    }

    /// Low-speed (kinematic) yaw rate gain [1/m].
    /// At zero speed: d(r)/d(delta) = V/L.
    /// At speed V: d(r)/d(delta) = V / (L + K_us * V^2).
    Real yaw_rate_gain(Real V) const
    {
        const Real K = understeer_gradient();
        return V / (params.wheelbase() + K * V * V);
    }

    // --- Linear steady-state steering angle ---

    /// Required steering angle at speed V and lateral acceleration a_y [rad].
    /// Linear model: delta = (L/V^2 + K_us) * a_y.
    Real linear_steering_angle(Real V, Real a_y) const
    {
        const Real L = params.wheelbase();
        const Real K = understeer_gradient();
        return (L / (V * V) + K) * a_y;
    }

    // --- Nonlinear steady-state (Pacejka-based) ---

    /// Invert Pacejka lateral force to find slip angle.
    /// Finds alpha such that 2 * Fy(alpha, Fz_per_tire) = F_axle_required.
    /// Returns alpha in radians. Returns NaN if the required force exceeds the tire limit.
    static Real invert_axle_force(const PacejkaTire& tire,
                                  Real F_axle_required,
                                  Real Fz_per_tire,
                                  Real alpha_guess = 0.0)
    {
        // Check if force is achievable (peak force per axle)
        const Real mu = tire.peak_mu_lateral(Fz_per_tire);
        const Real F_peak_axle = Real(2.0) * mu * Fz_per_tire;

        if (std::abs(F_axle_required) > F_peak_axle * Real(0.99)) {
            return std::numeric_limits<Real>::quiet_NaN();
        }

        // Newton-Raphson: find alpha such that 2*Fy(alpha, Fz) = F_target
        Real alpha = alpha_guess;
        if (std::abs(alpha) < Real(1e-10)) {
            // Initial guess from linear approximation
            const Real C_axle = Real(2.0) * tire.cornering_stiffness(Fz_per_tire);
            if (std::abs(C_axle) > Real(1e-6)) {
                alpha = F_axle_required / C_axle;
            }
        }

        const Real eps_fd = Real(1e-7);

        for (int iter = 0; iter < 30; ++iter) {
            auto r = tire.compute(0.0, alpha, Fz_per_tire);
            const Real F_current = Real(2.0) * r.Fy;
            const Real error = F_current - F_axle_required;

            if (std::abs(error) < Real(1e-4)) {
                return alpha;
            }

            // Finite difference derivative
            auto r_p = tire.compute(0.0, alpha + eps_fd, Fz_per_tire);
            const Real dF = Real(2.0) * (r_p.Fy - r.Fy) / eps_fd;

            if (std::abs(dF) < Real(1e-6)) break;

            Real d_alpha = -error / dF;

            // Damped step to stay in reasonable range
            const Real max_step = Real(0.05);
            d_alpha = std::clamp(d_alpha, -max_step, max_step);

            alpha += d_alpha;
        }

        return alpha;
    }

    /// Compute the nonlinear steady-state steering angle for given speed and
    /// lateral acceleration.
    /// Returns NaN if the required forces exceed the tire limit.
    Real nonlinear_steering_angle(Real V, Real a_y) const
    {
        const Real m = params.mass;
        const Real L = params.wheelbase();
        const Real a = params.front_axle_x;
        const Real b = params.rear_axle_x;

        // Required axle forces
        const Real F_yf = m * a_y * b / L;
        const Real F_yr = m * a_y * a / L;

        // Invert Pacejka to find slip angles
        const Real alpha_f = invert_axle_force(
            tire_front, F_yf, front_tire_load());
        const Real alpha_r = invert_axle_force(
            tire_rear, F_yr, rear_tire_load());

        if (std::isnan(alpha_f) || std::isnan(alpha_r)) {
            return std::numeric_limits<Real>::quiet_NaN();
        }

        // Steering angle: kinematic + slip angle difference
        const Real delta = L * a_y / (V * V) + alpha_f - alpha_r;
        return delta;
    }

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
        int n_steps = 41) const
    {
        const Real m = params.mass;
        const Real L = params.wheelbase();
        const Real a = params.front_axle_x;
        const Real b = params.rear_axle_x;

        // Default max: 90% of friction limit
        if (a_y_max <= Real(0.0)) {
            const Real mu_f = tire_front.peak_mu_lateral(front_tire_load());
            const Real mu_r = tire_rear.peak_mu_lateral(rear_tire_load());
            a_y_max = std::min(mu_f, mu_r) * g_accel * Real(0.90);
        }

        std::vector<CorneringPoint> points;
        points.reserve(n_steps);

        for (int i = 0; i < n_steps; ++i) {
            const Real a_y = a_y_max * i / std::max(n_steps - 1, 1);

            CorneringPoint pt;
            pt.a_y = a_y;

            // Linear model
            pt.delta_linear = linear_steering_angle(V, a_y);

            // Nonlinear model
            const Real F_yf = m * a_y * b / L;
            const Real F_yr = m * a_y * a / L;

            pt.alpha_f = invert_axle_force(tire_front, F_yf, front_tire_load());
            pt.alpha_r = invert_axle_force(tire_rear, F_yr, rear_tire_load());

            if (std::isnan(pt.alpha_f) || std::isnan(pt.alpha_r)) {
                pt.valid = false;
                pt.delta_nonlinear = std::numeric_limits<Real>::quiet_NaN();
            } else {
                pt.delta_nonlinear = L * a_y / (V * V) + pt.alpha_f - pt.alpha_r;
            }

            points.push_back(pt);
        }

        return points;
    }

    /// Maximum lateral acceleration [m/s^2] before either axle saturates.
    Real max_lateral_acceleration() const
    {
        const Real m = params.mass;
        const Real L = params.wheelbase();
        const Real a = params.front_axle_x;
        const Real b = params.rear_axle_x;

        const Real mu_f = tire_front.peak_mu_lateral(front_tire_load());
        const Real mu_r = tire_rear.peak_mu_lateral(rear_tire_load());

        // Front-limited: F_yf_max = 2*mu_f*Fz_f = mu_f*W_f
        // Required F_yf = m*a_y*b/L → a_y_max_f = mu_f*W_f*L/(m*b) = mu_f*g
        const Real a_y_max_f = mu_f * g_accel;
        const Real a_y_max_r = mu_r * g_accel;

        return std::min(a_y_max_f, a_y_max_r);
    }
};

} // namespace mbd