#include "mbd/analysis/bicycle_model.hpp"

namespace mbd {

BicycleModelParams BicycleModelParams::FromVehicle(const VehicleParams& vp)
{
    BicycleModelParams bp;
    bp.mass = vp.total_mass();
    bp.front_axle_x = vp.front_axle_x;
    bp.rear_axle_x  = vp.rear_axle_x;
    bp.tire_front = vp.tire_params;
    bp.tire_rear  = vp.tire_params;
    return bp;
}

Real BicycleModel::understeer_gradient() const
{
    const Real m = params.mass;
    const Real L = params.wheelbase();
    const Real a = params.front_axle_x;
    const Real b = params.rear_axle_x;
    const Real C_f = front_axle_cornering_stiffness();
    const Real C_r = rear_axle_cornering_stiffness();

    return (m / L) * (b / C_f - a / C_r);
}

Real BicycleModel::characteristic_speed() const
{
    const Real K = understeer_gradient();
    if (K <= Real(0.0)) return std::numeric_limits<Real>::infinity();
    return std::sqrt(params.wheelbase() / K);
}

Real BicycleModel::yaw_rate_gain(Real V) const
{
    const Real K = understeer_gradient();
    return V / (params.wheelbase() + K * V * V);
}

Real BicycleModel::linear_steering_angle(Real V, Real a_y) const
{
    const Real L = params.wheelbase();
    const Real K = understeer_gradient();
    return (L / (V * V) + K) * a_y;
}

Real BicycleModel::invert_axle_force(const PacejkaTire& tire,
                                     Real F_axle_required,
                                     Real Fz_per_tire,
                                     Real alpha_guess)
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

Real BicycleModel::nonlinear_steering_angle(Real V, Real a_y) const
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

std::vector<BicycleModel::CorneringPoint> BicycleModel::cornering_diagram(
    Real V,
    Real a_y_max,
    int n_steps) const
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

Real BicycleModel::max_lateral_acceleration() const
{
    const Real mu_f = tire_front.peak_mu_lateral(front_tire_load());
    const Real mu_r = tire_rear.peak_mu_lateral(rear_tire_load());

    // Front-limited: F_yf_max = 2*mu_f*Fz_f = mu_f*W_f
    // Required F_yf = m*a_y*b/L → a_y_max_f = mu_f*W_f*L/(m*b) = mu_f*g
    const Real a_y_max_f = mu_f * g_accel;
    const Real a_y_max_r = mu_r * g_accel;

    return std::min(a_y_max_f, a_y_max_r);
}

} // namespace mbd
