#pragma once

// Local speed limit V_max(s) for QSS lap simulation.
//
// At each track point with curvature κ, banking φ, and slope:
//   V²_max = (μ·g·cos(φ) + g·sin(φ·sgn(κ))) /
//            (|κ|·cos(φ) - μ·|κ|·sin(φ·sgn(κ)) - 0.5·ρ·μ·ClA·cos(φ)/m)
//
// Slope does not enter V_max directly (it's a longitudinal effect).
// For straights (κ=0), V_max = +infinity.

#include "mbd/core.hpp"
#include "mbd/track.hpp"
#include "mbd/lap_vehicle.hpp"

#include <vector>
#include <cmath>
#include <limits>

namespace mbd {

// ============================================================================
// Single-point V_max
// ============================================================================

/// Compute the maximum cornering speed at a given track point.
/// Returns +infinity for straights (κ ≈ 0).
/// Returns 0 if the off-camber bank is too steep to support any speed.
inline Real lap_vmax_at(const TrackPoint& pt, const LapVehicle& lv,
                        Real kappa_min = 1e-6)
{
    const Real abs_k = std::abs(pt.kappa);
    if (abs_k < kappa_min) {
        return std::numeric_limits<Real>::infinity();
    }

    const Real phi = pt.bank;
    const Real eff_bank = phi * (pt.kappa >= 0.0 ? 1.0 : -1.0);

    const Real cos_phi = std::cos(phi);
    const Real sin_eff = std::sin(eff_bank);

    const Real numerator   = lv.mu * g_accel * cos_phi + g_accel * sin_eff;
    const Real downforce_term = 0.5 * lv.air_density * lv.mu * lv.ClA * cos_phi / lv.mass;
    const Real denominator = abs_k * cos_phi - lv.mu * abs_k * sin_eff - downforce_term;

    if (numerator <= 0.0) {
        // Off-camber too severe (gravity+bank don't support any positive lateral force)
        return 0.0;
    }
    if (denominator <= 0.0) {
        // High downforce overcomes the curvature limit; speed unbounded by this constraint
        return std::numeric_limits<Real>::infinity();
    }

    const Real V_squared = numerator / denominator;
    if (V_squared <= 0.0) return 0.0;
    return std::sqrt(V_squared);
}

// ============================================================================
// Sampled V_max profile along the track
// ============================================================================

struct SpeedProfile {
    std::vector<Real> s;     ///< Arc length samples [m]
    std::vector<Real> V_max; ///< V_max(s) at each sample [m/s]
};

/// Sample V_max along the track at uniform arc-length intervals.
/// `n_samples` includes both endpoints. `n_samples >= 2`.
inline SpeedProfile sample_vmax_profile(const Track& track, const LapVehicle& lv,
                                        int n_samples)
{
    MBD_THROW_IF(n_samples < 2, "sample_vmax_profile: n_samples must be >= 2");

    SpeedProfile prof;
    prof.s.reserve(n_samples);
    prof.V_max.reserve(n_samples);

    const Real L = track.total_length();
    for (int i = 0; i < n_samples; ++i) {
        const Real s = L * i / (n_samples - 1);
        const TrackPoint pt = track.query(s);
        prof.s.push_back(s);
        prof.V_max.push_back(lap_vmax_at(pt, lv));
    }

    return prof;
}

// ============================================================================
// Lap simulation: forward + backward integration
// ============================================================================

/// Compute available longitudinal acceleration (positive = accelerating forward)
/// given current speed V at track point pt, considering:
///   - drivetrain thrust capped by power/traction
///   - friction-circle limit (longitudinal grip after lateral demand)
///   - aerodynamic drag (opposing motion)
///   - gravity component along slope (negative when uphill)
inline Real lap_a_long(Real V, const TrackPoint& pt, const LapVehicle& lv)
{
    // Lateral demand from cornering
    const Real F_lat_required = lv.mass * V * V * std::abs(pt.kappa);
    const Real grip_total = lv.grip_force(V);

    // Longitudinal grip available after lateral demand (friction circle)
    Real F_grip_long;
    if (F_lat_required >= grip_total) {
        F_grip_long = 0.0;
    } else {
        F_grip_long = std::sqrt(grip_total * grip_total
                                - F_lat_required * F_lat_required);
    }

    // Drivetrain thrust capped by friction circle
    const Real F_drive = std::min(lv.F_drive_max(V), F_grip_long);

    // Aero drag and gravity
    const Real F_drag = lv.drag(V);
    const Real F_grav_along_motion = lv.mass * g_accel * pt.slope;

    // Net forward force
    const Real F_net = F_drive - F_drag - F_grav_along_motion;
    return F_net / lv.mass;
}

/// Compute available braking deceleration (positive = decelerating).
/// Considers brake force, friction circle, drag (assists braking),
/// and gravity (assists when uphill).
inline Real lap_a_brake(Real V, const TrackPoint& pt, const LapVehicle& lv)
{
    const Real F_lat_required = lv.mass * V * V * std::abs(pt.kappa);
    const Real grip_total = lv.grip_force(V);

    Real F_grip_long;
    if (F_lat_required >= grip_total) {
        F_grip_long = 0.0;
    } else {
        F_grip_long = std::sqrt(grip_total * grip_total
                                - F_lat_required * F_lat_required);
    }

    const Real F_brake = std::min(lv.F_brake_max(V), F_grip_long);
    const Real F_drag  = lv.drag(V);
    const Real F_grav_along_motion = lv.mass * g_accel * pt.slope;

    // Net deceleration: brake + drag both decelerate; uphill gravity also decelerates
    const Real F_dec = F_brake + F_drag + F_grav_along_motion;
    return F_dec / lv.mass;
}

// ============================================================================
// Lap result
// ============================================================================

struct LapResult {
    std::vector<Real> s;       ///< Arc length samples [m]
    std::vector<Real> V;       ///< Realized speed profile [m/s]
    std::vector<Real> V_max;   ///< Cornering limit (for diagnostics) [m/s]
    Real lap_time{0.0};        ///< Total lap time [s]
    Real total_length{0.0};    ///< Track length [m]
};

// ============================================================================
// Lap simulation
// ============================================================================

/// Compute the realized speed profile and lap time using the QSS algorithm.
///
/// Algorithm:
///   1. Start with V(s) = V_max(s) (cornering limit)
///   2. Forward pass: integrate forward, capping at V_max
///   3. Backward pass: integrate backward, capping at V_max
///   4. Lap time = sum of ds/V averaged over each segment
///
/// `n_samples` is the number of points along the track (>= 2).
/// `is_closed_lap` indicates whether to wrap the integration passes for a
/// closed track. For an open track (point-to-point), set false.
inline LapResult simulate_lap(const Track& track, const LapVehicle& lv,
                              int n_samples = 1000,
                              bool is_closed_lap = true)
{
    MBD_THROW_IF(n_samples < 2, "simulate_lap: n_samples must be >= 2");

    LapResult result;
    result.total_length = track.total_length();
    result.s.reserve(n_samples);
    result.V.reserve(n_samples);
    result.V_max.reserve(n_samples);

    // Sample track at uniform arc-length intervals
    std::vector<TrackPoint> pts;
    pts.reserve(n_samples);

    const Real L = track.total_length();
    const Real ds = L / (n_samples - 1);

    for (int i = 0; i < n_samples; ++i) {
        const Real s = i * ds;
        TrackPoint pt = track.query(s);
        pts.push_back(pt);
        result.s.push_back(s);

        const Real V_max = lap_vmax_at(pt, lv);
        result.V_max.push_back(V_max);
        result.V.push_back(V_max);  // initial: bounded by cornering
    }

    // Cap any infinite V_max by a generous absolute limit (e.g. 200 m/s = 720 km/h).
    // The forward/backward passes will reduce this naturally where physics dictates.
    const Real V_cap = 200.0;
    for (auto& v : result.V) {
        if (!std::isfinite(v) || v > V_cap) v = V_cap;
    }

    // --- Forward pass ---
    // Always applies: when a_x < 0 (drag exceeds drive thrust), the car
    // decelerates naturally, and V_next < V[i] correctly captures this.
    // For closed tracks, run twice to allow wrap-around.
    const int n_forward_sweeps = is_closed_lap ? 2 : 1;
    for (int sweep = 0; sweep < n_forward_sweeps; ++sweep) {
        for (int i = 0; i < n_samples - 1; ++i) {
            const Real a_x = lap_a_long(result.V[i], pts[i], lv);
            const Real V_next_squared = result.V[i] * result.V[i] + 2.0 * a_x * ds;
            if (V_next_squared <= 0.0) {
                // Vehicle would stall (can happen with very steep uphill at low speed).
                // Cap at zero.
                if (result.V[i + 1] > 0.0) result.V[i + 1] = 0.0;
                continue;
            }
            const Real V_next = std::sqrt(V_next_squared);
            if (V_next < result.V[i + 1]) {
                result.V[i + 1] = V_next;
            }
        }
        if (is_closed_lap && sweep + 1 < n_forward_sweeps) {
            if (result.V.front() > result.V.back()) {
                result.V.front() = result.V.back();
            }
        }
    }

    // --- Backward pass ---
    const int n_backward_sweeps = is_closed_lap ? 2 : 1;
    for (int sweep = 0; sweep < n_backward_sweeps; ++sweep) {
        for (int i = n_samples - 1; i > 0; --i) {
            const Real a_b = lap_a_brake(result.V[i], pts[i], lv);
            if (a_b <= 0.0) continue;
            const Real V_prev_squared = result.V[i] * result.V[i] + 2.0 * a_b * ds;
            if (V_prev_squared <= 0.0) continue;
            const Real V_prev = std::sqrt(V_prev_squared);
            if (V_prev < result.V[i - 1]) {
                result.V[i - 1] = V_prev;
            }
        }
        if (is_closed_lap && sweep + 1 < n_backward_sweeps) {
            if (result.V.back() > result.V.front()) {
                result.V.back() = result.V.front();
            }
        }
    }

    // --- Lap time: trapezoidal integration of dt = ds/V_avg ---
    Real T = 0.0;
    for (int i = 0; i < n_samples - 1; ++i) {
        const Real V_avg = 0.5 * (result.V[i] + result.V[i + 1]);
        if (V_avg > 1e-6) {
            T += ds / V_avg;
        }
    }
    result.lap_time = T;

    return result;
}

} // namespace mbd