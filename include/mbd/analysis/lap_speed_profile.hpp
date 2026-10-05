#pragma once

// Local speed limit V_max(s) for QSS lap simulation.
//
// At each track point with curvature κ, banking φ, and slope:
//   V²_max = (μ·g·cos(φ) + g·sin(φ·sgn(κ))) /
//            (|κ|·cos(φ) - μ·|κ|·sin(φ·sgn(κ)) - 0.5·ρ·μ·ClA·cos(φ)/m)
//
// Slope does not enter V_max directly (it's a longitudinal effect).
// For straights (κ=0), V_max = +infinity.

#include "mbd/core/core.hpp"
#include "mbd/analysis/track.hpp"
#include "mbd/analysis/lap_vehicle.hpp"

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
Real lap_vmax_at(const TrackPoint& pt, const LapVehicle& lv,
                        Real kappa_min = 1e-6);

// ============================================================================
// Sampled V_max profile along the track
// ============================================================================

struct SpeedProfile {
    std::vector<Real> s;     ///< Arc length samples [m]
    std::vector<Real> V_max; ///< V_max(s) at each sample [m/s]
};

/// Sample V_max along the track at uniform arc-length intervals.
/// `n_samples` includes both endpoints. `n_samples >= 2`.
SpeedProfile sample_vmax_profile(const Track& track, const LapVehicle& lv,
                                        int n_samples);

// ============================================================================
// Lap simulation: forward + backward integration
// ============================================================================

/// Compute available longitudinal acceleration (positive = accelerating forward)
/// given current speed V at track point pt, considering:
///   - drivetrain thrust capped by power/traction
///   - friction-circle limit (longitudinal grip after lateral demand)
///   - aerodynamic drag (opposing motion)
///   - gravity component along slope (negative when uphill)
Real lap_a_long(Real V, const TrackPoint& pt, const LapVehicle& lv);

/// Compute available braking deceleration (positive = decelerating).
/// Considers brake force, friction circle, drag (assists braking),
/// and gravity (assists when uphill).
Real lap_a_brake(Real V, const TrackPoint& pt, const LapVehicle& lv);

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
LapResult simulate_lap(const Track& track, const LapVehicle& lv,
                              int n_samples = 1000,
                              bool is_closed_lap = true);

} // namespace mbd
