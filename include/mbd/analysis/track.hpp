#pragma once

// 2D track representation parametrized by arc length.
//
// A track is a sequence of straight and arc segments (or a sampled polyline).
// Querying at arc length s returns: position (x,y), heading psi, curvature kappa.
//
// Conventions:
//   - Position in 2D ground plane (x, y)
//   - Heading psi: angle from +X axis, positive counterclockwise
//   - Signed radius for arcs: positive = left turn, negative = right turn
//   - Curvature kappa = d(psi)/ds = 1/R (with sign matching radius)

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"

#include <vector>
#include <cmath>

namespace mbd {

// ============================================================================
// Track query result
// ============================================================================

struct TrackPoint {
    Real s{0.0};         ///< Arc length [m]
    Real x{0.0};         ///< X position [m] (in horizontal plane)
    Real y{0.0};         ///< Y position [m] (in horizontal plane)
    Real z{0.0};         ///< Elevation [m]
    Real psi{0.0};       ///< Heading [rad] (in horizontal plane)
    Real kappa{0.0};     ///< Horizontal-plane curvature [1/m] (signed)
    Real slope{0.0};     ///< dz/ds (dimensionless, +ve = uphill)
    Real bank{0.0};      ///< Banking angle [rad] (+ve = tilted toward inside of left turn)
};

// ============================================================================
// Track
// ============================================================================

class Track {
public:
    enum class SegmentType { Straight, Arc, Clothoid };

    struct Segment {
        SegmentType type;
        Real s_start{0.0};   ///< Arc length at segment start [m]
        Real length{0.0};    ///< Segment length [m]
        Real x_start{0.0};   ///< X position at segment start
        Real y_start{0.0};   ///< Y position at segment start
        Real z_start{0.0};   ///< Elevation at segment start [m]
        Real psi_start{0.0}; ///< Heading at segment start [rad]
        Real kappa{0.0};     ///< Curvature [1/m] (0 for straights, signed for arcs)
        Real kappa_end{0.0}; ///< Only used for Clothoid (kappa_start = kappa)
        Real slope{0.0};     ///< dz/ds (dimensionless)
        Real bank{0.0};      ///< Banking [rad] (constant within segment)
    };

private:
    std::vector<Segment> segments_;
    Real total_length_{0.0};

public:
    Track() = default;

    /// Total length of the track [m].
    Real total_length() const { return total_length_; }

    /// Number of segments.
    std::size_t segment_count() const { return segments_.size(); }

    /// Access segment by index.
    const Segment& segment(std::size_t i) const { return segments_.at(i); }

    /// Append a straight segment of given length.
    /// Continues from the end of the previous segment (or from origin if first).
    void add_straight(Real length, Real delta_z = 0.0, Real bank = 0.0);

    /// Append an arc segment of given length and signed radius.
    /// Positive radius = left turn (counterclockwise), negative = right turn.
    void add_arc(Real length, Real signed_radius, Real delta_z = 0.0, Real bank = 0.0);

    /// Convenience: arc specified by sweep angle (radians, signed) and radius magnitude.
    /// Positive sweep = left turn, negative = right turn.
    void add_arc_by_angle(Real sweep_rad, Real radius_magnitude,
                          Real delta_z = 0.0, Real bank = 0.0);

    /// Append a clothoid (Euler spiral) segment with linearly varying curvature
    /// from kappa_start to kappa_end. Useful for smooth corner entry/exit.
    /// kappa_start, kappa_end: curvatures [1/m] at start and end of segment.
    /// delta_z, bank: as for other segment types.
    void add_clothoid(Real length, Real kappa_start, Real kappa_end,
                      Real delta_z = 0.0, Real bank = 0.0);

    /// Build a track from sampled centerline points (open polyline).
    /// Curvature is computed via 3-point circumradius. Endpoint curvatures
    /// inherit from neighbors. The result has (n-1) straight-or-arc segments
    /// approximated as straights of the chord length, with stored curvature
    /// for slope analysis. For higher fidelity, use add_straight/add_arc directly.
    static Track from_polyline(const std::vector<Vec2>& points);

    /// Query the track at arc length s.
    /// For s outside [0, total_length], the result is clamped to endpoints.
    TrackPoint query(Real s) const;

    /// Wrap s into [0, total_length) for closed tracks.
    Real wrap_s(Real s) const;

    /// Check if the track is approximately closed.
    /// Returns true if start and end positions match within pos_tol AND
    /// start and end headings match (mod 2π) within angle_tol.
    bool is_closed(Real pos_tol = 1e-6, Real angle_tol = 1e-6) const;

private:
    /// Set segment's start position/heading from previous segment's end.
    void compute_segment_start(Segment& seg) const;

    /// Evaluate a segment at local arc length `local_s` ∈ [0, segment.length].
    static TrackPoint query_segment(const Segment& seg, Real local_s);

    /// Compute signed curvature at p1 using points p0, p1, p2.
    /// Returns 0 for collinear points. Positive = left turn.
    static Real compute_curvature_3pt(const Vec2& p0, const Vec2& p1, const Vec2& p2);
};

} // namespace mbd
