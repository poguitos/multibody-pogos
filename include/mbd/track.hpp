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

#include "mbd/core.hpp"
#include "mbd/math.hpp"

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
    void add_straight(Real length, Real delta_z = 0.0, Real bank = 0.0)
    {
        MBD_THROW_IF(length <= 0.0, "Track::add_straight: length must be positive");

        Segment seg;
        seg.type     = SegmentType::Straight;
        seg.length   = length;
        seg.kappa    = 0.0;
        seg.slope    = delta_z / length;
        seg.bank     = bank;
        compute_segment_start(seg);

        segments_.push_back(seg);
        total_length_ += length;
    }

    /// Append an arc segment of given length and signed radius.
    /// Positive radius = left turn (counterclockwise), negative = right turn.
    void add_arc(Real length, Real signed_radius, Real delta_z = 0.0, Real bank = 0.0)
    {
        MBD_THROW_IF(length <= 0.0, "Track::add_arc: length must be positive");
        MBD_THROW_IF(std::abs(signed_radius) < 1e-9,
                     "Track::add_arc: |radius| must be > 0");

        Segment seg;
        seg.type   = SegmentType::Arc;
        seg.length = length;
        seg.kappa  = 1.0 / signed_radius;
        seg.slope  = delta_z / length;
        seg.bank   = bank;
        compute_segment_start(seg);

        segments_.push_back(seg);
        total_length_ += length;
    }

    /// Convenience: arc specified by sweep angle (radians, signed) and radius magnitude.
    /// Positive sweep = left turn, negative = right turn.
    void add_arc_by_angle(Real sweep_rad, Real radius_magnitude,
                          Real delta_z = 0.0, Real bank = 0.0)
    {
        MBD_THROW_IF(radius_magnitude <= 0.0,
                     "Track::add_arc_by_angle: radius must be positive");
        MBD_THROW_IF(std::abs(sweep_rad) < 1e-12,
                     "Track::add_arc_by_angle: sweep must be nonzero");

        const Real signed_radius = (sweep_rad > 0.0) ? radius_magnitude
                                                     : -radius_magnitude;
        const Real length = std::abs(sweep_rad) * radius_magnitude;
        add_arc(length, signed_radius, delta_z, bank);
    }

    /// Append a clothoid (Euler spiral) segment with linearly varying curvature
    /// from kappa_start to kappa_end. Useful for smooth corner entry/exit.
    /// kappa_start, kappa_end: curvatures [1/m] at start and end of segment.
    /// delta_z, bank: as for other segment types.
    void add_clothoid(Real length, Real kappa_start, Real kappa_end,
                      Real delta_z = 0.0, Real bank = 0.0)
    {
        MBD_THROW_IF(length <= 0.0, "Track::add_clothoid: length must be positive");

        Segment seg;
        seg.type      = SegmentType::Clothoid;
        seg.length    = length;
        seg.kappa     = kappa_start;
        seg.kappa_end = kappa_end;
        seg.slope     = delta_z / length;
        seg.bank      = bank;
        compute_segment_start(seg);

        segments_.push_back(seg);
        total_length_ += length;
    }

    /// Build a track from sampled centerline points (open polyline).
    /// Curvature is computed via 3-point circumradius. Endpoint curvatures
    /// inherit from neighbors. The result has (n-1) straight-or-arc segments
    /// approximated as straights of the chord length, with stored curvature
    /// for slope analysis. For higher fidelity, use add_straight/add_arc directly.
    static Track from_polyline(const std::vector<Vec2>& points)
    {
        MBD_THROW_IF(points.size() < 2,
                     "Track::from_polyline: need at least 2 points");

        Track t;

        for (std::size_t i = 0; i + 1 < points.size(); ++i) {
            const Vec2 d = points[i + 1] - points[i];
            const Real len = d.norm();
            MBD_THROW_IF(len < 1e-9,
                         "Track::from_polyline: zero-length segment");

            // Straight segment with length = chord; curvature stored separately
            t.add_straight(len);

            // Override segment kappa from local geometry if we have neighbors
            if (i >= 1 && i + 1 < points.size()) {
                const Real kappa = compute_curvature_3pt(
                    points[i - 1], points[i], points[i + 1]);
                t.segments_.back().kappa = kappa;
            }
        }

        return t;
    }

    /// Query the track at arc length s.
    /// For s outside [0, total_length], the result is clamped to endpoints.
    TrackPoint query(Real s) const
    {
        MBD_THROW_IF(segments_.empty(), "Track::query: empty track");

        // Clamp to track bounds
        if (s <= 0.0) {
            return query_segment(segments_.front(), 0.0);
        }
        if (s >= total_length_) {
            const auto& last = segments_.back();
            return query_segment(last, last.length);
        }

        // Find the segment containing s. Linear search; for many segments,
        // a binary search could be added later.
        for (const auto& seg : segments_) {
            if (s < seg.s_start + seg.length) {
                const Real local_s = s - seg.s_start;
                return query_segment(seg, local_s);
            }
        }

        // Fallback (numerical edge case): use last segment
        const auto& last = segments_.back();
        return query_segment(last, last.length);
    }

    /// Wrap s into [0, total_length) for closed tracks.
    Real wrap_s(Real s) const
    {
        const Real L = total_length_;
        if (L <= 0.0) return 0.0;

        Real r = std::fmod(s, L);
        if (r < 0.0) r += L;
        return r;
    }

    /// Check if the track is approximately closed.
    /// Returns true if start and end positions match within pos_tol AND
    /// start and end headings match (mod 2π) within angle_tol.
    bool is_closed(Real pos_tol = 1e-6, Real angle_tol = 1e-6) const
    {
        if (segments_.empty()) return false;

        TrackPoint p0 = query(0.0);
        TrackPoint p1 = query(total_length_);

        const Real dx = p1.x - p0.x;
        const Real dy = p1.y - p0.y;
        const Real dz = p1.z - p0.z;
        if (std::sqrt(dx * dx + dy * dy + dz * dz) > pos_tol) return false;

        // Heading difference modulo 2π
        Real dpsi = std::fmod(p1.psi - p0.psi, 2.0 * pi);
        if (dpsi > pi)  dpsi -= 2.0 * pi;
        if (dpsi < -pi) dpsi += 2.0 * pi;
        if (std::abs(dpsi) > angle_tol) return false;

        return true;
    }

private:
    /// Set segment's start position/heading from previous segment's end.
    void compute_segment_start(Segment& seg) const
    {
        seg.s_start = total_length_;

        if (segments_.empty()) {
            seg.x_start = 0.0;
            seg.y_start = 0.0;
            seg.z_start = 0.0;
            seg.psi_start = 0.0;
            return;
        }

        const Segment& prev = segments_.back();
        TrackPoint end_of_prev = query_segment(prev, prev.length);
        seg.x_start = end_of_prev.x;
        seg.y_start = end_of_prev.y;
        seg.z_start = end_of_prev.z;
        seg.psi_start = end_of_prev.psi;
    }

    /// Evaluate a segment at local arc length `local_s` ∈ [0, segment.length].
    static TrackPoint query_segment(const Segment& seg, Real local_s)
    {
        TrackPoint p;
        p.s = seg.s_start + local_s;
        p.slope = seg.slope;
        p.bank  = seg.bank;
        p.z = seg.z_start + seg.slope * local_s;

        switch (seg.type) {
            case SegmentType::Straight: {
                const Real cp = std::cos(seg.psi_start);
                const Real sp = std::sin(seg.psi_start);
                p.x = seg.x_start + local_s * cp;
                p.y = seg.y_start + local_s * sp;
                p.psi = seg.psi_start;
                p.kappa = seg.kappa;  // 0 for pure straights, may be nonzero for polylines
                break;
            }

            case SegmentType::Arc: {
                const Real R = 1.0 / seg.kappa;
                const Real psi0 = seg.psi_start;
                const Real x_c = seg.x_start - R * std::sin(psi0);
                const Real y_c = seg.y_start + R * std::cos(psi0);

                const Real dpsi = local_s * seg.kappa;
                const Real psi_now = psi0 + dpsi;

                p.x = x_c + R * std::sin(psi_now);
                p.y = y_c - R * std::cos(psi_now);
                p.psi = psi_now;
                p.kappa = seg.kappa;
                break;
            }

            case SegmentType::Clothoid: {
                // kappa(s) = kappa_start + (kappa_end - kappa_start) * (s / length)
                const Real k0 = seg.kappa;
                const Real k1 = seg.kappa_end;
                const Real L  = seg.length;
                const Real kp = (k1 - k0) / L;  // d(kappa)/ds, constant

                // Heading: psi(s) = psi_0 + k0*s + 0.5*kp*s^2
                p.kappa = k0 + kp * local_s;
                p.psi   = seg.psi_start + k0 * local_s + 0.5 * kp * local_s * local_s;

                // Position: integrate cos/sin of psi from 0 to local_s.
                // Use composite Simpson's rule with adaptive N based on segment "strength"
                const Real heading_change = std::abs(k0 * L + 0.5 * kp * L * L);
                int N = 16;
                if (heading_change > 0.5)  N = 32;
                if (heading_change > 1.0)  N = 64;
                if (heading_change > 2.0)  N = 128;
                // Make N even (Simpson requirement)
                if (N % 2 == 1) ++N;

                const Real h = local_s / static_cast<Real>(N);

                auto psi_of_s = [&](Real s) {
                    return seg.psi_start + k0 * s + 0.5 * kp * s * s;
                };

                Real sum_cos = 0.0;
                Real sum_sin = 0.0;

                if (local_s > 0.0) {
                    // Simpson: I = h/3 * (f0 + 4*f1 + 2*f2 + 4*f3 + ... + fN)
                    sum_cos += std::cos(psi_of_s(0.0));
                    sum_sin += std::sin(psi_of_s(0.0));
                    sum_cos += std::cos(psi_of_s(local_s));
                    sum_sin += std::sin(psi_of_s(local_s));

                    for (int i = 1; i < N; ++i) {
                        const Real s_i = i * h;
                        const Real psi_i = psi_of_s(s_i);
                        const Real coef = (i % 2 == 1) ? 4.0 : 2.0;
                        sum_cos += coef * std::cos(psi_i);
                        sum_sin += coef * std::sin(psi_i);
                    }

                    p.x = seg.x_start + (h / 3.0) * sum_cos;
                    p.y = seg.y_start + (h / 3.0) * sum_sin;
                } else {
                    p.x = seg.x_start;
                    p.y = seg.y_start;
                }
                break;
            }
        }

        return p;
    }

    /// Compute signed curvature at p1 using points p0, p1, p2.
    /// Returns 0 for collinear points. Positive = left turn.
    static Real compute_curvature_3pt(const Vec2& p0, const Vec2& p1, const Vec2& p2)
    {
        const Vec2 d1 = p1 - p0;
        const Vec2 d2 = p2 - p1;

        const Real cross = d1.x() * d2.y() - d1.y() * d2.x();
        const Real l1 = d1.norm();
        const Real l2 = d2.norm();
        const Real l3 = (p2 - p0).norm();

        if (l1 < 1e-12 || l2 < 1e-12 || l3 < 1e-12) return 0.0;

        // Signed curvature: 2 * (signed area) / (l1 * l2 * l3)
        return 2.0 * cross / (l1 * l2 * l3);
    }
};

} // namespace mbd