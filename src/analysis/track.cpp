#include "mbd/analysis/track.hpp"

namespace mbd {

void Track::add_straight(Real length, Real delta_z, Real bank)
{
    MBD_THROW_IF(length <= 0.0, "MBD-A001: Track::add_straight: length must be positive");

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

void Track::add_arc(Real length, Real signed_radius, Real delta_z, Real bank)
{
    MBD_THROW_IF(length <= 0.0, "MBD-A001: Track::add_arc: length must be positive");
    MBD_THROW_IF(std::abs(signed_radius) < 1e-9,
                 "MBD-A001: Track::add_arc: |radius| must be > 0");

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

void Track::add_arc_by_angle(Real sweep_rad, Real radius_magnitude,
                             Real delta_z, Real bank)
{
    MBD_THROW_IF(radius_magnitude <= 0.0,
                 "MBD-A001: Track::add_arc_by_angle: radius must be positive");
    MBD_THROW_IF(std::abs(sweep_rad) < 1e-12,
                 "MBD-A001: Track::add_arc_by_angle: sweep must be nonzero");

    const Real signed_radius = (sweep_rad > 0.0) ? radius_magnitude
                                                 : -radius_magnitude;
    const Real length = std::abs(sweep_rad) * radius_magnitude;
    add_arc(length, signed_radius, delta_z, bank);
}

void Track::add_clothoid(Real length, Real kappa_start, Real kappa_end,
                         Real delta_z, Real bank)
{
    MBD_THROW_IF(length <= 0.0, "MBD-A001: Track::add_clothoid: length must be positive");

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

Track Track::from_polyline(const std::vector<Vec2>& points)
{
    MBD_THROW_IF(points.size() < 2,
                 "MBD-A002: Track::from_polyline: need at least 2 points");

    Track t;

    for (std::size_t i = 0; i + 1 < points.size(); ++i) {
        const Vec2 d = points[i + 1] - points[i];
        const Real len = d.norm();
        MBD_THROW_IF(len < 1e-9,
                     "MBD-A002: Track::from_polyline: zero-length segment");

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

TrackPoint Track::query(Real s) const
{
    MBD_THROW_IF(segments_.empty(), "MBD-A003: Track::query: empty track");

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

Real Track::wrap_s(Real s) const
{
    const Real L = total_length_;
    if (L <= 0.0) return 0.0;

    Real r = std::fmod(s, L);
    if (r < 0.0) r += L;
    return r;
}

bool Track::is_closed(Real pos_tol, Real angle_tol) const
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

void Track::compute_segment_start(Segment& seg) const
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

TrackPoint Track::query_segment(const Segment& seg, Real local_s)
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

Real Track::compute_curvature_3pt(const Vec2& p0, const Vec2& p1, const Vec2& p2)
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

} // namespace mbd
