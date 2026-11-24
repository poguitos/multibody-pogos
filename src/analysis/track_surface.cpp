#include "mbd/analysis/track_surface.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace mbd {

TrackSurfaceRoad::TrackSurfaceRoad(Track track, Real friction, Real sample_spacing)
    : track_(std::move(track)), friction_(friction), closed_(false)
{
    MBD_THROW_IF(track_.segment_count() == 0, "MBD-F050: TrackSurfaceRoad: the track has no segment");
    MBD_THROW_IF(!(sample_spacing > 0.0), "MBD-F050: TrackSurfaceRoad: the sample spacing must be > 0");
    MBD_THROW_IF(friction < 0.0, "MBD-F050: TrackSurfaceRoad: the friction must be >= 0");
    closed_ = track_.is_closed();

    // Samples of the centreline, and a grid of buckets of four sample spacings
    // (at least 1 m) holding them, so that a query looks at its neighbours.
    const Real length = track_.total_length();
    const auto n = static_cast<int>(std::ceil(length / sample_spacing));
    cell_ = std::max(4.0 * sample_spacing, Real(1.0));
    for (int k = 0; k <= n; ++k) {
        const Real s = std::min(length, k * sample_spacing);
        const TrackPoint p = track_.query(s);
        sample_s_.push_back(s);
        sample_xy_.emplace_back(p.x, p.y);
        const auto i = static_cast<long long>(std::floor(p.x / cell_));
        const auto j = static_cast<long long>(std::floor(p.y / cell_));
        buckets_[bucket_key(i, j)].push_back(k);
    }
}

void TrackSurfaceRoad::project(Real x, Real y, Real& s, Real& d) const
{
    // The nearest sample: in the 3 x 3 buckets around the point, or, for a
    // point farther from the centreline than a bucket, among all of them.
    const Vec2 p(x, y);
    const auto ci = static_cast<long long>(std::floor(x / cell_));
    const auto cj = static_cast<long long>(std::floor(y / cell_));
    int best = -1;
    Real best_d2 = std::numeric_limits<Real>::infinity();
    for (long long i = ci - 1; i <= ci + 1; ++i) {
        for (long long j = cj - 1; j <= cj + 1; ++j) {
            const auto it = buckets_.find(bucket_key(i, j));
            if (it == buckets_.end()) continue;
            for (const int k : it->second) {
                const Real d2 = (sample_xy_[static_cast<std::size_t>(k)] - p).squaredNorm();
                if (d2 < best_d2) { best_d2 = d2; best = k; }
            }
        }
    }
    if (best < 0) {
        for (std::size_t k = 0; k < sample_xy_.size(); ++k) {
            const Real d2 = (sample_xy_[k] - p).squaredNorm();
            if (d2 < best_d2) { best_d2 = d2; best = static_cast<int>(k); }
        }
    }

    // Newton on f(s) = (p - c(s)) . t(s) = 0, the foot of the perpendicular:
    // f'(s) = -1 + kappa d, with d the distance along the left normal.
    const Real length = track_.total_length();
    s = sample_s_[static_cast<std::size_t>(best)];
    for (int iteration = 0; iteration < 30; ++iteration) {
        const TrackPoint c = track_.query(s);
        const Vec2 e = p - Vec2(c.x, c.y);
        const Vec2 t(std::cos(c.psi), std::sin(c.psi));
        const Vec2 left(-t.y(), t.x());
        const Real f = e.dot(t);
        d = e.dot(left);
        const Real slope = 1.0 - c.kappa * d;
        Real next = s + f / (std::abs(slope) > 1e-3 ? slope : 1.0);
        next = closed_ ? track_.wrap_s(next) : std::clamp(next, Real(0.0), length);
        const Real change = std::abs(next - s);
        s = next;
        if (change < 1e-12 * (1.0 + length)) break;
    }
    const TrackPoint c = track_.query(s);
    d = (p - Vec2(c.x, c.y)).dot(Vec2(-std::sin(c.psi), std::cos(c.psi)));
}

RoadPoint TrackSurfaceRoad::at(Real x, Real y) const
{
    Real s = 0.0, d = 0.0;
    project(x, y, s, d);
    const TrackPoint c = track_.query(s);
    const Real tb = std::tan(c.bank);
    const Real stretch = 1.0 - c.kappa * d;
    const Real sp = std::sin(c.psi), cp = std::cos(c.psi);
    const Vec3 normal(-stretch * sp * tb - c.slope * cp, stretch * cp * tb - c.slope * sp, stretch);
    return {c.z - d * tb, normal.normalized(), friction_};
}

} // namespace mbd
