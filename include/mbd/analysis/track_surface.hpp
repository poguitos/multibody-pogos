#pragma once

// The three-dimensional surface of a lap track as a road (plan task 7.2,
// D61): the track's centreline in the ground plane, its elevation and its
// banking, swept across its width.
//
// A point (x, y) is projected onto the centreline: the arc length s of the
// nearest centreline point and the signed distance d from it, positive to
// the left of the direction of travel (ISO 8855). The surface there is
//
//     P(s, d) = (x_c(s) - d sin psi, y_c(s) + d cos psi, z_c(s) - d tan(bank)),
//
// with psi the heading, z_c the elevation and bank the banking angle
// (positive toward the inside of a left turn, so the left side is lower).
// Its normal is dP/ds x dP/dd, normalised:
//
//     ( -(1 - kappa d) sin psi tan(bank) - slope cos psi,
//        (1 - kappa d) cos psi tan(bank) - slope sin psi,
//        1 - kappa d ),
//
// kappa the curvature and slope dz_c/ds. Banking is constant within a
// segment of the track, so where two segments of different banking meet the
// surface steps by d times the change of tan(bank) away from the centreline.

#include <unordered_map>
#include <vector>

#include "mbd/analysis/track.hpp"
#include "mbd/forces/road.hpp"

namespace mbd {

class TrackSurfaceRoad : public Road {
public:
    /// `sample_spacing`: the centreline is sampled at this arc-length
    /// interval to find the nearest point of a query, then refined by Newton
    /// iterations on s.
    explicit TrackSurfaceRoad(Track track, Real friction = 1.0, Real sample_spacing = 1.0);

    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "track surface"; }

    /// The arc length and the signed lateral distance (positive to the left)
    /// of the ground point (x, y) from the centreline.
    void project(Real x, Real y, Real& s, Real& d) const;

    const Track& track() const { return track_; }

private:
    Track track_;
    Real friction_;
    bool closed_;
    Real cell_;
    std::vector<Real> sample_s_;
    std::vector<Vec2> sample_xy_;
    std::unordered_map<long long, std::vector<int>> buckets_;

    long long bucket_key(long long i, long long j) const { return i * 2000003LL + j; }
};

} // namespace mbd
