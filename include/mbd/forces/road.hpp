#pragma once

// The road (plan task 7.2, D61): one interface returning, at a point (x, y)
// of the ground, the road's height, its unit normal and its friction. World
// axes are ISO 8855's (mbd/vehicle/axes.hpp): the height is along Z, and the
// normal points up, out of the road.
//
// The friction is a scale on the tyre's own friction: 1 is the surface its
// data was measured on (a dry road), 0.5 a surface with half its grip. The
// tyre multiplies its peak forces by it and keeps its slip stiffnesses, as
// the Magic Formula's friction scaling does.
//
// Roads here: flat; analytic profiles (a ramp, a bump, a kerb, a sine wave);
// a gridded height map; ISO 8608 random roads of classes A to H. The
// three-dimensional surface of a lap track (its elevation and banking) is
// mbd/analysis/track_surface.hpp.

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "mbd/core/core.hpp"

namespace mbd {

/// What a road is at a point of the ground.
struct RoadPoint {
    Real height{0.0};                ///< Z of the surface above (x, y) [m]
    Vec3 normal{Vec3::UnitZ()};      ///< Unit normal, up, world axes
    Real friction{1.0};              ///< Scale on the tyre's friction [-]
};

/// A road: the surface z = height(x, y), its normal and its friction.
class Road {
public:
    virtual ~Road() = default;

    /// The road at the ground point (x, y), world axes [m].
    virtual RoadPoint at(Real x, Real y) const = 0;

    virtual const char* name() const = 0;
};

/// The normal of the surface z = h(x, y) from its slopes dh/dx and dh/dy:
/// (-dh/dx, -dh/dy, 1), normalised.
Vec3 surface_normal(Real dh_dx, Real dh_dy);

/// A level road at a constant height.
class FlatRoad : public Road {
public:
    explicit FlatRoad(Real height = 0.0, Real friction = 1.0);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "flat road"; }

private:
    Real height_, friction_;
};

/// A ramp across the road: level at 0 before x_start, rising linearly by
/// `rise` over `length`, level at `rise` after. The slope changes abruptly at
/// both ends; there the normal is the one of the side x is on (the lower x
/// for x exactly at an end).
class RampRoad : public Road {
public:
    RampRoad(Real x_start, Real length, Real rise, Real friction = 1.0);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "ramp"; }

private:
    Real x_start_, length_, rise_, friction_;
};

/// A bump across the road: a raised cosine of `height` and `length` starting
/// at x_start, h = height (1 - cos(2 pi (x - x_start) / length)) / 2; level
/// at 0 elsewhere, with a continuous slope.
class BumpRoad : public Road {
public:
    BumpRoad(Real x_start, Real length, Real height, Real friction = 1.0);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "bump"; }

private:
    Real x_start_, length_, height_, friction_;
};

/// A kerb along the road: a strip between y_from and y_to (y_from < y_to),
/// `height` high, its two edges chamfered over `chamfer` (rising inside the
/// strip); level at 0 elsewhere. A kerb on the left side of a road has
/// positive y (ISO 8855).
class KerbRoad : public Road {
public:
    KerbRoad(Real y_from, Real y_to, Real height, Real chamfer, Real friction = 1.0);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "kerb"; }

private:
    Real y_from_, y_to_, height_, chamfer_, friction_;
};

/// A sine wave along the road: h = amplitude sin(2 pi x / wavelength + phase).
class SineRoad : public Road {
public:
    SineRoad(Real amplitude, Real wavelength, Real phase = 0.0, Real friction = 1.0);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "sine road"; }

private:
    Real amplitude_, wavelength_, phase_, friction_;
};

/// A height map on a regular grid: heights[i + nx j] at (x0 + i dx, y0 + j dy),
/// i < nx, j < ny, interpolated bilinearly within each cell (so the normal is
/// constant along neither axis and changes across cell edges). Beyond the grid
/// the height at its edge continues level.
class GridRoad : public Road {
public:
    GridRoad(Real x0, Real y0, Real dx, Real dy, int nx, int ny, std::vector<Real> heights,
             Real friction = 1.0);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "gridded road"; }

private:
    Real x0_, y0_, dx_, dy_;
    int nx_, ny_;
    std::vector<Real> heights_;
    Real friction_;
};

// ============================================================================
// ISO 8608 random roads
// ============================================================================

/// The road classes of ISO 8608 (2016), A (the smoothest) to H.
enum class Iso8608Class { A, B, C, D, E, F, G, H };

/// A class's displacement spectral density at the reference spatial frequency
/// n0 = 0.1 cycles/m, the geometric mean of the class's band in ISO 8608's
/// table: 16e-6 m^3 for class A, four times more for each class after it.
Real iso8608_psd_at_n0(Iso8608Class road_class);

/// A class's displacement spectral density, one-sided, at spatial frequency n
/// [cycles/m]: G_d(n) = G_d(n0) (n / n0)^-2, n0 = 0.1 cycles/m [m^3].
Real iso8608_psd(Iso8608Class road_class, Real n);

struct Iso8608Params {
    Iso8608Class road_class{Iso8608Class::A};
    std::uint64_t seed{0};
    Real length{1000.0};   ///< L: the profile repeats every L [m]
    Real n_min{0.011};     ///< Lowest spatial frequency [cycles/m] (ISO 8608's band)
    Real n_max{2.83};      ///< Highest spatial frequency [cycles/m]
    int samples_per_wavelength{10};  ///< At n_max, for the sampling
    Real friction{1.0};
};

/// An ISO 8608 random road: one profile along X, the same across the road,
///
///     h(x) = sum_k A_k cos(2 pi n_k x + phi_k),
///
/// over the spatial frequencies n_k = k / L in [n_min, n_max], with
/// A_k = sqrt(2 G_d(n_k) / L): the variance between n_k and n_k + 1/L is the
/// class's. The phases are uniform in [0, 2 pi), drawn from the seed by a
/// 64-bit Mersenne twister mapped to doubles without the standard library's
/// distributions, so that a seed gives the same road on every compiler. The
/// profile is sampled once, at L / N with N = ceil(samples_per_wavelength
/// n_max L), and interpolated by cubic Hermite (Catmull-Rom) segments: it
/// passes through the samples and its slope is continuous. It repeats every
/// L along X.
class Iso8608Road : public Road {
public:
    explicit Iso8608Road(const Iso8608Params& params);
    RoadPoint at(Real x, Real y) const override;
    const char* name() const override { return "ISO 8608 road"; }

    const Iso8608Params& params() const { return params_; }
    Real spacing() const { return spacing_; }
    const std::vector<Real>& samples() const { return samples_; }

private:
    Iso8608Params params_;
    Real spacing_{0.0};
    std::vector<Real> samples_;
};

} // namespace mbd
