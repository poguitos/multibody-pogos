#include "mbd/forces/road.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <random>
#include <string>

namespace mbd {

namespace {

void check(bool ok, const char* road, const char* what)
{
    MBD_THROW_IF(!ok, std::string("MBD-F050: ") + road + ": " + what);
}

} // namespace

Vec3 surface_normal(Real dh_dx, Real dh_dy)
{
    return Vec3(-dh_dx, -dh_dy, 1.0).normalized();
}

// --- Flat ----------------------------------------------------------------------

FlatRoad::FlatRoad(Real height, Real friction) : height_(height), friction_(friction)
{
    check(friction >= 0.0, "FlatRoad", "the friction must be >= 0");
}

RoadPoint FlatRoad::at(Real, Real) const
{
    return {height_, Vec3::UnitZ(), friction_};
}

// --- Ramp ----------------------------------------------------------------------

RampRoad::RampRoad(Real x_start, Real length, Real rise, Real friction)
    : x_start_(x_start), length_(length), rise_(rise), friction_(friction)
{
    check(length > 0.0, "RampRoad", "the length must be > 0");
    check(friction >= 0.0, "RampRoad", "the friction must be >= 0");
}

RoadPoint RampRoad::at(Real x, Real) const
{
    const Real u = x - x_start_;
    if (u <= 0.0) return {0.0, Vec3::UnitZ(), friction_};
    if (u > length_) return {rise_, Vec3::UnitZ(), friction_};
    const Real slope = rise_ / length_;
    return {slope * u, surface_normal(slope, 0.0), friction_};
}

// --- Bump ----------------------------------------------------------------------

BumpRoad::BumpRoad(Real x_start, Real length, Real height, Real friction)
    : x_start_(x_start), length_(length), height_(height), friction_(friction)
{
    check(length > 0.0, "BumpRoad", "the length must be > 0");
    check(friction >= 0.0, "BumpRoad", "the friction must be >= 0");
}

RoadPoint BumpRoad::at(Real x, Real) const
{
    const Real u = x - x_start_;
    if (u <= 0.0 || u >= length_) return {0.0, Vec3::UnitZ(), friction_};
    const Real w = 2.0 * pi / length_;
    const Real h = 0.5 * height_ * (1.0 - std::cos(w * u));
    const Real dh = 0.5 * height_ * w * std::sin(w * u);
    return {h, surface_normal(dh, 0.0), friction_};
}

// --- Kerb ----------------------------------------------------------------------

KerbRoad::KerbRoad(Real y_from, Real y_to, Real height, Real chamfer, Real friction)
    : y_from_(y_from), y_to_(y_to), height_(height), chamfer_(chamfer), friction_(friction)
{
    check(y_to > y_from, "KerbRoad", "y_to must be greater than y_from");
    check(chamfer > 0.0 && 2.0 * chamfer <= y_to - y_from, "KerbRoad",
          "the chamfer must be > 0 and at most half the kerb's width");
    check(friction >= 0.0, "KerbRoad", "the friction must be >= 0");
}

RoadPoint KerbRoad::at(Real, Real y) const
{
    if (y <= y_from_ || y >= y_to_) return {0.0, Vec3::UnitZ(), friction_};
    const Real slope = height_ / chamfer_;
    if (y < y_from_ + chamfer_) return {slope * (y - y_from_), surface_normal(0.0, slope), friction_};
    if (y > y_to_ - chamfer_) return {slope * (y_to_ - y), surface_normal(0.0, -slope), friction_};
    return {height_, Vec3::UnitZ(), friction_};
}

// --- Sine ----------------------------------------------------------------------

SineRoad::SineRoad(Real amplitude, Real wavelength, Real phase, Real friction)
    : amplitude_(amplitude), wavelength_(wavelength), phase_(phase), friction_(friction)
{
    check(wavelength > 0.0, "SineRoad", "the wavelength must be > 0");
    check(friction >= 0.0, "SineRoad", "the friction must be >= 0");
}

RoadPoint SineRoad::at(Real x, Real) const
{
    const Real w = 2.0 * pi / wavelength_;
    const Real h = amplitude_ * std::sin(w * x + phase_);
    const Real dh = amplitude_ * w * std::cos(w * x + phase_);
    return {h, surface_normal(dh, 0.0), friction_};
}

// --- Grid ----------------------------------------------------------------------

GridRoad::GridRoad(Real x0, Real y0, Real dx, Real dy, int nx, int ny, std::vector<Real> heights,
                   Real friction)
    : x0_(x0), y0_(y0), dx_(dx), dy_(dy), nx_(nx), ny_(ny), heights_(std::move(heights)),
      friction_(friction)
{
    check(dx > 0.0 && dy > 0.0, "GridRoad", "the spacings must be > 0");
    check(nx >= 2 && ny >= 2, "GridRoad", "the grid needs at least 2 x 2 points");
    check(heights_.size() == static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny), "GridRoad",
          "the heights need nx * ny values");
    check(friction >= 0.0, "GridRoad", "the friction must be >= 0");
}

RoadPoint GridRoad::at(Real x, Real y) const
{
    // The cell (i, j) and the position in it (u, v) in [0, 1]; beyond the grid
    // the edge continues level, so the slope across that edge is zero.
    auto locate = [](Real coordinate, Real origin, Real spacing, int n, int& i, Real& u, bool& inside) {
        const Real s = (coordinate - origin) / spacing;
        inside = s >= 0.0 && s <= n - 1;
        const Real clamped = std::clamp(s, Real(0.0), Real(n - 1));
        i = std::min(static_cast<int>(clamped), n - 2);
        u = clamped - i;
    };
    int i = 0, j = 0;
    Real u = 0.0, v = 0.0;
    bool in_x = true, in_y = true;
    locate(x, x0_, dx_, nx_, i, u, in_x);
    locate(y, y0_, dy_, ny_, j, v, in_y);
    const auto h = [this](int a, int b) { return heights_[static_cast<std::size_t>(a + nx_ * b)]; };
    const Real h00 = h(i, j), h10 = h(i + 1, j), h01 = h(i, j + 1), h11 = h(i + 1, j + 1);
    const Real height = (1.0 - u) * (1.0 - v) * h00 + u * (1.0 - v) * h10 + (1.0 - u) * v * h01 + u * v * h11;
    const Real dh_dx = in_x ? ((1.0 - v) * (h10 - h00) + v * (h11 - h01)) / dx_ : 0.0;
    const Real dh_dy = in_y ? ((1.0 - u) * (h01 - h00) + u * (h11 - h10)) / dy_ : 0.0;
    return {height, surface_normal(dh_dx, dh_dy), friction_};
}

// --- ISO 8608 ------------------------------------------------------------------

Real iso8608_psd_at_n0(Iso8608Class road_class)
{
    // ISO 8608, table: class A 16e-6 m^3 (geometric mean of 0 to 32e-6), and
    // each class after it four times the one before: B 64e-6 ... H 262144e-6.
    return 16e-6 * std::pow(4.0, static_cast<int>(road_class));
}

Real iso8608_psd(Iso8608Class road_class, Real n)
{
    const Real n0 = 0.1;   // cycles/m
    return iso8608_psd_at_n0(road_class) * (n0 / n) * (n0 / n);   // waviness w = 2
}

Iso8608Road::Iso8608Road(const Iso8608Params& params) : params_(params)
{
    check(params.length > 0.0, "Iso8608Road", "the length must be > 0");
    check(params.n_min > 0.0 && params.n_max > params.n_min, "Iso8608Road", "it needs 0 < n_min < n_max");
    check(params.samples_per_wavelength >= 4, "Iso8608Road", "it needs at least 4 samples per wavelength");
    check(params.friction >= 0.0, "Iso8608Road", "the friction must be >= 0");

    const Real L = params.length;
    const auto n_samples = static_cast<std::size_t>(std::ceil(params.samples_per_wavelength * params.n_max * L));
    spacing_ = L / static_cast<Real>(n_samples);
    samples_.assign(n_samples, 0.0);

    // Phases from the seed: the generator's 64-bit outputs are fixed by the
    // standard, the distributions' are not, so the top 53 bits are mapped to
    // [0, 1) here.
    std::mt19937_64 engine(params.seed);
    const auto k_first = static_cast<long long>(std::ceil(params.n_min * L));
    const auto k_last = static_cast<long long>(std::floor(params.n_max * L));
    for (long long k = k_first; k <= k_last; ++k) {
        const Real n = static_cast<Real>(k) / L;
        const Real amplitude = std::sqrt(2.0 * iso8608_psd(params.road_class, n) / L);
        const Real phase = 2.0 * pi * static_cast<Real>(engine() >> 11) * 0x1.0p-53;
        // cos(2 pi n x_j + phase) at x_j = j L / N, the angle advancing by
        // 2 pi k / N per sample: a rotation, recomputed every 256 samples to
        // keep its rounding from building up.
        const Real step = 2.0 * pi * static_cast<Real>(k) / static_cast<Real>(n_samples);
        const std::complex<Real> rotate(std::cos(step), std::sin(step));
        std::complex<Real> z;
        for (std::size_t j = 0; j < n_samples; ++j) {
            if (j % 256 == 0) z = std::polar(Real(1.0), step * static_cast<Real>(j) + phase);
            samples_[j] += amplitude * z.real();
            z *= rotate;
        }
    }
}

RoadPoint Iso8608Road::at(Real x, Real) const
{
    // The profile repeats every L: the sample interval of x and its fraction.
    const auto n = static_cast<long long>(samples_.size());
    const Real s = x / spacing_;
    const Real cell = std::floor(s);
    const Real t = s - cell;
    const long long j = static_cast<long long>(cell);
    const auto sample = [&](long long i) { return samples_[static_cast<std::size_t>(((i % n) + n) % n)]; };
    const Real p0 = sample(j - 1), p1 = sample(j), p2 = sample(j + 1), p3 = sample(j + 2);
    // Catmull-Rom: Hermite between p1 and p2 with slopes (p2 - p0) / 2 and
    // (p3 - p1) / 2 per sample interval.
    const Real m1 = 0.5 * (p2 - p0), m2 = 0.5 * (p3 - p1);
    const Real t2 = t * t, t3 = t2 * t;
    const Real h = (2.0 * t3 - 3.0 * t2 + 1.0) * p1 + (t3 - 2.0 * t2 + t) * m1 + (-2.0 * t3 + 3.0 * t2) * p2
                 + (t3 - t2) * m2;
    const Real dh_dt = (6.0 * t2 - 6.0 * t) * p1 + (3.0 * t2 - 4.0 * t + 1.0) * m1 + (-6.0 * t2 + 6.0 * t) * p2
                     + (3.0 * t2 - 2.0 * t) * m2;
    return {h, surface_normal(dh_dt / spacing_, 0.0), params_.friction};
}

} // namespace mbd
