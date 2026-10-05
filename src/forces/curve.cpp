#include "mbd/forces/curve.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mbd {

namespace {

Real sign(Real x) { return (x > 0.0) - (x < 0.0); }

// End slope of the shape-preserving three-point formula (as in Moler's pchip):
// h0, d0 the end interval and its secant, h1, d1 the next.
Real end_slope(Real h0, Real h1, Real d0, Real d1)
{
    Real m = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1);
    if (sign(m) != sign(d0)) {
        m = 0.0;
    } else if (sign(d0) != sign(d1) && std::abs(m) > std::abs(3.0 * d0)) {
        m = 3.0 * d0;
    }
    return m;
}

} // namespace

Curve Curve::linear(Real slope, Real offset)
{
    Curve c;
    c.slope_ = slope;
    c.offset_ = offset;
    return c;
}

Curve Curve::table(std::vector<Real> x, std::vector<Real> y)
{
    MBD_THROW_IF(x.size() != y.size() || x.size() < 2,
                 "MBD-F002: Curve::table: x and y need the same length, at least 2");
    for (std::size_t k = 0; k < x.size(); ++k) {
        MBD_THROW_IF(!std::isfinite(x[k]) || !std::isfinite(y[k]),
                     "MBD-F002: Curve::table: the points must be finite");
        MBD_THROW_IF(k > 0 && !(x[k] > x[k - 1]),
                     "MBD-F002: Curve::table: x must be strictly increasing");
    }

    const std::size_t n = x.size();
    std::vector<Real> h(n - 1), d(n - 1);
    for (std::size_t k = 0; k + 1 < n; ++k) {
        h[k] = x[k + 1] - x[k];
        d[k] = (y[k + 1] - y[k]) / h[k];
    }

    // Fritsch-Carlson slopes: zero at a local extremum of the data, otherwise
    // a weighted harmonic mean of the neighbouring secants, which keeps each
    // interval monotone where the data are.
    std::vector<Real> m(n, 0.0);
    if (n == 2) {
        m[0] = m[1] = d[0];
    } else {
        for (std::size_t k = 1; k + 1 < n; ++k) {
            if (sign(d[k - 1]) * sign(d[k]) <= 0.0) {
                m[k] = 0.0;
            } else {
                const Real w1 = 2.0 * h[k] + h[k - 1];
                const Real w2 = h[k] + 2.0 * h[k - 1];
                m[k] = (w1 + w2) / (w1 / d[k - 1] + w2 / d[k]);
            }
        }
        m[0] = end_slope(h[0], h[1], d[0], d[1]);
        m[n - 1] = end_slope(h[n - 2], h[n - 3], d[n - 2], d[n - 3]);
    }

    Curve c;
    c.xs_ = std::move(x);
    c.ys_ = std::move(y);
    c.ms_ = std::move(m);
    return c;
}

Real Curve::value(Real x) const
{
    if (xs_.empty()) return offset_ + slope_ * x;
    if (x <= xs_.front()) return ys_.front() + ms_.front() * (x - xs_.front());
    if (x >= xs_.back()) return ys_.back() + ms_.back() * (x - xs_.back());

    const std::size_t k = static_cast<std::size_t>(
        std::upper_bound(xs_.begin(), xs_.end(), x) - xs_.begin()) - 1;
    const Real h = xs_[k + 1] - xs_[k];
    const Real s = (x - xs_[k]) / h;
    // Cubic Hermite basis on [0, 1].
    const Real h00 = (1.0 + 2.0 * s) * (1.0 - s) * (1.0 - s);
    const Real h10 = s * (1.0 - s) * (1.0 - s);
    const Real h01 = s * s * (3.0 - 2.0 * s);
    const Real h11 = s * s * (s - 1.0);
    return h00 * ys_[k] + h10 * h * ms_[k] + h01 * ys_[k + 1] + h11 * h * ms_[k + 1];
}

Real Curve::slope(Real x) const
{
    if (xs_.empty()) return slope_;
    if (x <= xs_.front()) return ms_.front();
    if (x >= xs_.back()) return ms_.back();

    const std::size_t k = static_cast<std::size_t>(
        std::upper_bound(xs_.begin(), xs_.end(), x) - xs_.begin()) - 1;
    const Real h = xs_[k + 1] - xs_[k];
    const Real s = (x - xs_[k]) / h;
    // Derivatives of the basis with respect to s, divided by h for x.
    const Real d00 = 6.0 * s * s - 6.0 * s;
    const Real d10 = 3.0 * s * s - 4.0 * s + 1.0;
    const Real d01 = -6.0 * s * s + 6.0 * s;
    const Real d11 = 3.0 * s * s - 2.0 * s;
    return (d00 * ys_[k] + d01 * ys_[k + 1]) / h + d10 * ms_[k] + d11 * ms_[k + 1];
}

} // namespace mbd
