#include "mbd/forces/curve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
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

// A bushing shape's force, slope and integral from 0 at x >= 0 (BushingShape:
// odd force, even slope and integral). With w = b - a and s = (x - a) / w:
//   slope  k + (k_s - k)(3 s^2 - 2 s^3)          in the progressive range,
//   force  k x + (k_s - k) w (s^3 - s^4 / 2)     its integral from a,
//   energy k x^2 / 2 + (k_s - k) w^2 (s^4 / 4 - s^5 / 10).
// At s = 1 the last two add (k_s - k) w / 2 and 0.15 (k_s - k) w^2.
struct ShapeValue {
    Real force, slope, energy;
};

ShapeValue shape_at(const BushingShape& p, Real x)
{
    const Real k = p.stiffness, ks = p.stop_stiffness, a = p.linear_range, b = p.stop;
    if (x <= a) return {k * x, k, 0.5 * k * x * x};
    const Real w = b - a;
    if (x < b) {
        const Real s = (x - a) / w;
        const Real s2 = s * s, s3 = s2 * s;
        return {k * x + (ks - k) * w * (s3 - 0.5 * s3 * s),
                k + (ks - k) * (3.0 * s2 - 2.0 * s3),
                0.5 * k * x * x + (ks - k) * w * w * (0.25 * s3 * s - 0.1 * s3 * s2)};
    }
    const Real f_b = k * b + 0.5 * (ks - k) * w;
    const Real e_b = 0.5 * k * b * b + 0.15 * (ks - k) * w * w;
    const Real d = x - b;
    return {f_b + ks * d, ks, e_b + f_b * d + 0.5 * ks * d * d};
}

// The x in [lo, hi] at which f(x) = y, for f rising on the interval with
// f(lo) <= y <= f(hi): Newton's method kept inside a bracket that shrinks
// at every step, bisecting where a Newton step would leave it.
template <class F, class DF>
Real solve_rising(const F& f, const DF& df, Real y, Real lo, Real hi)
{
    const Real f_lo = f(lo), f_hi = f(hi);
    if (y <= f_lo) return lo;
    if (y >= f_hi) return hi;
    Real x = lo + (hi - lo) * (y - f_lo) / (f_hi - f_lo);
    for (int it = 0; it < 200; ++it) {
        const Real g = f(x) - y;
        if (g == 0.0) return x;
        if (g < 0.0) lo = x; else hi = x;
        const Real d = df(x);
        Real next = (d > 0.0) ? x - g / d : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);
        const Real eps = std::numeric_limits<Real>::epsilon();
        if (std::abs(next - x) <= 2.0 * eps * (std::abs(x) + std::abs(next))
            || hi - lo <= 2.0 * eps * (std::abs(lo) + std::abs(hi))) {
            return next;
        }
        x = next;
    }
    return x;
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

    // Integral of each cubic piece: h (y_k + y_k+1) / 2 + h^2 (m_k - m_k+1) / 12.
    std::vector<Real> cum(n, 0.0);
    for (std::size_t k = 0; k + 1 < n; ++k) {
        cum[k + 1] = cum[k] + h[k] * (y[k] + y[k + 1]) / 2.0 + h[k] * h[k] * (m[k] - m[k + 1]) / 12.0;
    }

    Curve c;
    c.xs_ = std::move(x);
    c.ys_ = std::move(y);
    c.ms_ = std::move(m);
    c.cum_ = std::move(cum);
    return c;
}

Curve Curve::bushing(const BushingShape& shape)
{
    const BushingShape& p = shape;
    const bool finite = std::isfinite(p.stiffness) && std::isfinite(p.linear_range) && std::isfinite(p.stop)
                     && std::isfinite(p.stop_stiffness);
    MBD_THROW_IF(!finite || p.stiffness < 0.0 || p.linear_range < 0.0 || p.stop_stiffness < 0.0,
                 "MBD-F056: Curve::bushing: the stiffnesses, linear range and stop must be finite and >= 0");
    MBD_THROW_IF(!(p.stop >= p.linear_range),
                 "MBD-F056: Curve::bushing: the stop must not come before the end of the linear range");
    Curve c;
    c.shaped_ = true;
    c.shape_ = p;
    return c;
}

bool Curve::is_increasing() const
{
    if (shaped_) return shape_.stiffness > 0.0 && shape_.stop_stiffness > 0.0;
    if (xs_.empty()) return slope_ > 0.0;
    for (std::size_t k = 1; k < ys_.size(); ++k) {
        if (!(ys_[k] > ys_[k - 1])) return false;
    }
    return ms_.front() > 0.0 && ms_.back() > 0.0;
}

Real Curve::inverse(Real y) const
{
    MBD_THROW_IF(!is_increasing(),
                 "MBD-F057: Curve::inverse: the curve must rise everywhere (a line with a positive slope, a table "
                 "with rising values and positive end slopes, a bushing shape with positive stiffnesses)");
    if (std::isnan(y)) return y;
    if (shaped_) {
        const BushingShape& p = shape_;
        const Real m = std::abs(y);
        const Real f_a = p.stiffness * p.linear_range;
        const Real f_b = p.stiffness * p.stop + 0.5 * (p.stop_stiffness - p.stiffness) * (p.stop - p.linear_range);
        Real x;
        if (m <= f_a) {
            x = m / p.stiffness;
        } else if (m >= f_b) {
            x = p.stop + (m - f_b) / p.stop_stiffness;
        } else {
            x = solve_rising([&](Real u) { return shape_at(p, u).force; },
                             [&](Real u) { return shape_at(p, u).slope; }, m, p.linear_range, p.stop);
        }
        return y < 0.0 ? -x : x;
    }
    if (xs_.empty()) return (y - offset_) / slope_;
    if (y <= ys_.front()) return xs_.front() + (y - ys_.front()) / ms_.front();
    if (y >= ys_.back()) return xs_.back() + (y - ys_.back()) / ms_.back();
    const std::size_t k = static_cast<std::size_t>(
        std::upper_bound(ys_.begin(), ys_.end(), y) - ys_.begin()) - 1;
    return solve_rising([this](Real u) { return value(u); }, [this](Real u) { return slope(u); }, y, xs_[k],
                        xs_[k + 1]);
}

Real Curve::primitive(Real x) const
{
    if (shaped_) return shape_at(shape_, std::abs(x)).energy;
    if (xs_.empty()) return offset_ * x + 0.5 * slope_ * x * x;
    if (std::isnan(x)) return x;   // no interval to look up (see value())
    if (x <= xs_.front()) {
        const Real d = x - xs_.front();
        return ys_.front() * d + 0.5 * ms_.front() * d * d;
    }
    if (x >= xs_.back()) {
        const Real d = x - xs_.back();
        return cum_.back() + ys_.back() * d + 0.5 * ms_.back() * d * d;
    }
    const std::size_t k = static_cast<std::size_t>(
        std::upper_bound(xs_.begin(), xs_.end(), x) - xs_.begin()) - 1;
    const Real h = xs_[k + 1] - xs_[k];
    const Real s = (x - xs_[k]) / h;
    // Integrals from 0 to s of the cubic Hermite basis.
    const Real s2 = s * s, s3 = s2 * s, s4 = s3 * s;
    const Real a00 = 0.5 * s4 - s3 + s;
    const Real a10 = 0.25 * s4 - 2.0 * s3 / 3.0 + 0.5 * s2;
    const Real a01 = -0.5 * s4 + s3;
    const Real a11 = 0.25 * s4 - s3 / 3.0;
    return cum_[k] + h * (a00 * ys_[k] + a10 * h * ms_[k] + a01 * ys_[k + 1] + a11 * h * ms_[k + 1]);
}

Real Curve::integral(Real a, Real b) const
{
    return primitive(b) - primitive(a);
}

Real Curve::value(Real x) const
{
    if (shaped_) {
        if (std::isnan(x)) return x;
        const Real f = shape_at(shape_, std::abs(x)).force;
        return x < 0.0 ? -f : f;
    }
    if (xs_.empty()) return offset_ + slope_ * x;
    // NaN fails every comparison: past both ends' tests, upper_bound would
    // return end() and the interval would be the one past the last point,
    // read out of bounds (task 5.4: a deliberately unstable run fed its NaN
    // states to a spring's table, which the address sanitizer stopped).
    if (std::isnan(x)) return x;
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
    if (shaped_) return std::isnan(x) ? x : shape_at(shape_, std::abs(x)).slope;
    if (xs_.empty()) return slope_;
    if (std::isnan(x)) return x;   // no interval to look up (see value())
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
