#pragma once

// Characteristics of force elements: a force as a function of one variable
// (a deflection, a rate), with its slope (plan task 3.1, decision D24).
//
// A tabulated curve is interpolated by a monotone piecewise cubic (PCHIP,
// Fritsch and Carlson): it passes through every point, has a continuous slope,
// and does not overshoot, so a monotone table gives a monotone force. Beyond
// the table it continues as a straight line with the end slope.

#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"

namespace mbd {

/// A force law evaluated at an element's own coordinate x and its rate
/// x_dot (decision D24): the force and its two derivatives.
struct LawValue {
    Real force{0.0};
    Real d_position{0.0};   ///< d force / d x
    Real d_rate{0.0};       ///< d force / d x_dot
};

class Curve {
public:
    /// The zero function.
    Curve() = default;

    /// f(x) = offset + slope * x.
    static Curve linear(Real slope, Real offset = 0.0);

    /// Through the points (x[k], y[k]); x strictly increasing, at least two
    /// points.
    static Curve table(std::vector<Real> x, std::vector<Real> y);

    Real value(Real x) const;
    Real slope(Real x) const;

    bool is_zero() const { return xs_.empty() && slope_ == 0.0 && offset_ == 0.0; }

private:
    // Linear: offset_ + slope_ x. Tabulated: xs_, ys_ and the Hermite slopes ms_.
    Real slope_{0.0};
    Real offset_{0.0};
    std::vector<Real> xs_, ys_, ms_;
};

} // namespace mbd
