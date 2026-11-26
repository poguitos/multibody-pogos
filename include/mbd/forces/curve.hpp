#pragma once

// Characteristics of force elements: a force as a function of one variable
// (a deflection, a rate), with its slope (plan task 3.1, decision D24).
//
// A tabulated curve is interpolated by a monotone piecewise cubic (PCHIP,
// Fritsch and Carlson): it passes through every point, has a continuous slope,
// and does not overshoot, so a monotone table gives a monotone force. Beyond
// the table it continues as a straight line with the end slope.
//
// A bushing's curve can instead be given by its shape, four numbers that an
// optimisation can move (task 7.6, D42, D66): see BushingShape.

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

/// The shape of a bushing's characteristic (task 7.6, D66): an odd function
/// of the deflection x, straight with slope `stiffness` over |x| <=
/// linear_range; then progressive, its slope rising smoothly from stiffness
/// to stop_stiffness until |x| = stop (the slope a cubic in |x|, flat at both
/// ends, so that the force has a continuous second derivative); beyond the
/// stop, straight with slope stop_stiffness. With w = stop - linear_range and
/// s = (|x| - linear_range) / w, the force in the progressive range is
/// stiffness |x| + (stop_stiffness - stiffness) w (s^3 - s^4 / 2), with the
/// sign of x. A stop equal to linear_range makes the change of slope abrupt;
/// a stop_stiffness below stiffness makes the end soften instead.
struct BushingShape {
    Real stiffness{0.0};       ///< The slope through the linear range [N/m or N m/rad], >= 0
    Real linear_range{0.0};    ///< Half-width of the linear range [m or rad], >= 0
    Real stop{0.0};            ///< Where the slope reaches stop_stiffness [m or rad], >= linear_range
    Real stop_stiffness{0.0};  ///< The slope beyond the stop, >= 0
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

    /// A bushing's characteristic of the given shape (task 7.6, D66). Throws
    /// MBD-F056 for a number that is negative or not finite, or a stop before
    /// the end of the linear range.
    static Curve bushing(const BushingShape& shape);

    Real value(Real x) const;
    Real slope(Real x) const;

    /// The integral of the curve from a to b, exact for the cubic pieces and
    /// the straight continuations: a spring's stored energy.
    Real integral(Real a, Real b) const;

    /// True when the curve rises everywhere, so that it has an inverse: a
    /// line with a positive slope, a table whose values rise and whose end
    /// slopes are positive, a bushing shape with both stiffnesses positive.
    bool is_increasing() const;

    /// The x at which value(x) = y, to rounding, for a curve that rises
    /// everywhere (a damper's force against its rate, inverted: task 7.6).
    /// Throws MBD-F057 for a curve that does not.
    Real inverse(Real y) const;

    bool is_zero() const { return !shaped_ && xs_.empty() && slope_ == 0.0 && offset_ == 0.0; }

    /// What the curve was made from (task 6.4, for model files): a table's
    /// points, a line's slope and offset (zero for a table or a shape), or a
    /// bushing's shape.
    bool is_table() const { return !xs_.empty(); }
    const std::vector<Real>& table_x() const { return xs_; }
    const std::vector<Real>& table_y() const { return ys_; }
    Real line_slope() const { return slope_; }
    Real line_offset() const { return offset_; }
    bool is_bushing() const { return shaped_; }
    const BushingShape& bushing_shape() const { return shape_; }

private:
    /// The integral from the first point (tabulated) or from 0 (linear or
    /// shaped) to x.
    Real primitive(Real x) const;

    // Linear: offset_ + slope_ x. Tabulated: xs_, ys_, the Hermite slopes ms_
    // and the integral from xs_[0] to each point, cum_. Shaped: shape_.
    Real slope_{0.0};
    Real offset_{0.0};
    std::vector<Real> xs_, ys_, ms_, cum_;
    bool shaped_{false};
    BushingShape shape_;
};

} // namespace mbd
