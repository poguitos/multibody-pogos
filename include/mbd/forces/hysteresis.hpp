#pragma once

// Hysteresis as state elements (plan task 7.6, decisions D42 and D66).
//
// MaxwellDamper: a damper with a stiffness in series between two points (a
// Maxwell element). The stiffness stands for the damper's top mount and the
// compressibility of its oil and gas; with it the force lags the stroke, and
// a damper driven through a sine traces a loop against its velocity, not a
// curve. One state: the series stiffness's extension s. The damper's own
// extension rate is then the length's rate less s's, so
//
//     F = series(s),   ds/dt = dL/dt - damper^-1(F),
//
// F the force pulling the points together (tension positive), damper(r) the
// damper's force at its own extension rate r (inverted: it must rise
// everywhere). The state starts at zero, the element relaxed: a Maxwell
// element carries no force at rest, and statics, which holds the states
// where they are, sees none from it.
//
// BoucWenBushing: a six-axis bushing (mbd/forces/bushing.hpp) with Bouc-Wen
// hysteresis on any of its axes (Wen 1976; the form and the bounds on the
// parameters as Ismail, Ikhouane and Rodellar's 2009 review gives them). On
// such an axis the load gains k z, z a state of the deflection's units
// following
//
//     dz/dt = A x_dot - beta |x_dot| |z|^(n-1) z - gamma x_dot |z|^n,
//
// x the axis's deflection. The loop it traces does not depend on how fast
// the deflection changes (z depends on the path of x alone). Under a long
// deflection one way, z tends to (A / (beta + gamma))^(1/n): the hysteretic
// force saturates at k times that. The elastic and viscous parts are the
// bushing's own curves, as a plain Bushing has them: Bouc-Wen's usual
// alpha k x is the elastic curve here, and its (1 - alpha) k is k.
//
// Both follow the kernel's pattern for state elements (the tyre's lag): what
// the rates need from the bodies is taken in apply() and kept for rates().

#include <array>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/state_element.hpp"

namespace mbd {

// ============================================================================
// A damper with a stiffness in series
// ============================================================================

struct MaxwellDamperParams {
    /// The damper: the force opposing its own extension (tension positive)
    /// against its rate of extension [N against m/s]. It must rise
    /// everywhere (Curve::is_increasing): the element inverts it.
    Curve damper;
    /// The stiffness in series with the damper (top mount, oil, gas): the
    /// force against its extension [N against m]; it must rise everywhere.
    Curve series;
};

class MaxwellDamper : public ForceElement, public kernel::StateElement {
public:
    /// Between point a1_B on body b1 and point a2_B on body b2, each in its
    /// body's frame. Throws MBD-F058 for a damper or series curve that does
    /// not rise everywhere.
    MaxwellDamper(BodyIndex b1, BodyIndex b2, const Vec3& a1_B, const Vec3& a2_B, MaxwellDamperParams params);

    const char* name() const override { return "Maxwell damper"; }
    std::vector<BodyIndex> bodies() const override { return {body1_, body2_}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    // --- The series extension, as a state element ---
    int size() const override { return 1; }
    void initial_states(kernel::VecRef x) const override { x.setZero(); }
    void begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x) override;
    void rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot) override;

    /// The force pulling the points together at series extension s [N].
    Real force(Real s) const { return params_.series.value(s); }
    /// The rate of s at s and the length's rate dL/dt: dL/dt - damper^-1(force(s)).
    Real state_rate(Real s, Real length_rate) const;

    /// The series extension the next apply() takes, as begin_evaluation
    /// hands it: for calling apply() without a simulator.
    void set_state(Real s) { s_ = s; }
    Real state() const { return s_; }

    /// The length and its rate at the last apply() [m, m/s].
    Real last_length() const { return last_length_; }
    Real last_length_rate() const { return last_rate_; }

    const MaxwellDamperParams& params() const { return params_; }
    const Vec3& anchor1() const { return a1_B_; }
    const Vec3& anchor2() const { return a2_B_; }

private:
    BodyIndex body1_, body2_;
    Vec3 a1_B_, a2_B_;
    MaxwellDamperParams params_;
    Real s_{0.0};
    mutable Real last_length_{0.0};
    mutable Real last_rate_{0.0};
};

// ============================================================================
// A bushing with Bouc-Wen hysteresis
// ============================================================================

/// Bouc-Wen hysteresis on one axis: the load k z, z following the equation
/// at the top of this file. A stiffness of zero leaves the axis without it.
struct BoucWen {
    Real stiffness{0.0};   ///< k [N/m or N m/rad], >= 0
    Real A{1.0};           ///< > 0
    Real beta{0.0};        ///< [1 / (m or rad)^n]; beta + gamma > 0 and beta >= gamma
    Real gamma{0.0};       ///< [1 / (m or rad)^n]
    Real n{1.0};           ///< The sharpness of the transition, >= 1

    /// dz/dt at z and the deflection's rate x_dot.
    Real rate(Real z, Real x_dot) const;
    /// The value z tends to under a long deflection one way: (A / (beta + gamma))^(1/n).
    Real ultimate() const;
};

struct BoucWenBushingParams {
    /// The elastic and viscous curves, axis by axis, as a Bushing's.
    BushingParams bushing;
    /// Hysteresis about marker 1's X, Y and Z, and along them.
    std::array<BoucWen, 3> rotational;
    std::array<BoucWen, 3> translational;
};

class BoucWenBushing : public ForceElement, public kernel::StateElement {
public:
    /// Between marker X_1M on body b1 and marker X_2M on body b2, each in its
    /// body's frame, as a Bushing. Throws MBD-F059 for an axis with
    /// hysteresis whose parameters are outside the bounds above.
    BoucWenBushing(BodyIndex b1, const Transform3& X_1M, BodyIndex b2, const Transform3& X_2M,
                   BoucWenBushingParams params);

    const char* name() const override { return "Bouc-Wen bushing"; }
    std::vector<BodyIndex> bodies() const override { return bushing_.bodies(); }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    // --- The hysteretic variables, as a state element: one for each axis
    // with hysteresis, in the order of the bushing's coordinates ---
    int size() const override { return static_cast<int>(axes_with_hysteresis_.size()); }
    void initial_states(kernel::VecRef x) const override { x.setZero(); }
    void begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x) override;
    void rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot) override;

    /// The load [moment; force] on body 2, in marker 1's axes, at the
    /// coordinates x, rates x_dot and hysteretic variables z (one per
    /// coordinate; those of axes without hysteresis are not read).
    Vec6 load(const Vec6& x, const Vec6& x_dot, const Vec6& z) const;

    /// The coordinate (0 to 5: rotations, then displacements) of each state.
    const std::vector<int>& axes_with_hysteresis() const { return axes_with_hysteresis_; }
    /// Each coordinate's hysteresis, [rotational; translational].
    const BoucWen& axis(int coordinate) const { return axes_[static_cast<std::size_t>(coordinate)]; }

    /// The hysteretic variables the next apply() takes, one per coordinate,
    /// as begin_evaluation hands them: for calling apply() without a
    /// simulator.
    void set_states(const Vec6& z) { z_ = z; }
    const Vec6& states() const { return z_; }

    /// The deflection's rates at the last apply().
    const Vec6& last_rates() const { return last_x_dot_; }

    /// The bushing's geometry and its elastic and viscous curves.
    const Bushing& bushing() const { return bushing_; }

private:
    Bushing bushing_;
    std::array<BoucWen, 6> axes_;
    std::vector<int> axes_with_hysteresis_;
    Vec6 z_{Vec6::Zero()};
    mutable Vec6 last_x_dot_{Vec6::Zero()};
};

} // namespace mbd
