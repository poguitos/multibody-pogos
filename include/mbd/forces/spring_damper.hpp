#pragma once

// Spring-damper between two points, with tabulated characteristics, preload
// and bump and rebound stops (plan task 3.1). Its coordinate is the distance
// between the points; the force acts along the line through them.

#include <limits>
#include <vector>

#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

struct SpringDamperParams {
    /// Length at which the spring carries no force [m].
    Real free_length{0.0};
    /// Force pushing the points apart, against the compression
    /// free_length - L.
    Curve spring;
    /// Constant force pushing the points apart (an installed preload) [N].
    Real preload{0.0};
    /// Force opposing the rate of extension dL/dt; positive for a positive
    /// rate. A table may differ in bump (dL/dt < 0) and rebound.
    Curve damper;
    /// The bump stop engages when the compression exceeds this clearance [m];
    /// then the curve gives its force, pushing apart, against the stop's own
    /// compression.
    Real bump_clearance{std::numeric_limits<Real>::infinity()};
    Curve bump_stop;
    /// The rebound stop engages when the extension L - free_length exceeds
    /// this clearance; then the curve gives its force, pulling together,
    /// against the stop's own extension.
    Real rebound_clearance{std::numeric_limits<Real>::infinity()};
    Curve rebound_stop;
};

class SpringDamper : public ForceElement {
public:
    /// Between point a1_B on body b1 and point a2_B on body b2, each in its
    /// body's frame.
    SpringDamper(BodyIndex b1, BodyIndex b2, const Vec3& a1_B, const Vec3& a2_B,
                 SpringDamperParams params);

    const char* name() const override { return "nonlinear spring-damper"; }
    std::vector<BodyIndex> bodies() const override { return {body1_, body2_}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// The force pushing the points apart at length L and rate dL/dt, with its
    /// derivatives with respect to both.
    LawValue law(Real length, Real rate) const;

    const SpringDamperParams& params() const { return params_; }

private:
    BodyIndex body1_, body2_;
    Vec3 a1_B_, a2_B_;
    SpringDamperParams params_;
};

} // namespace mbd
