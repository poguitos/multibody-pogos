#pragma once

// Rotational spring-damper between two bodies, about an axis (plan task 3.1):
// a torsion bar or a hinge spring between bodies that are not parent and child.
// Its coordinate is the twist of marker j relative to marker i about marker
// i's Z axis, from the swing-twist split of their relative rotation, so a
// small misalignment of the two Z axes does not change it to first order.
// Across a single revolute joint, JointCoordinateForce is the exact and
// cheaper choice.

#include <vector>

#include "mbd/core/math.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

struct RotationalSpringDamperParams {
    /// Twist at which the spring carries no torque [rad], in (-pi, pi].
    Real reference{0.0};
    /// Restoring torque against the twist - reference.
    Curve spring;
    /// Constant torque in the direction of increasing twist [N m].
    Real preload{0.0};
    /// Torque opposing the twist rate.
    Curve damper;
};

class RotationalSpringDamper : public ForceElement {
public:
    /// Between marker X_1M on body b1 and marker X_2M on body b2, each in its
    /// body's frame; the axis is the Z axis of the first.
    RotationalSpringDamper(BodyIndex b1, const Transform3& X_1M, BodyIndex b2,
                           const Transform3& X_2M, RotationalSpringDamperParams params);

    const char* name() const override { return "rotational spring-damper"; }
    std::vector<BodyIndex> bodies() const override { return {body1_, body2_}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// Twist of marker 2 relative to marker 1 about marker 1's Z axis, in
    /// (-pi, pi], and its rate.
    void twist(const std::vector<RigidBodyState>& states, Real& angle, Real& rate) const;

    /// The torque on body 2 about the axis at the given twist and rate, with
    /// its derivatives.
    LawValue law(Real angle, Real rate) const;

    /// Energy stored at a twist by the spring and the preload, zero at the
    /// reference: the conservative torque is -dV/d(angle).
    Real potential_energy(Real angle) const;

private:
    BodyIndex body1_, body2_;
    Transform3 X_1M_, X_2M_;
    RotationalSpringDamperParams params_;
};

} // namespace mbd
