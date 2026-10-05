#include "mbd/forces/rotational_spring_damper.hpp"

#include <cmath>
#include <utility>

namespace mbd {

RotationalSpringDamper::RotationalSpringDamper(BodyIndex b1, const Transform3& X_1M, BodyIndex b2,
                                               const Transform3& X_2M,
                                               RotationalSpringDamperParams params)
    : body1_(b1), body2_(b2), X_1M_(X_1M), X_2M_(X_2M), params_(std::move(params))
{}

void RotationalSpringDamper::twist(const std::vector<RigidBodyState>& states, Real& angle,
                                   Real& rate) const
{
    const auto& s1 = states[static_cast<std::size_t>(body1_)];
    const auto& s2 = states[static_cast<std::size_t>(body2_)];
    const Quat q1 = s1.q_WB * X_1M_.q;
    const Quat q2 = s2.q_WB * X_2M_.q;
    // Swing-twist: the twist about Z of the relative rotation q1^-1 q2 is the
    // rotation about Z whose quaternion has the relative one's z and w parts.
    const Quat rel = q1.conjugate() * q2;
    angle = 2.0 * std::atan2(rel.z(), rel.w());
    if (angle > pi) angle -= 2.0 * pi;
    if (angle <= -pi) angle += 2.0 * pi;
    rate = (s2.w_WB - s1.w_WB).dot(q1 * Vec3::UnitZ());
}

LawValue RotationalSpringDamper::law(Real angle, Real rate) const
{
    const RotationalSpringDamperParams& p = params_;
    LawValue out;
    out.force = p.preload - p.spring.value(angle - p.reference) - p.damper.value(rate);
    out.d_position = -p.spring.slope(angle - p.reference);
    out.d_rate = -p.damper.slope(rate);
    return out;
}

Real RotationalSpringDamper::potential_energy(Real angle) const
{
    const RotationalSpringDamperParams& p = params_;
    return p.spring.integral(0.0, angle - p.reference) - p.preload * (angle - p.reference);
}

void RotationalSpringDamper::apply(const std::vector<RigidBodyState>& states,
                                   std::vector<RigidBodyForces>& forces) const
{
    Real angle = 0.0, rate = 0.0;
    twist(states, angle, rate);
    const Vec3 axis_W = (states[static_cast<std::size_t>(body1_)].q_WB * X_1M_.q) * Vec3::UnitZ();
    const Vec3 T2 = law(angle, rate).force * axis_W;   // on body 2; body 1 takes the reaction
    forces[static_cast<std::size_t>(body2_)].tau_W += T2;
    forces[static_cast<std::size_t>(body1_)].tau_W -= T2;
}

} // namespace mbd
