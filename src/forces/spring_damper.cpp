#include "mbd/forces/spring_damper.hpp"

#include <cmath>
#include <utility>

namespace mbd {

SpringDamper::SpringDamper(BodyIndex b1, BodyIndex b2, const Vec3& a1_B, const Vec3& a2_B,
                           SpringDamperParams params)
    : body1_(b1), body2_(b2), a1_B_(a1_B), a2_B_(a2_B), params_(std::move(params))
{
    MBD_THROW_IF(!(params_.free_length >= 0.0) || !(params_.bump_clearance >= 0.0)
                     || !(params_.rebound_clearance >= 0.0),
                 "MBD-F003: SpringDamper: free length and stop clearances must be >= 0");
}

LawValue SpringDamper::law(Real length, Real rate) const
{
    const SpringDamperParams& p = params_;
    const Real compression = p.free_length - length;

    LawValue out;
    out.force = p.preload + p.spring.value(compression) - p.damper.value(rate);
    out.d_position = -p.spring.slope(compression);   // d(compression)/dL = -1
    out.d_rate = -p.damper.slope(rate);

    const Real bump = compression - p.bump_clearance;
    if (bump > 0.0) {
        out.force += p.bump_stop.value(bump);
        out.d_position -= p.bump_stop.slope(bump);
    }
    const Real rebound = -compression - p.rebound_clearance;
    if (rebound > 0.0) {
        out.force -= p.rebound_stop.value(rebound);
        out.d_position -= p.rebound_stop.slope(rebound);
    }
    return out;
}

void SpringDamper::apply(const std::vector<RigidBodyState>& states,
                         std::vector<RigidBodyForces>& forces) const
{
    const auto& s1 = states[static_cast<std::size_t>(body1_)];
    const auto& s2 = states[static_cast<std::size_t>(body2_)];
    const Vec3 r1_W = s1.q_WB * a1_B_;
    const Vec3 r2_W = s2.q_WB * a2_B_;
    const Vec3 d = (s2.p_WB + r2_W) - (s1.p_WB + r1_W);
    const Real length = d.norm();
    if (length < Real(1e-9)) return;   // no direction to push along
    const Vec3 dir = d / length;
    const Vec3 v1 = s1.v_WB + s1.w_WB.cross(r1_W);
    const Vec3 v2 = s2.v_WB + s2.w_WB.cross(r2_W);
    const Real rate = (v2 - v1).dot(dir);

    const Vec3 F2 = law(length, rate).force * dir;   // on body 2, away from body 1
    forces[static_cast<std::size_t>(body1_)].f_W -= F2;
    forces[static_cast<std::size_t>(body1_)].tau_W -= r1_W.cross(F2);
    forces[static_cast<std::size_t>(body2_)].f_W += F2;
    forces[static_cast<std::size_t>(body2_)].tau_W += r2_W.cross(F2);
}

} // namespace mbd
