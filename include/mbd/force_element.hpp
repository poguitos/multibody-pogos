#pragma once

#include "mbd/core.hpp"
#include "mbd/rigid_body.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

/// Abstract base class for force-generating elements.
///
/// The apply() method reads from `states` and accumulates into `forces`.
/// Each concrete element stores which body indices it operates on.
class ForceElement {
public:
    virtual ~ForceElement() = default;

    virtual void apply(const std::vector<RigidBodyState>& states,
                       std::vector<RigidBodyForces>& forces) const = 0;
};

/// Linear spring-damper connecting a point on body1 to a point on body2.
/// For a world-anchored spring, use body1_idx = kGroundIndex (0).
class LinearSpringDamper : public ForceElement {
public:
    BodyIndex body1_idx;
    BodyIndex body2_idx;
    Vec3 anchor1_B;     // attachment in body1 local frame
    Vec3 anchor2_B;     // attachment in body2 local frame
    Real k;
    Real c;
    Real rest_length;

    LinearSpringDamper(BodyIndex b1, BodyIndex b2,
                       const Vec3& a1_local, const Vec3& a2_local,
                       Real stiffness, Real damping, Real length_0)
        : body1_idx(b1), body2_idx(b2)
        , anchor1_B(a1_local), anchor2_B(a2_local)
        , k(stiffness), c(damping), rest_length(length_0)
    {
        MBD_THROW_IF(k < 0 || c < 0 || rest_length < 0,
            "SpringDamper parameters must be non-negative");
    }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override
    {
        const auto& s1 = states[static_cast<size_t>(body1_idx)];
        const auto& s2 = states[static_cast<size_t>(body2_idx)];

        // Attachment point kinematics in world frame
        const Vec3 r1_W = s1.q_WB * anchor1_B;
        const Vec3 r2_W = s2.q_WB * anchor2_B;
        const Vec3 p1_W = s1.p_WB + r1_W;
        const Vec3 p2_W = s2.p_WB + r2_W;

        const Vec3 diff = p2_W - p1_W;
        const Real dist = diff.norm();
        if (dist < Real(1e-9)) return;

        const Vec3 dir = diff / dist;

        // Relative velocity along spring axis
        const Vec3 v1_W = s1.v_WB + s1.w_WB.cross(r1_W);
        const Vec3 v2_W = s2.v_WB + s2.w_WB.cross(r2_W);
        const Real vel_rel = (v2_W - v1_W).dot(dir);

        const Real force_mag = -k * (dist - rest_length) - c * vel_rel;

        const Vec3 F_on_2 = dir * force_mag;
        const Vec3 F_on_1 = -F_on_2;

        // Equal and opposite on both bodies
        forces[static_cast<size_t>(body1_idx)].f_W   += F_on_1;
        forces[static_cast<size_t>(body1_idx)].tau_W += r1_W.cross(F_on_1);
        forces[static_cast<size_t>(body2_idx)].f_W   += F_on_2;
        forces[static_cast<size_t>(body2_idx)].tau_W += r2_W.cross(F_on_2);
    }
};

} // namespace mbd