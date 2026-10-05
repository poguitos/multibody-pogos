#pragma once

#include "mbd/core/core.hpp"
#include "mbd/model/rigid_body.hpp"

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

    /// Short name, for messages.
    virtual const char* name() const = 0;

    /// The bodies the element acts on, for the checks.
    virtual std::vector<BodyIndex> bodies() const = 0;
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
                       Real stiffness, Real damping, Real length_0);

    const char* name() const override { return "spring-damper"; }
    std::vector<BodyIndex> bodies() const override { return {body1_idx, body2_idx}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;
};

} // namespace mbd
