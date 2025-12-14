#pragma once

#include "mbd/core.hpp"
#include "mbd/math.hpp"
#include "mbd/rigid_body.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

// Abstract base class for any element that generates forces/torques
class ForceElement {
public:
    virtual ~ForceElement() = default;

    // Calculate forces based on state and accumulate them into 'forces'
    // internal_time can be used for time-varying forces (optional)
    virtual void apply(const RigidBodyState& state, 
                       RigidBodyForces& forces) const = 0;
};

// A linear spring-damper connecting a fixed point in World to a point on Body.
class LinearSpringDamper : public ForceElement {
public:
    Vec3 anchor_W;      // Fixed anchor position in World Frame
    Vec3 anchor_B;      // Attachment point in Body Frame (relative to COM)
    Real k;             // Stiffness [N/m]
    Real c;             // Damping [N*s/m]
    Real rest_length;   // Unstretched length [m]

    LinearSpringDamper(const Vec3& anchor_world,
                       const Vec3& anchor_body,
                       Real stiffness,
                       Real damping,
                       Real length_0)
        : anchor_W(anchor_world)
        , anchor_B(anchor_body)
        , k(stiffness)
        , c(damping)
        , rest_length(length_0)
    {
        MBD_THROW_IF(k < 0 || c < 0 || rest_length < 0, 
            "SpringDamper parameters must be non-negative");
    }

    void apply(const RigidBodyState& state, RigidBodyForces& forces) const override
    {
        // 1. Calculate kinematics of the body attachment point in World
        const Mat3 R_WB = state.q_WB.toRotationMatrix();
        const Vec3 r_W = R_WB * anchor_B; // vector from COM to attachment in W
        const Vec3 pos_B_in_W = state.p_WB + r_W;

        // 2. Vector from World Anchor to Body Anchor
        const Vec3 diff = pos_B_in_W - anchor_W;
        const Real dist = diff.norm();

        // Avoid division by zero if anchors coincide
        if (dist < Real(1e-9)) {
            return; 
        }

        const Vec3 dir = diff / dist; // Unit vector pointing towards body

        // 3. Velocity of the body attachment point
        // v_point = v_com + w x r
        const Vec3 vel_point_W = state.v_WB + state.w_WB.cross(r_W);
        
        // Projected velocity along the spring axis (scalar)
        // positive means separating (lengthening)
        const Real vel_rel = vel_point_W.dot(dir);

        // 4. Spring-Damper Force Magnitude (Hooke's Law + Damping)
        // F_spring = -k * (current_len - rest_len)
        // F_damper = -c * velocity
        const Real force_mag = -k * (dist - rest_length) - c * vel_rel;

        // Force vector acting on the body (along direction dir)
        const Vec3 F_vec = dir * force_mag;

        // 5. Accumulate
        forces.f_W += F_vec;
        
        // Torque = r x F
        forces.tau_W += r_W.cross(F_vec);
    }
};

} // namespace mbd