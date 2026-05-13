#pragma once

// Anti-roll bar (ARB): torsion spring coupling two wheels on an axle.
//
// Measures vertical displacement (in chassis frame) of each wheel center
// from reference, and applies restoring forces proportional to the
// difference. Opposes roll without affecting symmetric bounce.

#include "mbd/core.hpp"
#include "mbd/force_element.hpp"

namespace mbd {

class AntiRollBar : public ForceElement {
public:
    BodyIndex chassis_idx;
    BodyIndex left_wheel_idx;
    BodyIndex right_wheel_idx;

    /// Roll stiffness [N/m] per unit travel difference.
    /// Typical values: 20,000-80,000 N/m for passenger cars.
    Real k_arb{30000.0};

    /// Roll damping [Ns/m] per unit travel difference velocity.
    Real c_arb{0.0};

    /// Reference Y coordinate of left wheel in chassis frame.
    /// Usually set automatically from initial configuration.
    Real z_L_ref{0.0};

    /// Reference Y coordinate of right wheel in chassis frame.
    Real z_R_ref{0.0};

    AntiRollBar(BodyIndex chassis,
                BodyIndex left_wheel,
                BodyIndex right_wheel,
                Real stiffness,
                Real damping = 0.0)
        : chassis_idx(chassis)
        , left_wheel_idx(left_wheel)
        , right_wheel_idx(right_wheel)
        , k_arb(stiffness)
        , c_arb(damping)
    {
        MBD_THROW_IF(stiffness < 0.0, "AntiRollBar: stiffness must be >= 0");
        MBD_THROW_IF(damping < 0.0, "AntiRollBar: damping must be >= 0");
    }

    /// Capture reference heights from current state.
    /// Call this ONCE after the vehicle is built and positioned at equilibrium,
    /// before starting the simulation.
    void capture_reference(const std::vector<RigidBodyState>& states)
    {
        const auto& chassis = states[chassis_idx];
        const auto& L = states[left_wheel_idx];
        const auto& R = states[right_wheel_idx];

        const Transform3 T_CW = chassis.pose_WB().inverse();
        z_L_ref = T_CW.apply(L.p_WB).y();
        z_R_ref = T_CW.apply(R.p_WB).y();
    }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override
    {
        const auto& chassis = states[chassis_idx];
        const auto& L_state = states[left_wheel_idx];
        const auto& R_state = states[right_wheel_idx];

        // Transform wheel centers into chassis frame
        const Transform3 T_CW = chassis.pose_WB().inverse();
        const Vec3 p_L_C = T_CW.apply(L_state.p_WB);
        const Vec3 p_R_C = T_CW.apply(R_state.p_WB);

        // Displacement from reference (positive = wheel moved up in chassis frame)
        const Real dz_L = p_L_C.y() - z_L_ref;
        const Real dz_R = p_R_C.y() - z_R_ref;
        const Real delta = dz_L - dz_R;

        // Velocity of wheel centers in chassis frame.
        // v_wheel_in_chassis = R_CW * (v_wheel_W - v_contact_point_chassis_W)
        // where v_contact_point = v_chassis + w_chassis x (p_wheel_W - p_chassis_W)
        const Mat3 R_CW = chassis.q_WB.toRotationMatrix().transpose();
        const Vec3 r_L = L_state.p_WB - chassis.p_WB;
        const Vec3 r_R = R_state.p_WB - chassis.p_WB;
        const Vec3 v_L_rel_W = L_state.v_WB - chassis.v_WB - chassis.w_WB.cross(r_L);
        const Vec3 v_R_rel_W = R_state.v_WB - chassis.v_WB - chassis.w_WB.cross(r_R);
        const Real dz_L_dot = (R_CW * v_L_rel_W).y();
        const Real dz_R_dot = (R_CW * v_R_rel_W).y();
        const Real delta_dot = dz_L_dot - dz_R_dot;

        // Force magnitude (opposes delta)
        const Real F_mag = k_arb * delta + c_arb * delta_dot;

        // Apply in chassis Y direction, expressed in world frame
        const Vec3 chassis_Y_W = chassis.q_WB * Vec3::UnitY();

        // Left wheel: force in -chassis_Y direction when delta > 0 (L is up)
        const Vec3 F_L = -F_mag * chassis_Y_W;
        const Vec3 F_R = +F_mag * chassis_Y_W;

        // Apply at wheel body origin (no torque arm, so no moment)
        forces[left_wheel_idx].f_W  += F_L;
        forces[right_wheel_idx].f_W += F_R;

        // Reaction on chassis: opposite forces at the wheel positions,
        // which creates a roll moment about chassis CG.
        forces[chassis_idx].f_W  -= (F_L + F_R);  // Net = 0
        forces[chassis_idx].tau_W -= r_L.cross(F_L) + r_R.cross(F_R);
    }
};

} // namespace mbd