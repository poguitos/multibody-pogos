#pragma once

// Tire vertical contact force model.
//
// Monitors a contact point on a wheel body. When the contact point
// penetrates the ground plane, applies a vertical spring-damper force.
// The tire only pushes (lifts off when no contact).
//
// Ground plane: y = road_height (default 0).
//
// This is L2.3.1: the simplest tire model — vertical forces only,
// no slip, no lateral dynamics. Suitable for quarter-car validation,
// ride comfort analysis, and as the foundation for more advanced models.

#include "mbd/core.hpp"
#include "mbd/force_element.hpp"
#include "mbd/pacejka.hpp"

namespace mbd {

class TireContactForce : public ForceElement {
public:
    BodyIndex wheel_body_idx;

    /// Contact point in the wheel body frame.
    /// For a wheel whose origin is at the axle center and Y is up,
    /// this is typically (0, -R_free, 0).
    Vec3 contact_point_B;

    Real k_z;        ///< Vertical stiffness [N/m]
    Real c_z;        ///< Vertical damping [N·s/m]
    Real R_free;     ///< Unloaded tire radius [m]
    Real road_height; ///< Ground plane height [m]

    TireContactForce(BodyIndex wheel_idx,
                     Real free_radius,
                     Real stiffness,
                     Real damping,
                     Real ground_y = 0.0)
        : wheel_body_idx(wheel_idx)
        , contact_point_B(0.0, -free_radius, 0.0)
        , k_z(stiffness)
        , c_z(damping)
        , R_free(free_radius)
        , road_height(ground_y)
    {
        MBD_THROW_IF(k_z < 0.0, "TireContactForce: stiffness must be >= 0");
        MBD_THROW_IF(c_z < 0.0, "TireContactForce: damping must be >= 0");
        MBD_THROW_IF(R_free <= 0.0, "TireContactForce: free radius must be > 0");
    }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override
    {
        const auto& s = states[static_cast<size_t>(wheel_body_idx)];

        // Contact point position and velocity in world frame
        const Vec3 r_W = s.q_WB * contact_point_B;
        const Vec3 p_contact_W = s.p_WB + r_W;
        const Vec3 v_contact_W = s.v_WB + s.w_WB.cross(r_W);

        // Penetration: positive when contact point is below ground
        const Real penetration = road_height - p_contact_W.y();

        if (penetration <= Real(0.0)) {
            return; // No contact — tire has lifted off
        }

        // Vertical velocity of contact point (positive = moving up)
        const Real v_z = v_contact_W.y();

        // Normal force: spring + damper, only pushes (clamp to >= 0)
        Real f_z = k_z * penetration - c_z * v_z;
        if (f_z < Real(0.0)) {
            f_z = Real(0.0); // Tire cannot pull
        }

        // Apply vertical force at the contact point
        const Vec3 F_W(0.0, f_z, 0.0);

        forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
        forces[static_cast<size_t>(wheel_body_idx)].tau_W += r_W.cross(F_W);
    }

    /// Current tire deflection (positive = compressed). Returns 0 if no contact.
    Real get_deflection(const std::vector<RigidBodyState>& states) const
    {
        const auto& s = states[static_cast<size_t>(wheel_body_idx)];
        const Vec3 r_W = s.q_WB * contact_point_B;
        const Vec3 p_contact_W = s.p_WB + r_W;
        const Real penetration = road_height - p_contact_W.y();
        return std::max(Real(0.0), penetration);
    }

    /// Current vertical force. Returns 0 if no contact.
    Real get_vertical_force(const std::vector<RigidBodyState>& states) const
    {
        const auto& s = states[static_cast<size_t>(wheel_body_idx)];
        const Vec3 r_W = s.q_WB * contact_point_B;
        const Vec3 p_contact_W = s.p_WB + r_W;
        const Vec3 v_contact_W = s.v_WB + s.w_WB.cross(r_W);

        const Real penetration = road_height - p_contact_W.y();
        if (penetration <= Real(0.0)) return Real(0.0);

        Real f_z = k_z * penetration - c_z * v_contact_W.y();
        return std::max(Real(0.0), f_z);
    }
};

// ============================================================================
// Full tire force element: vertical contact + Pacejka slip forces
// ============================================================================
//
// Combines vertical spring-damper contact with Pacejka Magic Formula
// lateral and longitudinal forces computed from slip kinematics.
//
// Wheel body frame convention:
//   X: forward (travel direction)
//   Y: up (axle direction, left side)
//   Z: lateral (to the right)
//   Contact point at (0, -R_free, 0) in body frame.

class FullTireForce : public ForceElement {
public:
    BodyIndex wheel_body_idx;

    // Tire geometry
    Real R_free;          ///< Unloaded radius [m]
    Real R_loaded_approx; ///< Approximate loaded radius for effective radius calc

    // Vertical contact
    Real k_z;             ///< Vertical stiffness [N/m]
    Real c_z;             ///< Vertical damping [Ns/m]
    Real road_height;     ///< Ground plane Y coordinate [m]

    // Pacejka model
    PacejkaTire pacejka;

    // Low-speed threshold to avoid singularity in slip computation
    Real V_low{1.0};     ///< [m/s] — below this speed, slip is clamped

    // Wheel spin rate (set externally, e.g. by drivetrain model)
    // For free-rolling, set to Vx / R_eff after each step.
    Real omega_wheel{0.0};

    /// When true, omega_wheel is computed automatically from forward velocity
    /// (free-rolling assumption: zero drive/brake torque). When false, the
    /// externally-set omega_wheel value is used.
    bool auto_free_roll{true};
    /// Steering angle applied to this tire [rad].
    /// Positive = wheel heading rotates toward +Z (left in vehicle frame).
    Real steer_angle{0.0};

    // --- Cached output (updated each apply() call) ---
    mutable TireForceResult last_result;
    mutable Real last_Fz{0.0};
    mutable Real last_deflection{0.0};
    mutable Vec3 last_contact_pos_W{Vec3::Zero()};
    mutable Vec3 last_forward_W{Vec3::UnitX()};
    mutable Vec3 last_lateral_W{Vec3::UnitZ()};

    FullTireForce(BodyIndex wheel_idx,
                  Real free_radius,
                  Real vert_stiffness,
                  Real vert_damping,
                  const PacejkaTireParams& tire_params,
                  Real ground_y = 0.0)
        : wheel_body_idx(wheel_idx)
        , R_free(free_radius)
        , R_loaded_approx(free_radius * 0.97)
        , k_z(vert_stiffness)
        , c_z(vert_damping)
        , road_height(ground_y)
        , pacejka(tire_params)
    {
        MBD_THROW_IF(R_free <= 0.0, "FullTireForce: free radius must be > 0");
        MBD_THROW_IF(k_z < 0.0, "FullTireForce: vertical stiffness must be >= 0");
        MBD_THROW_IF(c_z < 0.0, "FullTireForce: vertical damping must be >= 0");
    }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override
    {
        const auto& s = states[static_cast<size_t>(wheel_body_idx)];

        // --- Contact point kinematics ---
        const Vec3 contact_B(0.0, -R_free, 0.0);
        const Vec3 r_W = s.q_WB * contact_B;
        const Vec3 p_contact_W = s.p_WB + r_W;
        const Vec3 v_contact_W = s.v_WB + s.w_WB.cross(r_W);

        last_contact_pos_W = p_contact_W;

        // --- Vertical contact ---
        const Real penetration = road_height - p_contact_W.y();
        last_deflection = std::max(Real(0.0), penetration);

        if (penetration <= Real(0.0)) {
            // No contact: clear cached results
            last_Fz = 0.0;
            last_result = TireForceResult{};
            return;
        }

        Real Fz = k_z * penetration - c_z * v_contact_W.y();
        if (Fz < Real(0.0)) Fz = Real(0.0);
        last_Fz = Fz;

        // --- Tire frame directions in world ---
        // Forward = steered body X projected onto ground plane and normalized.
        // Ry(-steer_angle) rotates body X toward +Z for positive steer_angle (left).
        const Quat q_steer(Eigen::AngleAxisd(-steer_angle, Vec3::UnitY()));
        const Vec3 steered_fwd_B = q_steer * Vec3::UnitX();
        Vec3 body_x_W = s.q_WB * steered_fwd_B;
        body_x_W.y() = 0.0;  // Project onto ground plane
        const Real fwd_len = body_x_W.norm();

        Vec3 forward_W, lateral_W;
        if (fwd_len > Real(1e-6)) {
            forward_W = body_x_W / fwd_len;
        } else {
            forward_W = Vec3::UnitX(); // Fallback
        }

        // Lateral = up x forward (points to the right in SAE convention)
        lateral_W = Vec3::UnitY().cross(forward_W);
        const Real lat_len = lateral_W.norm();
        if (lat_len > Real(1e-6)) {
            lateral_W /= lat_len;
        } else {
            lateral_W = Vec3::UnitZ();
        }

        last_forward_W = forward_W;
        last_lateral_W = lateral_W;

        // --- Slip computation ---
        const Real Vx = v_contact_W.dot(forward_W);
        const Real Vy = v_contact_W.dot(lateral_W);

        const Real V_abs = std::max(std::abs(Vx), V_low);

        // Effective rolling radius
        const Real R_eff = R_free - last_deflection * 0.5;

        // Slip ratio: kappa = (omega*R_eff - Vx) / V_abs
        const Real actual_omega = auto_free_roll ? (Vx / R_eff) : omega_wheel;
        const Real kappa = (actual_omega * R_eff - Vx) / V_abs;
        // Slip angle: alpha = -atan(Vy / V_abs)
        // The sign convention: positive alpha → positive Fy
        // Vy positive means tire moving to the right, which requires
        // a leftward force (positive Fy in SAE).
        const Real alpha = -std::atan2(Vy, V_abs);

        // --- Pacejka forces ---
        last_result = pacejka.compute(kappa, alpha, Fz);

        // --- Apply forces in world frame at contact patch ---
        const Vec3 F_W = last_result.Fx * forward_W
                       + last_result.Fy * lateral_W
                       + Fz * Vec3::UnitY();

        // Force applied at wheel body
        forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
        forces[static_cast<size_t>(wheel_body_idx)].tau_W += r_W.cross(F_W);
    }

    // --- Telemetry accessors ---

    Real get_vertical_force() const { return last_Fz; }
    Real get_deflection() const { return last_deflection; }
    Real get_slip_ratio() const { return last_result.kappa; }
    Real get_slip_angle() const { return last_result.alpha; }
    Real get_Fx() const { return last_result.Fx; }
    Real get_Fy() const { return last_result.Fy; }
    const TireForceResult& get_last_result() const { return last_result; }
};

} // namespace mbd