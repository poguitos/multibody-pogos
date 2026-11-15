#include "mbd/forces/tire.hpp"

namespace mbd {

TireContactForce::TireContactForce(BodyIndex wheel_idx,
                                   Real free_radius,
                                   Real stiffness,
                                   Real damping,
                                   Real ground_height)
    : wheel_body_idx(wheel_idx)
    , contact_point_B(0.0, 0.0, -free_radius)
    , k_z(stiffness)
    , c_z(damping)
    , R_free(free_radius)
    , road_height(ground_height)
{
    MBD_THROW_IF(k_z < 0.0, "MBD-F010: TireContactForce: stiffness must be >= 0");
    MBD_THROW_IF(c_z < 0.0, "MBD-F010: TireContactForce: damping must be >= 0");
    MBD_THROW_IF(R_free <= 0.0, "MBD-F010: TireContactForce: free radius must be > 0");
}

void TireContactForce::apply(const std::vector<RigidBodyState>& states,
                             std::vector<RigidBodyForces>& forces) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];

    // Contact point position and velocity in world frame
    const Vec3 r_W = s.q_WB * contact_point_B;
    const Vec3 p_contact_W = s.p_WB + r_W;
    const Vec3 v_contact_W = s.v_WB + s.w_WB.cross(r_W);

    // Penetration: positive when contact point is below ground
    const Real penetration = road_height - ground_normal_W.dot(p_contact_W);

    if (penetration <= Real(0.0)) {
        return; // No contact — tire has lifted off
    }

    // Velocity of contact point along the normal (positive = moving up)
    const Real v_z = ground_normal_W.dot(v_contact_W);

    // Normal force: spring + damper, only pushes (clamp to >= 0)
    Real f_z = k_z * penetration - c_z * v_z;
    if (f_z < Real(0.0)) {
        f_z = Real(0.0); // Tire cannot pull
    }

    // Apply the normal force at the contact point
    const Vec3 F_W = f_z * ground_normal_W;

    forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
    forces[static_cast<size_t>(wheel_body_idx)].tau_W += r_W.cross(F_W);
}

Real TireContactForce::get_deflection(const std::vector<RigidBodyState>& states) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];
    const Vec3 r_W = s.q_WB * contact_point_B;
    const Vec3 p_contact_W = s.p_WB + r_W;
    const Real penetration = road_height - ground_normal_W.dot(p_contact_W);
    return std::max(Real(0.0), penetration);
}

Real TireContactForce::get_vertical_force(const std::vector<RigidBodyState>& states) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];
    const Vec3 r_W = s.q_WB * contact_point_B;
    const Vec3 p_contact_W = s.p_WB + r_W;
    const Vec3 v_contact_W = s.v_WB + s.w_WB.cross(r_W);

    const Real penetration = road_height - ground_normal_W.dot(p_contact_W);
    if (penetration <= Real(0.0)) return Real(0.0);

    Real f_z = k_z * penetration - c_z * ground_normal_W.dot(v_contact_W);
    return std::max(Real(0.0), f_z);
}

FullTireForce::FullTireForce(BodyIndex wheel_idx,
                             Real free_radius,
                             Real vert_stiffness,
                             Real vert_damping,
                             const PacejkaTireParams& tire_params,
                             Real ground_height)
    : wheel_body_idx(wheel_idx)
    , R_free(free_radius)
    , R_loaded_approx(free_radius * 0.97)
    , k_z(vert_stiffness)
    , c_z(vert_damping)
    , road_height(ground_height)
    , pacejka(tire_params)
{
    MBD_THROW_IF(R_free <= 0.0, "MBD-F011: FullTireForce: free radius must be > 0");
    MBD_THROW_IF(k_z < 0.0, "MBD-F011: FullTireForce: vertical stiffness must be >= 0");
    MBD_THROW_IF(c_z < 0.0, "MBD-F011: FullTireForce: vertical damping must be >= 0");
}

void FullTireForce::apply(const std::vector<RigidBodyState>& states,
                          std::vector<RigidBodyForces>& forces) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];

    // --- Contact point kinematics ---
    const Vec3 contact_B(0.0, 0.0, -R_free);
    const Vec3& n = ground_normal_W;
    const Vec3 r_W = s.q_WB * contact_B;
    const Vec3 p_contact_W = s.p_WB + r_W;
    const Vec3 v_contact_W = s.v_WB + s.w_WB.cross(r_W);

    last_contact_pos_W = p_contact_W;

    // --- Contact along the ground's normal ---
    const Real penetration = road_height - n.dot(p_contact_W);
    last_deflection = std::max(Real(0.0), penetration);

    if (penetration <= Real(0.0)) {
        // No contact: clear cached results
        last_Fz = 0.0;
        last_result = TireForceResult{};
        return;
    }

    Real Fz = k_z * penetration - c_z * n.dot(v_contact_W);
    if (Fz < Real(0.0)) Fz = Real(0.0);
    last_Fz = Fz;

    // --- Tire frame directions in world ---
    // Forward = steered body X projected onto ground plane and normalized.
    // Rz(steer_angle) turns body X to the left for a positive steer_angle.
    const Quat q_steer(Eigen::AngleAxisd(steer_angle, Vec3::UnitZ()));
    const Vec3 steered_fwd_B = q_steer * Vec3::UnitX();
    Vec3 body_x_W = s.q_WB * steered_fwd_B;
    body_x_W -= n.dot(body_x_W) * n;  // Project onto ground plane
    const Real fwd_len = body_x_W.norm();

    Vec3 forward_W, lateral_W;
    if (fwd_len > Real(1e-6)) {
        forward_W = body_x_W / fwd_len;
    } else {
        forward_W = Vec3::UnitX(); // Fallback
    }

    // Lateral = up x forward, to the left
    lateral_W = n.cross(forward_W);
    const Real lat_len = lateral_W.norm();
    if (lat_len > Real(1e-6)) {
        lateral_W /= lat_len;
    } else {
        lateral_W = Vec3::UnitY();
    }

    last_forward_W = forward_W;
    last_lateral_W = lateral_W;

    // --- Slip computation ---
    const Real Vx = v_contact_W.dot(forward_W);
    const Real Vy = v_contact_W.dot(lateral_W);

    const Real V_abs = std::max(std::abs(Vx), V_low);

    // Effective rolling radius
    const Real R_eff = R_free - last_deflection * 0.5;

    last_V_abs = V_abs;
    last_R_eff = R_eff;

    // Slip ratio: kappa = (omega*R_eff - Vx) / V_abs
    const Real actual_omega = auto_free_roll ? (Vx / R_eff) : omega_wheel;
    const Real kappa = (actual_omega * R_eff - Vx) / V_abs;
    // Slip angle: alpha = -atan(Vy / V_abs)
    // The sign convention: positive alpha → positive Fy
    // Vy positive means tire moving to the left, which requires
    // a rightward force: negative alpha, negative Fy.
    const Real alpha = -std::atan2(Vy, V_abs);

    // --- Pacejka forces ---
    last_result = pacejka.compute(kappa, alpha, Fz);

    // --- Apply forces in world frame at contact patch ---
    const Vec3 F_W = last_result.Fx * forward_W
                   + last_result.Fy * lateral_W
                   + Fz * n;

    // Force applied at wheel body
    forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
    forces[static_cast<size_t>(wheel_body_idx)].tau_W += r_W.cross(F_W);
}

Real FullTireForce::longitudinal_force_slope() const
{
    if (last_Fz <= Real(0.0) || last_R_eff <= Real(0.0)) {
        return Real(0.0);
    }
    const Real eps = Real(1e-4);
    const Real Fx_plus  = pacejka.compute(last_result.kappa + eps,
                                          last_result.alpha, last_Fz).Fx;
    const Real Fx_minus = pacejka.compute(last_result.kappa - eps,
                                          last_result.alpha, last_Fz).Fx;
    const Real dFx_dkappa = (Fx_plus - Fx_minus) / (Real(2.0) * eps);
    return std::max(dFx_dkappa, Real(0.0)) * last_R_eff / last_V_abs;
}

} // namespace mbd
