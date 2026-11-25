#include "mbd/forces/tire.hpp"

namespace mbd {

namespace {

/// The road given, or a level one at height 0.
std::shared_ptr<const Road> road_or_level(std::shared_ptr<const Road> road)
{
    return road ? std::move(road) : std::make_shared<FlatRoad>();
}

/// `v` turned by `angle` about the unit axis `n` (Rodrigues).
Vec3 turned_about(const Vec3& v, const Vec3& n, Real angle)
{
    if (angle == 0.0) return v;
    const Real c = std::cos(angle);
    const Real s = std::sin(angle);
    return v * c + n.cross(v) * s + n * (n.dot(v) * (1.0 - c));
}

/// The direction in the wheel plane (normal to `axis`) furthest against the
/// road's normal `n`: the projection of -n on the plane, normalized. A wheel
/// lying on its side, its plane the road's, has none; -n is taken then.
Vec3 towards_road(const Vec3& axis, const Vec3& n)
{
    const Vec3 in_plane = n - n.dot(axis) * axis;
    const Real length = in_plane.norm();
    return length > Real(1e-12) ? Vec3(-in_plane / length) : Vec3(-n);
}

/// The wheel's angular velocity without its spin about its own axis: the
/// motion of the wheel plane, which carries the contact point.
Vec3 without_spin(const RigidBodyState& s, const Vec3& spin_axis)
{
    return s.w_WB - s.w_WB.dot(spin_axis) * spin_axis;
}

} // namespace

WheelContact wheel_contact(const std::shared_ptr<const Road>& road, const Vec3& centre,
                           const Vec3& spin_axis, Real free_radius, Real steer)
{
    MBD_THROW_IF(!road, "MBD-F051: tyre: its road is null; give it a road, or keep the level one it starts with");
    // The direction to the lowest point, from the road's normal under the
    // centre; the road again at that point.
    const Vec3 n_centre = road->at(centre.x(), centre.y()).normal;
    const Vec3 lowest = centre + free_radius * towards_road(turned_about(spin_axis, n_centre, steer), n_centre);
    const RoadPoint r = road->at(lowest.x(), lowest.y());

    WheelContact c;
    c.lowest = lowest;
    c.normal = r.normal;
    c.friction = r.friction;
    c.penetration = (r.height - lowest.z()) * r.normal.z();
    c.point = lowest + c.penetration * r.normal;

    // The heading: the wheel plane's line in the road plane.
    const Vec3 axis = turned_about(spin_axis, r.normal, steer);
    const Vec3 heading = axis.cross(r.normal);
    const Real length = heading.norm();
    c.forward = length > Real(1e-12) ? Vec3(heading / length) : Vec3(Vec3::UnitX());
    c.lateral = r.normal.cross(c.forward);
    return c;
}

TireContactForce::TireContactForce(BodyIndex wheel_idx,
                                   Real free_radius,
                                   Real stiffness,
                                   Real damping,
                                   std::shared_ptr<const Road> on_road)
    : wheel_body_idx(wheel_idx)
    , k_z(stiffness)
    , c_z(damping)
    , R_free(free_radius)
    , road(road_or_level(std::move(on_road)))
{
    MBD_THROW_IF(k_z < 0.0, "MBD-F010: TireContactForce: stiffness must be >= 0");
    MBD_THROW_IF(c_z < 0.0, "MBD-F010: TireContactForce: damping must be >= 0");
    MBD_THROW_IF(R_free <= 0.0, "MBD-F010: TireContactForce: free radius must be > 0");
}

void TireContactForce::apply(const std::vector<RigidBodyState>& states,
                             std::vector<RigidBodyForces>& forces) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];
    const Vec3 axis = s.q_WB * Vec3::UnitY();
    const WheelContact contact = wheel_contact(road, s.p_WB, axis, R_free);
    if (contact.penetration <= Real(0.0)) {
        return; // No contact — tire has lifted off
    }

    // Normal force: spring + damper (the contact point's velocity along the
    // normal is positive moving up), only pushes (clamp to >= 0)
    const Vec3 r_W = contact.point - s.p_WB;
    const Vec3 v_contact = s.v_WB + without_spin(s, axis).cross(r_W);
    Real f_z = k_z * contact.penetration - c_z * contact.normal.dot(v_contact);
    if (f_z < Real(0.0)) {
        f_z = Real(0.0); // Tire cannot pull
    }

    // Apply the normal force at the contact point
    const Vec3 F_W = f_z * contact.normal;

    forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
    forces[static_cast<size_t>(wheel_body_idx)].tau_W += r_W.cross(F_W);
}

Real TireContactForce::get_deflection(const std::vector<RigidBodyState>& states) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];
    return std::max(Real(0.0), wheel_contact(road, s.p_WB, s.q_WB * Vec3::UnitY(), R_free).penetration);
}

Real TireContactForce::get_vertical_force(const std::vector<RigidBodyState>& states) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];
    const Vec3 axis = s.q_WB * Vec3::UnitY();
    const WheelContact contact = wheel_contact(road, s.p_WB, axis, R_free);
    if (contact.penetration <= Real(0.0)) return Real(0.0);
    const Vec3 v_contact = s.v_WB + without_spin(s, axis).cross(contact.point - s.p_WB);
    const Real f_z = k_z * contact.penetration - c_z * contact.normal.dot(v_contact);
    return std::max(Real(0.0), f_z);
}

FullTireForce::FullTireForce(BodyIndex wheel_idx,
                             Real free_radius,
                             Real vert_stiffness,
                             Real vert_damping,
                             const PacejkaTireParams& tire_params,
                             std::shared_ptr<const Road> on_road)
    : wheel_body_idx(wheel_idx)
    , R_free(free_radius)
    , R_loaded_approx(free_radius * 0.97)
    , k_z(vert_stiffness)
    , c_z(vert_damping)
    , road(road_or_level(std::move(on_road)))
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

    // --- The contact, from the wheel plane and the road ---
    const Vec3 axis = s.q_WB * Vec3::UnitY();
    const WheelContact contact = wheel_contact(road, s.p_WB, axis, R_free, steer_angle);
    const Vec3& n = contact.normal;
    const Vec3 r_W = contact.point - s.p_WB;
    last_contact_pos_W = contact.point;
    last_deflection = std::max(Real(0.0), contact.penetration);
    last_friction = contact.friction;
    last_spin = s.w_WB.dot(axis);

    if (contact.penetration <= Real(0.0)) {
        // No contact: clear cached results
        last_Fz = 0.0;
        last_axle_moment = 0.0;
        last_result = TireForceResult{};
        return;
    }

    // The contact point's velocity, carried by the wheel without its spin.
    const Vec3 v_contact_W = s.v_WB + without_spin(s, axis).cross(r_W);

    Real Fz = k_z * contact.penetration - c_z * n.dot(v_contact_W);
    if (Fz < Real(0.0)) Fz = Real(0.0);
    last_Fz = Fz;

    const Vec3& forward_W = contact.forward;
    const Vec3& lateral_W = contact.lateral;
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
    const Real kappa = (last_spin * R_eff - Vx) / V_abs;
    // Slip angle: alpha = -atan2(Vy, V_abs)
    // The sign convention: positive alpha → positive Fy
    // Vy positive means tire moving to the left, which requires
    // a rightward force: negative alpha, negative Fy.
    const Real alpha = -std::atan2(Vy, V_abs);

    // --- Pacejka forces ---
    last_result = pacejka.compute(kappa, alpha, Fz, contact.friction);

    // --- Apply forces in world frame at the contact point ---
    const Vec3 F_W = last_result.Fx * forward_W
                   + last_result.Fy * lateral_W
                   + Fz * n;
    const Vec3 moment_W = r_W.cross(F_W);
    last_axle_moment = moment_W.dot(axis);

    forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
    forces[static_cast<size_t>(wheel_body_idx)].tau_W += moment_W;
}

Real FullTireForce::free_rolling_spin(const std::vector<RigidBodyState>& states) const
{
    const auto& s = states[static_cast<size_t>(wheel_body_idx)];
    const Vec3 axis = s.q_WB * Vec3::UnitY();
    const WheelContact contact = wheel_contact(road, s.p_WB, axis, R_free, steer_angle);
    const Vec3 v_contact_W = s.v_WB + without_spin(s, axis).cross(contact.point - s.p_WB);
    const Real R_eff = R_free - std::max(Real(0.0), contact.penetration) * 0.5;
    return v_contact_W.dot(contact.forward) / R_eff;
}

Real FullTireForce::longitudinal_force_slope() const
{
    if (last_Fz <= Real(0.0) || last_R_eff <= Real(0.0)) {
        return Real(0.0);
    }
    const Real eps = Real(1e-4);
    const Real Fx_plus  = pacejka.compute(last_result.kappa + eps,
                                          last_result.alpha, last_Fz, last_friction).Fx;
    const Real Fx_minus = pacejka.compute(last_result.kappa - eps,
                                          last_result.alpha, last_Fz, last_friction).Fx;
    const Real dFx_dkappa = (Fx_plus - Fx_minus) / (Real(2.0) * eps);
    return std::max(dFx_dkappa, Real(0.0)) * last_R_eff / last_V_abs;
}

} // namespace mbd
