#include "mbd/forces/tire.hpp"

#include <algorithm>
#include <cmath>

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

namespace {

std::shared_ptr<const TyreModel> not_null(std::shared_ptr<const TyreModel> model)
{
    MBD_THROW_IF(!model, "MBD-F011: FullTireForce: the tyre model is null");
    return model;
}

} // namespace

FullTireForce::FullTireForce(BodyIndex wheel_idx,
                             const MagicFormulaParameters& parameters,
                             std::shared_ptr<const Road> on_road)
    : FullTireForce(wheel_idx, std::make_shared<MagicFormulaTyre>(parameters), std::move(on_road))
{
}

FullTireForce::FullTireForce(BodyIndex wheel_idx,
                             std::shared_ptr<const TyreModel> model,
                             std::shared_ptr<const Road> on_road)
    : wheel_body_idx(wheel_idx)
    , tyre(not_null(std::move(model)))
    , R_free(tyre->body().free_radius)
    , k_z(tyre->body().vertical_stiffness)
    , c_z(tyre->body().vertical_damping)
    , road(road_or_level(std::move(on_road)))
    , V_low(tyre->body().low_speed)
{
    MBD_THROW_IF(R_free <= 0.0, "MBD-F011: FullTireForce: free radius must be > 0");
    MBD_THROW_IF(k_z < 0.0, "MBD-F011: FullTireForce: vertical stiffness must be >= 0");
    MBD_THROW_IF(c_z < 0.0, "MBD-F011: FullTireForce: vertical damping must be >= 0");
    MBD_THROW_IF(!(V_low > 0.0), "MBD-F011: FullTireForce: VXLOW must be > 0");
    const TyreOutput nominal = tyre->characteristics(tyre->body().nominal_load, 0.0);
    sigma_kappa_nominal_ = nominal.sigma_kappa;
    sigma_alpha_nominal_ = nominal.sigma_alpha;
    MBD_THROW_IF(!(sigma_kappa_nominal_ > 0.0) || !(sigma_alpha_nominal_ > 0.0),
                 "MBD-F054: FullTireForce: the tyre's relaxation lengths at its nominal load are "
                     + std::to_string(sigma_kappa_nominal_) + " and " + std::to_string(sigma_alpha_nominal_)
                     + " m; the lag needs both positive (MF 6.1: LONGITUDINAL_STIFFNESS and LATERAL_STIFFNESS;"
                       " MF 5.2: PTX1 and PTY1, PTY2)");
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
    last_forward_W = contact.forward;
    last_lateral_W = contact.lateral;

    // The contact point's velocity, carried by the wheel without its spin,
    // and the slip speeds; the rates need them off the road too.
    const Vec3 v_contact_W = s.v_WB + without_spin(s, axis).cross(r_W);
    const Real Vx = v_contact_W.dot(contact.forward);
    const Real Vy = v_contact_W.dot(contact.lateral);
    const Real R_eff = R_free - last_deflection * 0.5;
    last_Vx = Vx;
    last_R_eff = R_eff;
    last_Vsx = Vx - last_spin * R_eff;
    last_Vsy = Vy;

    Real Fz = contact.penetration > 0.0 ? k_z * contact.penetration - c_z * n.dot(v_contact_W) : 0.0;
    if (Fz < Real(0.0)) Fz = Real(0.0);   // the tyre cannot pull
    last_Fz = Fz;
    in_contact_ = Fz > 0.0;
    if (!in_contact_) {
        last_output = TyreOutput{};
        last_kappa = 0.0;
        last_alpha = 0.0;
        last_My = 0.0;
        last_axle_moment = 0.0;
        return;
    }

    // --- The slips: the deflections over the relaxation lengths, with the
    // low-speed damping; or the steady-state slips ---
    last_gamma = std::asin(std::clamp(axis.dot(n), Real(-1.0), Real(1.0)));
    const Real side = mirrored ? -1.0 : 1.0;   // a mirrored tyre: alpha and gamma negated
    const TyreOutput c = tyre->characteristics(Fz, side * last_gamma, -1.0, contact.friction);
    last_sigma_kappa = std::max(c.sigma_kappa, sigma_min);
    last_sigma_alpha = std::max(c.sigma_alpha, sigma_min);
    kappa_sliding_ = c.Kxk > 0.0 ? 3.0 * c.mux * Fz / c.Kxk : 0.0;                          // (8.114)
    alpha_sliding_ = c.Kya != 0.0 ? 3.0 * std::abs(c.muy) * Fz / std::abs(c.Kya) : 0.0;
    Real kappa, tan_alpha;
    if (lag) {
        const Real speed = std::abs(Vx);
        const Real k_low = speed < V_low ? 0.5 * k_V_low0 * (1.0 + std::cos(pi * speed / V_low)) : 0.0;   // (8.128)
        kappa = u_ / last_sigma_kappa - (c.Kxk > 0.0 ? k_low * last_Vsx / c.Kxk : 0.0);              // (8.111), (8.127)
        tan_alpha = v_ / last_sigma_alpha - (c.Kya != 0.0 ? k_low * last_Vsy / c.Kya : 0.0);
    } else {
        const Real speed = std::max(std::abs(Vx), V_low);
        kappa = -last_Vsx / speed;
        tan_alpha = last_Vsy / speed;
    }
    last_kappa = kappa;
    last_alpha = std::atan(tan_alpha);

    // --- The Magic Formula ---
    TyreInput in;
    in.Fz = Fz;
    in.kappa = kappa;
    in.alpha = side * last_alpha;
    in.gamma = side * last_gamma;
    in.Vcx = Vx;
    in.friction = contact.friction;
    last_output = tyre->evaluate(in);
    if (mirrored) {
        for (Real* m : {&last_output.Fy, &last_output.Fy0, &last_output.Mx, &last_output.Mz, &last_output.Mz0,
                        &last_output.Mzr}) {
            *m = -*m;
        }
    }
    const TyreOutput& o = last_output;

    // The rolling resistance against the wheel's rolling: the formula's
    // moment for rolling forwards, scaled by the rolling speed over V_roll
    // below it, so that a wheel at rest has none.
    const Real forwards = Vx < 0.0 ? -o.My : o.My;
    last_My = forwards * std::clamp(last_spin * R_eff / V_roll, Real(-1.0), Real(1.0));

    // --- Forces and moments at the contact point, in the W axes ---
    const Vec3 F_W = o.Fx * contact.forward + o.Fy * contact.lateral + Fz * n;
    const Vec3 moment_W = r_W.cross(F_W) + o.Mx * contact.forward + last_My * contact.lateral + o.Mz * n;
    last_axle_moment = moment_W.dot(axis);

    forces[static_cast<size_t>(wheel_body_idx)].f_W   += F_W;
    forces[static_cast<size_t>(wheel_body_idx)].tau_W += moment_W;
}

void FullTireForce::begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x)
{
    (void)sim;
    (void)t;
    if (x.size() == 2) {
        u_ = x(0);
        v_ = x(1);
    }
}

void FullTireForce::rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot)
{
    (void)sim;
    (void)t;
    if (x.size() != 2) return;
    const Real u = x(0);
    const Real v = x(1);
    const Real speed = std::abs(last_Vx);
    if (!in_contact_) {
        // Off the road the tread's deflections relax over their lengths.
        x_dot(0) = -(speed + V_low) * u / sigma_kappa_nominal_;
        x_dot(1) = -(speed + V_low) * v / sigma_alpha_nominal_;
        return;
    }
    Real du = -last_Vsx - speed * u / last_sigma_kappa;                       // (8.110)
    Real dv = last_Vsy - speed * v / last_sigma_alpha;
    // (8.112): at low speed, a deflection past the slip of full sliding is
    // not let grow.
    if (speed < V_low) {
        if (std::abs(u / last_sigma_kappa) > limit_factor * kappa_sliding_ && du * u > 0.0) du = 0.0;
        if (std::abs(v / last_sigma_alpha) > limit_factor * alpha_sliding_ && dv * v > 0.0) dv = 0.0;
    }
    x_dot(0) = du;
    x_dot(1) = dv;
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

} // namespace mbd
