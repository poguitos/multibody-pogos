#include "mbd/forces/plane_contact.hpp"

#include <cmath>
#include <utility>

namespace mbd {

namespace {

// The damper's ramp from first touch to full damping: 3x^2 - 2x^3 on [0, 1],
// with its derivative; continuous in value and slope at both ends.
Real ramp(Real x, Real& slope)
{
    if (x >= 1.0) {
        slope = 0.0;
        return 1.0;
    }
    if (x <= 0.0) {
        slope = 0.0;
        return 0.0;
    }
    slope = 6.0 * x * (1.0 - x);
    return x * x * (3.0 - 2.0 * x);
}

// tanh(x) / x, continued to 1 at x = 0.
Real tanh_over_x(Real x)
{
    if (x < 1e-6) return 1.0 - x * x / 3.0;
    return std::tanh(x) / x;
}

} // namespace

PlaneContact::PlaneContact(BodyIndex body, std::vector<ContactSphere> spheres,
                           PlaneContactParams params, BodyIndex plane_body,
                           const Transform3& plane_frame)
    : body_(body)
    , spheres_(std::move(spheres))
    , params_(params)
    , plane_body_(plane_body)
    , plane_frame_(plane_frame)
{
    const PlaneContactParams& p = params_;
    MBD_THROW_IF(!(p.stiffness >= 0.0) || !(p.exponent >= 1.0) || !(p.damping >= 0.0)
                     || !(p.damping_depth > 0.0) || !(p.friction >= 0.0) || !(p.slip_speed > 0.0),
                 "MBD-F040: PlaneContact: stiffness, damping and friction must be >= 0, the exponent "
                 ">= 1, and the damping depth and slip speed > 0");
    MBD_THROW_IF(spheres_.empty(), "MBD-F041: PlaneContact: no contact points given");
    for (const auto& s : spheres_) {
        MBD_THROW_IF(!(s.radius >= 0.0) || !s.center_B.allFinite(),
                     "MBD-F041: PlaneContact: a contact point is not finite, or its radius is negative");
    }
}

LawValue PlaneContact::normal_law(Real depth, Real depth_rate) const
{
    LawValue out;
    if (depth <= 0.0) return out;
    const PlaneContactParams& p = params_;
    Real slope = 0.0;
    const Real s = ramp(depth / p.damping_depth, slope);
    const Real spring = p.stiffness * std::pow(depth, p.exponent);
    out.force = spring + p.damping * s * depth_rate;
    if (out.force <= 0.0) {
        // Separating fast: the damper would pull, and the plane never does.
        out.force = 0.0;
        return out;
    }
    out.d_position = p.stiffness * p.exponent * std::pow(depth, p.exponent - 1.0)
                     + p.damping * slope / p.damping_depth * depth_rate;
    out.d_rate = p.damping * s;
    return out;
}

LawValue PlaneContact::friction_law(Real slip, Real normal_force) const
{
    LawValue out;
    const PlaneContactParams& p = params_;
    const Real th = std::tanh(slip / p.slip_speed);
    out.force = p.friction * normal_force * th;
    out.d_rate = p.friction * normal_force * (1.0 - th * th) / p.slip_speed;
    return out;
}

Real PlaneContact::potential_energy(Real depth) const
{
    if (depth <= 0.0) return 0.0;
    return params_.stiffness * std::pow(depth, params_.exponent + 1.0) / (params_.exponent + 1.0);
}

ContactState PlaneContact::contact(const std::vector<RigidBodyState>& states, std::size_t i) const
{
    const RigidBodyState& b = states[static_cast<std::size_t>(body_)];
    const RigidBodyState& pl = states[static_cast<std::size_t>(plane_body_)];
    const ContactSphere& sphere = spheres_[i];

    ContactState out;
    const Vec3 n = pl.q_WB * (plane_frame_.q * Vec3::UnitZ());
    const Vec3 origin = pl.p_WB + pl.q_WB * plane_frame_.p;
    const Vec3 center = b.p_WB + b.q_WB * sphere.center_B;
    out.normal_W = n;
    out.depth = sphere.radius - (center - origin).dot(n);
    out.point_W = center - sphere.radius * n;
    if (out.depth <= 0.0) return out;

    // The velocities of the two material points at the contact point. Their
    // relative normal component is the rate of the height, the plane's
    // turning included.
    const Vec3 v_body = b.v_WB + b.w_WB.cross(out.point_W - b.p_WB);
    const Vec3 v_plane = pl.v_WB + pl.w_WB.cross(out.point_W - pl.p_WB);
    const Vec3 v_rel = v_body - v_plane;
    out.depth_rate = -v_rel.dot(n);
    out.normal_force = normal_law(out.depth, out.depth_rate).force;
    out.touching = out.normal_force > 0.0;

    const Vec3 u = v_rel + out.depth_rate * n;   // tangential part
    out.slip = u.norm();
    // F_t = -mu F_n tanh(|u| / v_s) u / |u|, written to stay smooth at u = 0.
    const Real vs = params_.slip_speed;
    out.friction_W = -(params_.friction * out.normal_force * tanh_over_x(out.slip / vs) / vs) * u;
    return out;
}

void PlaneContact::apply(const std::vector<RigidBodyState>& states,
                         std::vector<RigidBodyForces>& forces) const
{
    const Vec3 p_body = states[static_cast<std::size_t>(body_)].p_WB;
    const Vec3 p_plane = states[static_cast<std::size_t>(plane_body_)].p_WB;
    RigidBodyForces& fb = forces[static_cast<std::size_t>(body_)];
    RigidBodyForces& fp = forces[static_cast<std::size_t>(plane_body_)];
    for (std::size_t i = 0; i < spheres_.size(); ++i) {
        const ContactState c = contact(states, i);
        if (!c.touching) continue;
        const Vec3 F = c.normal_force * c.normal_W + c.friction_W;
        fb.f_W += F;
        fb.tau_W += (c.point_W - p_body).cross(F);
        fp.f_W -= F;
        fp.tau_W -= (c.point_W - p_plane).cross(F);
    }
}

} // namespace mbd
