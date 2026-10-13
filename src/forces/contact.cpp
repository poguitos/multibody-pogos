#include "mbd/forces/contact.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace mbd {

namespace {

/// The cubic step from (x0, h0) to (x1, h1), flat at both ends, with its
/// derivative: h0 before x0, h1 after x1.
std::pair<Real, Real> step(Real x, Real x0, Real h0, Real x1, Real h1)
{
    if (x <= x0) return {h0, 0.0};
    if (x >= x1) return {h1, 0.0};
    const Real s = (x - x0) / (x1 - x0);
    return {h0 + (h1 - h0) * s * s * (3.0 - 2.0 * s), (h1 - h0) * 6.0 * s * (1.0 - s) / (x1 - x0)};
}

} // namespace

PlaneContact::PlaneContact(BodyIndex body, std::vector<Vec3> centres_B, Real radius,
                           BodyIndex plane_body, const Transform3& plane_X, ContactParams params)
    : body_(body)
    , plane_body_(plane_body)
    , centres_B_(std::move(centres_B))
    , radius_(radius)
    , plane_X_(plane_X)
    , params_(params)
{
    const ContactParams& p = params_;
    MBD_THROW_IF(!(radius_ >= 0.0) || !(p.stiffness >= 0.0) || !(p.exponent >= 1.0) || !(p.damping >= 0.0)
                     || !(p.damping_depth > 0.0) || !(p.mu_static >= 0.0) || !(p.mu_dynamic >= 0.0)
                     || !(p.v_static > 0.0) || !(p.v_dynamic > p.v_static),
                 "MBD-F040: PlaneContact: needs radius, stiffness, damping and friction coefficients >= 0, "
                 "exponent >= 1, damping depth > 0 and 0 < v_static < v_dynamic");
    MBD_THROW_IF(body_ == plane_body_, "MBD-F041: PlaneContact: the spheres and the plane are on the same body");
}

LawValue PlaneContact::normal_law(Real delta, Real delta_rate) const
{
    LawValue out;
    if (delta <= 0.0) return out;
    const ContactParams& p = params_;
    const auto [c, dc] = step(delta, 0.0, 0.0, p.damping_depth, p.damping);
    const Real power = std::pow(delta, p.exponent);
    out.force = p.stiffness * power + c * delta_rate;
    if (out.force <= 0.0) return LawValue{};   // pushes, never pulls
    out.d_position = p.stiffness * p.exponent * power / delta + dc * delta_rate;
    out.d_rate = c;
    return out;
}

LawValue PlaneContact::friction(Real speed) const
{
    const ContactParams& p = params_;
    const auto [mu, dmu] = speed <= p.v_static ? step(speed, 0.0, 0.0, p.v_static, p.mu_static)
                                                : step(speed, p.v_static, p.mu_static, p.v_dynamic, p.mu_dynamic);
    LawValue out;
    out.force = mu;
    out.d_rate = dmu;
    return out;
}

std::vector<Real> PlaneContact::penetrations(const std::vector<RigidBodyState>& states) const
{
    const auto& sA = states[static_cast<std::size_t>(body_)];
    const auto& sB = states[static_cast<std::size_t>(plane_body_)];
    const Vec3 p_W = sB.p_WB + sB.q_WB * plane_X_.p;
    const Vec3 n_W = sB.q_WB * (plane_X_.q * Vec3::UnitZ());
    std::vector<Real> out;
    out.reserve(centres_B_.size());
    for (const Vec3& c_B : centres_B_) out.push_back(radius_ - n_W.dot(sA.p_WB + sA.q_WB * c_B - p_W));
    return out;
}

void PlaneContact::apply(const std::vector<RigidBodyState>& states,
                         std::vector<RigidBodyForces>& forces) const
{
    const auto& sA = states[static_cast<std::size_t>(body_)];
    const auto& sB = states[static_cast<std::size_t>(plane_body_)];
    const Vec3 p_W = sB.p_WB + sB.q_WB * plane_X_.p;
    const Vec3 n_W = sB.q_WB * (plane_X_.q * Vec3::UnitZ());
    auto& fA = forces[static_cast<std::size_t>(body_)];
    auto& fB = forces[static_cast<std::size_t>(plane_body_)];

    for (const Vec3& c_B : centres_B_) {
        const Vec3 c_W = sA.p_WB + sA.q_WB * c_B;
        const Real delta = radius_ - n_W.dot(c_W - p_W);
        if (delta <= 0.0) continue;

        // The contact point, and its velocity on each body. The normal
        // component of the relative velocity is exactly -d(delta)/dt, the
        // plane's turning included.
        const Vec3 x_W = c_W - radius_ * n_W;
        const Vec3 vA = sA.v_WB + sA.w_WB.cross(x_W - sA.p_WB);
        const Vec3 vB = sB.v_WB + sB.w_WB.cross(x_W - sB.p_WB);
        const Vec3 v_rel = vA - vB;
        const Real normal_speed = n_W.dot(v_rel);
        const Real Fn = normal_law(delta, -normal_speed).force;
        if (Fn <= 0.0) continue;

        Vec3 F = Fn * n_W;
        const Vec3 v_t = v_rel - normal_speed * n_W;
        const Real speed = v_t.norm();
        if (speed > 0.0) F -= (friction(speed).force * Fn / speed) * v_t;

        fA.f_W += F;
        fA.tau_W += (x_W - sA.p_WB).cross(F);
        fB.f_W -= F;
        fB.tau_W -= (x_W - sB.p_WB).cross(F);
    }
}

} // namespace mbd
