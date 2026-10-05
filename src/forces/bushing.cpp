#include "mbd/forces/bushing.hpp"

#include <utility>

#include "mbd/spatial/spatial.hpp"

namespace mbd {

Bushing::Bushing(BodyIndex b1, const Transform3& X_1M, BodyIndex b2, const Transform3& X_2M,
                 BushingParams params)
    : body1_(b1), body2_(b2), X_1M_(X_1M), X_2M_(X_2M), params_(std::move(params))
{}

void Bushing::deflection(const std::vector<RigidBodyState>& states, Vec6& x, Vec6& x_dot) const
{
    const auto& s1 = states[static_cast<std::size_t>(body1_)];
    const auto& s2 = states[static_cast<std::size_t>(body2_)];
    const Quat q1 = s1.q_WB * X_1M_.q;
    const Quat q2 = s2.q_WB * X_2M_.q;
    const Vec3 r1 = s1.q_WB * X_1M_.p;   // marker origins from the body origins
    const Vec3 r2 = s2.q_WB * X_2M_.p;
    const Vec3 d_W = (s2.p_WB + r2) - (s1.p_WB + r1);
    const Quat R1t = q1.conjugate();

    x.head<3>() = log3(R1t * q2);
    x.tail<3>() = R1t * d_W;
    // The displacement's rate as seen in marker 1, which turns with body 1.
    const Vec3 dd_W = (s2.v_WB + s2.w_WB.cross(r2)) - (s1.v_WB + s1.w_WB.cross(r1));
    x_dot.head<3>() = R1t * (s2.w_WB - s1.w_WB);
    x_dot.tail<3>() = R1t * (dd_W - s1.w_WB.cross(d_W));
}

BushingLaw Bushing::law(const Vec6& x, const Vec6& x_dot) const
{
    BushingLaw out;
    for (int i = 0; i < 3; ++i) {
        const Curve& kr = params_.rotational_stiffness[static_cast<std::size_t>(i)];
        const Curve& k = params_.stiffness[static_cast<std::size_t>(i)];
        const Curve& cr = params_.rotational_damping[static_cast<std::size_t>(i)];
        const Curve& c = params_.damping[static_cast<std::size_t>(i)];
        out.load(i) = -kr.value(x(i)) - cr.value(x_dot(i));
        out.load(3 + i) = -k.value(x(3 + i)) - c.value(x_dot(3 + i));
        out.d_position(i, i) = -kr.slope(x(i));
        out.d_position(3 + i, 3 + i) = -k.slope(x(3 + i));
        out.d_rate(i, i) = -cr.slope(x_dot(i));
        out.d_rate(3 + i, 3 + i) = -c.slope(x_dot(3 + i));
    }
    return out;
}

void Bushing::apply(const std::vector<RigidBodyState>& states,
                    std::vector<RigidBodyForces>& forces) const
{
    Vec6 x, x_dot;
    deflection(states, x, x_dot);
    const Vec6 load = law(x, x_dot).load;

    const auto& s1 = states[static_cast<std::size_t>(body1_)];
    const auto& s2 = states[static_cast<std::size_t>(body2_)];
    const Quat q1 = s1.q_WB * X_1M_.q;
    const Vec3 M_W = q1 * load.head<3>();
    const Vec3 F_W = q1 * load.tail<3>();
    // The force acts at marker 2's origin, on body 2 and, reversed, on body 1.
    const Vec3 p2 = s2.p_WB + s2.q_WB * X_2M_.p;
    auto& f2 = forces[static_cast<std::size_t>(body2_)];
    auto& f1 = forces[static_cast<std::size_t>(body1_)];
    f2.f_W += F_W;
    f2.tau_W += (p2 - s2.p_WB).cross(F_W) + M_W;
    f1.f_W -= F_W;
    f1.tau_W -= (p2 - s1.p_WB).cross(F_W) + M_W;
}

} // namespace mbd
