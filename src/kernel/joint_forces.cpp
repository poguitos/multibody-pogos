#include "mbd/kernel/joint_forces.hpp"

#include <cmath>
#include <utility>

namespace mbd::kernel {

JointCoordinateForce::JointCoordinateForce(const Model& model, int body,
                                           JointCoordinateForceParams params)
    : body_(body), iq_(0), iv_(0), params_(std::move(params))
{
    MBD_THROW_IF(body < 1 || body >= model.nbodies(),
                 "MBD-K026: kernel::JointCoordinateForce: no such body");
    MBD_THROW_IF(model.nqs[body] != 1 || model.nvs[body] != 1,
                 "MBD-K026: kernel::JointCoordinateForce: the joint must have one coordinate "
                 "(revolute or prismatic)");
    const JointCoordinateForceParams& p = params_;
    MBD_THROW_IF(!(p.friction >= 0.0) || !(p.friction_velocity > 0.0) || !(p.limit_stiffness >= 0.0)
                     || !(p.limit_damping >= 0.0) || !(p.lower_limit <= p.upper_limit),
                 "MBD-K027: kernel::JointCoordinateForce: friction, stop stiffness and damping must "
                 "be >= 0, the friction velocity > 0, and the lower limit not above the upper");
    iq_ = model.idx_q[body];
    iv_ = model.idx_v[body];
}

LawValue JointCoordinateForce::law(Real q, Real v) const
{
    const JointCoordinateForceParams& p = params_;
    LawValue out;
    out.force = p.preload - p.spring.value(q - p.reference) - p.damper.value(v);
    out.d_position = -p.spring.slope(q - p.reference);
    out.d_rate = -p.damper.slope(v);

    if (p.friction > 0.0) {
        const Real th = std::tanh(v / p.friction_velocity);
        out.force -= p.friction * th;
        out.d_rate -= p.friction / p.friction_velocity * (1.0 - th * th);
    }

    // A stop pushes back with k * penetration + c * (rate of penetration),
    // but never pulls the joint into it.
    if (q > p.upper_limit) {
        const Real push = p.limit_stiffness * (q - p.upper_limit) + p.limit_damping * v;
        if (push > 0.0) {
            out.force -= push;
            out.d_position -= p.limit_stiffness;
            out.d_rate -= p.limit_damping;
        }
    } else if (q < p.lower_limit) {
        const Real push = p.limit_stiffness * (p.lower_limit - q) - p.limit_damping * v;
        if (push > 0.0) {
            out.force += push;
            out.d_position -= p.limit_stiffness;
            out.d_rate -= p.limit_damping;
        }
    }
    return out;
}

Real JointCoordinateForce::potential_energy(Real q) const
{
    const JointCoordinateForceParams& p = params_;
    Real V = p.spring.integral(0.0, q - p.reference) - p.preload * (q - p.reference);
    if (q > p.upper_limit) V += 0.5 * p.limit_stiffness * (q - p.upper_limit) * (q - p.upper_limit);
    if (q < p.lower_limit) V += 0.5 * p.limit_stiffness * (p.lower_limit - q) * (p.lower_limit - q);
    return V;
}

void JointCoordinateForce::apply(const VecX& q, const VecX& v, VecX& tau) const
{
    tau(iv_) += law(q(iq_), v(iv_)).force;
}

} // namespace mbd::kernel
