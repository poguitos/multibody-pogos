#pragma once

// Random models and states for the kernel tests.

#include <memory>
#include <string>
#include <vector>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/model.hpp"

#include "support/rng.hpp"

namespace mbd_test {

enum class KJoint { Revolute, Prismatic, Fixed, Universal, Cylindrical, Planar, Spherical, Free };

inline const std::vector<KJoint>& all_kernel_joints()
{
    static const std::vector<KJoint> all{KJoint::Revolute,  KJoint::Prismatic, KJoint::Fixed,
                                         KJoint::Universal, KJoint::Cylindrical, KJoint::Planar,
                                         KJoint::Spherical, KJoint::Free};
    return all;
}

inline std::shared_ptr<const mbd::kernel::JointModel> make_kernel_joint(KJoint k)
{
    using namespace mbd::kernel;
    switch (k) {
        case KJoint::Revolute:    return std::make_shared<RevoluteJointModel>();
        case KJoint::Prismatic:   return std::make_shared<PrismaticJointModel>();
        case KJoint::Fixed:       return std::make_shared<FixedJointModel>();
        case KJoint::Universal:   return std::make_shared<UniversalJointModel>();
        case KJoint::Cylindrical: return std::make_shared<CylindricalJointModel>();
        case KJoint::Planar:      return std::make_shared<PlanarJointModel>();
        case KJoint::Spherical:   return std::make_shared<SphericalJointModel>();
        case KJoint::Free:        return std::make_shared<FreeJointModel>();
    }
    return nullptr;
}

/// A generic body: box inertia with tilted principal axes and an offset
/// centre of mass.
inline mbd::RigidBodyInertia random_inertia(Rng& rng)
{
    mbd::RigidBodyInertia I = mbd::RigidBodyInertia::from_solid_box(
        rng.range(0.5, 3.0),
        mbd::Vec3(rng.range(0.05, 0.3), rng.range(0.05, 0.3), rng.range(0.05, 0.3)));
    const mbd::Mat3 R = rng.quat().toRotationMatrix();
    I.I_com_B = R * I.I_com_B * R.transpose();
    I.com_B   = rng.vec(0.15);
    return I;
}

/// Add a body on a joint of the given kind with generic joint frames.
inline int add_random_body(mbd::kernel::Model& model, Rng& rng, KJoint kind, int parent)
{
    return model.add_body(parent, make_kernel_joint(kind), rng.frame(0.4), rng.frame(0.3),
                          random_inertia(rng));
}

/// Ground - root joint - body - each of `children` in a chain.
inline mbd::kernel::Model random_chain(Rng& rng, KJoint root, const std::vector<KJoint>& children)
{
    mbd::kernel::Model model;
    int parent = add_random_body(model, rng, root, 0);
    for (KJoint k : children) parent = add_random_body(model, rng, k, parent);
    return model;
}

/// A tree that exercises branching: a root body with two branches of two
/// bodies each, the joint kinds taken in turn from `kinds`.
inline mbd::kernel::Model random_tree(Rng& rng, KJoint root, const std::vector<KJoint>& kinds)
{
    mbd::kernel::Model model;
    const int b1 = add_random_body(model, rng, root, 0);
    std::size_t k = 0;
    auto next = [&] { return kinds[k++ % kinds.size()]; };
    const int b2 = add_random_body(model, rng, next(), b1);
    add_random_body(model, rng, next(), b2);
    const int b4 = add_random_body(model, rng, next(), b1);
    add_random_body(model, rng, next(), b4);
    return model;
}

/// A generic configuration: random angles and displacements, random unit
/// quaternions.
inline mbd::VecX random_configuration(const mbd::kernel::Model& model, Rng& rng)
{
    mbd::VecX q = model.neutral_configuration();
    for (int i = 1; i < model.nbodies(); ++i) {
        auto qi = q.segment(model.idx_q[i], model.nqs[i]);
        const std::string name = model.joint[i]->name();
        if (name == "spherical") {
            qi = rng.quat().coeffs();
        } else if (name == "free") {
            qi.head<3>() = rng.vec(0.5);
            qi.tail<4>() = rng.quat().coeffs();
        } else {
            for (Eigen::Index k = 0; k < qi.size(); ++k) qi(k) = rng.range(-1.0, 1.0);
        }
    }
    return q;
}

inline mbd::VecX random_vector(int n, Rng& rng, mbd::Real magnitude)
{
    mbd::VecX x(n);
    for (int k = 0; k < n; ++k) x(k) = rng.range(-magnitude, magnitude);
    return x;
}

/// One classical Runge-Kutta step of the unforced system. The coordinates of
/// the stages are combined linearly and normalized once, at the end (see
/// kernel::q_dot).
inline void rk4_step(const mbd::kernel::Model& model, mbd::kernel::Data& data,
                     mbd::VecX& q, mbd::VecX& v, mbd::Real dt)
{
    using mbd::VecX;
    using namespace mbd::kernel;
    const VecX zero = VecX::Zero(model.nv);
    VecX k1q, k2q, k3q, k4q;
    q_dot(model, q, v, k1q);
    const VecX k1v = aba(model, data, q, v, zero);
    const VecX q2 = q + 0.5 * dt * k1q, v2 = v + 0.5 * dt * k1v;
    q_dot(model, q2, v2, k2q);
    const VecX k2v = aba(model, data, q2, v2, zero);
    const VecX q3 = q + 0.5 * dt * k2q, v3 = v + 0.5 * dt * k2v;
    q_dot(model, q3, v3, k3q);
    const VecX k3v = aba(model, data, q3, v3, zero);
    const VecX q4 = q + dt * k3q, v4 = v + dt * k3v;
    q_dot(model, q4, v4, k4q);
    const VecX k4v = aba(model, data, q4, v4, zero);
    q += dt / 6.0 * (k1q + 2.0 * k2q + 2.0 * k3q + k4q);
    v += dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    normalize(model, q);
}

} // namespace mbd_test
