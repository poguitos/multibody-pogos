#pragma once

// The small models the how-to guides' examples run on (task H.6), with the
// closed-form values their checks derive from.

#include <cmath>
#include <memory>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd_howto {

using mbd::Real;

/// A 1 kg rod 1 m long hanging from a hinge along Z at its top end, gravity
/// along -Y; the hinge's coordinate is the rod's angle from hanging.
inline int build_pendulum(mbd::kernel::System& sys)
{
    using namespace mbd;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia rod = RigidBodyInertia::from_solid_box(1.0, Vec3(0.02, 0.5, 0.02));
    rod.com_B = Vec3(0.0, -0.5, 0.0);
    return sys.model.add_body(0, std::make_shared<kernel::RevoluteJointModel>(), Transform3(), Transform3(), rod,
                              "rod");
}

/// The rod's moment of inertia about the hinge: m (b^2 + w^2) / 3 + m d^2,
/// half-length b = 0.5, half-width w = 0.02, centre of mass d = 0.5 away.
inline Real pendulum_inertia() { return (0.5 * 0.5 + 0.02 * 0.02) / 3.0 + 0.5 * 0.5; }

/// Its small swings: w^2 = m g d / I.
inline Real pendulum_w() { return std::sqrt(mbd::g_accel * 0.5 / pendulum_inertia()); }

/// A 1 kg block on a slider with a spring of k N/m, no gravity.
inline void build_spring_block(mbd::kernel::System& sys, Real k)
{
    using namespace mbd;
    sys.model.gravity = Vec3::Zero();
    const int block = sys.model.add_body(0, std::make_shared<kernel::PrismaticJointModel>(), Transform3(),
                                         Transform3(), RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)),
                                         "block");
    kernel::JointCoordinateForceParams spring;
    spring.spring = Curve::linear(k);
    sys.joint_forces.push_back(std::make_shared<kernel::JointCoordinateForce>(sys.model, block, spring));
}

/// A 10 kg block on a vertical slider (its coordinate its height, gravity
/// along -Z), hanging from a spring-damper of 1e4 N/m and free length 0.8 m
/// between a point 1 m up on the ground and the block's origin: at height
/// 0.2 m the spring is at its free length, and it holds the block m g / k =
/// 9.81 mm lower.
struct HangingBlock {
    static constexpr Real m = 10.0, k = 1e4, free_length = 0.8;
    static Real equilibrium() { return 1.0 - free_length - m * mbd::g_accel / k; }
};

inline int build_hanging_block(mbd::kernel::System& sys, Real damping)
{
    using namespace mbd;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    const int block = sys.model.add_body(0, std::make_shared<kernel::PrismaticJointModel>(), Transform3(),
                                         Transform3(),
                                         RigidBodyInertia::from_solid_box(HangingBlock::m, Vec3(0.1, 0.1, 0.1)),
                                         "block");
    SpringDamperParams p;
    p.free_length = HangingBlock::free_length;
    p.spring = Curve::linear(HangingBlock::k);
    p.damper = Curve::linear(damping);
    sys.force_elements.push_back(std::make_shared<SpringDamper>(0, block, Vec3(0.0, 0.0, 1.0), Vec3::Zero(), p));
    return block;
}

} // namespace mbd_howto
