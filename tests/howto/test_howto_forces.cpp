// The examples of the how-to guide docs/help/howto/forces-and-drives.md
// (task H.6, Phase 3's tasks 3.1 and 3.2), each run and checked against a
// value derived from the physics. The guide quotes the code between the
// example markers exactly (test_howto_pages.cpp).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"

#include "howto/howto_models.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_howto::HangingBlock;

TEST_CASE("How-to forces: a spring-damper between two points", "[howto]")
{
    // A 10 kg block on a vertical slider hangs from the spring, released at
    // the spring's free length. Damped critically (c = 2 sqrt(k m) = 632 N
    // s/m, w = 31.6 rad/s), it settles within a second where k (L - L0) = m
    // g: 9.81 mm lower. After 3 s what is left of the motion is e^(-w t) (1 +
    // w t) of 9.81 mm, below 1e-30; RK4 at 1 ms resolves w. Allowed 1e-9 m.
    System sys;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    // [example: spring-damper]
    const int block = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                         RigidBodyInertia::from_solid_box(10.0, Vec3(0.1, 0.1, 0.1)), "block");
    SpringDamperParams p;
    p.free_length = 0.8;                         // [m]
    p.spring = Curve::linear(1e4);               // force pushing apart against the compression [N/m]
    p.damper = Curve::linear(632.0);             // force against the rate of extension [N s/m]
    p.rebound_clearance = 0.05;                  // a rebound stop 5 cm beyond the free length
    p.rebound_stop = Curve::linear(1e6);
    sys.force_elements.push_back(
        std::make_shared<SpringDamper>(0, block, Vec3(0.0, 0.0, 1.0), Vec3::Zero(), p));   // ground point, block point
    // [end example]
    Simulator sim(sys);
    sim.q(0) = 0.2;   // 0.8 m below the ground point: the free length
    sim.initialize();
    sim.run(3.0, 0.001);
    INFO("height " << sim.q(0) << " against " << HangingBlock::equilibrium());
    CHECK(std::abs(sim.q(0) - HangingBlock::equilibrium()) < 1e-9);
}

TEST_CASE("How-to forces: a torsion spring on a hinge", "[howto]")
{
    // A rod 1 m long on a hinge along Z, pointing along X at angle 0, gravity
    // along -Y: gravity's torque about the hinge is -m g d cos q. A torsion
    // spring of 50 N m/rad with no torque at 0.2 rad holds it where 50 (0.2 -
    // q) = m g d cos q (solved below by Newton). Its damper, near critical
    // for the rod's inertia, settles it within 3 s (w = 12.2 rad/s). Allowed
    // 1e-9 rad.
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia bar = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.02, 0.02));
    bar.com_B = Vec3(0.5, 0.0, 0.0);
    const int rod = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), bar,
                                       "rod");
    // [example: torsion-spring]
    JointCoordinateForceParams hinge;
    hinge.reference = 0.2;                       // where the spring carries no torque [rad]
    hinge.spring = Curve::linear(50.0);          // [N m/rad]
    hinge.damper = Curve::linear(8.0);           // [N m s/rad]
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, rod, hinge));
    // [end example]
    Simulator sim(sys);
    sim.initialize();
    sim.run(3.0, 0.001);
    Real q = 0.1;
    for (int i = 0; i < 20; ++i) {
        const Real f = 50.0 * (0.2 - q) - g_accel * 0.5 * std::cos(q);
        q -= f / (-50.0 + g_accel * 0.5 * std::sin(q));
    }
    INFO("angle " << sim.q(0) << " against " << q);
    CHECK(std::abs(sim.q(0) - q) < 1e-9);
}

TEST_CASE("How-to forces: a hinge driven along a prescribed motion", "[howto]")
{
    // The hanging rod (howto_models.hpp) driven along q = A sin(w t). The
    // drive supplies what the motion needs: I q'' + m g d sin q, with I the
    // inertia about the hinge and d = 0.5 m. RK4 at 1 ms holds the drive to
    // the projection's tolerance; the torque is that of the dynamics at the
    // state, to rounding: allowed 1e-8 N m.
    System sys;
    const int rod = mbd_howto::build_pendulum(sys);
    // [example: joint-driver]
    const Real A = 0.5, w = 3.0;                 // amplitude [rad], frequency [rad/s]
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, rod,
        TimeFunction([=](Real t) { return A * std::sin(w * t); },              // the angle,
                     [=](Real t) { return A * w * std::cos(w * t); },          // its rate
                     [=](Real t) { return -A * w * w * std::sin(w * t); })));  // and its acceleration
    Simulator sim(sys);
    sim.initialize();                            // the velocity follows the drive: A w at t = 0
    sim.run(1.0, 0.001);
    const Loads loads = compute_loads(sim);      // at the state at 1 s
    const Real torque = loads.lambda(0);         // what the drive exerts [N m]
    // [end example]
    const Real t = sim.time, q = A * std::sin(w * t), q_ddot = -A * w * w * std::sin(w * t);
    const Real expected = mbd_howto::pendulum_inertia() * q_ddot + 1.0 * g_accel * 0.5 * std::sin(q);
    INFO("angle " << sim.q(0) << " against " << q << "; torque " << torque << " against " << expected);
    CHECK(std::abs(sim.q(0) - q) < 1e-9);
    CHECK(std::abs(torque - expected) < 1e-8);
}
