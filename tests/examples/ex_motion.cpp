// How-to examples (task H.6): prescribed motion and kinematic analysis.
// Quoted by docs/help/howto/motion.md.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;

TEST_CASE("Example: drive a joint and read the drive's torque", "[examples]")
{
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia arm = RigidBodyInertia::from_solid_box(1.0, Vec3(0.2, 0.02, 0.02));
    arm.com_B = Vec3(0.2, 0.0, 0.0);
    const int body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), arm);
    // [driver]
    // The hinge follows s(t) = 0.3 sin(2 t): the function and its first two
    // derivatives.
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, body,
        TimeFunction([](Real t) { return 0.3 * std::sin(2.0 * t); },
                     [](Real t) { return 0.6 * std::cos(2.0 * t); },
                     [](Real t) { return -1.2 * std::sin(2.0 * t); })));
    Simulator sim(sys);
    sim.initialize();                                  // velocities consistent with the drive
    sim.run(1.0, 1e-3);
    sim.acceleration(sim.q, sim.v, sim.time);          // evaluate at the state reached
    const Real torque = sim.solver().lambda()(0);      // the drive's torque [N m]
    // [driver]
    CHECK(std::abs(sim.q(0) - 0.3 * std::sin(2.0)) <= 1e-9);
    // I q'' + m g l cos q for an arm along X at angle q, gravity -Y.
    const Real I = arm.I_com_B(2, 2) + 1.0 * 0.2 * 0.2;
    const Real q = sim.q(0);
    CHECK(std::abs(torque - (I * (-1.2 * std::sin(2.0)) + 1.0 * g_accel * 0.2 * std::cos(q))) <= 1e-8);
}

TEST_CASE("Example: a mechanism's motion ratios by kinematic analysis", "[examples]")
{
    // [kinematics]
    // A four-bar (crank 0.3 m, coupler 0.8, rocker 0.6, ground 0.7), its
    // crank driven to the angle t: in a kinematic analysis t is the driving
    // parameter, not a time.
    System sys;
    const auto hinge = std::make_shared<RevoluteJointModel>();
    const RigidBodyInertia link = RigidBodyInertia::from_solid_box(1.0, Vec3(0.2, 0.02, 0.02));
    const int crank = sys.model.add_body(0, hinge, Transform3(), Transform3(), link, "crank");
    const int coupler = sys.model.add_body(crank, hinge, Transform3::FromTranslation(Vec3(0.3, 0, 0)), Transform3(), link, "coupler");
    const int rocker = sys.model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(0.8, 0, 0)), Transform3(), link, "rocker");
    sys.constraints.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(0.7, 0, 0))},
                                               Marker{rocker, Transform3::FromTranslation(Vec3(0.6, 0, 0))}));
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, crank, TimeFunction([](Real t) { return t; }, [](Real) { return 1.0; }, [](Real) { return 0.0; })));

    Kinematics k(sys);
    k.q << 1.0, -0.6, -2.3;             // near the closed loop
    k.t = 1.0;                          // crank at 1 rad
    const bool closed = k.solve();
    const VecX& ratio = k.velocities(); // d(angles)/d(crank angle)
    // [kinematics]
    REQUIRE(closed);
    CHECK(std::abs(ratio(0) - 1.0) <= 1e-12);   // the crank itself
    CHECK(k.phi().norm() <= 1e-10);
}
