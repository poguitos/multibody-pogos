// How-to examples (task H.6): force elements. Quoted by
// docs/help/howto/forces.md; the regions between "// [name]" markers must
// match the page (tests/core/test_howto_examples.cpp checks it).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

#include "mbd/forces/curve.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

using namespace mbd;
using namespace mbd::kernel;

TEST_CASE("Example: a spring with a tabulated curve and a bump stop", "[examples]")
{
    // [spring]
    System sys;
    const int slider = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                          RigidBodyInertia::from_solid_box(10.0, Vec3(0.1, 0.1, 0.1)), "slider");
    SpringDamperParams p;
    p.free_length = 0.3;                                                  // [m]
    p.spring = Curve::table({0.0, 0.02, 0.05, 0.1}, {0.0, 400.0, 1200.0, 3000.0});   // force [N] against compression [m]
    p.damper = Curve::linear(150.0);                                      // [N per m/s] of extension rate
    p.bump_clearance = 0.08;                                              // the stop engages beyond 8 cm of compression
    p.bump_stop = Curve::linear(5e4);
    auto spring = std::make_shared<SpringDamper>(0, slider, Vec3::Zero(), Vec3::Zero(), p);
    sys.force_elements.push_back(spring);

    Simulator sim(sys);
    sim.q(0) = 0.3;                                    // the slider's height: the spring at its free length
    const StaticsReport rest = static_equilibrium(sim);
    const LawValue f = spring->law(sim.q(0), 0.0);     // force, d/dlength, d/drate
    // [spring]
    REQUIRE(rest.converged);
    CHECK(std::abs(f.force - 10.0 * g_accel) <= 1e-5);   // it carries the weight
}

TEST_CASE("Example: a torsion spring with friction and limits on a hinge", "[examples]")
{
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia arm = RigidBodyInertia::from_solid_box(2.0, Vec3(0.25, 0.02, 0.02));
    arm.com_B = Vec3(0.25, 0.0, 0.0);
    const int hinge = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), arm, "arm");
    // [joint]
    JointCoordinateForceParams h;
    h.spring = Curve::linear(20.0);       // [N m per rad] about the reference
    h.reference = 0.0;
    h.friction = 0.5;                     // [N m], regularised over friction_velocity (1e-3 rad/s)
    h.lower_limit = -1.0;                 // [rad]: beyond the limits a stop pushes back
    h.upper_limit = 1.0;
    h.limit_stiffness = 1e4;
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, hinge, h));
    // [joint]
    Simulator sim(sys);
    sim.run(2.0, 1e-3);
    CHECK(sim.q(0) < 0.0);       // the arm sags under gravity
    CHECK(sim.q(0) > -1.0);      // the spring holds it above the stop
}
