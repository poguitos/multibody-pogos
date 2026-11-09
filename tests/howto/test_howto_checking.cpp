// The examples of the how-to guide docs/help/howto/checking-a-model.md (task
// H.6, Phase 5's tasks 5.1, 5.2 and 5.4), each run and checked against a value
// derived from the geometry or the physics. The guide quotes the code between
// the example markers exactly (test_howto_pages.cpp).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

using namespace mbd;
using namespace mbd::kernel;

TEST_CASE("How-to checking: constraints that contradict each other are named", "[howto]")
{
    // A 1 kg bob on a free joint, held 1 m from (0, 0, 2) and 1 m from
    // (3, 0, 2): the anchors are 3 m apart, more than 1 + 1, so no point
    // satisfies both. Started between them, the assembly cannot close the
    // positions and names the two distance equations, rows 0 and 1, as the
    // smallest set that cannot hold together (task 5.2, D44).
    System sys;
    const int bob = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)), "bob");
    sys.constraints.push_back(std::make_shared<Distance>(Marker{0, Transform3::FromTranslation(Vec3(0.0, 0.0, 2.0))},
                                                         Marker{bob, Transform3()}, 1.0));
    sys.constraints.push_back(std::make_shared<Distance>(Marker{0, Transform3::FromTranslation(Vec3(3.0, 0.0, 2.0))},
                                                         Marker{bob, Transform3()}, 1.0));
    Simulator sim(sys);
    sim.q.head<3>() = Vec3(1.2, 0.3, 2.1);
    // [example: conflicts]
    const AssemblyReport report = assemble(sim);
    if (!report.positions_assembled) {
        // report.conflicting_equations: the rows (the constraints' equations, stacked in
        // order) that cannot all hold together; the MBD-K062 warning names their constraints.
        for (const std::string& w : report.warnings) std::cout << w << "\n";
    }
    // [end example]
    REQUIRE_FALSE(report.positions_assembled);
    CHECK(report.conflicting_equations == std::vector<int>{0, 1});
}

TEST_CASE("How-to checking: where a mode's energy is", "[howto]")
{
    // Two blocks of 1 and 4 kg on sliders in a line between two walls, three
    // springs of k = 100 N/m (wall, block, block, wall) at their rest
    // lengths, no gravity. K = k [[2, -1], [-1, 2]], M = diag(1, 4):
    // det(K - w^2 M) = 0 gives 4 w^4 - 10 k w^2 + 3 k^2 = 0, w^2 = k (10 -+
    // sqrt(52)) / 8. In the slow mode x2 / x1 = 2 - w^2 / k, and block i's
    // share of the kinetic energy is m_i x_i^2 / (m_1 x_1^2 + m_2 x_2^2):
    // 0.0840 and 0.9160, the heavy block carrying almost all of it. The
    // model is linear, so the differences are exact to rounding (allowed
    // 1e-7 relative on the frequency, 1e-7 on the shares).
    const Real k = 100.0, a = 0.25;
    System sys;
    sys.model.gravity = Vec3::Zero();
    const auto slider = std::make_shared<PrismaticJointModel>();
    const int b1 = sys.model.add_body(0, slider, Transform3::FromTranslation(Vec3(0.0, 0.0, a)), Transform3(),
                                      RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)), "light");
    const int b2 = sys.model.add_body(0, slider, Transform3::FromTranslation(Vec3(0.0, 0.0, 2.0 * a)), Transform3(),
                                      RigidBodyInertia::from_solid_box(4.0, Vec3(0.05, 0.05, 0.05)), "heavy");
    const Vec3 o = Vec3::Zero();
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0, b1, o, o, k, 0.0, a));
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(b1, b2, o, o, k, 0.0, a));
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(b2, 0, o, Vec3(0.0, 0.0, 3.0 * a), k, 0.0, a));
    Simulator sim(sys);
    // [example: energy-shares]
    const LinearModel lin = linearize(sim);
    for (const Mode& mode : lin.modes) {
        // mode.natural_frequency [rad/s]; mode.energy_fraction(i): the share of the mode's
        // kinetic energy in velocity coordinate i, adding up to 1 over the coordinates.
        std::cout << mode.natural_frequency << " rad/s: " << mode.energy_fraction.transpose() << "\n";
    }
    // [end example]
    REQUIRE(lin.modes.size() == 2);
    const Real w2 = k * (10.0 - std::sqrt(52.0)) / 8.0;
    const Real ratio = 2.0 - w2 / k;
    const Real light = 1.0 / (1.0 + 4.0 * ratio * ratio);
    const Mode& slow = lin.modes[0];
    INFO("slow mode " << slow.natural_frequency << " rad/s, shares " << slow.energy_fraction.transpose()
                      << "; closed form " << std::sqrt(w2) << " rad/s, " << light << ", " << 1.0 - light);
    CHECK(std::abs(slow.natural_frequency - std::sqrt(w2)) < 1e-7 * std::sqrt(w2));
    CHECK(std::abs(slow.energy_fraction(0) - light) < 1e-7);
    CHECK(std::abs(slow.energy_fraction(1) - (1.0 - light)) < 1e-7);
}

TEST_CASE("How-to checking: settling parts of very different stiffness", "[howto]")
{
    // Two 1 kg blocks on vertical sliders in series (gravity along -Z), a
    // spring of 1e8 N/m under the first and one of 100 N/m between them
    // (task 5.4, D47): at rest q1 = -2 g / 1e8 and q2 = -g / 100, one Newton
    // step for this linear model once the step limit allows the 9.8 cm. The
    // relaxation step, used only if Newton stalls, is set below 2 / w for
    // the stiff spring (w = 1e4 rad/s). Allowed 1e-11 m (the soft spring's
    // rounding, as in the statics tests).
    System sys;
    const int stiff = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                         RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), "stiff");
    const int soft = sys.model.add_body(stiff, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                        RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), "soft");
    for (const auto& [body, stiffness] : {std::pair{stiff, 1e8}, std::pair{soft, 100.0}}) {
        JointCoordinateForceParams spring;
        spring.spring = Curve::linear(stiffness);
        sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, body, spring));
    }
    Simulator sim(sys);
    // [example: stiff-statics]
    StaticOptions options;
    options.max_translation = 1.0;               // let one step move the soft spring 9.8 cm
    options.relaxation_step = 1e-4;              // below 2 / w for the stiffest part, should Newton stall
    const StaticReport report = static_equilibrium(sim, options);
    // report.supported: the directions with stiffness (here 2), down to 1e-7 of the stiffest;
    // report.relaxation_undone: relaxation runs that grew instead of settling, and were undone.
    // [end example]
    INFO(report.summary(sys.model));
    REQUIRE(report.converged);
    CHECK(report.supported == 2);
    CHECK(std::abs(sim.q(0) + 2.0 * g_accel / 1e8) < 1e-15);
    CHECK(std::abs(sim.q(1) + g_accel / 100.0) < 1e-11);
}
