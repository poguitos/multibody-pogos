// The examples of the how-to guide docs/help/howto/analyses.md (task H.6,
// Phase 3's tasks 3.4 to 3.7), each run and checked against a value derived
// from the physics or the geometry. The guide quotes the code between the
// example markers exactly (test_howto_pages.cpp).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/core/core.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

#include "howto/howto_models.hpp"
#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_howto::HangingBlock;
using mbd_test::FourBar;

TEST_CASE("How-to analyses: assembling a closed loop with a coordinate held", "[howto]")
{
    // The four-bar of the tests (a tree of three hinges closed at the fourth
    // by a revolute closure), its loop open by a few degrees. Holding the
    // crank's angle (1 rad) and its rate (2 rad/s), assembly must close the
    // loop where the geometry puts it, FourBar::configuration(1), and give
    // the other joints the rates that keep it closed, 2 FourBar::motion(q):
    // both in closed form. Allowed 1e-9 (the tolerance is 1e-10 on the
    // residuals, and the loop's Jacobian is well conditioned here).
    FourBar fb;
    const Eigen::Vector3d closed = FourBar::configuration(1.0);
    // [example: assembly]
    Simulator sim(fb.sys);                       // the four-bar: three hinges, a closure at the fourth
    sim.q.head(3) = closed + Eigen::Vector3d(0.0, 0.05, -0.04);   // the loop a few degrees open
    sim.v(0) = 2.0;                              // the crank's rate [rad/s]; the others unknown
    AssemblyOptions options;
    options.hold_position = {0};                 // keep the crank's angle as given
    options.hold_velocity = {0};                 // and its rate
    const AssemblyReport report = assemble(sim, options);
    // report.ok(): positions and velocities on the constraints; sim.q, sim.v assembled.
    // report.summary(fb.sys.model) lists every coordinate's given and assembled value.
    // [end example]
    const Eigen::Vector3d rates = 2.0 * FourBar::motion(sim.q);
    INFO(report.summary(fb.sys.model));
    REQUIRE(report.ok());
    CHECK(sim.q(0) == 1.0);
    CHECK((sim.q.head(3) - closed).norm() < 1e-9);
    CHECK(sim.v(0) == 2.0);
    CHECK((sim.v.head(3) - rates).norm() < 1e-9);
}

TEST_CASE("How-to analyses: static equilibrium", "[howto]")
{
    // The block hanging from its spring (howto_models.hpp), started at the
    // spring's free length: at rest k (L - L0) = m g, 9.81 mm lower. The
    // model is linear, so Newton lands there at once; the tolerance on the
    // acceleration at rest (1e-6 m/s^2) bounds the position by 1e-6 / w^2 =
    // 1e-9 m.
    System sys;
    mbd_howto::build_hanging_block(sys, 632.0);
    // [example: statics]
    Simulator sim(sys);                          // the block hanging from its spring
    sim.q(0) = 0.2;                              // where the spring is at its free length
    const StaticReport report = static_equilibrium(sim);
    // report.converged; sim.q is the equilibrium, at rest; report.loads the forces there;
    // report.summary(sys.model) says how it went.
    // [end example]
    INFO(report.summary(sys.model));
    REQUIRE(report.converged);
    CHECK(std::abs(sim.q(0) - HangingBlock::equilibrium()) < 1e-9);
    CHECK(sim.v.norm() == 0.0);
}

TEST_CASE("How-to analyses: kinematics along a driving parameter", "[howto]")
{
    // The four-bar with a driver turning its crank to the angle t: no degree
    // of freedom left. At t = 1 the solved configuration is
    // FourBar::configuration(1), and the velocities are the joints' rates
    // per unit rate of t, FourBar::motion(q): both in closed form. Allowed
    // 1e-9.
    FourBar fb;
    fb.sys.constraints.push_back(std::make_shared<JointDriver>(
        fb.sys.model, fb.crank,
        TimeFunction([](Real t) { return t; }, [](Real) { return 1.0; }, [](Real) { return 0.0; })));
    // [example: kinematics]
    Kinematics kin(fb.sys);                      // the four-bar with a driver turning its crank to angle t
    kin.q.head(3) = FourBar::configuration(0.9); // a start near the solution
    kin.t = 1.0;                                 // the driving parameter: here the crank's angle [rad]
    const bool closed = kin.solve();             // phi(q, t) = 0
    const VecX& rates = kin.velocities();        // dq/dt: each joint's rate per unit rate of t
    // [end example]
    REQUIRE(closed);
    INFO("configuration " << kin.q.transpose() << "; rates " << rates.transpose());
    CHECK((kin.q.head(3) - FourBar::configuration(1.0)).norm() < 1e-9);
    CHECK((rates.head(3) - FourBar::motion(kin.q)).norm() < 1e-9);
}

TEST_CASE("How-to analyses: natural frequencies and damping", "[howto]")
{
    // The hanging block with a damper of 63.2 N s/m: one mode, w = sqrt(k /
    // m) = 31.62 rad/s, damping ratio c / (2 sqrt(k m)) = 0.0999. The model
    // is linear, so the central differences are exact to rounding (1e-10 of
    // the entries at the perturbation of 1e-6): allowed 1e-7 relative.
    System sys;
    mbd_howto::build_hanging_block(sys, 63.2);
    Simulator sim(sys);
    sim.q(0) = HangingBlock::equilibrium();
    // [example: linearize]
    const LinearModel lm = linearize(sim);       // about the simulator's state: here the equilibrium
    const Mode& slowest = lm.modes.front();      // the modes come by natural frequency
    // slowest.natural_frequency [rad/s], slowest.damping_ratio, slowest.shape (velocity coordinates),
    // slowest.energy_fraction (its kinetic energy's share in each coordinate, adding up to 1);
    // lm.summary(sys.model) lists every mode with its largest shape components and energy shares.
    // [end example]
    const Real w = std::sqrt(HangingBlock::k / HangingBlock::m);
    const Real zeta = 63.2 / (2.0 * std::sqrt(HangingBlock::k * HangingBlock::m));
    INFO(lm.summary(sys.model));
    REQUIRE(lm.modes.size() == 1);
    CHECK(std::abs(slowest.natural_frequency - w) < 1e-7 * w);
    CHECK(std::abs(slowest.damping_ratio - zeta) < 1e-7);
}
