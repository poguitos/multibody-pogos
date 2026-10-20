// The examples of the how-to guide docs/help/howto/outputs-contact-events.md
// (task H.6, Phase 3's tasks 3.3, 3.8 and 3.9, and the debugging aids of
// H.4), each run and checked against a value derived from the physics. The
// guide quotes the code between the example markers exactly
// (test_howto_pages.cpp).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "mbd/analysis/recorder.hpp"
#include "mbd/core/core.hpp"
#include "mbd/forces/contact.hpp"
#include "mbd/kernel/describe.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/validate.hpp"

#include "howto/howto_models.hpp"
#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
namespace fs = std::filesystem;

namespace {

/// A 2 kg ball on a vertical slider, its coordinate its height, gravity
/// along -Z.
int add_ball(System& sys)
{
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    return sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                              RigidBodyInertia::from_solid_box(2.0, Vec3(0.05, 0.05, 0.05)), "ball");
}

} // namespace

TEST_CASE("How-to outputs: the loads a joint carries", "[howto]")
{
    // The rod hanging at rest from its hinge (howto_models.hpp): the hinge
    // carries its weight, m g = 9.81 N upward (+Y), and no moment, all forces
    // on one vertical line; the joint frame is the world's. To rounding.
    System sys;
    mbd_howto::build_pendulum(sys);
    // [example: loads]
    Simulator sim(sys);                          // the rod hanging at rest from its hinge
    sim.initialize();
    const Loads loads = compute_loads(sim);      // at the simulator's state
    const Vec6& hinge = loads.joint_reaction[1]; // [moment; force] the hinge passes to the rod, in its joint frame
    // loads.v_dot: the accelerations; loads.lambda and loads.constraint_W: the constraints' forces.
    // [end example]
    INFO("reaction " << hinge.transpose());
    CHECK(hinge.head<3>().norm() < 1e-12);
    CHECK((hinge.tail<3>() - Vec3(0.0, g_accel, 0.0)).norm() < 1e-12);
}

TEST_CASE("How-to outputs: recording channels", "[howto]")
{
    // The rod swinging 1 mrad: the recorded angle at every sample is the
    // small-angle solution q0 cos(w t) to RK4's and the nonlinearity's error
    // (below 1e-9 rad over 1 s, as in the time-integration guide); the file
    // has a header row and one row per sample.
    System sys;
    mbd_howto::build_pendulum(sys);
    Simulator sim(sys);
    sim.q(0) = 0.001;
    sim.initialize();
    const std::string path = (fs::temp_directory_path() / "mbd_howto_recorder.csv").string();
    // [example: recorder]
    Recorder rec;
    rec.add("angle", [&] { return sim.q(0); });  // any function of anything: read at every sample
    rec.add("rate", [&] { return sim.v(0); });
    for (int k = 0; k < 100; ++k) {
        sim.step(0.01);
        rec.sample(sim.time);
    }
    const std::vector<Real>& angle = rec.column("angle");   // one value per sample
    rec.write_csv(path);                         // a header row, then time and the channels, 17 digits
    // [end example]
    const Real w = mbd_howto::pendulum_w();
    Real worst = 0.0;
    for (std::size_t k = 0; k < angle.size(); ++k) {
        worst = std::max(worst, std::abs(angle[k] - 0.001 * std::cos(w * rec.times()[k])));
    }
    std::ifstream in(path);
    int lines = 0;
    for (std::string line; std::getline(in, line);) ++lines;
    in.close();
    fs::remove(path);
    INFO("largest difference from the small-angle solution " << worst);
    CHECK(rec.samples() == 100);
    CHECK(worst < 1e-9);
    CHECK(lines == 101);
}

TEST_CASE("How-to contact: a ball on the ground", "[howto]")
{
    // A 2 kg ball of radius 0.1 m dropped from 5 cm above the ground. At
    // rest the contact carries the weight: k delta^1.5 = m g, delta = (m g /
    // k)^(2/3) = 0.73 mm into the ground. The damping (c / m = 1000 per
    // second at full depth) settles it within a second; after 2 s the rest
    // is far below 1e-9 m.
    System sys;
    const int ball = add_ball(sys);
    // [example: contact]
    ContactParams contact;
    contact.stiffness = 1e6;                     // k [N/m^1.5]: force k delta^e at penetration delta
    contact.exponent = 1.5;                      // e: 1.5 for a sphere (Hertz)
    contact.damping = 2000.0;                    // [N s/m], reached at damping_depth
    contact.damping_depth = 1e-4;                // [m]
    contact.mu_static = 0.8;                     // friction, for sliding (none here)
    contact.mu_dynamic = 0.6;
    sys.force_elements.push_back(std::make_shared<PlaneContact>(
        ball, std::vector<Vec3>{Vec3::Zero()}, 0.1,   // a sphere of 0.1 m at the ball's origin
        0, Transform3(), contact));              // against the ground's XY plane, solid below
    Simulator sim(sys);
    sim.q(0) = 0.15;                             // the ball's centre 5 cm above touching
    sim.initialize();
    sim.run(2.0, 0.0005);
    // [end example]
    const Real sink = std::pow(2.0 * g_accel / 1e6, 1.0 / 1.5);
    INFO("sunk " << 0.1 - sim.q(0) << " against " << sink);
    CHECK(std::abs((0.1 - sim.q(0)) - sink) < 1e-9);
}

TEST_CASE("How-to events: acting when a function crosses zero", "[howto]")
{
    // The ball without contact, dropped from 1 m: it passes 0.5 m at t =
    // sqrt(2 x 0.5 / g) = 0.3193 s with speed sqrt(2 g 0.5) = 3.13 m/s. Free
    // fall is quadratic in time, which RK4 integrates exactly, so the event
    // is found to its tolerance (1e-10 s); the speed then to g x 1e-10.
    System sys;
    add_ball(sys);
    Simulator sim(sys);
    sim.q(0) = 1.0;
    sim.initialize();
    Real speed_at_crossing = 0.0;
    // [example: event]
    Event half_way;
    half_way.name = "half way";
    half_way.function = [](const Simulator& s) { return s.q(0) - 0.5; };   // zero at 0.5 m
    half_way.direction = -1;                     // falling crossings only
    half_way.action = [&](Simulator& s) { speed_at_crossing = -s.v(0); };  // may change q and v
    sim.events.push_back(half_way);
    sim.run(1.0, 0.01);
    // sim.event_log(): every event that fired, and when.
    // [end example]
    const Real t_cross = std::sqrt(2.0 * 0.5 / g_accel), v_cross = std::sqrt(2.0 * g_accel * 0.5);
    REQUIRE(sim.event_log().size() == 1);
    INFO("at " << sim.event_log()[0].time << " s against " << t_cross << "; speed " << speed_at_crossing
               << " against " << v_cross);
    CHECK(std::abs(sim.event_log()[0].time - t_cross) < 1e-9);
    CHECK(std::abs(speed_at_crossing - v_cross) < 1e-8);
}

TEST_CASE("How-to checks: validate, describe, trace and dump", "[howto]")
{
    // The four-bar closed in 3D: 3 velocities, 5 equations of which 2 are
    // independent (a planar loop), so 1 degree of freedom and 3 redundant
    // equations. 50 traced steps make 50 records.
    mbd_test::FourBar fb;
    const std::string trace_path = (fs::temp_directory_path() / "mbd_howto_trace.csv").string();
    // [example: checks]
    const VecX q0 = mbd_test::FourBar::configuration(1.0);   // a configuration with the loop closed
    const ValidationReport check = validate(fb.sys, q0); // before simulating: errors, warnings, counts
    // check.ok(), check.degrees_of_freedom, check.redundant_equations; check.summary() as text.
    const std::string text = describe(fb.sys);           // bodies, joints, constraints, forces
    Simulator sim(fb.sys);
    sim.q = q0;
    sim.trace_steps = true;                              // a record of every step
    sim.initialize();
    sim.run(0.05, 0.001);
    sim.write_trace(trace_path);                         // projection, residuals, energy... per step, as CSV
    const std::string state = dump_state(sim);           // the state now, every coordinate named
    // [end example]
    fs::remove(trace_path);
    INFO(check.summary());
    CHECK(check.ok());
    CHECK(check.velocities == 3);
    CHECK(check.constraint_equations == 5);
    CHECK(check.degrees_of_freedom == 1);
    CHECK(check.redundant_equations == 3);
    CHECK(text.find("crank") != std::string::npos);
    CHECK(sim.trace().size() == 50);
    CHECK(state.find("coupler") != std::string::npos);
}
