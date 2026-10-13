// Events (plan task 3.9).
//
//   - a bouncing ball's impact times, each found by shortening the step to
//     the crossing, match the closed form;
//   - several events in one step are handled in the order they happen, and
//     the step still advances by its length;
//   - a chain of events that never ends (an action that sends its function
//     back across zero, both directions watched) is cut and reported.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;

namespace {

/// A ball on a vertical slider, gravity along -Z, its height the slider's
/// coordinate.
System falling_ball()
{
    System sys;
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)), "ball");
    return sys;
}

/// The heights' crossing of `level`, falling.
Event level_crossing(const char* name, Real level)
{
    Event e;
    e.name = name;
    e.direction = -1;
    e.function = [level](const Simulator& s) { return s.q(0) - level; };
    return e;
}

std::vector<std::string>& captured()
{
    static std::vector<std::string> messages;
    return messages;
}

} // namespace

TEST_CASE("Events: a bouncing ball's impact times match the closed form", "[kernel][events]")
{
    // Dropped from h = 1 m, restitution e = 0.8: the first impact at
    // sqrt(2 h / g), then flights of 2 e^k v1 / g, v1 = sqrt(2 g h). Gravity
    // alone makes the motion quadratic in time, which RK4 integrates
    // exactly, so the event times are the exact ones to within the event
    // tolerance (1e-10 s); each bounce starts at most v1 x 1e-10 below the
    // ground, which shifts later impacts by a further 1e-10 s or so. The
    // steps are 10 ms: every impact falls inside one.
    const Real h = 1.0, e = 0.8;
    System sys = falling_ball();
    Simulator sim(sys);
    sim.q(0) = h;
    std::vector<Real> impacts;
    Event ground = level_crossing("ground", 0.0);
    ground.action = [&](Simulator& s) {
        impacts.push_back(s.time);
        s.v(0) = -e * s.v(0);
    };
    sim.events.push_back(ground);
    sim.run(3.0, 0.01);

    const Real v1 = std::sqrt(2.0 * g_accel * h);
    std::vector<Real> expected{std::sqrt(2.0 * h / g_accel)};
    for (Real speed = e * v1; expected.back() + 2.0 * speed / g_accel < 3.0; speed *= e) {
        expected.push_back(expected.back() + 2.0 * speed / g_accel);
    }
    REQUIRE(expected.size() == 6);
    REQUIRE(impacts.size() == expected.size());
    Real worst = 0.0;
    for (std::size_t k = 0; k < impacts.size(); ++k) worst = std::max(worst, std::abs(impacts[k] - expected[k]));
    INFO("worst impact time error " << worst << " s");
    CHECK(worst < 1e-9);
    CHECK(sim.event_log().size() == 6);
    CHECK(std::abs(sim.time - 3.0) < 1e-12);
}

TEST_CASE("Events: several in one step are handled in order, and the step keeps its length",
          "[kernel][events]")
{
    // A ball falling from 1 m crosses 0.5 m at sqrt(2 x 0.5 / g) = 0.3193 s
    // and 0.3 m at sqrt(2 x 0.7 / g) = 0.3778 s: both inside the step from
    // 0.3 to 0.4 s. No actions, so the motion stays the exact free fall.
    System sys = falling_ball();
    Simulator sim(sys);
    sim.q(0) = 1.0;
    sim.events.push_back(level_crossing("lower", 0.3));
    sim.events.push_back(level_crossing("upper", 0.5));
    sim.run(0.5, 0.1);

    const auto& log = sim.event_log();
    REQUIRE(log.size() == 2);
    CHECK(log[0].event == 1);
    CHECK(log[1].event == 0);
    CHECK(std::abs(log[0].time - std::sqrt(2.0 * 0.5 / g_accel)) < 1e-9);
    CHECK(std::abs(log[1].time - std::sqrt(2.0 * 0.7 / g_accel)) < 1e-9);
    CHECK(std::abs(sim.time - 0.5) < 1e-12);
    CHECK(std::abs(sim.q(0) - (1.0 - 0.5 * g_accel * 0.25)) < 1e-12);
}

TEST_CASE("Events: a chain that never ends is cut and reported", "[kernel][events]")
{
    // Both directions watched at 0.5 m, and an action that puts the ball
    // back 1 micrometre above that level, still falling, so it crosses
    // again some 3e-7 s later, and again. The step stops watching after
    // max_events_per_step events, warns once, and still ends at its time.
    // (Reflecting the height across the level instead dies out: the
    // crossings land so close to 0.5 that the reflection rounds to 0.5
    // itself, where the function has no sign.)
    System sys = falling_ball();
    Simulator sim(sys);
    sim.q(0) = 1.0;
    Event level = level_crossing("level", 0.5);
    level.direction = 0;
    level.action = [](Simulator& s) { s.q(0) = 0.5 + 1e-6; };
    sim.events.push_back(level);

    captured().clear();
    DiagnosticSink saved = diagnostic_sink();
    diagnostic_sink() = [](const std::string& m) { captured().push_back(m); };
    sim.run(0.5, 0.1);
    diagnostic_sink() = saved;

    CHECK(sim.event_log().size() == static_cast<std::size_t>(sim.max_events_per_step));
    REQUIRE(captured().size() == 1);
    CHECK(captured()[0].find("MBD-K052") != std::string::npos);
    CHECK(std::abs(sim.time - 0.5) < 1e-12);
}
