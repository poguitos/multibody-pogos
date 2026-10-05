// Events (plan task 3.9).
//
//   - a bouncing ball (the plan's "done when"): impact times against the
//     closed form, whatever the step;
//   - a pendulum stopped by a terminal event at the bottom of its swing, a
//     quarter period from release (the complete elliptic integral);
//   - directions: rising and falling crossings of a driven coordinate, and
//     two events inside one step, found in order;
//   - switched friction at rest chatters, and is reported once;
//   - an event without a function is refused.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

/// A ball on a vertical slider (Z, gravity -Z) of radius r above a floor at
/// z = 0: the event is the contact, falling; the action the rebound with
/// restitution e.
System ball_on_slider()
{
    System sys;
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)), "ball");
    return sys;
}

Event bounce(Real r, Real e)
{
    Event ev;
    ev.name = "impact";
    ev.function = [r](const Simulator& sim) { return sim.q(0) - r; };
    ev.direction = -1;
    ev.action = [e](Simulator& sim) { sim.v(0) = -e * sim.v(0); };
    return ev;
}

std::vector<std::string>& captured()
{
    static std::vector<std::string> w;
    return w;
}

/// The complete elliptic integral of the first kind, K(k), by the
/// arithmetic-geometric mean: K = pi / (2 AGM(1, sqrt(1 - k^2))).
Real elliptic_K(Real k)
{
    Real a = 1.0, b = std::sqrt(1.0 - k * k);
    for (int n = 0; n < 40; ++n) {
        const Real an = 0.5 * (a + b);
        b = std::sqrt(a * b);
        a = an;
    }
    return 0.5 * pi / a;
}

} // namespace

TEST_CASE("Events: a bouncing ball's impacts at their exact times", "[kernel][events]")
{
    // Dropped from h0 above the floor, the ball lands at t1 = sqrt(2 h0 / g)
    // with speed v1 = g t1, and after the k-th impact flies for 2 e^k v1 / g.
    // Free flight under gravity is quadratic in time, which RK4 integrates
    // exactly; the only errors are the event's placement, at most
    // event_tolerance (1e-10 s) after the crossing, and what it does to the
    // next flight: a rebound speed late by g x 1e-10, a flight longer by
    // 2 e x 1e-10. Per impact at most (1 + 2e) x 1e-10 = 2.6e-10 s, so after
    // ten impacts 2.6e-9 s: bound 1e-8. The step does not matter.
    const Real h0 = 1.0, r = 0.1, e = 0.8;
    const Real t1 = std::sqrt(2.0 * h0 / g_accel), v1 = g_accel * t1;
    std::vector<Real> expected{t1};
    for (int k = 1; k < 10; ++k) expected.push_back(expected.back() + 2.0 * std::pow(e, k) * v1 / g_accel);

    for (const Real dt : {0.01, 0.0137, 0.001}) {
        System sys = ball_on_slider();
        Simulator sim(sys);
        sim.q(0) = r + h0;
        sim.events.push_back(bounce(r, e));
        sim.run(3.7, dt);   // the tenth impact is at 3.58 s
        INFO("dt = " << dt);
        REQUIRE(sim.event_log().size() >= 10);
        for (int k = 0; k < 10; ++k) {
            CHECK(std::abs(sim.event_log()[static_cast<std::size_t>(k)].time - expected[static_cast<std::size_t>(k)]) <= 1e-8);
        }
        // run() ends on its time grid, events or not.
        CHECK(std::abs(sim.time - dt * std::round(3.7 / dt)) <= 1e-12);
    }
}

TEST_CASE("Events: a terminal event stops a pendulum at the bottom of its swing", "[kernel][events]")
{
    // A compound pendulum released from 1 rad reaches the bottom (angle 0) a
    // quarter period later: K(sin(th0 / 2)) / w0, w0^2 = m g L / I. RK4 at
    // 1 ms errs by about (w0 dt)^4 relative per unit of phase, below 1e-9;
    // the event's placement by 1e-10 s. Bound 1e-8 s.
    const Real m = 1.5, L = 0.6, th0 = 1.0;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, Vec3(0.03, 0.03, 0.03));
    I.com_B = Vec3(0.0, -L, 0.0);
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), I);
    Simulator sim(sys);
    sim.q(0) = th0;

    Event bottom;
    bottom.name = "bottom";
    bottom.function = [](const Simulator& s) { return s.q(0); };
    bottom.direction = -1;
    bottom.stop = true;
    sim.events.push_back(bottom);

    const int steps = sim.run(2.0, 1e-3);
    const Real w0 = std::sqrt(m * g_accel * L / (I.I_com_B(2, 2) + m * L * L));
    const Real quarter = elliptic_K(std::sin(0.5 * th0)) / w0;
    CHECK(sim.stopped());
    CHECK(steps < 2000);
    REQUIRE(sim.event_log().size() == 1);
    CHECK(std::abs(sim.time - quarter) <= 1e-8);
    CHECK(std::abs(sim.event_log()[0].time - sim.time) == 0.0);
    CHECK(sim.q(0) <= 0.0);   // just past the crossing
}

TEST_CASE("Events: directions, and two events in one step found in order", "[kernel][events]")
{
    // A slider driven along s(t) = sin(2 pi t). Event "half", rising only,
    // at q = 0.5: t = 1/12 and 1 + 1/12. Event "ninety", both ways, at
    // q = 0.9: t = a, 1/2 - a and 1 + a, with a = asin(0.9) / (2 pi). With
    // steps of 0.2 s, [0, 0.2] and [1, 1.2] each hold two events, which must
    // come out in order; half's falling crossing at 5/12 is left out. The
    // driver holds q = s(t) at every time the step is evaluated at
    // (projection tolerance 1e-10, a time error of 1e-10 / |s'| below 3e-11 s),
    // so the times are right to the event tolerance: bound 1e-9 s.
    //
    // Detection compares the signs at the ends of a step, so two crossings of
    // one function inside one step (q rising through 0.9 and falling back)
    // would go unseen; the steps here are shorter than any such pair.
    System sys;
    const int b = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                     RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)));
    const Real w = 2.0 * pi;
    sys.constraints.push_back(std::make_shared<JointDriver>(
        sys.model, b,
        TimeFunction([w](Real t) { return std::sin(w * t); }, [w](Real t) { return w * std::cos(w * t); },
                     [w](Real t) { return -w * w * std::sin(w * t); })));
    Simulator sim(sys);
    sim.initialize();

    Event half, ninety;
    half.name = "half";
    half.function = [](const Simulator& s) { return s.q(0) - 0.5; };
    half.direction = 1;
    ninety.name = "ninety";
    ninety.function = [](const Simulator& s) { return s.q(0) - 0.9; };
    ninety.direction = 0;
    sim.events = {half, ninety};

    sim.run(1.2, 0.2);
    const Real a = std::asin(0.9) / w;
    const std::vector<std::pair<std::size_t, Real>> expected{
        {0, 1.0 / 12.0}, {1, a}, {1, 0.5 - a}, {0, 1.0 + 1.0 / 12.0}, {1, 1.0 + a}};
    REQUIRE(sim.event_log().size() == expected.size());
    for (std::size_t k = 0; k < expected.size(); ++k) {
        CHECK(sim.event_log()[k].event == expected[k].first);
        CHECK(std::abs(sim.event_log()[k].time - expected[k].second) <= 1e-9);
    }
}

TEST_CASE("Events: switched friction at rest chatters, and is reported once", "[kernel][events]")
{
    // A slider pushed by P against Coulomb friction F0 > P, the friction's
    // direction switched by an event when the velocity crosses zero. Once the
    // slider stops, every switch reverses the net force (P - F0, then
    // P + F0), the velocity crosses zero again at once, and the events pile
    // up a tolerance apart: chattering, the reason friction is regularised
    // (task 3.8). After 100 events in one step the rest of the step is taken
    // without looking, and that is reported once.
    // P - F0 = -7 stops 0.1 m/s at 1/70 s, between steps of 0.01 s.
    const Real F0 = 10.0, P = 3.0;
    System sys;
    sys.model.gravity = Vec3::Zero();
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)));
    Simulator sim(sys);
    sim.v(0) = 0.1;
    auto direction = std::make_shared<Real>(-1.0);   // friction along -v
    sim.force_callback = [=](Simulator&, Real, VecX& tau) { tau(0) += P + *direction * F0; };
    Event zero_speed;
    zero_speed.name = "zero speed";
    zero_speed.function = [](const Simulator& s) { return s.v(0); };
    zero_speed.action = [=](Simulator&) { *direction = -*direction; };   // the motion reversed
    sim.events.push_back(zero_speed);

    captured().clear();
    const DiagnosticSink previous = diagnostic_sink();
    diagnostic_sink() = [](const std::string& m) { captured().push_back(m); };
    sim.run(0.1, 0.01);
    diagnostic_sink() = previous;
    REQUIRE(captured().size() == 1);
    CHECK_THAT(captured()[0], ContainsSubstring("MBD-K091"));
    CHECK(sim.event_log().size() >= 100);
    // The first stop is where the net force P - F0 brings 0.1 m/s to rest.
    CHECK(std::abs(sim.event_log()[0].time - 0.1 / (F0 - P)) <= 1e-9);
    CHECK(std::abs(sim.time - 0.1) <= 1e-12);
}

TEST_CASE("Events: an event without a function is refused", "[kernel][events]")
{
    System sys = ball_on_slider();
    Simulator sim(sys);
    Event empty;
    empty.name = "nothing";
    sim.events.push_back(empty);
    CHECK_THROWS_WITH(sim.step(0.01), ContainsSubstring("MBD-K090"));
}
