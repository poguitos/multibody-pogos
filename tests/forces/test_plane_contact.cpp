// Contact of points and spheres against a plane (plan task 3.8).
//
//   - the normal and friction laws against central differences, and the
//     penalty spring against its potential;
//   - a Hertzian sphere at rest sinks to (m g / k)^(1/e), found by statics,
//     free to roll and slide;
//   - a sphere dropped without damping rebounds to the height it fell from;
//   - a block on an incline (the plan's "done when"): one degree above the
//     friction angle atan(mu) it slides at g (sin th - mu cos th); one degree
//     below it creeps at the regularisation's speed v_s atanh(tan th / mu);
//   - a plane on a moving body takes the reaction: a ball on a sprung plate;
//   - malformed parameters are refused.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/plane_contact.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

Real relative(Real a, Real b) { return std::abs(a - b) / std::abs(b); }

bool has_code(const std::vector<std::string>& messages, const std::string& code)
{
    for (const auto& m : messages) {
        if (m.rfind(code, 0) == 0) return true;
    }
    return false;
}

} // namespace

TEST_CASE("Plane contact: the laws match their derivatives and potential", "[forces][contact]")
{
    PlaneContactParams p;
    p.stiffness = 2e5;
    p.exponent = 1.5;
    p.damping = 300.0;
    p.damping_depth = 1e-3;
    p.friction = 0.7;
    p.slip_speed = 2e-3;
    const PlaneContact pc(1, {ContactSphere{}}, p);

    // Depths inside the damper's ramp and beyond it, rates both ways (but not
    // so fast that the force is clamped at zero, where it has a kink: at
    // 0.2 mm the spring gives 0.57 N, so separating at 0.05 m/s would already
    // be clamped; at 0.005 m/s the damper takes 0.16 N of it). Central
    // differences of step h: truncation h^2 / 6 of the third derivative,
    // roundoff eps |F| / h; 1e-6 relative covers both.
    const Real h = 1e-8;
    for (const Real d : {2e-4, 5e-4, 9e-4, 3e-3}) {
        for (const Real rate : {-0.005, 0.0, 0.2}) {
            const LawValue v = pc.normal_law(d, rate);
            REQUIRE(v.force > 0.0);
            const Real dF_dd = (pc.normal_law(d + h, rate).force - pc.normal_law(d - h, rate).force) / (2.0 * h);
            const Real dF_dr = (pc.normal_law(d, rate + 1e-6).force - pc.normal_law(d, rate - 1e-6).force) / 2e-6;
            CHECK(std::abs(v.d_position - dF_dd) <= 1e-6 * std::abs(dF_dd) + 1e-6);
            CHECK(std::abs(v.d_rate - dF_dr) <= 1e-6 * std::abs(dF_dr) + 1e-9);
            // The spring part is -dV/d(height) = dV/d(depth).
            const Real dV = (pc.potential_energy(d + h) - pc.potential_energy(d - h)) / (2.0 * h);
            CHECK(relative(dV, p.stiffness * std::pow(d, p.exponent)) <= 1e-6);
        }
    }
    // Out of contact, nothing; separating fast, never a pull.
    CHECK(pc.normal_law(-1e-3, -1.0).force == 0.0);
    CHECK(pc.normal_law(2e-4, -1e3).force == 0.0);

    // Friction: its slope against central differences, step 1e-9 against a
    // slip speed of 2e-3 (relative truncation 1e-13, roundoff 1e-7 / 3e4);
    // at most mu F_n, and mu F_n once sliding well above the slip speed.
    for (const Real slip : {5e-4, 1e-3, 4e-3}) {
        const LawValue f = pc.friction_law(slip, 50.0);
        const Real df = (pc.friction_law(slip + 1e-9, 50.0).force - pc.friction_law(slip - 1e-9, 50.0).force) / 2e-9;
        CHECK(std::abs(f.d_rate - df) <= 1e-6 * f.d_rate);
        CHECK(f.force < p.friction * 50.0);
    }
    CHECK(pc.friction_law(0.0, 50.0).force == 0.0);
    CHECK(relative(pc.friction_law(0.1, 50.0).force, p.friction * 50.0) <= 1e-15);
}

TEST_CASE("Plane contact: a Hertzian sphere at rest sinks to (m g / k)^(1/e)", "[forces][contact]")
{
    // A free sphere on the ground plane, Hertz's exponent 1.5. At rest only
    // the vertical direction has stiffness; it may roll and slide freely, and
    // spin about its contact: five directions without stiffness.
    const Real m = 2.0, r = 0.1, k = 5e6, e = 1.5;
    System sys;
    const int ball = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                        RigidBodyInertia::from_solid_box(m, Vec3(0.07, 0.07, 0.07)));
    PlaneContactParams p;
    p.stiffness = k;
    p.exponent = e;
    p.friction = 0.8;
    sys.force_elements.push_back(std::make_shared<PlaneContact>(ball, std::vector<ContactSphere>{{Vec3::Zero(), r}}, p));
    Simulator sim(sys);
    sim.q(2) = r - 1e-4;

    const StaticsReport rep = static_equilibrium(sim);
    INFO(rep.summary(sys.model));
    REQUIRE(rep.converged);
    const Real depth = std::pow(m * g_accel / k, 1.0 / e);
    // |a| <= 1e-6 leaves the depth off by m |a| / (dF/dd), dF/dd = e k d^(e-1).
    const Real bound = m * 1e-6 / (e * k * std::pow(depth, e - 1.0));
    CHECK(std::abs((r - sim.q(2)) - depth) <= bound);
    CHECK(rep.neutral_directions == 5);
    CHECK(has_code(rep.notes, "MBD-K073"));
}

TEST_CASE("Plane contact: an undamped sphere rebounds to the height it fell from", "[forces][contact]")
{
    // Dropped from 0.1 m onto a Hertzian contact without damping or friction:
    // the contact stores and returns the energy, so the ball comes back to the
    // same height. RK4 at 1e-5 s is accurate away from first touch; across it,
    // where the force k d^1.5 has no second derivative, one step travels
    // 1.4e-5 m, where the force is at most k (1.4e-5)^1.5 = 0.05 N: an impulse
    // error of at most 5e-7 N s against a momentum of 1.4 N s, so the height
    // is right to about 1e-6 relative; the bound is 1e-5.
    const Real m = 1.0, r = 0.05, k = 1e6, h0 = 0.1;
    System sys;
    const int ball = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                        RigidBodyInertia::from_solid_box(m, Vec3(0.03, 0.03, 0.03)));
    PlaneContactParams p;
    p.stiffness = k;
    p.exponent = 1.5;
    auto contact = std::make_shared<PlaneContact>(ball, std::vector<ContactSphere>{{Vec3::Zero(), r}}, p);
    sys.force_elements.push_back(contact);
    Simulator sim(sys);
    sim.q(2) = r + h0;

    Real deepest = 0.0, apex_after = 0.0;
    bool bounced = false;
    const Real dt = 1e-5;
    for (int n = 0; n < 40000; ++n) {
        sim.step(dt);
        const Real z = sim.q(2);
        deepest = std::max(deepest, r - z);
        const Real vz = sim.states()[static_cast<std::size_t>(ball)].v_WB.z();
        if (deepest > 0.0 && vz > 0.0) bounced = true;
        if (bounced) apex_after = std::max(apex_after, z - r);
    }
    REQUIRE(bounced);
    // Energy balance at the deepest point: m g (h0 + d) = k d^2.5 / 2.5. The
    // deepest point is sampled every step; at rest there, the sample misses
    // it by at most a dt^2 / 2 = 2e-8 m (a = k d^1.5 / m = 450 m/s^2), 4e-6 of
    // d and 1e-5 of the energy: bound 1e-4.
    CHECK(relative(m * g_accel * (h0 + deepest), contact->potential_energy(deepest)) <= 1e-4);
    CHECK(relative(apex_after, h0) <= 1e-5);
}

TEST_CASE("Plane contact: a block on an incline slides at the friction angle", "[forces][contact]")
{
    // A 10 kg block on four corner points, friction mu = 0.5, on a plane
    // tilted by th (gravity tilted instead: g (-sin th, 0, -cos th)). Coulomb's
    // law gives the friction angle atan(mu): steeper, the block slides with
    // a = g (sin th - mu cos th); shallower, it sticks. With friction
    // regularised over v_s, sticking becomes a steady creep at the speed where
    // mu tanh(v / v_s) = tan th: v = v_s atanh(tan th / mu).
    //
    // The block starts at the static depth of its corners, at rest. Sliding,
    // after 0.5 s its speed is 100 v_s and tanh differs from 1 by e^-200: the
    // friction is Coulomb's, and the acceleration measured from 0.5 to 1 s is
    // exact but for RK4's error with the contact's damped pitch long settled;
    // creeping, the speed is steady. Both to 1e-6 relative.
    const Real m = 10.0, k = 1e5, c = 1000.0, mu = 0.5, vs = 1e-3;
    const Real hx = 0.2, hy = 0.15, hz = 0.1;

    auto run = [&](Real th, Real& v_half, Real& v_end) {
        System sys;
        sys.model.gravity = g_accel * Vec3(-std::sin(th), 0.0, -std::cos(th));
        const int block = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                             RigidBodyInertia::from_solid_box(m, Vec3(hx, hy, hz)));
        PlaneContactParams p;
        p.stiffness = k;
        p.damping = c;               // critical for one corner's quarter of the mass
        p.damping_depth = 1e-4;      // full at the static depth, 2e-4 m
        p.friction = mu;
        p.slip_speed = vs;
        std::vector<ContactSphere> corners;
        for (const Real sx : {-1.0, 1.0}) {
            for (const Real sy : {-1.0, 1.0}) corners.push_back({Vec3(sx * hx, sy * hy, -hz), 0.0});
        }
        sys.force_elements.push_back(std::make_shared<PlaneContact>(block, corners, p));
        Simulator sim(sys);
        sim.q(2) = hz - m * g_accel * std::cos(th) / (4.0 * k);
        // RK4 is stable while dt x (mu F_n / v_s) / (m / 4) < 2.8: dt < 5.6e-4.
        const Real dt = 2e-4;
        for (int n = 1; n <= 5000; ++n) {
            sim.step(dt);
            if (n == 2500) v_half = sim.states()[static_cast<std::size_t>(block)].v_WB.x();
        }
        v_end = sim.states()[static_cast<std::size_t>(block)].v_WB.x();
    };

    const Real friction_angle = std::atan(mu);
    const Real degree = pi / 180.0;
    Real v_half = 0.0, v_end = 0.0;

    SECTION("one degree steeper: it slides")
    {
        const Real th = friction_angle + degree;
        run(th, v_half, v_end);
        const Real a = -g_accel * (std::sin(th) - mu * std::cos(th));   // down the slope is -X
        CHECK(relative((v_end - v_half) / 0.5, a) <= 1e-6);
    }

    SECTION("one degree shallower: it creeps at the regularisation's speed")
    {
        const Real th = friction_angle - degree;
        run(th, v_half, v_end);
        const Real creep = -vs * std::atanh(std::tan(th) / mu);
        CHECK(relative(v_end, creep) <= 1e-6);
        CHECK(relative(v_half, creep) <= 1e-6);
    }
}

TEST_CASE("Plane contact: a plane on a moving body takes the reaction", "[forces][contact]")
{
    // A plate on a vertical slider over a spring (k_p, free length L0), a ball
    // resting on its top face. The plate's spring carries both weights:
    // its compression is (m_p + m_b) g / k_p; the ball sinks into the plate by
    // m_b g / k_c.
    const Real mp = 5.0, mb = 1.0, kp = 2e4, L0 = 0.5, kc = 1e6, r = 0.05, t = 0.02;
    System sys;
    const int plate = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                         RigidBodyInertia::from_solid_box(mp, Vec3(0.3, 0.3, t)), "plate");
    const int ball = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                        RigidBodyInertia::from_solid_box(mb, Vec3(0.03, 0.03, 0.03)), "ball");
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0, plate, Vec3::Zero(), Vec3::Zero(), kp, 0.0, L0));
    PlaneContactParams p;
    p.stiffness = kc;
    p.friction = 0.5;
    // The plate's top face: its XY plane moved up by its half thickness.
    sys.force_elements.push_back(std::make_shared<PlaneContact>(
        ball, std::vector<ContactSphere>{{Vec3::Zero(), r}}, p, plate, Transform3::FromTranslation(Vec3(0.0, 0.0, t))));
    Simulator sim(sys);
    sim.q(0) = L0;
    sim.q(1 + 2) = L0 + t + r;   // the ball's free joint: translation first

    const StaticsReport rep = static_equilibrium(sim);
    INFO(rep.summary(sys.model));
    REQUIRE(rep.converged);
    // |a| <= 1e-6: positions off by at most (m_p + m_b) 1e-6 / k_p = 3e-10 m.
    CHECK(std::abs(sim.q(0) - (L0 - (mp + mb) * g_accel / kp)) <= 1e-9);
    const Real depth = (sim.q(0) + t + r) - sim.q(3);
    CHECK(std::abs(depth - mb * g_accel / kc) <= 1e-9);
}

TEST_CASE("Plane contact: malformed parameters are refused", "[forces][contact]")
{
    PlaneContactParams p;
    p.exponent = 0.5;
    CHECK_THROWS_WITH(PlaneContact(1, {ContactSphere{}}, p), ContainsSubstring("MBD-F040"));
    p = PlaneContactParams{};
    p.slip_speed = 0.0;
    CHECK_THROWS_WITH(PlaneContact(1, {ContactSphere{}}, p), ContainsSubstring("MBD-F040"));
    CHECK_THROWS_WITH(PlaneContact(1, {}, PlaneContactParams{}), ContainsSubstring("MBD-F041"));
    CHECK_THROWS_WITH(PlaneContact(1, {ContactSphere{Vec3::Zero(), -0.1}}, PlaneContactParams{}),
                      ContainsSubstring("MBD-F041"));
}
