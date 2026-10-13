// Contact of points and spheres with a plane (plan task 3.8).
//
//   - the normal and friction laws' derivatives match central differences,
//     and the normal force is continuous where contact begins;
//   - a ball at rest on the ground sinks by (m g / k)^(1/e);
//   - a ball striking a free plate conserves the momentum of the pair;
//   - a block on an incline creeps below the friction angle, at the speed
//     the regularisation predicts, and above it slides with
//     a = g (sin th - mu_dynamic cos th).

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/contact.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

using namespace mbd;
using namespace mbd::kernel;

namespace {

/// A body on a free joint, its frame the free joint's.
int add_free_body(System& sys, const RigidBodyInertia& inertia, const char* name)
{
    return sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), inertia, name);
}

void place(Simulator& sim, int body, const Vec3& p, const Quat& R)
{
    const int iq = sim.system.model.idx_q[static_cast<std::size_t>(body)];
    sim.q.segment<3>(iq) = p;
    sim.q.segment<4>(iq + 3) = R.coeffs();
}

} // namespace

TEST_CASE("Contact: the laws' derivatives match central differences", "[forces][contact]")
{
    ContactParams p;
    p.stiffness = 2e5;
    p.exponent = 1.5;
    p.damping = 300.0;
    p.damping_depth = 2e-3;
    p.mu_static = 0.8;
    p.mu_dynamic = 0.6;
    p.v_static = 0.01;
    p.v_dynamic = 0.05;
    const PlaneContact contact(1, {Vec3::Zero()}, 0.05, 0, Transform3(), p);

    // Inside the damping ramp and beyond it, approaching and parting.
    for (const Real delta : {1e-4, 7e-4, 1.5e-3, 3e-3}) {
        for (const Real rate : {-0.05, 0.0, 0.2}) {
            const LawValue law = contact.normal_law(delta, rate);
            if (law.force <= 0.0) continue;   // parting fast: no pull
            const Real h = 1e-8, hv = 1e-6;
            const Real dd = (contact.normal_law(delta + h, rate).force - contact.normal_law(delta - h, rate).force) / (2 * h);
            const Real dv = (contact.normal_law(delta, rate + hv).force - contact.normal_law(delta, rate - hv).force) / (2 * hv);
            CHECK(std::abs(law.d_position - dd) < 1e-6 * std::abs(dd));
            CHECK(std::abs(law.d_rate - dv) < 1e-6 * (1.0 + std::abs(dv)));
        }
    }
    // The force starts from zero, whatever the rate.
    CHECK(contact.normal_law(1e-12, 1.0).force < 1e-6);
    CHECK(contact.normal_law(-1e-6, 1.0).force == 0.0);

    // Friction on its rising branch, its falling branch, and beyond.
    for (const Real v : {0.002, 0.006, 0.02, 0.04, 0.1}) {
        const Real h = 1e-8;
        const Real d = (contact.friction(v + h).force - contact.friction(v - h).force) / (2 * h);
        CHECK(std::abs(contact.friction(v).d_rate - d) < 1e-6 * (1.0 + std::abs(d)));
    }
    CHECK(contact.friction(0.0).force == 0.0);
    CHECK(contact.friction(0.01).force == 0.8);
    CHECK(contact.friction(0.2).force == 0.6);
}

TEST_CASE("Contact: a ball at rest on the ground sinks by (m g / k)^(1/e)", "[forces][contact]")
{
    // A 2 kg ball of radius 0.1 m on a vertical slider (gravity along -Z),
    // Hertzian contact k delta^1.5 with k = 1e6: at rest k delta^1.5 = m g.
    const Real m = 2.0, r = 0.1, k = 1e6;
    System sys;
    const int ball = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                        RigidBodyInertia::from_solid_box(m, Vec3(0.05, 0.05, 0.05)), "ball");
    ContactParams p;
    p.stiffness = k;
    p.exponent = 1.5;
    p.damping = 50.0;
    sys.force_elements.push_back(std::make_shared<PlaneContact>(
        ball, std::vector<Vec3>{Vec3::Zero()}, r, 0, Transform3(), p));
    Simulator sim(sys);
    sim.q(0) = r;   // touching

    const StaticReport report = static_equilibrium(sim);
    INFO(report.summary(sys.model));
    REQUIRE(report.converged);
    const Real sink = std::pow(m * g_accel / k, 1.0 / 1.5);
    CHECK(std::abs((r - sim.q(0)) - sink) < 1e-9);
}

TEST_CASE("Contact: a ball striking a free plate conserves the momentum of the pair", "[forces][contact]")
{
    // No gravity. The contact force acts on both bodies at one point, equal
    // and opposite, so the pair's linear momentum and its angular momentum
    // about any point are conserved, to the integrator's error: RK4 combines
    // body-frame velocities from stages at different orientations, so it
    // conserves momentum to its truncation error, about (w dt)^5 per step
    // with w = sqrt(k / m) = 470 rad/s and dt = 1e-5 s, 2e-12 over each of the
    // contact's 700 or so steps: 2e-9 in all.
    System sys;
    sys.model.gravity = Vec3::Zero();
    const int plate = add_free_body(sys, RigidBodyInertia::from_solid_box(5.0, Vec3(0.3, 0.3, 0.02)), "plate");
    const int ball = add_free_body(sys, RigidBodyInertia::from_solid_box(0.5, Vec3(0.04, 0.04, 0.04)), "ball");
    ContactParams p;
    p.stiffness = 1e5;
    p.damping = 20.0;
    p.mu_static = 0.3;
    p.mu_dynamic = 0.3;
    p.v_static = 1e-3;
    p.v_dynamic = 2e-3;
    // The plate's top face (z = 0.02 in its frame) is the plane.
    sys.force_elements.push_back(std::make_shared<PlaneContact>(
        ball, std::vector<Vec3>{Vec3::Zero()}, 0.05, plate, Transform3::FromTranslation(Vec3(0, 0, 0.02)), p));
    Simulator sim(sys);
    place(sim, plate, Vec3::Zero(), Quat::Identity());
    place(sim, ball, Vec3(0.1, 0.05, 0.08), Quat::Identity());
    // The ball comes down and sideways onto the plate, off its centre.
    const int bv = sys.model.idx_v[static_cast<std::size_t>(ball)];
    sim.v.segment<3>(bv + 3) = Vec3(0.4, 0.0, -1.0);
    sim.initialize();

    Data data(sys.model);
    auto momentum = [&](Vec3& linear, Vec3& angular) {
        forward_kinematics(sys.model, data, sim.q, sim.v);
        const Vec6 h = momentum_world(sys.model, data);
        angular = h.head<3>();
        linear = h.tail<3>();
    };
    Vec3 L0, H0, L1, H1;
    momentum(L0, H0);
    sim.run(0.03, 1e-5);
    momentum(L1, H1);
    // It did bounce: the ball moves up afterwards.
    CHECK(sim.states()[static_cast<std::size_t>(ball)].v_WB.z() > 0.0);
    CHECK((L1 - L0).norm() < 1e-8 * L0.norm());
    CHECK((H1 - H0).norm() < 1e-8 * (H0.norm() + L0.norm()));
}

TEST_CASE("Contact: a block on an incline creeps below the friction angle and slides above it",
          "[forces][contact]")
{
    // A 1 kg block, 0.2 x 0.2 x 0.1 m, touching the incline at its four bottom
    // corners (k = 1e5 N/m each, linear; c = 200 N s/m over 1e-5 m), friction
    // mu_static = 0.5 at 1 mm/s, mu_dynamic = 0.4 from 1 cm/s. The friction
    // angle is atan(0.5) = 26.57 degrees.
    //
    // 1 degree below it the block cannot slide: friction holds it once the
    // coefficient reaches tan(th) = 0.4826, on the rising branch, where the
    // cubic step gives the creep speed v_c = s v_static with s^2 (3 - 2 s) =
    // tan(th) / mu_static (s = 0.893: v_c = 0.89 mm/s, below v_static).
    //
    // 1 degree above, friction cannot hold it: it slides, and once past
    // v_dynamic its acceleration is g (sin th - mu_dynamic cos th), the normal
    // force being m g cos th in steady sliding.
    const Real m = 1.0, mu_s = 0.5, mu_d = 0.4, v_s = 1e-3, v_d = 1e-2, k = 1e5;
    const Real friction_angle = std::atan(mu_s);

    auto slide = [&](Real th, Real duration, std::vector<Real>& speeds, Real dt_sample) {
        System sys;
        const int block = add_free_body(sys, RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.05)), "block");
        ContactParams p;
        p.stiffness = k;
        p.damping = 200.0;
        p.damping_depth = 1e-5;
        p.mu_static = mu_s;
        p.mu_dynamic = mu_d;
        p.v_static = v_s;
        p.v_dynamic = v_d;
        const Quat R(Eigen::AngleAxisd(th, Vec3::UnitY()));
        std::vector<Vec3> corners;
        for (const Real x : {-0.1, 0.1}) {
            for (const Real y : {-0.1, 0.1}) corners.emplace_back(x, y, -0.05);
        }
        sys.force_elements.push_back(std::make_shared<PlaneContact>(
            block, corners, 0.0, 0, Transform3(R, Vec3::Zero()), p));
        Simulator sim(sys);
        // Aligned with the incline, each corner pressed in by its share of
        // m g cos(th).
        const Vec3 n = R * Vec3::UnitZ(), downhill = R * Vec3::UnitX();
        const Real delta0 = m * g_accel * std::cos(th) / (4.0 * k);
        place(sim, block, (0.05 - delta0) * n, R);
        sim.initialize();
        const int steps = static_cast<int>(std::round(duration / 1e-4));
        const int every = static_cast<int>(std::round(dt_sample / 1e-4));
        speeds.clear();
        for (int i = 1; i <= steps; ++i) {
            sim.step(1e-4);
            if (i % every == 0) speeds.push_back(sim.states()[static_cast<std::size_t>(block)].v_WB.dot(downhill));
        }
    };

    SECTION("1 degree below: it creeps at the regularisation's speed")
    {
        const Real th = friction_angle - pi / 180.0;
        Real lo = 0.0, hi = 1.0;
        for (int i = 0; i < 100; ++i) {
            const Real s = 0.5 * (lo + hi);
            if (s * s * (3.0 - 2.0 * s) < std::tan(th) / mu_s) lo = s;
            else hi = s;
        }
        const Real v_creep = 0.5 * (lo + hi) * v_s;
        std::vector<Real> speeds;
        slide(th, 0.5, speeds, 0.01);
        Real fastest = 0.0;
        for (const Real v : speeds) fastest = std::max(fastest, std::abs(v));
        INFO("creep speed " << speeds.back() << " m/s, predicted " << v_creep << ", fastest " << fastest);
        CHECK(fastest < v_s);
        CHECK(std::abs(speeds.back() - v_creep) < 1e-3 * v_creep);
    }
    SECTION("1 degree above: it slides with a = g (sin th - mu_dynamic cos th)")
    {
        const Real th = friction_angle + pi / 180.0;
        std::vector<Real> speeds;
        slide(th, 0.5, speeds, 0.1);   // speeds at 0.1, 0.2, ..., 0.5 s
        REQUIRE(speeds.size() == 5);
        REQUIRE(speeds[1] > v_d);      // past v_dynamic from 0.2 s on
        const Real a = (speeds[4] - speeds[1]) / 0.3;
        const Real expected = g_accel * (std::sin(th) - mu_d * std::cos(th));
        INFO("acceleration " << a << " m/s^2, expected " << expected);
        CHECK(std::abs(a - expected) < 1e-4 * expected);
    }
}
