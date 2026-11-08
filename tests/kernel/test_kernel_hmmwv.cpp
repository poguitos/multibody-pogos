// The HMMWV benchmark of Haug, Negrut, Serban and Solis (1999) (plan task
// 5.4): the fourteen-body model of kernel/hmmwv.hpp, with joints and with the
// paper's bushings.
//
//   - fourteen bodies; ten degrees of freedom, eighteen with the bushings, as
//     the paper counts them;
//   - at rest on flat ground the tyres carry the vehicle's weight and its
//     moment, the arms rest at their design angles (the springs' free
//     lengths were chosen for that), and the bushings move the rest by less
//     than their compliance allows;
//   - the front tyres turn to the bump when the model says;
//   - over both bumps, an explicit method (RK4) converges to the implicit
//     one (Radau IIA) at its order;
//   - with the bushings, whose fastest mode is -2.06e6 s^-1, RK4 needs a step
//     below 2.785 / 2.06e6 = 1.35e-6 s: inside it, it agrees with Radau IIA
//     over the start of a bump, and at twice it, it fails within a
//     millisecond; the implicit methods agree with each other at steps a
//     thousand times longer.
//
// The costs of every method, over the paper's 5 s, are measured by
// bench/hmmwv.cpp (mbd_hmmwv) and tabulated in docs/hmmwv.md.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/validate.hpp"

#include "kernel/hmmwv.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::Hmmwv;

namespace {

Real total_mass(const Model& model)
{
    Real m = 0.0;
    for (int i = 1; i < model.nbodies(); ++i) m += model.inertia[static_cast<std::size_t>(i)].mass;
    return m;
}

/// The whole vehicle's centre of mass.
Vec3 centre_of_mass(const Simulator& sim)
{
    const Model& model = sim.system.model;
    Vec3 c = Vec3::Zero();
    for (int i = 1; i < model.nbodies(); ++i) {
        const auto& s = sim.states()[static_cast<std::size_t>(i)];
        const auto& I = model.inertia[static_cast<std::size_t>(i)];
        c += I.mass * (s.p_WB + s.q_WB * I.com_B);
    }
    return c / total_mass(model);
}

/// Every body's origin, sampled every `every` seconds for `samples` samples.
std::vector<std::vector<Vec3>> trajectory(Simulator& sim, Real every, int samples, Real dt)
{
    std::vector<std::vector<Vec3>> path;
    for (int k = 0; k < samples; ++k) {
        sim.run(every, dt);
        std::vector<Vec3> p;
        for (const auto& s : sim.states()) p.push_back(s.p_WB);
        path.push_back(p);
    }
    return path;
}

Real largest_difference(const std::vector<std::vector<Vec3>>& a, const std::vector<std::vector<Vec3>>& b)
{
    Real d = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k)
        for (std::size_t i = 0; i < a[k].size(); ++i) d = std::max(d, (a[k][i] - b[k][i]).norm());
    return d;
}

} // namespace

TEST_CASE("HMMWV: fourteen bodies, ten degrees of freedom, eighteen with bushings", "[kernel][hmmwv]")
{
    // The paper: 98 Cartesian coordinates (14 bodies x 7) under 88
    // constraints leave 10 with the steering held; the bushed model has 19,
    // 18 with the rack locked. Here: a free chassis (6), the rack (1), and at
    // each corner two arms (1 each, or 3 for the bushed upper arm) and the
    // spindle (3), closed by an upper ball joint (3) and a tie rod (1) each,
    // and the steering driver (1): 27 - 17 = 10, and 35 - 17 = 18.
    for (const bool bushed : {false, true}) {
        Hmmwv car(bushed);
        const ValidationReport v = validate(car.sys);
        INFO(v.summary());
        CHECK(v.ok());
        CHECK(v.warnings.empty());
        CHECK(v.bodies == 14);
        CHECK(v.velocities == (bushed ? 35 : 27));
        CHECK(v.constraint_equations == 17);
        CHECK(v.redundant_equations == 0);
        CHECK(v.degrees_of_freedom == (bushed ? 18 : 10));
    }
}

TEST_CASE("HMMWV: at rest the tyres carry the weight and the arms sit at their design angles", "[kernel][hmmwv]")
{
    // At rest on flat ground every tyre pushes straight up, k (R - z) with z
    // its wheel centre's height. Their sum must be the weight and their
    // moments about the vehicle's centre of mass zero, to within what the
    // static tolerance leaves: accelerations of 1e-8, over 2,400 kg, 2.4e-5 N
    // (bounds 1e-3 N and 1e-3 N m). The springs' free lengths (kFreeLength)
    // were found so that the lower arms rest at their design angles (zero),
    // to 1e-15 rad at a tolerance of 1e-11; at 1e-8 the angles may move by
    // the residual over the arms' stiffness, 1e-8 / 100 s^-2 (bound 1e-8 rad).
    // The bushings carry the upper arms' joint loads, below a corner's load
    // (about 6 kN), so at 2e7 N/m they give way by less than 3e-4 m: the
    // chassis rests no further from where it rests on joints, and the arms,
    // at least 0.45 m long, turn by less than 7e-4 rad (seen: 2.3e-5 m, and
    // 2.7e-6 and 7.9e-6 rad).
    std::vector<Real> height;
    for (const bool bushed : {false, true}) {
        Hmmwv car(bushed);
        Simulator sim(car.sys);
        car.design(sim);
        StaticOptions options;
        options.tolerance = 1e-8;
        const StaticReport r = static_equilibrium(sim, options);
        INFO(r.summary(car.sys.model));
        REQUIRE(r.converged);
        CHECK(r.relaxation_runs == 0);
        CHECK(r.supported == (bushed ? 15 : 7));
        CHECK(r.neutral == 3);   // along the road, across it, and the heading

        const Real weight = total_mass(car.sys.model) * g_accel;
        const Vec3 G = centre_of_mass(sim);
        Real lift = 0.0, pitch = 0.0, roll = 0.0;
        for (std::size_t k = 0; k < 4; ++k) {
            const Vec3 c = car.wheel_centre(sim, k);
            const Real f = Hmmwv::kTyreStiffness * (Hmmwv::kTyreRadius - c.z());
            lift += f;
            pitch += f * (c.x() - G.x());
            roll += f * (c.y() - G.y());
        }
        CHECK(std::abs(lift - weight) < 1e-3);
        CHECK(std::abs(pitch) < 1e-3);
        CHECK(std::abs(roll) < 1e-3);

        const Model& m = car.sys.model;
        for (const auto& c : car.corners) {
            INFO("lower arm angle " << sim.q(m.idx_q[static_cast<std::size_t>(c.lca)]) << " rad");
            CHECK(std::abs(sim.q(m.idx_q[static_cast<std::size_t>(c.lca)])) < (bushed ? 7e-4 : 1e-8));
            if (!bushed) CHECK(std::abs(sim.q(m.idx_q[static_cast<std::size_t>(c.uca)])) < 1e-8);
        }
        height.push_back(car.chassis_com(sim).z());
    }
    INFO("chassis height on joints " << height[0] << " m, on bushings " << height[1] << " m");
    CHECK(std::abs(height[1] - height[0]) < 3e-4);
}

TEST_CASE("HMMWV: the front tyres turn to the bump at the time asked for", "[kernel][hmmwv]")
{
    // start() puts the bump where the front tyres turn to it at t = 0.5 s, the
    // vehicle coasting at 10 mph from rest on its tyres. Before the bump the
    // motion is uniform but for what the static residual leaves (1e-8 m/s^2:
    // 1e-9 m over 0.5 s, 3e-10 s at 4.47 m/s); the event is located to 1e-10 s.
    Hmmwv car;
    Simulator sim(car.sys);
    (void)car.start(sim, 0.5, true);
    sim.run(0.6, 1e-3);
    const std::string front = "front left tyre: surface";
    Real first = -1.0;
    for (const auto& e : sim.event_log()) {
        if (sim.events[e.event].name == front) {
            first = e.time;
            break;
        }
    }
    INFO("the front left tyre turned to the bump at " << first << " s");
    CHECK(std::abs(first - 0.5) < 1e-8);
}

TEST_CASE("HMMWV: an explicit and an implicit method agree over both bumps", "[kernel][hmmwv]")
{
    // The model with joints over its first 1.5 s, both bumps (the rear tyres
    // turn to it at 1.26 s), without events: every body's origin every 20 ms.
    // Radau IIA at a tolerance of 1e-8 is the reference (over 5 s within
    // 2.3e-9 m of Radau at 1e-10). Dormand-Prince, explicit, must converge to it: from a
    // tolerance of 1e-6 to 1e-8 its difference must fall at least tenfold
    // (over 5 s the benchmark measured 5.2e-5 and 3.8e-7 m against Radau at
    // 1e-10), and at 1e-8 be within 1e-6 m. RK4 at fixed steps converges
    // only slowly here, because the tyres' forces jump where they turn to
    // the bump (by about 7 kN through the damper) and the steps straddle the
    // jumps: over 5 s, 4.4e-3 m at 1 ms and 1.8e-3 m at 0.5 ms (benchmark).
    // From 1 ms to 0.5 ms its difference must fall, and at 0.5 ms be within
    // 3e-3 m.
    auto path = [](Integrator method, Real parameter) {
        Hmmwv car;
        Simulator sim(car.sys);
        (void)car.start(sim, 0.5, false);
        sim.method = method;
        const bool adaptive = method == Integrator::DormandPrince || method == Integrator::RadauIIA;
        if (adaptive) {
            sim.step_control.relative_tolerance = parameter;
            sim.step_control.absolute_tolerance = parameter;
        }
        return trajectory(sim, 0.02, 75, adaptive ? 0.02 : parameter);
    };
    const auto reference = path(Integrator::RadauIIA, 1e-8);
    const Real dp6 = largest_difference(path(Integrator::DormandPrince, 1e-6), reference);
    const Real dp8 = largest_difference(path(Integrator::DormandPrince, 1e-8), reference);
    const Real rk1 = largest_difference(path(Integrator::RK4, 1e-3), reference);
    const Real rk4 = largest_difference(path(Integrator::RK4, 5e-4), reference);
    INFO("Dormand-Prince " << dp6 << " m at 1e-6, " << dp8 << " m at 1e-8; RK4 " << rk1 << " m at 1 ms, " << rk4
                           << " m at 0.5 ms");
    CHECK(dp8 < dp6 / 10.0);
    CHECK(dp8 < 1e-6);
    CHECK(rk4 < rk1);
    CHECK(rk4 < 3e-3);
}

TEST_CASE("HMMWV with bushings: RK4 needs a step below 1.35e-6 s, and then agrees", "[kernel][hmmwv]")
{
    // The bushings' fastest mode is -2.06e6 s^-1 (linearize at rest: a
    // bushing's damping, 2e6 N s/m, on about 1 kg of upper arm), so RK4,
    // stable for |lambda h| up to 2.785, needs h below 1.35e-6 s. At 2.5e-6 s
    // (|lambda h| = 5.15) RK4's amplification is R(-5.15) = 1 + z + z^2/2 +
    // z^3/6 + z^4/24 = 15.7 per step, so rounding grows past the range of a
    // double within 260 steps and the run fails within 1 ms. At 1.25e-6 s it
    // agrees with Radau IIA over the first 15 ms, the bump moved to 5 ms so
    // that the run, 12,000 steps, stays short; a step that straddles a
    // tyre's force jump leaves an error of about the jump's impulse over the
    // step, 7 kN x 1.25e-6 s on 90 kg, 1e-4 m/s, for a few milliseconds:
    // below 1e-6 m (seen 7.9e-8).
    auto start = [](Simulator& sim, const Hmmwv& car) { (void)car.start(sim, 0.005, false); };
    {
        Hmmwv car(true);
        Simulator sim(car.sys);
        start(sim, car);
        sim.method = Integrator::RK4;
        bool failed = false;
        try {
            sim.run(1e-3, 2.5e-6);
            failed = !sim.q.allFinite() || sim.projection_failures() > 0;
        } catch (const std::exception&) {
            failed = true;
        }
        INFO("after 1 ms at 2.5e-6 s: |q| " << sim.q.norm() << ", |v| " << sim.v.norm() << ", projection failures " << sim.projection_failures());
        CHECK(failed);
    }
    Hmmwv car(true);
    Simulator rk4(car.sys), radau(car.sys);
    start(rk4, car);
    rk4.method = Integrator::RK4;
    const auto explicit_path = trajectory(rk4, 0.001, 15, 1.25e-6);
    start(radau, car);
    radau.method = Integrator::RadauIIA;
    radau.step_control.relative_tolerance = radau.step_control.absolute_tolerance = 1e-9;
    const auto implicit_path = trajectory(radau, 0.001, 15, 0.001);
    const Real d = largest_difference(explicit_path, implicit_path);
    INFO("RK4 at 1.25e-6 s against Radau IIA at 1e-9: " << d << " m over 15 ms; " << rk4.projection_failures()
                                                         << " projection failures");
    CHECK(rk4.projection_failures() == 0);
    CHECK(d < 1e-6);
}

TEST_CASE("HMMWV with bushings: two implicit methods agree at steps a thousand times longer", "[kernel][hmmwv]")
{
    // Radau IIA and SDIRK4 at a tolerance of 1e-6 over the first 1.5 s, both
    // bumps, against Radau IIA at 1e-8, with steps of milliseconds where RK4
    // needs 1.35e-6 s. Radau IIA: within 5e-6 m (over 5 s the benchmark
    // measured 2.9e-7). SDIRK4's error estimate understates its error beside
    // stiff bushings (stage order 1, D33), and its error depends on where
    // its steps fall (4.6e-6 m over 5 s in the benchmark, 1.3e-5 m over 2 s
    // sampled every 20 ms, 4.3e-6 m here): within 5e-5 m. And the bushings
    // change the motion little: the chassis's height stays within 1e-3 m of
    // the model's with joints (the bushings' loads, up to a few times a
    // corner's 6 kN while the wheels climb, deflect them by 2e7 N/m by less
    // than that; seen 3.4e-5 m).
    auto run = [](bool bushed, Integrator method, Real tolerance, std::vector<Real>* z1) {
        Hmmwv car(bushed);
        Simulator sim(car.sys);
        (void)car.start(sim, 0.5, false);
        sim.method = method;
        sim.step_control.relative_tolerance = sim.step_control.absolute_tolerance = tolerance;
        std::vector<std::vector<Vec3>> path;
        for (int k = 0; k < 75; ++k) {
            sim.run(0.02, 0.02);
            std::vector<Vec3> p;
            for (const auto& s : sim.states()) p.push_back(s.p_WB);
            path.push_back(p);
            if (z1) z1->push_back(car.chassis_com(sim).z());
        }
        return path;
    };
    std::vector<Real> z1_bushed, z1_joints;
    const auto reference = run(true, Integrator::RadauIIA, 1e-8, &z1_bushed);
    const Real radau = largest_difference(run(true, Integrator::RadauIIA, 1e-6, nullptr), reference);
    const Real sdirk = largest_difference(run(true, Integrator::SDIRK4, 1e-6, nullptr), reference);
    (void)run(false, Integrator::RadauIIA, 1e-8, &z1_joints);
    Real dz = 0.0;
    for (std::size_t k = 0; k < z1_bushed.size(); ++k) dz = std::max(dz, std::abs(z1_bushed[k] - z1_joints[k]));
    INFO("Radau IIA " << radau << " m, SDIRK4 " << sdirk << " m at 1e-6; the chassis's height with bushings "
                      << dz << " m from its height with joints");
    CHECK(radau < 5e-6);
    CHECK(sdirk < 5e-5);
    CHECK(dz < 1e-3);
}
