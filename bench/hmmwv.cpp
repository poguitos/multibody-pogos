// The HMMWV benchmark (plan task 5.4): the fourteen-body HMMWV of Haug et al.
// (1999) driven at 10 mph over a half-cylinder bump, and its variant with
// bushings, run for 5 s by every integrator at several steps or tolerances.
// Each run is compared with a reference (Radau IIA at a tolerance of 1e-10)
// on every body's position every 10 ms, and its cost is counted: wall-clock
// time of the stepping alone, steps, evaluations of the accelerations, Newton
// iterations and Jacobians. The runs go without events, as the paper's did;
// a second table repeats some on the joint model with the tyres' events
// (Hmmwv::add_events). Prints the tables of docs/hmmwv.md (Markdown).
//
// Built on request only:  scripts\build.ps1 -Target mbd_hmmwv
// Run:                    build\bench\mbd_hmmwv.exe            the tables (about 45 minutes)
//                         build\bench\mbd_hmmwv.exe quick      without the explicit runs on the bushed model
//                         build\bench\mbd_hmmwv.exe calibrate  the springs' free lengths (kFreeLength)
//                         build\bench\mbd_hmmwv.exe modes      the modes at rest of both variants
//
// Run it pinned to a performance core at high priority (docs/performance.md)
// for timings that compare.

#include <algorithm>
#include <cmath>
#include <array>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/validate.hpp"

#include "kernel/hmmwv.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::Hmmwv;

namespace {

constexpr Real kDuration = 5.0;   // [s], the paper's
constexpr Real kSample = 0.01;    // [s] between samples
constexpr int kSamples = 500;

// --- Calibration and modes ---------------------------------------------------

/// The lower arms' joint angles, front and rear (left side), at rest on flat
/// ground with the given free lengths.
std::array<Real, 2> arm_angles(std::array<Real, 2> L0)
{
    Hmmwv car(false, L0);
    Simulator sim(car.sys);
    car.design(sim);
    StaticOptions o;
    o.tolerance = 1e-11;
    (void)static_equilibrium(sim, o);
    const auto& m = car.sys.model;
    return {sim.q(m.idx_q[static_cast<std::size_t>(car.corners[0].lca)]),
            sim.q(m.idx_q[static_cast<std::size_t>(car.corners[2].lca)])};
}

/// Newton's method on the free lengths, by differences: the lengths at which
/// the arms rest at their design angles.
int calibrate()
{
    std::array<Real, 2> L0{0.29, 0.34};
    for (int it = 0; it < 8; ++it) {
        const auto a = arm_angles(L0);
        std::array<Real, 2> d{};
        for (std::size_t k = 0; k < 2; ++k) {
            auto L = L0;
            L[k] += 1e-5;
            d[k] = (arm_angles(L)[k] - a[k]) / 1e-5;
        }
        std::printf("free lengths %.9f %.9f: arm angles %+.3e %+.3e rad\n", L0[0], L0[1], a[0], a[1]);
        for (std::size_t k = 0; k < 2; ++k) L0[k] -= a[k] / d[k];
    }
    return 0;
}

int modes()
{
    for (const bool bushed : {false, true}) {
        Hmmwv car(bushed);
        Simulator sim(car.sys);
        car.design(sim);
        StaticOptions o;
        o.tolerance = 1e-8;
        const StaticReport st = static_equilibrium(sim, o);
        const LinearModel lin = linearize(sim);
        std::printf("%s: statics %s in %d iterations (residual %.1e), %d of %d directions with stiffness; %zu modes, "
                    "the fastest:\n",
                    bushed ? "with bushings" : "with joints", st.converged ? "converged" : "NOT converged",
                    st.iterations, st.residual, st.supported, st.degrees_of_freedom, lin.modes.size());
        for (std::size_t i = lin.modes.size() > 6 ? lin.modes.size() - 6 : 0; i < lin.modes.size(); ++i) {
            const Mode& m = lin.modes[i];
            std::printf("  %+.4e %+.4ei  (|lambda| %.4e rad/s, damping ratio %.3f)\n", m.eigenvalue.real(),
                        m.eigenvalue.imag(), m.natural_frequency, m.damping_ratio);
        }
    }
    return 0;
}

// --- Runs --------------------------------------------------------------------

struct Method {
    const char* name;
    Integrator integrator;
    bool adaptive;
};

const Method kRK4{"RK4", Integrator::RK4, false};
const Method kDP{"Dormand-Prince", Integrator::DormandPrince, true};
const Method kGA{"generalised-alpha", Integrator::GeneralizedAlpha, false};
const Method kSDIRK{"SDIRK4", Integrator::SDIRK4, true};
const Method kRadau{"Radau IIA", Integrator::RadauIIA, true};
const Method kLIE{"linearly implicit Euler", Integrator::LinearlyImplicitEuler, false};

/// Every body's origin at every sample.
using Trajectory = std::vector<std::vector<Vec3>>;

struct Run {
    std::string method;
    Real parameter{0.0};      // step [s] or tolerance
    bool adaptive{false};
    bool ok{false};
    std::string failure;      // the exception's message
    bool with_events{false};
    Trajectory path;
    std::vector<Real> z1;     // the chassis's centre of mass, its height
    Real seconds{0.0};
    IntegratorStats stats;
    int projection_failures{0};
    std::size_t events{0};
    Real difference{0.0};     // from the reference: any body's origin [m]
    Real z1_difference{0.0};  // from the reference: the chassis's height [m]
    Real z1_min{0.0}, z1_max{0.0}, speed_end{0.0};
};

Run run(bool bushed, const Method& m, Real parameter, bool events = false)
{
    Hmmwv car(bushed);
    Simulator sim(car.sys);
    (void)car.start(sim, 0.5, events);
    sim.method = m.integrator;
    if (m.adaptive) {
        sim.step_control.relative_tolerance = parameter;
        sim.step_control.absolute_tolerance = parameter;
        sim.step_control.max_steps = 100000000;
    }
    const Real dt = m.adaptive ? kSample : parameter;
    Run r;
    r.method = m.name;
    r.parameter = parameter;
    r.adaptive = m.adaptive;
    r.with_events = events;
    r.z1_min = r.z1_max = car.chassis_com(sim).z();
    Real stepping = 0.0;
    try {
        for (int k = 0; k < kSamples; ++k) {
            const auto s0 = std::chrono::steady_clock::now();
            sim.run(kSample, dt);
            stepping += std::chrono::duration<Real>(std::chrono::steady_clock::now() - s0).count();
            std::vector<Vec3> p;
            for (const auto& s : sim.states()) p.push_back(s.p_WB);
            if (!p.back().allFinite()) throw std::runtime_error("not finite");
            r.path.push_back(p);
            const Real z1 = car.chassis_com(sim).z();
            r.z1.push_back(z1);
            r.z1_min = std::min(r.z1_min, z1);
            r.z1_max = std::max(r.z1_max, z1);
        }
        r.ok = true;
    } catch (const std::exception& e) {
        r.ok = false;
        r.failure = e.what();
    }
    r.seconds = stepping;
    r.stats = sim.integrator_stats();
    r.projection_failures = sim.projection_failures();
    r.speed_end = sim.states()[static_cast<std::size_t>(car.chassis)].v_WB.x();
    r.events = sim.event_log().size();
    return r;
}

Real largest_difference(const Trajectory& a, const Trajectory& b)
{
    Real d = 0.0;
    for (std::size_t k = 0; k < std::min(a.size(), b.size()); ++k)
        for (std::size_t i = 0; i < a[k].size(); ++i) d = std::max(d, (a[k][i] - b[k][i]).norm());
    return d;
}

std::string parameter_text(const Run& r)
{
    char s[64];
    if (r.adaptive)
        std::snprintf(s, sizeof s, "tol %.0e", r.parameter);
    else
        std::snprintf(s, sizeof s, "h %.3g ms", r.parameter * 1e3);
    return s;
}

void table(const char* title, const std::vector<Run>& runs)
{
    std::printf("\n### %s\n\n", title);
    std::printf("| method | step or tolerance | difference, any body [m] | difference, chassis height [m] | time [s] "
                "| steps (rejected) | evaluations | Newton iterations | Jacobians |\n"
                "|---|---|---|---|---|---|---|---|---|\n");
    for (const Run& r : runs) {
        if (!r.ok) {
            // The message's code and where: its first sentence.
            const std::string why = r.failure.substr(0, r.failure.find(" did not") != std::string::npos
                                                            ? r.failure.find(" did not")
                                                            : std::min<std::size_t>(r.failure.size(), 90));
            std::printf("| %s | %s | failed: %s | | %.3g | %lld | | | |\n", r.method.c_str(),
                        parameter_text(r).c_str(), why.c_str(), r.seconds, r.stats.steps);
            continue;
        }
        std::printf("| %s | %s | %.2e | %.2e | %.3g | %lld (%lld) | %lld | %lld | %lld |\n", r.method.c_str(),
                    parameter_text(r).c_str(), r.difference, r.z1_difference, r.seconds, r.stats.steps,
                    r.stats.rejected, r.stats.evaluations, r.stats.newton_iterations, r.stats.jacobians);
    }
}

/// Each method at each of its parameters, compared with the reference.
std::vector<Run> runs_of(bool bushed, bool events, const Run& reference,
                         const std::vector<std::pair<Method, std::vector<Real>>>& plan)
{
    std::vector<Run> runs;
    for (const auto& [method, parameters] : plan) {
        for (const Real p : parameters) {
            std::fprintf(stderr, "%s%s: %s %g...\n", bushed ? "with bushings" : "with joints",
                         events ? ", events" : "", method.name, p);
            Run r = run(bushed, method, p, events);
            if (r.ok) {
                r.difference = largest_difference(r.path, reference.path);
                for (std::size_t k = 0; k < r.z1.size(); ++k)
                    r.z1_difference = std::max(r.z1_difference, std::abs(r.z1[k] - reference.z1[k]));
            }
            r.path.clear();
            runs.push_back(r);
        }
    }
    return runs;
}

int benchmark(bool quick)
{
    for (const bool bushed : {false, true}) {
        const char* variant = bushed ? "with bushings" : "with joints";
        std::fprintf(stderr, "%s: reference...\n", variant);
        const Run reference = run(bushed, kRadau, 1e-10);
        std::printf("\n## The HMMWV %s\n\nReference: Radau IIA at tol 1e-10, %.3g s, %lld steps; z1 from %.6f to "
                    "%.6f m; speed at 5 s %.6f m/s.\n",
                    variant, reference.seconds, reference.stats.steps, reference.z1_min, reference.z1_max,
                    reference.speed_end);

        std::vector<std::pair<Method, std::vector<Real>>> plan;
        if (!bushed) {
            plan = {{kRK4, {4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4}},
                    {kDP, {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8}},
                    {kGA, {4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4}},
                    {kSDIRK, {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8}},
                    {kRadau, {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8}},
                    {kLIE, {1e-3, 5e-4, 2.5e-4, 1.25e-4}}};
        } else {
            plan = {{kGA, {4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4}},
                    {kSDIRK, {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8}},
                    {kRadau, {1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8}},
                    {kLIE, {1e-3, 5e-4, 2.5e-4, 1.25e-4}}};
            if (!quick) plan.insert(plan.begin(), {{kRK4, {1.25e-6}}, {kDP, {1e-6}}});
        }
        table(variant, runs_of(bushed, false, reference, plan));

        if (!bushed) {
            // The same model with the tyres' events (Hmmwv::add_events), against
            // the same reference; the reference with events, for its accuracy.
            std::fprintf(stderr, "%s, events: reference...\n", variant);
            Run with_events = run(false, kRadau, 1e-10, true);
            std::printf("\nThe reference with the tyres' events (Radau IIA at tol 1e-10, %.3g s, %lld steps, %zu "
                        "events) differs from it by %.2e m (any body), %.2e m (chassis height).\n",
                        with_events.seconds, with_events.stats.steps, with_events.events,
                        largest_difference(with_events.path, reference.path),
                        [&] {
                            Real d = 0.0;
                            for (std::size_t k = 0; k < with_events.z1.size(); ++k)
                                d = std::max(d, std::abs(with_events.z1[k] - reference.z1[k]));
                            return d;
                        }());
            const std::vector<std::pair<Method, std::vector<Real>>> event_plan = {
                {kRK4, {2e-3, 1e-3, 5e-4, 2.5e-4}}, {kDP, {1e-3, 1e-6}},  {kGA, {1e-3, 2.5e-4}},
                {kSDIRK, {1e-3, 1e-6}},              {kRadau, {1e-3, 1e-6}}, {kLIE, {1e-3, 5e-4}}};
            table("with joints, with the tyres' events", runs_of(false, true, reference, event_plan));
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "calibrate") return calibrate();
    if (mode == "modes") return modes();
    return benchmark(mode == "quick");
}
