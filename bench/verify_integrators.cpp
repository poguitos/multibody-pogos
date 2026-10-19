// The integrators' verification (plan task 4.9): convergence, energy and
// work against precision for every method, on six models from an oscillator
// with an exact solution to the sedan. Writes the figures (SVG) and the data
// (CSV) of docs/integrators.md into a folder, and prints the report's tables
// (Markdown) to standard output.
//
// Built on request only:  scripts\build.ps1 -Target mbd_verify
// Run:                    build\bench\mbd_verify.exe docs\integrators
//
// The timings are wall-clock time of the stepping alone (not the model's
// set-up), the least of several runs where one is short; run the program
// pinned to a performance core at high priority (docs/performance.md) for
// timings that compare.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/drivetrain.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

#include "kernel/bushed_pendulum.hpp"
#include "kernel/four_bar.hpp"

#include "svg_plot.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Complex = std::complex<Real>;

namespace {

// --- Methods ------------------------------------------------------------------

struct Method {
    std::string name;
    Integrator method;
    bool adaptive{false};     // error control (DormandPrince, SDIRK4, RadauIIA)
    bool stabilized{false};   // generalised-alpha in the stabilized index-2 form
    int multirate{1};         // RK4's fast steps per step
};

const Method kRK4{"RK4", Integrator::RK4};
const Method kSIE{"semi-implicit Euler", Integrator::SemiImplicitEuler};
const Method kDP{"Dormand-Prince", Integrator::DormandPrince};
const Method kGA{"generalised-alpha", Integrator::GeneralizedAlpha};
const Method kGAS{"generalised-alpha, stabilized", Integrator::GeneralizedAlpha, false, true};
const Method kSDIRK{"SDIRK4", Integrator::SDIRK4};
const Method kRadau{"Radau IIA", Integrator::RadauIIA};
const Method kLIE{"linearly implicit Euler", Integrator::LinearlyImplicitEuler};

Method adaptive(Method m)
{
    m.adaptive = true;
    m.name += ", error control";
    return m;
}

/// Newton's tolerance for the implicit methods at fixed steps: tight enough
/// that the methods' own errors are measured, not Newton's.
constexpr Real kNewtonTolerance = 1e-12;
/// On the stiff pendulum generalised-alpha's Newton failed at 2 ms with the
/// tolerances at 1e-12 and runs at 1e-9 (measured, task 4.9), as in its
/// test in tests/kernel/test_kernel_integrators.cpp: far below the errors
/// there (2.7e-5 m at 2 ms). Used there and for the four-bar's longest step.
constexpr Real kStiffNewton = 1e-9;

// --- A run ------------------------------------------------------------------------

/// A model ready to run: it owns its System and its Simulator, released at
/// the start.
struct Case {
    virtual ~Case() = default;
    virtual Simulator& sim() = 0;
    /// What is compared with the reference at the end.
    virtual VecX observe() = 0;
};

using Factory = std::function<std::unique_ptr<Case>(const Method&)>;

struct Run {
    std::string method;
    Real parameter{0.0};   // the step [s], or the tolerance with error control
    bool ok{true};
    std::string failure;
    VecX observed;
    Real error{std::numeric_limits<Real>::quiet_NaN()};
    IntegratorStats stats;
    Real seconds{0.0};
};

void configure(Simulator& sim, const Method& m, Real tolerance)
{
    sim.method = m.method;
    sim.step_control.adaptive = m.adaptive;
    sim.step_control.relative_tolerance = tolerance;
    sim.step_control.absolute_tolerance = tolerance;
    sim.implicit_control.stabilized_index2 = m.stabilized;
    sim.multirate = m.multirate;
}

/// T seconds of the case with method m: fixed steps of h, Newton (in the
/// implicit methods) at `newton`, or with error control at tolerance tol
/// within output steps of h. Timed, again if short.
Run run(const Factory& make, const Method& m, Real T, Real h, Real tol, Real newton = kNewtonTolerance)
{
    Run r;
    r.method = m.name;
    r.parameter = m.adaptive ? tol : h;
    const int steps = static_cast<int>(std::lround(T / h));
    Real best = HUGE_VAL, total = 0.0;
    for (int repeat = 0; repeat < 5 && total < 0.5; ++repeat) {
        std::unique_ptr<Case> c = make(m);
        Simulator& sim = c->sim();
        configure(sim, m, m.adaptive ? tol : newton);
        try {
            const auto t0 = std::chrono::steady_clock::now();
            for (int k = 0; k < steps; ++k) sim.step(h);
            const Real s = std::chrono::duration<Real>(std::chrono::steady_clock::now() - t0).count();
            best = std::min(best, s);
            total += s;
        } catch (const std::exception& e) {
            r.ok = false;
            r.failure = e.what();
            r.failure = r.failure.substr(0, r.failure.find(':'));   // the code
            return r;
        }
        r.observed = c->observe();
        r.stats = sim.integrator_stats();
        if (!r.observed.allFinite()) {
            r.ok = false;
            r.failure = "diverged";
            return r;
        }
    }
    r.seconds = best;
    return r;
}

// --- Output -----------------------------------------------------------------------

std::string g(Real v, int digits = 3)
{
    if (!std::isfinite(v)) return "-";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.*g", digits, v);
    return buf;
}

std::string param(Real p, bool adaptive)
{
    if (adaptive) return "tol " + g(p, 2);
    if (p >= 1e-3) return g(p * 1e3, 4) + " ms";
    return g(p * 1e6, 4) + " us";
}

struct Study {
    std::string name;   // the file stem
    std::vector<Run> runs;

    void csv(const std::string& dir) const
    {
        std::ofstream f(dir + "/" + name + ".csv");
        f << "method,parameter,ok,error,evaluations,steps,rejected,newton_iterations,jacobians,seconds,failure\n";
        for (const Run& r : runs) {
            f << '"' << r.method << "\"," << g(r.parameter, 6) << ',' << (r.ok ? 1 : 0) << ',' << g(r.error, 6)
              << ',' << r.stats.evaluations << ',' << r.stats.steps << ',' << r.stats.rejected << ','
              << r.stats.newton_iterations << ',' << r.stats.jacobians << ',' << g(r.seconds, 6) << ','
              << r.failure << '\n';
        }
    }
};

/// The runs of one method, in order.
std::vector<const Run*> of(const Study& s, const std::string& method)
{
    std::vector<const Run*> out;
    for (const Run& r : s.runs) {
        if (r.method == method) out.push_back(&r);
    }
    return out;
}

/// Error against the step, one row per method with the observed order
/// between successive steps (log2 of the ratio for halved steps).
void convergence_table(const Study& s, const std::vector<std::string>& methods, const std::vector<Real>& steps)
{
    std::printf("\n| method |");
    for (const Real h : steps) std::printf(" %s |", param(h, false).c_str());
    std::printf(" observed orders |\n|---|");
    for (std::size_t i = 0; i < steps.size(); ++i) std::printf("---|");
    std::printf("---|\n");
    for (const std::string& m : methods) {
        std::printf("| %s |", m.c_str());
        const auto rs = of(s, m);
        std::vector<Real> e;
        for (const Real h : steps) {
            const Run* hit = nullptr;
            for (const Run* r : rs) {
                if (std::abs(r->parameter - h) < 1e-12 * h) hit = r;
            }
            const Real err = hit && hit->ok ? hit->error : std::numeric_limits<Real>::quiet_NaN();
            e.push_back(err);
            std::printf(" %s |", hit && !hit->ok ? hit->failure.c_str() : g(err).c_str());
        }
        std::string orders;
        for (std::size_t i = 0; i + 1 < e.size(); ++i) {
            if (!(e[i] > 0.0 && e[i + 1] > 0.0)) continue;
            const Real p = std::log(e[i] / e[i + 1]) / std::log(steps[i] / steps[i + 1]);
            orders += (orders.empty() ? "" : ", ") + g(p, 3);
        }
        std::printf(" %s |\n", orders.c_str());
    }
}

/// The cheapest run of each method within each error bound.
void cheapest_table(const Study& s, const std::vector<std::string>& methods, const std::vector<Real>& bounds)
{
    std::printf("\n| method |");
    for (const Real b : bounds) std::printf(" error <= %s |", g(b, 2).c_str());
    std::printf("\n|---|");
    for (std::size_t i = 0; i < bounds.size(); ++i) std::printf("---|");
    std::printf("\n");
    for (const std::string& m : methods) {
        std::printf("| %s |", m.c_str());
        for (const Real b : bounds) {
            const Run* best = nullptr;
            for (const Run* r : of(s, m)) {
                if (r->ok && r->error <= b && (!best || r->seconds < best->seconds)) best = r;
            }
            if (best) {
                const bool adaptive = m.find("error control") != std::string::npos;
                std::printf(" %s ms (%s, %lld evaluations) |", g(best->seconds * 1e3).c_str(),
                            param(best->parameter, adaptive).c_str(), best->stats.evaluations);
            } else {
                std::printf(" - |");
            }
        }
        std::printf("\n");
    }
}

/// Guide lines of order p under a method's errors: from its longest step, a
/// quarter of its error there, over three halvings of the step.
std::vector<mbd_plot::Slope> guides(const Study& s, const std::vector<std::pair<std::string, Real>>& orders)
{
    std::vector<mbd_plot::Slope> out;
    for (const auto& [method, p] : orders) {
        std::vector<const Run*> rs;
        for (const Run* r : of(s, method)) {
            if (r->ok && r->error > 0.0) rs.push_back(r);
        }
        if (rs.size() < 2) continue;
        std::sort(rs.begin(), rs.end(), [](const Run* a, const Run* b) { return a->parameter > b->parameter; });
        const Run* far = rs[std::min<std::size_t>(3, rs.size() - 1)];
        out.push_back({p, rs[0]->parameter, 0.25 * rs[0]->error, far->parameter, "order " + g(p)});
    }
    return out;
}

/// Error against the step (convergence) or against the cost (work).
void plot(const Study& s, const std::string& dir, const std::string& file, const std::string& title,
          const std::vector<std::string>& methods, bool against_step, bool seconds,
          const std::vector<mbd_plot::Slope>& slopes = {}, const std::string& ylabel = "error")
{
    mbd_plot::Plot p;
    p.title = title;
    p.ylabel = ylabel;
    p.xlabel = against_step ? "step h [s]" : seconds ? "time to run [s]" : "evaluations of the accelerations";
    p.slopes = slopes;
    for (const std::string& m : methods) {
        mbd_plot::Series series;
        series.name = m;
        series.dashed = m.find("error control") != std::string::npos;
        for (const Run* r : of(s, m)) {
            if (!r->ok) continue;
            series.x.push_back(against_step ? r->parameter : seconds ? r->seconds
                                                                      : static_cast<Real>(r->stats.evaluations));
            series.y.push_back(r->error);
        }
        p.series.push_back(series);
    }
    p.write(dir + "/" + file);
}

// --- Model 1: the oscillator ---------------------------------------------------

/// A linear spring on a slider's coordinate.
class Spring : public JointForce {
public:
    Spring(int body, Real k) : body_(body), k_(k) {}
    const char* name() const override { return "spring"; }
    std::vector<int> bodies() const override { return {body_}; }
    void apply(const VecX& q, const VecX&, VecX& tau) const override { tau(0) -= k_ * q(0); }

private:
    int body_;
    Real k_;
};

/// 1 kg on a slider with a spring of 1 N/m (w = 1 rad/s), no gravity,
/// released at rest from q = 1: q = cos t, v = -sin t.
struct OscillatorCase : Case {
    System sys;
    std::unique_ptr<Simulator> s;
    OscillatorCase()
    {
        sys.model.gravity = Vec3::Zero();
        sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                           RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), "block");
        sys.joint_forces.push_back(std::make_shared<Spring>(1, 1.0));
        s = std::make_unique<Simulator>(sys);
        s->q(0) = 1.0;
        s->initialize();
    }
    Simulator& sim() override { return *s; }
    VecX observe() override { return Eigen::Vector2d(s->q(0), s->v(0)); }
};

// The stability functions R(z): one step of a Runge-Kutta method on y' = A y
// multiplies the state by R(hA); on the oscillator w = q - i v obeys w' = i w.

Complex R_rk4(Complex z) { return 1.0 + z + z * z / 2.0 + std::pow(z, 3) / 6.0 + std::pow(z, 4) / 24.0; }

Complex R_dp(Complex z) { return R_rk4(z) + std::pow(z, 5) / 120.0 + std::pow(z, 6) / 600.0; }

Complex R_sdirk(Complex z)
{
    static constexpr Real A[5][5] = {{0.25, 0.0, 0.0, 0.0, 0.0},
                                     {0.5, 0.25, 0.0, 0.0, 0.0},
                                     {17.0 / 50.0, -1.0 / 25.0, 0.25, 0.0, 0.0},
                                     {371.0 / 1360.0, -137.0 / 2720.0, 15.0 / 544.0, 0.25, 0.0},
                                     {25.0 / 24.0, -49.0 / 48.0, 125.0 / 16.0, -85.0 / 12.0, 0.25}};
    Complex k[5];
    Complex r = 1.0;
    for (int i = 0; i < 5; ++i) {
        Complex sum = 1.0;
        for (int j = 0; j < i; ++j) sum += z * A[i][j] * k[j];
        k[i] = sum / (1.0 - z * A[i][i]);
        r += z * A[4][i] * k[i];
    }
    return r;
}

Complex R_radau(Complex z)
{
    const Real s6 = std::sqrt(6.0);
    Eigen::Matrix3d A;
    A << (88.0 - 7.0 * s6) / 360.0, (296.0 - 169.0 * s6) / 1800.0, (-2.0 + 3.0 * s6) / 225.0,
        (296.0 + 169.0 * s6) / 1800.0, (88.0 + 7.0 * s6) / 360.0, (-2.0 - 3.0 * s6) / 225.0,
        (16.0 - s6) / 36.0, (16.0 + s6) / 36.0, 1.0 / 9.0;
    const Eigen::Matrix3cd M = Eigen::Matrix3cd::Identity() - z * A.cast<Complex>();
    const Eigen::Vector3cd Z = M.inverse() * (z * A.cast<Complex>() * Eigen::Vector3cd::Ones());
    return 1.0 + Z(2);
}

Complex R_backward_euler(Complex z) { return 1.0 / (1.0 - z); }

/// The state (q, v) after N steps of h predicted by each method's own
/// recurrence on the oscillator, from (1, 0).
Eigen::Vector2d predicted_oscillator(const Method& m, Real h, int N)
{
    auto rk = [&](Complex (*R)(Complex)) {
        const Complex w = std::pow(R(Complex(0.0, h)), N);   // w0 = 1
        return Eigen::Vector2d(w.real(), -w.imag());
    };
    switch (m.method) {
        case Integrator::RK4: return rk(R_rk4);
        case Integrator::DormandPrince: return rk(R_dp);
        case Integrator::SDIRK4: return rk(R_sdirk);
        case Integrator::RadauIIA: return rk(R_radau);
        case Integrator::LinearlyImplicitEuler: return rk(R_backward_euler);
        case Integrator::SemiImplicitEuler: {
            // v+ = v - h q, q+ = q + h v+.
            Real q = 1.0, v = 0.0;
            for (int k = 0; k < N; ++k) {
                v -= h * q;
                q += h * v;
            }
            return {q, v};
        }
        case Integrator::GeneralizedAlpha: {
            // Chung and Hulbert's recurrence for q'' = -q at rho_infinity 0.8,
            // started from a = q'' (tests: "the steps are the method's
            // recurrence").
            const Real rho = 0.8, am = (2.0 * rho - 1.0) / (rho + 1.0), af = rho / (rho + 1.0);
            const Real gam = 0.5 - am + af, beta = 0.25 * (gam + 0.5) * (gam + 0.5);
            Real q = 1.0, v = 0.0, a = -1.0;
            for (int k = 0; k < N; ++k) {
                const Real a1 = (-(1.0 - af) * (q + h * v + h * h * (0.5 - beta) * a) - am * a + af * (-q))
                                / ((1.0 - am) + (1.0 - af) * beta * h * h);
                q += h * v + h * h * ((0.5 - beta) * a + beta * a1);
                v += h * ((1.0 - gam) * a + gam * a1);
                a = a1;
            }
            return {q, v};
        }
    }
    return {0.0, 0.0};
}

void oscillator(const std::string& dir)
{
    std::printf("\n## The oscillator\n");
    const Factory make = [](const Method&) { return std::make_unique<OscillatorCase>(); };
    const std::vector<Method> methods = {kRK4, kSIE, kDP, kGA, kSDIRK, kRadau, kLIE};

    // Convergence over 10 s.
    const Real T = 10.0;
    const std::vector<Real> steps = {0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625};
    Study s{"oscillator_convergence", {}};
    for (const Method& m : methods) {
        for (const Real h : steps) {
            Run r = run(make, m, T, h, 0.0);
            if (r.ok) r.error = std::hypot(r.observed(0) - std::cos(T), r.observed(1) + std::sin(T));
            s.runs.push_back(r);
        }
    }
    s.csv(dir);
    std::vector<std::string> names;
    for (const Method& m : methods) names.push_back(m.name);
    std::printf("\nError |(q, v) - (cos t, -sin t)| at t = 10 s, released from q = 1 m:\n");
    convergence_table(s, names, steps);
    plot(s, dir, "oscillator_convergence.svg", "Oscillator: error at 10 s against the step", names, true, false,
         guides(s, {{kSIE.name, 1.0}, {kGA.name, 2.0}, {kSDIRK.name, 4.0}, {kRadau.name, 5.0}}));

    // Energy over 1000 s, against each method's own recurrence.
    std::printf("\nEnergy after 1000 s, E / E0, measured and predicted by the method's recurrence on the "
                "oscillator; and its range over the run:\n\n");
    std::printf("| method | w h | steps | E / E0 measured | predicted | relative difference "
                "| least, largest E / E0 |\n");
    std::printf("|---|---|---|---|---|---|---|\n");
    for (const Real h : {0.1, 0.5}) {
        for (const Method& m : methods) {
            OscillatorCase c;
            configure(c.sim(), m, kNewtonTolerance);
            const int N = static_cast<int>(std::lround(1000.0 / h));
            Real lo = 1.0, hi = 1.0;
            for (int k = 0; k < N; ++k) {
                c.sim().step(h);
                const Real E = c.sim().q(0) * c.sim().q(0) + c.sim().v(0) * c.sim().v(0);   // 2 E, E0 = 1/2
                lo = std::min(lo, E);
                hi = std::max(hi, E);
            }
            const Eigen::Vector2d qv = c.observe(), pred = predicted_oscillator(m, h, N);
            const Real E = qv.squaredNorm(), Ep = pred.squaredNorm();
            std::printf("| %s | %s | %d | %s | %s | %s | %s, %s |\n", m.name.c_str(), g(h).c_str(), N,
                        g(E, 10).c_str(), g(Ep, 10).c_str(), g(std::abs(E - Ep) / Ep, 2).c_str(), g(lo, 6).c_str(),
                        g(hi, 6).c_str());
        }
    }
}

// --- Model 2 and 5: the bushed pendulum ------------------------------------------

struct PendulumCase : Case {
    mbd_test::BushedPendulum bp;
    std::unique_ptr<Simulator> s;
    PendulumCase(Real k, Real k_r) : bp(k, k_r)
    {
        s = std::make_unique<Simulator>(bp.sys);
        bp.release(*s);
    }
    Simulator& sim() override { return *s; }
    VecX observe() override { return bp.tip(*s); }
};

/// The reference: Dormand-Prince at tolerance 1e-13, checked against Radau
/// IIA at 1e-12; prints their difference.
VecX reference(const Factory& make, Real T, const char* what)
{
    const Run a = run(make, adaptive(kDP), T, 0.01, 1e-13);
    const Run b = run(make, adaptive(kRadau), T, 0.01, 1e-12);
    std::printf("\nReference (%s): Dormand-Prince at tolerance 1e-13 (%lld evaluations); Radau IIA at 1e-12 "
                "differs from it by %s.\n",
                what, a.stats.evaluations, g((a.observed - b.observed).norm(), 2).c_str());
    return a.observed;
}

void error_against(Study& s, const VecX& ref)
{
    for (Run& r : s.runs) {
        if (r.ok) r.error = (r.observed - ref).norm();
    }
}

/// The model's fastest motions at its start, from the linearisation (task
/// 3.7): what bounds an explicit method's step. A mode of natural frequency
/// w and damping ratio z has the eigenvalues -z w +- i w sqrt(1 - z^2).
void fastest_modes(Case& c, const char* what)
{
    LinearModel lm = linearize(c.sim());
    std::vector<Mode> modes = lm.modes;
    std::sort(modes.begin(), modes.end(),
              [](const Mode& a, const Mode& b) { return a.natural_frequency > b.natural_frequency; });
    std::printf("\nThe fastest motions of %s at the start (linearised): ", what);
    for (std::size_t i = 0; i < std::min<std::size_t>(4, modes.size()); ++i) {
        std::printf("%s%s rad/s, damping ratio %s", i ? "; " : "", g(modes[i].natural_frequency, 4).c_str(),
                    g(modes[i].damping_ratio, 3).c_str());
    }
    std::printf(".\n");
}

void smooth_pendulum(const std::string& dir)
{
    std::printf("\n## The soft bushed pendulum (smooth, spatial)\n");
    const Factory make = [](const Method&) { return std::make_unique<PendulumCase>(1e4, 1e2); };
    const Real T = 1.0;
    const VecX ref = reference(make, T, "the tip at 1 s");
    fastest_modes(*make(kRK4), "the pendulum");
    const std::vector<Method> fixed = {kRK4, kSIE, kDP, kGA, kSDIRK, kRadau, kLIE};
    const std::vector<Method> controlled = {adaptive(kDP), adaptive(kSDIRK), adaptive(kRadau)};
    const std::vector<Real> steps = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4};
    Study s{"pendulum", {}};
    for (const Method& m : fixed) {
        for (const Real h : steps) s.runs.push_back(run(make, m, T, h, 0.0));
    }
    for (const Method& m : controlled) {
        for (const Real tol : {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11}) {
            s.runs.push_back(run(make, m, T, 0.01, tol));
        }
    }
    error_against(s, ref);
    s.csv(dir);
    std::vector<std::string> fixed_names, all;
    for (const Method& m : fixed) fixed_names.push_back(m.name);
    all = fixed_names;
    for (const Method& m : controlled) all.push_back(m.name);
    std::printf("\nError of the tip at 1 s [m], fixed steps:\n");
    convergence_table(s, fixed_names, steps);
    std::printf("\nThe cheapest run within each error:\n");
    cheapest_table(s, all, {1e-3, 1e-6, 1e-9});
    plot(s, dir, "pendulum_convergence.svg", "Soft bushed pendulum: error of the tip at 1 s", fixed_names, true,
         false, guides(s, {{kSIE.name, 1.0}, {kGA.name, 2.0}, {kSDIRK.name, 4.0}, {kRadau.name, 5.0}}),
         "error [m]");
    plot(s, dir, "pendulum_work.svg", "Soft bushed pendulum: error against time to run", all, false, true, {},
         "error [m]");
}

void stiff_pendulum(const std::string& dir)
{
    std::printf("\n## The stiff bushed pendulum (k = 1e8 N/m)\n");
    const Factory make = [](const Method&) { return std::make_unique<PendulumCase>(1e8, 1e6); };
    const Real T = 1.0;
    const VecX ref = reference(make, T, "the tip at 1 s");
    fastest_modes(*make(kRK4), "the pendulum");
    Study s{"stiff", {}};
    auto sweep = [&](const Method& m, std::initializer_list<Real> values) {
        for (const Real v : values) {
            s.runs.push_back(m.adaptive ? run(make, m, T, 0.01, v) : run(make, m, T, v, 0.0, kStiffNewton));
        }
    };
    sweep(kRK4, {1e-4, 5e-5, 2.5e-5});
    sweep(kSIE, {1e-4, 5e-5, 2.5e-5});
    sweep(kGA, {4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4});
    sweep(kLIE, {4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5});
    sweep(adaptive(kDP), {1e-4, 1e-6, 1e-8, 1e-10});
    sweep(adaptive(kSDIRK), {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8});
    sweep(adaptive(kRadau), {1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10});
    error_against(s, ref);
    s.csv(dir);
    const std::vector<std::string> names = {kRK4.name, kSIE.name, kGA.name, kLIE.name, adaptive(kDP).name,
                                            adaptive(kSDIRK).name, adaptive(kRadau).name};
    std::printf("\nEvery run (error of the tip at 1 s [m]):\n\n| method | step or tolerance | error | evaluations "
                "| steps | rejected | Newton iterations | time [ms] |\n|---|---|---|---|---|---|---|---|\n");
    for (const Run& r : s.runs) {
        const bool a = r.method.find("error control") != std::string::npos;
        std::printf("| %s | %s | %s | %lld | %lld | %lld | %lld | %s |\n", r.method.c_str(),
                    param(r.parameter, a).c_str(), r.ok ? g(r.error).c_str() : r.failure.c_str(),
                    r.stats.evaluations, r.stats.steps, r.stats.rejected, r.stats.newton_iterations,
                    g(r.seconds * 1e3).c_str());
    }
    std::printf("\nThe cheapest run within each error:\n");
    cheapest_table(s, names, {1e-3, 1e-5, 1e-7});
    plot(s, dir, "stiff_work.svg", "Stiff bushed pendulum: error against time to run", names, false, true, {},
         "error [m]");
    plot(s, dir, "stiff_evaluations.svg", "Stiff bushed pendulum: error against evaluations", names, false, false,
         {}, "error [m]");
}

// --- Model 3: the four-bar, and model 4: the heavy top ----------------------------

struct FourBarCase : Case {
    mbd_test::FourBar fb{true};
    std::unique_ptr<Simulator> s;
    explicit FourBarCase(const Method& m)
    {
        s = std::make_unique<Simulator>(fb.sys);
        s->implicit_control.stabilized_index2 = m.stabilized;
        s->q.head(3) = mbd_test::FourBar::configuration(1.0);
        s->q(3) = 0.3;
        s->v.head(3) = 2.0 * mbd_test::FourBar::motion(s->q);
        s->v(3) = 0.0;
        s->initialize();
    }
    Simulator& sim() override { return *s; }
    VecX observe() override { return s->q; }
};

/// A heavy symmetric top held at its tip (tests/kernel/test_kernel_drift.cpp):
/// a 1 kg disc 0.2 m across, the tip 0.1 m below its centre, tilted 0.5 rad
/// and spinning at 30 rad/s.
struct TopCase : Case {
    System sys;
    std::unique_ptr<Simulator> s;
    explicit TopCase(const Method& m)
    {
        const Real l = 0.1;
        sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
        sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                           RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.02)), "top");
        sys.constraints.push_back(spherical_closure(Marker{1, Transform3::FromTranslation(Vec3(0.0, 0.0, -l))},
                                                    Marker{0, Transform3()}));
        s = std::make_unique<Simulator>(sys);
        s->implicit_control.stabilized_index2 = m.stabilized;
        const Quat tilt(Eigen::AngleAxisd(0.5, Vec3::UnitX()));
        s->q.head<3>() = tilt * Vec3(0.0, 0.0, l);
        s->q.tail<4>() = tilt.coeffs();
        s->v.setZero();
        s->v(2) = 30.0;
        s->initialize();
    }
    Simulator& sim() override { return *s; }
    VecX observe() override { return s->q; }
};

Real energy(const Simulator& sim)
{
    Data d(sim.system.model);
    forward_kinematics(sim.system.model, d, sim.q, sim.v);
    return kinetic_energy(sim.system.model, d) + potential_energy(sim.system.model, d);
}

/// Energy over `duration` at output steps of h (fixed steps of h; with error
/// control, at the default tolerance within each): the change at the end and
/// the largest, relative to the start; the samples every `every` seconds.
struct EnergyRun {
    std::string method;
    bool ok{true};
    std::string failure;
    Real end{0.0}, largest{0.0};
    Real spin{0.0};   // the top's spin at the end
    Real energy_end{0.0};   // the energy itself at the end [J]
    long long newton_iterations{0};
    Real max_drift{0.0};   // the largest a step left the positions off the constraints
    std::vector<Real> t, change;
};

EnergyRun energy_run(const Factory& make, const Method& m, Real duration, Real h, Real every)
{
    EnergyRun e;
    e.method = m.name;
    std::unique_ptr<Case> c = make(m);
    Simulator& sim = c->sim();
    configure(sim, m, m.adaptive ? 1e-6 : kNewtonTolerance);
    const Real E0 = energy(sim);
    const int steps = static_cast<int>(std::lround(duration / h));
    const int sample = std::max(1, static_cast<int>(std::lround(every / h)));
    try {
        for (int k = 1; k <= steps; ++k) {
            sim.step(h);
            const Real dE = (energy(sim) - E0) / std::abs(E0);
            if (!std::isfinite(dE)) throw MbdError("diverged");
            if (std::abs(dE) > std::abs(e.largest)) e.largest = dE;
            if (k % sample == 0) {
                e.t.push_back(sim.time);
                e.change.push_back(dE);
            }
            e.end = dE;
        }
    } catch (const std::exception& ex) {
        e.ok = false;
        e.failure = std::string(ex.what()).substr(0, std::string(ex.what()).find(':')) + " at "
                    + g(sim.time, 3) + " s";
    }
    e.spin = sim.v(2);
    e.energy_end = energy(sim);
    e.newton_iterations = sim.integrator_stats().newton_iterations;
    e.max_drift = sim.constraint_drift().max_position;
    return e;
}

void four_bar(const std::string& dir)
{
    std::printf("\n## The four-bar with a pendulum (closed loop)\n");
    const Factory make = [](const Method& m) { return std::make_unique<FourBarCase>(m); };
    const Real T = 1.0;
    const VecX ref = reference(make, T, "the joint angles at 1 s");
    fastest_modes(*make(kRK4), "the four-bar");
    const std::vector<Method> fixed = {kRK4, kSIE, kDP, kGA, kGAS, kSDIRK, kRadau, kLIE};
    const std::vector<Real> steps = {8e-3, 4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4};
    Study s{"fourbar", {}};
    for (const Method& m : fixed) {
        for (const Real h : steps) s.runs.push_back(run(make, m, T, h, 0.0));
    }
    error_against(s, ref);
    s.csv(dir);
    std::vector<std::string> names;
    for (const Method& m : fixed) names.push_back(m.name);
    std::printf("\nError of the joint angles at 1 s [rad], fixed steps, projected after each (the stabilized "
                "form: on the constraints):\n");
    convergence_table(s, names, steps);
    {
        // Generalised-alpha's longest step with Newton at a looser tolerance.
        Run r = run(make, kGA, T, 8e-3, 0.0, kStiffNewton);
        if (r.ok) r.error = (r.observed - ref).norm();
        std::printf("\nGeneralised-alpha at 8 ms with Newton at 1e-9: %s.\n",
                    r.ok ? ("error " + g(r.error)).c_str() : r.failure.c_str());
    }
    plot(s, dir, "fourbar_convergence.svg", "Four-bar: error of the joint angles at 1 s", names, true, false,
         guides(s, {{kSIE.name, 1.0}, {kGA.name, 2.0}, {kSDIRK.name, 4.0}, {kRadau.name, 5.0}}),
         "error [rad]");

    // Energy over 100 s (no damping anywhere).
    const std::vector<Method> energy_methods = {kRK4, kSIE, adaptive(kDP), kGA, kGAS, adaptive(kSDIRK),
                                                adaptive(kRadau), kLIE};
    std::printf("\nEnergy over 100 s, relative to the start (17.83 J): the change at the end and the largest; "
                "error control at tolerance 1e-6 within each output step:\n\n| method | 5 ms: at 100 s | largest "
                "| 1 ms: at 100 s | largest |\n|---|---|---|---|---|\n");
    mbd_plot::Plot p;
    p.title = "Four-bar: energy change over 100 s at 5 ms";
    p.xlabel = "time [s]";
    p.ylabel = "|E - E0| / E0";
    p.logx = false;
    for (const Method& m : energy_methods) {
        const EnergyRun a = energy_run(make, m, 100.0, 5e-3, 0.5);
        const EnergyRun b = energy_run(make, m, 100.0, 1e-3, 0.5);
        std::printf("| %s | %s | %s | %s | %s |\n", m.name.c_str(), a.ok ? g(a.end).c_str() : a.failure.c_str(),
                    g(a.largest).c_str(), b.ok ? g(b.end).c_str() : b.failure.c_str(), g(b.largest).c_str());
        mbd_plot::Series series{m.name, a.t, {}, m.adaptive, false};
        for (const Real c : a.change) series.y.push_back(std::abs(c));
        p.series.push_back(series);
    }
    p.write(dir + "/fourbar_energy.svg");
}

void heavy_top()
{
    std::printf("\n## The heavy top (gyroscopic, held at its tip)\n");
    const Factory make = [](const Method& m) { return std::make_unique<TopCase>(m); };
    const std::vector<Method> methods = {kRK4, kSIE, adaptive(kDP), kGA, kGAS, adaptive(kSDIRK), adaptive(kRadau),
                                         kLIE};
    std::string details;
    std::printf("\nOver 100 s (energy 3.861 J, spin 30 rad/s; no damping): energy change at the end and the "
                "largest, relative; the spin at the end [rad/s]:\n\n| method | 5 ms: energy at 100 s | largest | spin "
                "| 1 ms: energy at 100 s | largest | spin |\n|---|---|---|---|---|---|---|\n");
    for (const Method& m : methods) {
        const EnergyRun a = energy_run(make, m, 100.0, 5e-3, 1.0);
        const EnergyRun b = energy_run(make, m, 100.0, 1e-3, 1.0);
        std::printf("| %s | %s | %s | %s | %s | %s | %s |\n", m.name.c_str(),
                    a.ok ? g(a.end).c_str() : a.failure.c_str(), g(a.largest).c_str(), g(a.spin, 5).c_str(),
                    b.ok ? g(b.end).c_str() : b.failure.c_str(), g(b.largest).c_str(), g(b.spin, 5).c_str());
        if (m.method == Integrator::GeneralizedAlpha) {
            details += "\n- " + m.name + ": energy at 100 s " + g(a.energy_end, 12) + " J at 5 ms, "
                       + g(b.energy_end, 12) + " J at 1 ms; Newton iterations " + std::to_string(a.newton_iterations)
                       + " and " + std::to_string(b.newton_iterations) + "; largest drift of a step "
                       + g(a.max_drift, 2) + " and " + g(b.max_drift, 2) + " m";
        }
    }
    std::printf("\nGeneralised-alpha projected and stabilized, in full:\n%s\n", details.c_str());
}

// --- Model 6: the sedan --------------------------------------------------------------

/// The double-wishbone sedan with its drivetrain, from its static
/// equilibrium: throttle 0.4 and the front wheels steered 0.02 rad from the
/// start, both held. Observed: the chassis's position.
struct SedanCase : Case {
    System sys;
    VehicleHandle vh;
    std::unique_ptr<Simulator> s;
    std::unique_ptr<Drivetrain> dt;
    SedanCase()
    {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
        tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
        vh = build_vehicle(sys, tmpl);
        s = std::make_unique<Simulator>(sys);
        set_vehicle_equilibrium(*s, vh);
        vh.set_steering(0.02);
        s->initialize();
        dt = std::make_unique<Drivetrain>(tmpl.drivetrain);
        dt->initialize(*s, vh);
        dt->connect(*s, vh);
        dt->throttle = 0.4;
    }
    Simulator& sim() override { return *s; }
    VecX observe() override { return s->states()[static_cast<std::size_t>(vh.chassis_body)].p_WB; }
};

void sedan(const std::string& dir)
{
    std::printf("\n## The sedan\n");
    const Factory make = [](const Method&) { return std::make_unique<SedanCase>(); };
    const Real T = 2.0;
    fastest_modes(*make(kRK4), "the sedan");
    // The reference: RK4 at 0.05 ms, checked against 0.1 ms.
    const Run ref = run(make, kRK4, T, 5e-5, 0.0);
    const Run half = run(make, kRK4, T, 1e-4, 0.0);
    {
        SedanCase c;
        std::printf("\nReference: RK4 at 0.05 ms (%.1f s to run); at 0.1 ms it differs by %s m. The chassis at 2 s: "
                    "(%s, %s, %s) m from its start (%s, %s, %s); gear %d at the start.\n",
                    ref.seconds, g((ref.observed - half.observed).norm(), 2).c_str(), g(ref.observed(0), 6).c_str(),
                    g(ref.observed(1), 6).c_str(), g(ref.observed(2), 6).c_str(), g(c.observe()(0), 6).c_str(),
                    g(c.observe()(1), 6).c_str(), g(c.observe()(2), 6).c_str(), c.dt->current_gear);
    }
    Study s{"sedan", {}};
    auto sweep = [&](const Method& m, std::initializer_list<Real> values) {
        for (const Real v : values) {
            s.runs.push_back(m.adaptive ? run(make, m, T, 0.01, v) : run(make, m, T, v, 0.0));
        }
    };
    Method mr = kRK4;
    mr.name = "multirate RK4 (fast steps 0.5 ms)";
    sweep(kRK4, {1e-3, 5e-4, 2.5e-4, 1e-4});
    for (const Real h : {4e-3, 2e-3, 1e-3}) {
        mr.multirate = static_cast<int>(std::lround(h / 5e-4));
        s.runs.push_back(run(make, mr, T, h, 0.0));
    }
    sweep(kSIE, {1e-3, 5e-4, 2.5e-4, 1e-4});
    sweep(kGA, {1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2.5e-4});
    sweep(kLIE, {5e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1e-4});
    sweep(adaptive(kDP), {1e-4, 1e-5, 1e-6, 1e-7, 1e-8});
    sweep(adaptive(kSDIRK), {1e-3, 1e-4, 1e-5, 1e-6});
    sweep(adaptive(kRadau), {1e-3, 1e-4, 1e-5, 1e-6, 1e-7});
    error_against(s, ref.observed);
    s.csv(dir);
    const std::vector<std::string> names = {kRK4.name, mr.name, kSIE.name, kGA.name, kLIE.name, adaptive(kDP).name,
                                            adaptive(kSDIRK).name, adaptive(kRadau).name};
    std::printf("\nEvery run (error of the chassis's position at 2 s [m]):\n\n| method | step or tolerance | error "
                "| evaluations | steps | rejected | Newton iterations | time [ms] |\n"
                "|---|---|---|---|---|---|---|---|\n");
    for (const Run& r : s.runs) {
        const bool a = r.method.find("error control") != std::string::npos;
        std::printf("| %s | %s | %s | %lld | %lld | %lld | %lld | %s |\n", r.method.c_str(),
                    param(r.parameter, a).c_str(), r.ok ? g(r.error).c_str() : r.failure.c_str(),
                    r.stats.evaluations, r.stats.steps, r.stats.rejected, r.stats.newton_iterations,
                    g(r.seconds * 1e3).c_str());
    }
    std::printf("\nThe cheapest run within each error:\n");
    cheapest_table(s, names, {1e-2, 1e-3, 1e-4, 1e-5});
    plot(s, dir, "sedan_work.svg", "Sedan: error of the chassis at 2 s against time to run", names, false, true, {},
         "error [m]");
}

} // namespace

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: mbd_verify <folder for the figures and data> [model ...]\n"
                             "models: oscillator pendulum stiff fourbar top sedan (default: all)\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::vector<std::string> which(argv + 2, argv + argc);
    auto wanted = [&](const char* name) {
        return which.empty() || std::find(which.begin(), which.end(), name) != which.end();
    };
    try {
        if (wanted("oscillator")) oscillator(dir);
        if (wanted("pendulum")) smooth_pendulum(dir);
        if (wanted("fourbar")) four_bar(dir);
        if (wanted("top")) heavy_top();
        if (wanted("stiff")) stiff_pendulum(dir);
        if (wanted("sedan")) sedan(dir);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "mbd_verify: %s\n", e.what());
        return 1;
    }
    return 0;
}
