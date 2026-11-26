// Task 7.6, part 1 (D66): a bushing's curve by its shape, a damper with a
// stiffness in series (a Maxwell element) and a bushing with Bouc-Wen
// hysteresis, both state elements, checked on the virtual damper dyno.
// Every expected value is a closed form written out here: the shape's
// pieces, the Maxwell element's response to a sine (transient included) and
// its complex stiffness, and Bouc-Wen's branches for n = 1 and n = 2, which
// are the exact solutions of its equation along a monotone deflection.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "mbd/analysis/damper_dyno.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/hysteresis.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace {

std::string error_of(void (*f)())
{
    try {
        f();
    } catch (const MbdError& e) {
        return e.what();
    }
    return "no error";
}

/// The Maxwell element driven through u = U sin(w t) from rest, its series
/// spring k and damper c linear, tau = c / k. The damper's stroke d obeys
/// tau d' + d = u, d(0) = 0:
///   d = U [sin(w t) - w tau cos(w t) + w tau exp(-t / tau)] / (1 + (w tau)^2),
/// and the force is k (u - d). In the steady state F = U (K' sin + K'' cos),
/// K* = k i w tau / (1 + i w tau): K' = k (w tau)^2 / (1 + (w tau)^2),
/// K'' = k w tau / (1 + (w tau)^2).
struct MaxwellExact {
    Real k, c, U, w;
    Real tau() const { return c / k; }
    Real wt() const { return w * tau(); }
    Real force(Real t) const
    {
        const Real d = U * (std::sin(w * t) - wt() * std::cos(w * t) + wt() * std::exp(-t / tau())) / (1.0 + wt() * wt());
        return k * (U * std::sin(w * t) - d);
    }
    Real storage() const { return k * wt() * wt() / (1.0 + wt() * wt()); }
    Real loss() const { return k * wt() / (1.0 + wt() * wt()); }
    Real steady(Real t) const { return U * (storage() * std::sin(w * t) + loss() * std::cos(w * t)); }
};

DynoElement maxwell(const MaxwellDamperParams& p)
{
    return [p](BodyIndex held, BodyIndex driven) {
        return std::make_shared<MaxwellDamper>(held, driven, Vec3::Zero(), Vec3::Zero(), p);
    };
}

MaxwellDamperParams linear_maxwell(Real k, Real c)
{
    MaxwellDamperParams p;
    p.damper = Curve::linear(c);
    p.series = Curve::linear(k);
    return p;
}

/// The exact flow of Bouc-Wen with n = 1 along a monotone change dx of the
/// deflection, from z. With d the distance travelled and dir the direction,
/// dz/dd = dir A - beta z - gamma dir |z|: where z has the direction's sign
/// (or is zero), dz/dd = dir A - (beta + gamma) z, tending to dir A / (beta +
/// gamma); against it, dz/dd = dir A - (beta - gamma) z, which reaches zero
/// after d0 = ln((z_m - z) / z_m) / (beta - gamma), z_m = dir A / (beta -
/// gamma). Each piece is linear in z, so the solution is an exponential.
Real flow_n1(const BoucWen& p, Real z, Real dx)
{
    if (dx == 0.0) return z;
    const Real dir = dx > 0.0 ? 1.0 : -1.0;
    Real d = std::abs(dx);
    if (z * dir < 0.0) {
        const Real r = p.beta - p.gamma;
        const Real zm = dir * p.A / r;
        const Real d0 = std::log((zm - z) / zm) / r;
        if (d <= d0) return zm + (z - zm) * std::exp(-r * d);
        z = 0.0;
        d -= d0;
    }
    const Real r = p.beta + p.gamma;
    const Real zi = dir * p.A / r;
    return zi + (z - zi) * std::exp(-r * d);
}

/// Bouc-Wen's z (n = 1) at each time of `t` along x = U sin(w t) from z = 0,
/// composed of exact flows; where a turning point (4 f t odd) lies between
/// two times, the flow goes to it and back.
std::vector<Real> exact_z(const BoucWen& p, const std::vector<Real>& t, Real U, Real f)
{
    const Real w = 2.0 * pi * f;
    std::vector<Real> z(t.size(), 0.0);
    for (std::size_t i = 0; i + 1 < t.size(); ++i) {
        const Real x0 = U * std::sin(w * t[i]);
        const Real x1 = U * std::sin(w * t[i + 1]);
        const Real phase0 = 4.0 * f * t[i], phase1 = 4.0 * f * t[i + 1];
        const Real odd = 2.0 * std::floor((phase0 + 1.0) / 2.0) + 1.0;   // the first odd integer above phase0
        if (odd < phase1) {
            const Real xt = U * std::sin(0.5 * pi * odd);
            z[i + 1] = flow_n1(p, flow_n1(p, z[i], xt - x0), x1 - xt);
        } else {
            z[i + 1] = flow_n1(p, z[i], x1 - x0);
        }
    }
    return z;
}

/// RK4's error over a step on which the rate along the solution, a function
/// of time, has a kink (a jump J in its first derivative): RK4 is then
/// Simpson's rule on that step, whose error for J max(t - theta h, 0) is at
/// most J h^2 / 24 over the kink's place theta (its largest, at theta = 1/2;
/// elsewhere J h^2 / 72). Bouc-Wen's rate dz/dt = A x' - beta |x'| z - gamma
/// x' |z| (n = 1) has two kinds: at a turning point |x'| turns, J = 2 beta
/// |z| U w^2 <= 2 beta z_u U w^2; where z crosses zero |z| turns while z' =
/// A x', J = 2 gamma A x'^2 <= 2 gamma A (U w)^2. Away from them the flows
/// contract (their exponents are negative), so the errors add, never grow.
Real kink_bound(const BoucWen& p, Real U, Real w, Real h, int turns, int crossings)
{
    const Real z_u = p.A / (p.beta + p.gamma);
    const Real at_turn = 2.0 * p.beta * z_u * U * w * w * h * h / 24.0;
    const Real at_zero = 2.0 * std::abs(p.gamma) * p.A * U * U * w * w * h * h / 24.0;
    return turns * at_turn + crossings * at_zero;
}

} // namespace

// ============================================================================
// A bushing's curve by its shape
// ============================================================================

TEST_CASE("Bushing shape: linear, then a smooth progression, then the stop's straight line",
          "[forces][hysteresis][shape]")
{
    // k = 1000, a = 0.01, b = 0.03, k_s = 21000: w = 0.02, k_s - k = 20000.
    // At 0.005 (linear): 5 N, slope 1000, energy 1000 * 0.005^2 / 2 = 0.0125.
    // At 0.02 (s = 0.5): 1000 * 0.02 + 20000 * 0.02 * (0.125 - 0.03125) =
    //   57.5 N; slope 1000 + 20000 * (0.75 - 0.25) = 11000; energy 1000 *
    //   0.02^2 / 2 + 20000 * 0.02^2 * (0.0625 / 4 - 0.03125 / 10) = 0.3.
    // At 0.03 (s = 1): 30 + 20000 * 0.02 / 2 = 230 N, slope 21000, energy
    //   0.45 + 0.15 * 20000 * 0.02^2 = 1.65.
    // At 0.04: 230 + 21000 * 0.01 = 440 N; energy 1.65 + 230 * 0.01 + 21000 *
    //   0.01^2 / 2 = 5.0. Odd in x: f(-0.02) = -57.5, the energy even.
    // All to rounding (the pieces are a few products).
    const Curve c = Curve::bushing({1000.0, 0.01, 0.03, 21000.0});
    CHECK(c.is_bushing());
    CHECK_FALSE(c.is_table());
    CHECK_FALSE(c.is_zero());
    CHECK_THAT(c.value(0.005), WithinRel(5.0, 1e-14));
    CHECK_THAT(c.value(0.02), WithinRel(57.5, 1e-14));
    CHECK_THAT(c.value(0.03), WithinRel(230.0, 1e-14));
    CHECK_THAT(c.value(0.04), WithinRel(440.0, 1e-14));
    CHECK_THAT(c.value(-0.02), WithinRel(-57.5, 1e-14));
    CHECK(c.slope(0.005) == 1000.0);
    CHECK_THAT(c.slope(0.02), WithinRel(11000.0, 1e-13));
    CHECK(c.slope(0.04) == 21000.0);
    CHECK_THAT(c.slope(-0.02), WithinRel(11000.0, 1e-13));
    CHECK_THAT(c.integral(0.0, 0.005), WithinRel(0.0125, 1e-14));
    CHECK_THAT(c.integral(0.0, 0.02), WithinRel(0.3, 1e-13));
    CHECK_THAT(c.integral(0.0, 0.03), WithinRel(1.65, 1e-14));
    CHECK_THAT(c.integral(0.0, 0.04), WithinRel(5.0, 1e-14));
    CHECK_THAT(c.integral(-0.04, 0.02), WithinRel(-4.7, 1e-14));

    // A stop at the end of the linear range changes the slope at once:
    // k = 1000 to 0.01, then 5000: f(0.02) = 10 + 5000 * 0.01 = 60 N.
    const Curve abrupt = Curve::bushing({1000.0, 0.01, 0.01, 5000.0});
    CHECK_THAT(abrupt.value(0.02), WithinRel(60.0, 1e-14));
    CHECK(abrupt.slope(0.009) == 1000.0);
    CHECK(abrupt.slope(0.011) == 5000.0);
}

TEST_CASE("Bushing shape: its slope and energy are the derivative and integral of its force",
          "[forces][hysteresis][shape]")
{
    // Central differences with h = 1e-6, at points in each range and across
    // the joints. The slope's own second derivative is at most 6 (k_s - k) /
    // w^2 = 3e8, so the difference of the force misses the slope by at most
    // h^2 / 6 * 3e8 = 5e-5 N/m, and rounding adds eps * 440 / h = 1e-7: allow
    // 1e-4 N/m. For the energy, whose third derivative is the force's
    // second, at most 1.5 (k_s - k) / w = 1.5e6: h^2 / 6 * 1.5e6 = 2.5e-7,
    // with rounding eps * 5 / h = 1e-9: allow 1e-6 N.
    const Curve c = Curve::bushing({1000.0, 0.01, 0.03, 21000.0});
    const Real h = 1e-6;
    Real worst_slope = 0.0, worst_force = 0.0;
    for (Real x : {0.004, 0.0099999, 0.010001, 0.012, 0.025, 0.029, 0.03, 0.05, -0.018, -0.0300001}) {
        const Real ds = (c.value(x + h) - c.value(x - h)) / (2.0 * h) - c.slope(x);
        const Real df = (c.integral(0.0, x + h) - c.integral(0.0, x - h)) / (2.0 * h) - c.value(x);
        worst_slope = std::max(worst_slope, std::abs(ds));
        worst_force = std::max(worst_force, std::abs(df));
    }
    INFO("worst slope difference " << worst_slope << " N/m, worst force difference " << worst_force << " N");
    CHECK(worst_slope < 1e-4);
    CHECK(worst_force < 1e-6);

    // The second derivative is continuous at both joints (the slope's cubic
    // is flat there): the slope moves from k by (k_s - k)(3 e^2 - 2 e^3),
    // e = h / w, within h of a, and from k_s by the same within h of b. With
    // h = 1e-6: 20000 * 3 * (5e-5)^2 = 1.5e-4 N/m, to the cubic's rounding.
    const Real e = h / 0.02;
    const Real moved = 20000.0 * (3.0 * e * e - 2.0 * e * e * e);
    CHECK_THAT(c.slope(0.01 + h) - 1000.0, WithinAbs(moved, 1e-10));
    CHECK_THAT(21000.0 - c.slope(0.03 - h), WithinAbs(moved, 1e-10));
}

TEST_CASE("Curves: a rising curve's inverse returns x to rounding, and a curve that does not rise has none",
          "[forces][hysteresis][shape]")
{
    // Each x through value() and back: the round trip loses only rounding,
    // a few ulps of x (allow 1e-14 relative, or 1e-18 absolute near zero).
    const Curve line = Curve::linear(2000.0, 10.0);
    CHECK_THAT(line.inverse(410.0), WithinRel(0.2, 1e-15));
    const Curve shape = Curve::bushing({1000.0, 0.01, 0.03, 21000.0});
    CHECK_THAT(shape.inverse(57.5), WithinRel(0.02, 1e-14));
    CHECK_THAT(shape.inverse(-440.0), WithinRel(-0.04, 1e-14));
    // A digressive damper, stiffer in rebound: velocities [m/s], forces [N].
    // Its secants fall outward slowly enough that the interpolation's end
    // slopes stay positive (at -1 m/s, ((2 0.5 + 0.3) 800 - 0.5 1333) / 0.8
    // = 467 N s/m; with steeper falls the end slope is set to zero, and the
    // curve is no longer invertible beyond its last point).
    const Curve damper = Curve::table({-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0},
                                      {-1200.0, -800.0, -400.0, 0.0, 900.0, 1600.0, 2400.0});
    CHECK(damper.is_increasing());
    CHECK(line.is_increasing());
    CHECK(shape.is_increasing());
    Real worst = 0.0;
    for (const Curve* c : {&line, &shape, &damper}) {
        for (Real x : {-1.7, -0.5, -0.1, -0.0123, -1e-9, 0.0, 3e-7, 0.015, 0.0299, 0.05, 0.25, 0.999, 2.5}) {
            const Real back = c->inverse(c->value(x));
            worst = std::max(worst, std::abs(back - x) / std::max(std::abs(x), 1e-4));
        }
    }
    INFO("worst relative round trip " << worst);
    CHECK(worst < 1e-14);

    CHECK_FALSE(Curve().is_increasing());
    CHECK_FALSE(Curve::table({0.0, 1.0, 2.0}, {0.0, 1.0, 1.0}).is_increasing());
    CHECK_FALSE(Curve::bushing({1000.0, 0.01, 0.03, 0.0}).is_increasing());
    CHECK_THAT(error_of([] { (void)Curve::table({0.0, 1.0, 2.0}, {0.0, 1.0, 1.0}).inverse(0.5); }),
               ContainsSubstring("MBD-F057"));
    CHECK_THAT(error_of([] { (void)Curve::bushing({1000.0, 0.02, 0.01, 5000.0}); }), ContainsSubstring("MBD-F056"));
    CHECK_THAT(error_of([] { (void)Curve::bushing({-1.0, 0.0, 0.01, 5000.0}); }), ContainsSubstring("MBD-F056"));
    CHECK_THAT(error_of([] { (void)Curve::bushing({1000.0, 0.0, std::nan(""), 5000.0}); }),
               ContainsSubstring("MBD-F056"));
}

// ============================================================================
// The Maxwell damper
// ============================================================================

TEST_CASE("Maxwell damper: on the dyno it follows its closed-form response from rest, transient included",
          "[forces][hysteresis][maxwell]")
{
    // k = 2e5 N/m, c = 2000 N s/m: tau = 0.01 s. U = 25 mm at 10 Hz (w tau =
    // 0.628), 1000 steps a cycle: h = 1e-4 s, h / tau = 0.01. RK4's error on
    // the relaxation, per step, is (h / tau)^5 / 120 = 8.3e-13 of the state;
    // the relaxation forgets earlier errors over tau / h = 100 steps, so they
    // stay below 8.3e-11 of the force, about 2700 N here: 2.2e-7 N. The rig's
    // motion is integrated by the same RK4 (w h = 0.0063, (w h)^5 / 120 =
    // 8e-14) and projected onto its driver. Allow 1e-6 N.
    const MaxwellExact exact{2e5, 2000.0, 0.025, 2.0 * pi * 10.0};
    DynoSettings s;
    s.amplitude = exact.U;
    s.frequency = 10.0;
    s.cycles = 3;
    s.steps_per_cycle = 1000;
    s.length = 0.3;
    const DynoRecord r = run_damper_dyno(maxwell(linear_maxwell(exact.k, exact.c)), s);
    REQUIRE(r.force.size() == 3001);
    Real worst = 0.0, largest = 0.0;
    for (std::size_t i = 0; i < r.force.size(); ++i) {
        worst = std::max(worst, std::abs(r.force[i] - exact.force(r.time[i])));
        largest = std::max(largest, std::abs(r.force[i]));
        CHECK_THAT(r.stroke[i], WithinAbs(exact.U * std::sin(exact.w * r.time[i]), 1e-9));
    }
    INFO("largest force " << largest << " N, worst difference from the closed form " << worst << " N");
    CHECK(worst < 1e-6);
}

TEST_CASE("Maxwell damper: its loop on the dyno is the closed-form steady-state response",
          "[forces][hysteresis][maxwell]")
{
    // The plan's test of task 7.6. The same damper and stroke; the third
    // cycle, from t = 0.2 s, when the transient is down to exp(-20) = 2.1e-9
    // of its start, k U w tau / (1 + (w tau)^2) = 2253 N: 4.6e-6 N, beside
    // the integrator's 2.2e-7 N. Allow 1e-5 N on the loop pointwise.
    //   K' = 2e5 * 0.39478 / 1.39478 = 56609 N/m: in phase, U K' = 1415.2 N;
    //   K'' = 2e5 * 0.62832 / 1.39478 = 90096 N/m: in quadrature, U K'' =
    //   2252.4 N; the energy a cycle, pi U^2 K'' = 176.90 J.
    // The force's first harmonic, by sums over the cycle's samples (exact for
    // a sine: the samples are equally spaced over a period), and the energy
    // by the dyno's rule (exact for a periodic loop) differ from these by
    // the loop's errors only: allow 1e-8 of each (1e-5 N in 2000 N).
    const MaxwellExact exact{2e5, 2000.0, 0.025, 2.0 * pi * 10.0};
    DynoSettings s;
    s.amplitude = exact.U;
    s.frequency = 10.0;
    s.cycles = 3;
    s.steps_per_cycle = 1000;
    s.length = 0.3;
    const DynoRecord r = run_damper_dyno(maxwell(linear_maxwell(exact.k, exact.c)), s);
    Real worst = 0.0, in_phase = 0.0, quadrature = 0.0;
    const std::size_t first = 2000;
    for (std::size_t i = first; i <= first + 1000; ++i) {
        worst = std::max(worst, std::abs(r.force[i] - exact.steady(r.time[i])));
        if (i < first + 1000) {
            in_phase += r.force[i] * std::sin(exact.w * r.time[i]) / 500.0;
            quadrature += r.force[i] * std::cos(exact.w * r.time[i]) / 500.0;
        }
    }
    const Real energy = r.energy(2);
    INFO("loop against the steady state: worst " << worst << " N; in phase " << in_phase << " N (closed form "
         << exact.U * exact.storage() << "), in quadrature " << quadrature << " N (" << exact.U * exact.loss()
         << "); energy " << energy << " J (" << pi * exact.U * exact.U * exact.loss() << ")");
    CHECK(worst < 1e-5);
    CHECK_THAT(in_phase, WithinRel(exact.U * exact.storage(), 1e-8));
    CHECK_THAT(quadrature, WithinRel(exact.U * exact.loss(), 1e-8));
    CHECK_THAT(energy, WithinRel(pi * exact.U * exact.U * exact.loss(), 1e-8));
}

TEST_CASE("Maxwell damper: the energy a cycle peaks at w tau = 1, as K'' does",
          "[forces][hysteresis][maxwell]")
{
    // pi U^2 k w tau / (1 + (w tau)^2) at 2 Hz, at 1 / (2 pi tau) = 15.92 Hz
    // (the peak, pi U^2 k / 2 = 196.35 J) and at 50 Hz, the last of enough
    // cycles that the transient's exp(-t / tau) is below 2e-9 when it
    // starts (t >= 20 tau). Steps enough that h <= tau / 100 and w h <=
    // 0.0063 (5000 a cycle at 2 Hz, 1000 at the others), as above: allow
    // 1e-8 relative.
    const Real k = 2e5, c = 2000.0, U = 0.025;
    for (Real f : {2.0, 1.0 / (2.0 * pi * 0.01), 50.0}) {
        const MaxwellExact exact{k, c, U, 2.0 * pi * f};
        DynoSettings s;
        s.amplitude = U;
        s.frequency = f;
        s.cycles = std::max(3, static_cast<int>(std::ceil(1.0 + 20.0 * exact.tau() * f)));
        s.steps_per_cycle = std::max(1000, static_cast<int>(std::ceil(100.0 / (f * exact.tau()))));
        s.length = 0.3;
        const DynoRecord r = run_damper_dyno(maxwell(linear_maxwell(k, c)), s);
        const Real energy = r.energy(s.cycles - 1);
        INFO(f << " Hz: " << energy << " J against " << pi * U * U * exact.loss() << " J");
        CHECK_THAT(energy, WithinRel(pi * U * U * exact.loss(), 1e-8));
    }
}

TEST_CASE("Maxwell damper: with a digressive damper, the rig's work is the damper's loss and the mount's energy",
          "[forces][hysteresis][maxwell]")
{
    // No closed form for a tabulated damper, but energy is conserved: the
    // rig's work F u' equals the damper's loss F d' (d' = damper^-1(F)) plus
    // the series stiffness's energy F^2 / (2 k), F s' summed: over any span,
    //   integral F u' dt = integral F damper^-1(F) dt + [F^2 / (2 k)].
    // Both integrals by the trapezoidal rule over the record. The force is
    // smooth but for the damper table's knots, where d'' jumps: the rule's
    // error is then of order h^2 times the integrands' second derivatives,
    // about w^2 times the work (h = 1.7e-4 s, w = 18.8 rad/s: (w h)^2 / 12 =
    // 8e-7 of 200 J, 1.6e-4 J). Allow 1e-3 J over the three cycles.
    MaxwellDamperParams p;
    p.damper = Curve::table({-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0},
                            {-1200.0, -800.0, -400.0, 0.0, 900.0, 1600.0, 2400.0});
    p.series = Curve::linear(5e5);
    DynoSettings s;
    s.amplitude = 0.02;
    s.frequency = 3.0;
    s.cycles = 3;
    s.steps_per_cycle = 2000;
    s.length = 0.3;
    const DynoRecord r = run_damper_dyno(maxwell(p), s);
    Real rig = 0.0, loss = 0.0;
    for (std::size_t i = 0; i + 1 < r.force.size(); ++i) {
        const Real dt = r.time[i + 1] - r.time[i];
        rig += 0.5 * (r.force[i] * r.velocity[i] + r.force[i + 1] * r.velocity[i + 1]) * dt;
        loss += 0.5 * (r.force[i] * p.damper.inverse(r.force[i]) + r.force[i + 1] * p.damper.inverse(r.force[i + 1])) * dt;
    }
    const Real stored = r.force.back() * r.force.back() / (2.0 * 5e5);   // from zero at the start
    INFO("rig's work " << rig << " J, damper's loss " << loss << " J, mount's energy " << stored
         << " J; the balance's residual " << rig - loss - stored << " J");
    CHECK(loss > 0.0);
    CHECK_THAT(rig, WithinAbs(loss + stored, 1e-3));
    // Rebound is the stiffer side: the force's largest tension exceeds its
    // largest compression.
    const auto [lo, hi] = std::minmax_element(r.force.begin(), r.force.end());
    CHECK(*hi > -*lo);
}

TEST_CASE("Maxwell damper: Radau IIA follows it at steps ten times its time constant",
          "[forces][hysteresis][maxwell]")
{
    // A stiff top mount: k = 2e7 N/m, c = 2000 N s/m, tau = 1e-4 s. At 10 Hz
    // with 100 steps a cycle, h = 1e-3 s = 10 tau, where RK4's step
    // multiplies the relaxation by 291 (1 - 10 + 50 - 166.7 + 416.7): it
    // cannot. Radau IIA, its step control off so that every step is h,
    // multiplies it by R(-10) = (1 - 4 + 5) / (1 + 6 + 15 + 16.7) = 0.052
    // (L-stable). What is left is its error on the forced motion, which its
    // coefficients give exactly for this linear system (the rig's velocity
    // at the stages from the driver's acceleration, exact at each step's
    // start after the projection; s's stages from it): its steady state
    // differs from the true one by 5.56e-11 of U |K*| = 3142 N, 1.748e-7 N
    // (journal data of task 7.6). Allow 2e-7 N on the third cycle against
    // the steady state (the transient, decaying by 0.052 a step, is gone).
    const MaxwellExact exact{2e7, 2000.0, 0.025, 2.0 * pi * 10.0};
    DynoSettings s;
    s.amplitude = exact.U;
    s.frequency = 10.0;
    s.cycles = 3;
    s.steps_per_cycle = 100;
    s.length = 0.3;
    s.method = kernel::Integrator::RadauIIA;
    s.step_control.adaptive = false;
    const DynoRecord r = run_damper_dyno(maxwell(linear_maxwell(exact.k, exact.c)), s);
    Real worst = 0.0;
    for (std::size_t i = 200; i < r.force.size(); ++i) {
        worst = std::max(worst, std::abs(r.force[i] - exact.steady(r.time[i])));
    }
    INFO("Radau IIA at h = 10 tau: worst " << worst << " N of " << exact.U * std::hypot(exact.storage(), exact.loss())
         << " N");
    CHECK(worst < 2e-7);
}

TEST_CASE("Maxwell damper: a mass on a spring with one settles where the spring alone holds it",
          "[forces][hysteresis][maxwell]")
{
    // 10 kg on a vertical slider, a spring of 1e4 N/m (free at the start) and
    // a Maxwell damper (1e6 N/m in series with 1000 N s/m) in parallel. The
    // Maxwell element carries no force at rest: statics, holding its state
    // at zero, finds the spring's deflection m g / k = 9.80665 mm (to
    // statics' 1e-6 m/s^2 over k / m = 1000 s^-2: 1e-9 m). Released from the
    // free length instead, the mass settles there too: the slowest root of
    // the motion decays at about k / c = 10 s^-1, exp(-50) over 5 s.
    auto build = [](kernel::System& sys) {
        sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
        const BodyIndex b = sys.model.add_body(0, std::make_shared<kernel::PrismaticJointModel>(), Transform3::Identity(),
                                               Transform3::Identity(),
                                               RigidBodyInertia::from_solid_box(10.0, Vec3(0.1, 0.1, 0.1)), "mass");
        SpringDamperParams sp;
        sp.free_length = 0.5;
        sp.spring = Curve::linear(1e4);
        sys.force_elements.push_back(
            std::make_shared<SpringDamper>(0, b, Vec3(0.0, 0.0, 0.5), Vec3::Zero(), sp));
        sys.force_elements.push_back(std::make_shared<MaxwellDamper>(0, b, Vec3(0.0, 0.0, 0.5), Vec3::Zero(),
                                                                     linear_maxwell(1e6, 1000.0)));
        return b;
    };
    const Real expected = -10.0 * g_accel / 1e4;
    {
        kernel::System sys;
        build(sys);
        kernel::Simulator sim(sys);
        sim.initialize();
        const kernel::StaticReport report = kernel::static_equilibrium(sim);
        INFO(report.summary(sys.model));
        CHECK(report.converged);
        CHECK_THAT(sim.q(0), WithinAbs(expected, 1e-9));
    }
    {
        kernel::System sys;
        build(sys);
        kernel::Simulator sim(sys);
        sim.initialize();
        sim.run(5.0, 1e-4);
        INFO("after 5 s: " << sim.q(0) << " m, Maxwell state " << sim.x(0) << " m");
        CHECK_THAT(sim.q(0), WithinAbs(expected, 1e-9));
        CHECK(std::abs(sim.x(0)) < 1e-12);
    }
}

TEST_CASE("Maxwell damper: equal and opposite forces along the line, and its refusals",
          "[forces][hysteresis][maxwell]")
{
    // Two free bodies in general poses, the state set to 1 mm: the forces on
    // them sum to zero and so do their moments about the world's origin
    // (each body's moment is about its own origin), to rounding; the force
    // is k s = 2e5 * 1e-3 = 200 N, pulling the anchors together.
    MaxwellDamper e(1, 2, Vec3(0.1, -0.2, 0.05), Vec3(-0.03, 0.04, 0.2), linear_maxwell(2e5, 2000.0));
    e.set_state(1e-3);
    std::vector<RigidBodyState> states(3);
    states[1].p_WB = Vec3(0.3, 0.1, -0.2);
    states[1].q_WB = Quat(Eigen::AngleAxis<Real>(0.4, Vec3(1.0, 2.0, 3.0).normalized()));
    states[1].v_WB = Vec3(0.2, -0.1, 0.3);
    states[1].w_WB = Vec3(-1.0, 0.5, 0.2);
    states[2].p_WB = Vec3(-0.4, 0.6, 0.5);
    states[2].q_WB = Quat(Eigen::AngleAxis<Real>(-0.7, Vec3(-2.0, 1.0, 0.5).normalized()));
    states[2].v_WB = Vec3(-0.3, 0.2, 0.1);
    states[2].w_WB = Vec3(0.3, -0.2, 1.1);
    std::vector<RigidBodyForces> f(3);
    e.apply(states, f);
    const Vec3 a1 = states[1].p_WB + states[1].q_WB * e.anchor1();
    const Vec3 a2 = states[2].p_WB + states[2].q_WB * e.anchor2();
    const Vec3 dir = (a2 - a1).normalized();
    CHECK((f[1].f_W + f[2].f_W).norm() < 1e-12);
    const Vec3 moment = states[1].p_WB.cross(f[1].f_W) + f[1].tau_W + states[2].p_WB.cross(f[2].f_W) + f[2].tau_W;
    CHECK(moment.norm() < 1e-12);
    CHECK_THAT(f[2].f_W.dot(dir), WithinRel(-200.0, 1e-14));
    CHECK((f[2].f_W + 200.0 * dir).norm() < 1e-12);
    // Its state's rate: the length's rate less the damper's, F / c = 0.1 m/s.
    const Real length_rate = ((states[2].v_WB + states[2].w_WB.cross(states[2].q_WB * e.anchor2()))
                              - (states[1].v_WB + states[1].w_WB.cross(states[1].q_WB * e.anchor1()))).dot(dir);
    CHECK_THAT(e.last_length_rate(), WithinRel(length_rate, 1e-14));
    CHECK_THAT(e.state_rate(1e-3, length_rate), WithinRel(length_rate - 0.1, 1e-14));

    CHECK_THAT(error_of([] {
                   MaxwellDamperParams p;
                   p.damper = Curve::table({-1.0, 0.0, 1.0}, {-500.0, 0.0, 0.0});
                   p.series = Curve::linear(1e6);
                   MaxwellDamper bad(0, 1, Vec3::Zero(), Vec3::Zero(), p);
               }),
               ContainsSubstring("MBD-F058"));
    CHECK_THAT(error_of([] { MaxwellDamper bad(0, 1, Vec3::Zero(), Vec3::Zero(), linear_maxwell(0.0, 1000.0)); }),
               ContainsSubstring("MBD-F058"));
    CHECK_THAT(error_of([] { (void)run_damper_dyno(maxwell(linear_maxwell(1e5, 1e3)), DynoSettings{0.0}); }),
               ContainsSubstring("MBD-A030"));
    CHECK_THAT(error_of([] { (void)run_damper_dyno(DynoElement{}); }), ContainsSubstring("MBD-A030"));
}

// ============================================================================
// The Bouc-Wen bushing
// ============================================================================

TEST_CASE("Bouc-Wen bushing: on the dyno its force follows the exact branches, loop after loop",
          "[forces][hysteresis][boucwen]")
{
    // Along Z: an elastic 2e4 N/m (Bouc-Wen's alpha k), damping 50 N s/m and
    // hysteresis k = 1e5 N/m, A = 1, beta = 300 /m, gamma = 100 /m, n = 1:
    // z_u = 1 / 400 = 2.5 mm, a hysteretic force of up to 250 N. U = 10 mm
    // at 1 Hz, two cycles, 10000 steps a cycle (h = 1e-4 s). The force
    // expected is 2e4 x + 50 x' + 1e5 z, z composed of exact flows. RK4's
    // error comes from the rate's kinks (kink_bound): four turning points
    // and four zero crossings of z in two cycles, at most 2.4e-10 m each
    // (the smooth parts' error, with rates of at most (beta + gamma) U w =
    // 25 s^-1, h times that 2.5e-3, is far below). Allow the bound in z,
    // times 1e5 N/m, plus 1e-9 N of rounding.
    BoucWenBushingParams p;
    p.bushing.stiffness[2] = Curve::linear(2e4);
    p.bushing.damping[2] = Curve::linear(50.0);
    p.translational[2] = BoucWen{1e5, 1.0, 300.0, 100.0, 1.0};
    DynoSettings s;
    s.amplitude = 0.01;
    s.frequency = 1.0;
    s.cycles = 2;
    s.steps_per_cycle = 10000;
    const DynoRecord r = run_damper_dyno(
        [p](BodyIndex held, BodyIndex driven) {
            return std::make_shared<BoucWenBushing>(held, Transform3::Identity(), driven, Transform3::Identity(), p);
        },
        s);
    const std::vector<Real> z = exact_z(p.translational[2], r.time, s.amplitude, s.frequency);
    const Real w = 2.0 * pi * s.frequency;
    const Real allowed = 1e5 * kink_bound(p.translational[2], s.amplitude, w, 1e-4, 4, 4) + 1e-9;
    Real worst = 0.0, largest_z = 0.0;
    for (std::size_t i = 0; i < r.force.size(); ++i) {
        const Real expected = 2e4 * r.stroke[i] + 50.0 * r.velocity[i] + 1e5 * z[i];
        worst = std::max(worst, std::abs(r.force[i] - expected));
        largest_z = std::max(largest_z, std::abs(z[i]));
    }
    INFO("worst " << worst << " N, allowed " << allowed << " N; largest |z| " << largest_z << " m (z_u 2.5e-3)");
    CHECK(worst < allowed);
    CHECK(largest_z > 0.99 * 2.5e-3);   // the loop reaches saturation: (beta + gamma) U = 4
    CHECK(largest_z < 2.5e-3);
    // Every loop dissipates: the energy a cycle is positive.
    CHECK(r.energy(0) > 0.0);
    CHECK(r.energy(1) > 0.0);
}

TEST_CASE("Bouc-Wen bushing: with n = 2 the first loading is z_u tanh(sqrt(A (beta + gamma)) x)",
          "[forces][hysteresis][boucwen]")
{
    // n = 2, loading from rest: dz/dx = A - (beta + gamma) z^2, so z = z_u
    // tanh(sqrt(A (beta + gamma)) x) with z_u = sqrt(A / (beta + gamma)).
    // A = 1, beta = 3e5, gamma = 1e5 /m^2: z_u = 1.5811 mm, the rate 632.5
    // /m. The first quarter cycle (x from 0 to U = 10 mm) has no kink: RK4's
    // error is that of a smooth function of time, h^5 z^(5) / 120 a step at
    // most, z^(5) of order z_u (632.5 U w)^5 = 1.58e-3 * 39.7^5 = 1.6e5 m/s^5:
    // 1.3e-17 m a step, 3.3e-14 m over 2500 steps. Allow 1e-12 m.
    BoucWenBushingParams p;
    p.translational[2] = BoucWen{1e5, 1.0, 3e5, 1e5, 2.0};
    DynoSettings s;
    s.amplitude = 0.01;
    s.frequency = 1.0;
    s.cycles = 1;
    s.steps_per_cycle = 10000;
    const DynoRecord r = run_damper_dyno(
        [p](BodyIndex held, BodyIndex driven) {
            return std::make_shared<BoucWenBushing>(held, Transform3::Identity(), driven, Transform3::Identity(), p);
        },
        s);
    const Real z_u = std::sqrt(1.0 / 4e5);
    CHECK_THAT(p.translational[2].ultimate(), WithinRel(z_u, 1e-15));
    Real worst = 0.0;
    for (std::size_t i = 0; i <= 2500; ++i) {
        worst = std::max(worst, std::abs(r.force[i] / 1e5 - z_u * std::tanh(std::sqrt(4e5) * r.stroke[i])));
    }
    INFO("worst " << worst << " m in z");
    CHECK(worst < 1e-12);
}

TEST_CASE("Bouc-Wen bushing: its loop does not depend on how fast it is traced", "[forces][hysteresis][boucwen]")
{
    // Without damping the force depends on the path of x alone: at 0.1 Hz
    // and at 10 Hz, with the same steps a cycle, the dyno takes the same
    // steps in the phase, and the two forces differ by rounding only (w h =
    // 2 pi / N computed two ways, a few ulps, carried through 4000 steps:
    // allow 1e-10 of z_u's 250 N).
    BoucWenBushingParams p;
    p.bushing.stiffness[2] = Curve::linear(2e4);
    p.translational[2] = BoucWen{1e5, 1.0, 300.0, 100.0, 1.0};
    auto run = [&p](Real f) {
        DynoSettings s;
        s.amplitude = 0.01;
        s.frequency = f;
        s.cycles = 2;
        s.steps_per_cycle = 2000;
        return run_damper_dyno(
            [p](BodyIndex held, BodyIndex driven) {
                return std::make_shared<BoucWenBushing>(held, Transform3::Identity(), driven, Transform3::Identity(),
                                                        p);
            },
            s);
    };
    const DynoRecord slow = run(0.1), fast = run(10.0);
    Real worst = 0.0;
    for (std::size_t i = 0; i < slow.force.size(); ++i) worst = std::max(worst, std::abs(slow.force[i] - fast.force[i]));
    INFO("worst difference " << worst << " N");
    CHECK(worst < 2.5e-8);
}

TEST_CASE("Bouc-Wen bushing: about a rotation axis, through a driven hinge, the exact branches again",
          "[forces][hysteresis][boucwen]")
{
    // A body on a hinge about Y, driven through theta = 0.05 sin(2 pi t)
    // rad; the bushing's hysteresis about Y: k = 500 N m/rad, A = 1, beta =
    // 60, gamma = 20 /rad (z_u = 0.0125 rad). The moment on the body about Y
    // is -500 z, z composed of exact flows; the bound as on the dyno, with U
    // = 0.05 rad, h = 1e-4 s, four turns and four crossings, times 500.
    kernel::System sys;
    sys.model.gravity = Vec3::Zero();
    const Mat3 R = Eigen::AngleAxis<Real>(-0.5 * pi, Vec3::UnitX()).toRotationMatrix();   // the joint's Z onto Y
    const BodyIndex b = sys.model.add_body(0, std::make_shared<kernel::RevoluteJointModel>(), Transform3::FromRotation(R),
                                           Transform3::FromRotation(R),
                                           RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), "arm");
    const Real Theta = 0.05, w = 2.0 * pi;
    sys.constraints.push_back(std::make_shared<kernel::JointDriver>(
        sys.model, b,
        kernel::TimeFunction([=](Real t) { return Theta * std::sin(w * t); },
                             [=](Real t) { return Theta * w * std::cos(w * t); },
                             [=](Real t) { return -Theta * w * w * std::sin(w * t); })));
    BoucWenBushingParams p;
    p.rotational[1] = BoucWen{500.0, 1.0, 60.0, 20.0, 1.0};
    auto bushing = std::make_shared<BoucWenBushing>(0, Transform3::Identity(), b, Transform3::Identity(), p);
    sys.force_elements.push_back(bushing);
    REQUIRE(bushing->size() == 1);
    REQUIRE(bushing->axes_with_hysteresis() == std::vector<int>{1});

    kernel::Simulator sim(sys);
    sim.initialize();
    std::vector<Real> t{0.0}, moment{0.0};
    std::vector<RigidBodyForces> f(2);
    for (int i = 0; i < 20000; ++i) {
        sim.step(1e-4);
        sim.acceleration(sim.q, sim.v, sim.x, sim.time);
        f.assign(2, RigidBodyForces{});
        bushing->apply(sim.states(), f);
        t.push_back(sim.time);
        moment.push_back(f[1].tau_W.y());
    }
    const std::vector<Real> z = exact_z(p.rotational[1], t, Theta, 1.0);
    const Real allowed = 500.0 * kink_bound(p.rotational[1], Theta, w, 1e-4, 4, 4) + 1e-10;
    Real worst = 0.0;
    for (std::size_t i = 0; i < t.size(); ++i) worst = std::max(worst, std::abs(moment[i] + 500.0 * z[i]));
    INFO("worst " << worst << " N m, allowed " << allowed << " N m");
    CHECK(worst < allowed);
}

TEST_CASE("Bouc-Wen bushing: balanced loads, and its refusals", "[forces][hysteresis][boucwen]")
{
    // On two bodies in general poses, z set on every axis: the forces sum to
    // zero and so do the moments about the world's origin, to rounding.
    BoucWenBushingParams p;
    for (std::size_t i = 0; i < 3; ++i) {
        p.rotational[i] = BoucWen{300.0 + 50.0 * i, 1.0, 40.0, 10.0, 1.5};
        p.translational[i] = BoucWen{1e5, 1.0, 300.0, 100.0, 1.0};
        p.bushing.stiffness[i] = Curve::linear(1e4);
        p.bushing.rotational_stiffness[i] = Curve::linear(200.0);
    }
    BoucWenBushing e(1, Transform3(Eigen::AngleAxis<Real>(0.3, Vec3::UnitZ()).toRotationMatrix(), Vec3(0.1, 0.0, 0.0)),
                     2, Transform3::FromTranslation(Vec3(0.0, 0.05, 0.0)), p);
    CHECK(e.size() == 6);
    Vec6 z;
    z << 0.01, -0.005, 0.002, 1e-3, -2e-3, 5e-4;
    e.set_states(z);
    std::vector<RigidBodyState> states(3);
    states[1].p_WB = Vec3(0.3, 0.1, -0.2);
    states[1].q_WB = Quat(Eigen::AngleAxis<Real>(0.4, Vec3(1.0, 2.0, 3.0).normalized()));
    states[2].p_WB = Vec3(0.41, 0.17, -0.15);
    states[2].q_WB = Quat(Eigen::AngleAxis<Real>(0.45, Vec3(1.0, 2.2, 2.9).normalized()));
    states[2].w_WB = Vec3(0.3, -0.2, 1.1);
    std::vector<RigidBodyForces> f(3);
    e.apply(states, f);
    CHECK((f[1].f_W + f[2].f_W).norm() < 1e-10);
    const Vec3 moment = states[1].p_WB.cross(f[1].f_W) + f[1].tau_W + states[2].p_WB.cross(f[2].f_W) + f[2].tau_W;
    CHECK(moment.norm() < 1e-10);
    // The hysteretic part is -k z on each axis, beside the plain bushing's
    // load: the load with z less the load without, axis by axis.
    Vec6 x, x_dot;
    e.bushing().deflection(states, x, x_dot);
    const Vec6 difference = e.load(x, x_dot, z) - e.load(x, x_dot, Vec6::Zero());
    for (int i = 0; i < 6; ++i) CHECK_THAT(difference(i), WithinAbs(-e.axis(i).stiffness * z(i), 1e-12));

    CHECK_THAT(error_of([] {
                   BoucWenBushingParams q;
                   q.translational[0] = BoucWen{1e5, 1.0, 100.0, 300.0, 1.0};   // beta < gamma
                   BoucWenBushing bad(0, Transform3::Identity(), 1, Transform3::Identity(), q);
               }),
               ContainsSubstring("MBD-F059"));
    CHECK_THAT(error_of([] {
                   BoucWenBushingParams q;
                   q.rotational[2] = BoucWen{100.0, 1.0, 30.0, 10.0, 0.5};   // n < 1
                   BoucWenBushing bad(0, Transform3::Identity(), 1, Transform3::Identity(), q);
               }),
               ContainsSubstring("MBD-F059"));
    CHECK_THAT(error_of([] {
                   BoucWenBushingParams q;
                   q.rotational[0] = BoucWen{100.0, 0.0, 30.0, 10.0, 1.0};   // A = 0
                   BoucWenBushing bad(0, Transform3::Identity(), 1, Transform3::Identity(), q);
               }),
               ContainsSubstring("MBD-F059"));
}
