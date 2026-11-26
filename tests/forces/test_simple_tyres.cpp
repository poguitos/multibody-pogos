// The simple tyres (plan task 7.5, D65): linear with saturation, Fiala, and
// the brush model, against their closed forms, each written out here from its
// source (Pacejka, Tire and Vehicle Dynamics, 3rd ed., 2012, chapter 3; for
// Fiala's longitudinal force and friction, MSC Adams's form as Project Chrono
// has it). The default parameters: Fz0 = 4000 N, R0 = 0.3135 m.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>

#include "mbd/forces/pacejka.hpp"
#include "mbd/forces/simple_tyres.hpp"

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace mbd;

namespace
{
    TyreInput at(Real Fz, Real kappa, Real alpha, Real Vcx = 10.0)
    {
        TyreInput in;
        in.Fz = Fz;
        in.kappa = kappa;
        in.alpha = alpha;
        in.Vcx = Vcx;
        return in;
    }
}

// ============================================================================
// Linear with saturation
// ============================================================================

TEST_CASE("Simple tyres: linear, its stiffnesses in proportion to the load, its force limited to the circle",
          "[tyre][simple][linear]")
{
    // The defaults: Kxk 86748 N and |Kya| 53356 N/rad at 4000 N, mu 1, a
    // trail of 0.028 m. At 4000 N and kappa = 0.01 the force is 867.48 N,
    // within the circle of radius mu Fz = 4000 N; at alpha = 0.02 rad, -C
    // alpha = -1067.12 N, and Mz = -0.028 Fy = 29.87936 N m. At half the
    // load the stiffnesses are half. At kappa = alpha = 0.05 the linear force
    // (4337.4, -2667.8) N has the magnitude 5092.3 N: brought back onto the
    // circle, 4000 N, along the same direction. All to rounding.
    const LinearTyre tyre{LinearTyreParams{}};
    const TyreOutput small = tyre.evaluate(at(4000.0, 0.01, 0.0));
    CHECK_THAT(small.Fx, WithinRel(867.48, 1e-14));
    CHECK(small.Fy == 0.0);
    const TyreOutput side = tyre.evaluate(at(4000.0, 0.0, 0.02));
    CHECK_THAT(side.Fy, WithinRel(-1067.12, 1e-14));
    CHECK_THAT(side.Mz, WithinRel(29.87936, 1e-13));
    CHECK(side.Kxk == 86748.0);
    CHECK(side.Kya == -53356.0);
    CHECK_THAT(tyre.evaluate(at(2000.0, 0.0, 0.0)).Kxk, WithinRel(43374.0, 1e-15));
    CHECK_THAT(tyre.evaluate(at(2000.0, 0.0, 0.0)).Kya, WithinRel(-26678.0, 1e-15));

    const TyreOutput both = tyre.evaluate(at(4000.0, 0.05, 0.05));
    CHECK_THAT(std::hypot(both.Fx, both.Fy), WithinRel(4000.0, 1e-14));
    CHECK_THAT(both.Fx / both.Fy, WithinRel(4337.4 / -2667.8, 1e-13));
    CHECK_THAT(both.Fx0, WithinRel(4000.0, 1e-15));    // pure slip: the limit along x
    CHECK_THAT(both.Fy0, WithinRel(-2667.8, 1e-14));   // within it along y

    // A road of friction 0.5 halves the circle.
    TyreInput slippery = at(4000.0, 0.05, 0.05);
    slippery.friction = 0.5;
    const TyreOutput wet = tyre.evaluate(slippery);
    CHECK_THAT(std::hypot(wet.Fx, wet.Fy), WithinRel(2000.0, 1e-14));
}

TEST_CASE("Simple tyres: the rolling resistance, the relaxation lengths and the refusals", "[tyre][simple]")
{
    // My = -f_r R0 Fz against the rolling: -0.007 x 0.3135 x 4000 = -8.778 N
    // m forwards, +8.778 backwards. The relaxation lengths are the slip
    // stiffnesses over the carcass's: 86748 / 358066 and 53356 / 102673 m.
    // The same for every simple tyre, the brush model's one stiffness for
    // both.
    const LinearTyre linear{LinearTyreParams{}};
    const FialaTyre fiala{FialaTyreParams{}};
    const BrushTyre brush{BrushTyreParams{}};
    for (const TyreModel* m : {static_cast<const TyreModel*>(&linear), static_cast<const TyreModel*>(&fiala),
                               static_cast<const TyreModel*>(&brush)}) {
        INFO(m->name());
        CHECK_THAT(m->evaluate(at(4000.0, 0.0, 0.0, 20.0)).My, WithinRel(-8.778, 1e-14));
        CHECK_THAT(m->evaluate(at(4000.0, 0.0, 0.0, -20.0)).My, WithinRel(8.778, 1e-14));
        const TyreOutput c = m->characteristics(4000.0, 0.0);
        CHECK_THAT(c.sigma_kappa, WithinRel(c.Kxk / 358066.0, 1e-15));
        CHECK_THAT(c.sigma_alpha, WithinRel(-c.Kya / 102673.0, 1e-15));
        const TyreBody b = m->body();
        CHECK(b.free_radius == 0.3135);
        CHECK(b.vertical_stiffness == 209651.0);
        CHECK(b.nominal_load == 4000.0);
        CHECK(m->evaluate(at(0.0, 0.1, 0.1)).Fx == 0.0);   // off the road
    }
    CHECK_THAT(linear.characteristics(4000.0, 0.0).sigma_kappa, WithinRel(86748.0 / 358066.0, 1e-15));
    CHECK_THAT(brush.characteristics(4000.0, 0.0).sigma_alpha, WithinRel(53356.0 / 102673.0, 1e-15));

    LinearTyreParams bad;
    bad.friction = 0.0;
    CHECK_THROWS_WITH(LinearTyre(bad), ContainsSubstring("MBD-F055"));
    BrushTyreParams no_patch;
    no_patch.half_contact_length = -0.01;
    CHECK_THROWS_WITH(BrushTyre(no_patch), ContainsSubstring("MBD-F055"));
    FialaTyreParams no_radius;
    no_radius.free_radius = 0.0;
    CHECK_THROWS_WITH(FialaTyre(no_radius), ContainsSubstring("MBD-F055"));
}

// ============================================================================
// Fiala
// ============================================================================

TEST_CASE("Simple tyres: Fiala's longitudinal force, linear then saturating, continuous at kappa_c",
          "[tyre][simple][fiala]")
{
    // The friction U = mu_max - (mu_max - mu_min) min(1, sqrt(kappa^2 +
    // tan^2 alpha)): at kappa = 0.01 alone, 1 - 0.1 x 0.01 = 0.999, so
    // kappa_c = U Fz / (2 C) = 3996 / 173496 = 0.02303; below it Fx = C kappa
    // = 867.48 N. At kappa = 0.1, U = 0.99 and Fx = U Fz - (U Fz)^2 / (4
    // kappa C) = 3960 - 3960^2 / 34699.2. At kappa_c the two branches meet at
    // U Fz / 2 with the same slope C: across it, between kappa_c -+ 1e-9,
    // the force changes by 2e-9 C (1.7e-4 N) and by U's change (1e-10 of it):
    // allowed 1e-3 N. Braking is the mirror image.
    const FialaTyre tyre{FialaTyreParams{}};
    CHECK_THAT(tyre.evaluate(at(4000.0, 0.01, 0.0)).Fx, WithinRel(867.48, 1e-14));
    CHECK_THAT(tyre.evaluate(at(4000.0, 0.1, 0.0)).Fx, WithinRel(3960.0 - 3960.0 * 3960.0 / 34699.2, 1e-13));
    CHECK_THAT(tyre.evaluate(at(4000.0, -0.1, 0.0)).Fx, WithinRel(-(3960.0 - 3960.0 * 3960.0 / 34699.2), 1e-13));

    // kappa_c where U = 1 - 0.1 kappa_c: kappa_c = (1 - 0.1 k) 4000 / 173496,
    // solved: k = 4000 / (173496 + 400).
    const Real kc = 4000.0 / (173496.0 + 400.0);
    const Real below = tyre.evaluate(at(4000.0, kc - 1e-9, 0.0)).Fx;
    const Real above = tyre.evaluate(at(4000.0, kc + 1e-9, 0.0)).Fx;
    INFO("below " << below << ", above " << above << " N");
    CHECK_THAT(above - below, WithinAbs(0.0, 1e-3));
    CHECK_THAT(below, WithinRel(0.5 * (1.0 - 0.1 * kc) * 4000.0, 1e-6));
}

TEST_CASE("Simple tyres: Fiala's side force and aligning torque are the brush model's at pure side slip",
          "[tyre][simple][fiala]")
{
    // With one friction (mu_max = mu_min = 1) the side force is Pacejka's
    // (3.11), Fy = -3 mu Fz theta s (1 - theta s + (theta s)^2 / 3) with s =
    // tan(alpha) and theta = C_alpha / (3 mu Fz), and the torque his (3.12)
    // with the width w for the half contact length, Mz = mu Fz w theta s (1 -
    // theta s)^3 in ISO's signs (H = 1 - theta s). Its peak is mu Fz w 27 /
    // 256 at theta s = 1/4. Past theta s = 1 (tan alpha = 3 mu Fz / C_alpha
    // = 0.2249) the tyre slides: Fy = -mu Fz, Mz = 0.
    FialaTyreParams p;
    p.friction_min = p.friction_max;
    const FialaTyre tyre{p};
    const Real theta = 53356.0 / (3.0 * 4000.0);
    for (const Real s : {0.02, 0.05, 0.1, 0.2}) {
        INFO("tan(alpha) " << s);
        const Real u = theta * s;
        const TyreOutput o = tyre.evaluate(at(4000.0, 0.0, std::atan(s)));
        CHECK_THAT(o.Fy, WithinRel(-3.0 * 4000.0 * u * (1.0 - u + u * u / 3.0), 1e-13));
        CHECK_THAT(o.Mz, WithinRel(4000.0 * 0.205 * u * std::pow(1.0 - u, 3.0), 1e-12));
        CHECK_THAT(tyre.evaluate(at(4000.0, 0.0, -std::atan(s))).Fy, WithinRel(-o.Fy, 1e-15));
    }
    CHECK_THAT(tyre.evaluate(at(4000.0, 0.0, std::atan(1.0 / (4.0 * theta)))).Mz,
               WithinRel(4000.0 * 0.205 * 27.0 / 256.0, 1e-13));
    const TyreOutput sliding = tyre.evaluate(at(4000.0, 0.0, std::atan(0.3)));
    CHECK(sliding.Fy == -4000.0);
    CHECK(sliding.Mz == 0.0);
}

TEST_CASE("Simple tyres: Fiala's friction falls with the combined slip, and its forces do not reduce each other",
          "[tyre][simple][fiala]")
{
    // At kappa = 0.8 and tan(alpha) = 0.6 the combined slip is 1: U = mu_min
    // = 0.9, the limit 3600 N. Fx = 3600 - 3600^2 / (4 x 0.8 x 86748); the
    // side force slides (0.6 > 3 x 3600 / 53356), -3600 N: together more than
    // 3600 N, as Adams's Fiala has it. At small combined slip the
    // longitudinal force is C kappa whatever the slip angle.
    const FialaTyre tyre{FialaTyreParams{}};
    const TyreOutput o = tyre.evaluate(at(4000.0, 0.8, std::atan(0.6)));
    CHECK_THAT(o.Fx, WithinRel(3600.0 - 3600.0 * 3600.0 / (4.0 * 0.8 * 86748.0), 1e-13));
    CHECK_THAT(o.Fy, WithinRel(-3600.0, 1e-14));
    CHECK(std::hypot(o.Fx, o.Fy) > 3600.0);
    CHECK_THAT(tyre.evaluate(at(4000.0, 0.005, 0.01)).Fx, WithinRel(86748.0 * 0.005, 1e-14));
    // Its stiffnesses are constants, whatever the load.
    CHECK(tyre.characteristics(2000.0, 0.0).Kxk == 86748.0);
    CHECK(tyre.characteristics(2000.0, 0.0).Kya == -53356.0);
}

// ============================================================================
// Brush
// ============================================================================

TEST_CASE("Simple tyres: the brush model at pure side slip is Pacejka's (3.11) to (3.14)", "[tyre][simple][brush]")
{
    // At the nominal load the half contact length is a = 0.075 m and the
    // slip stiffness C = 2 c_p a^2 = 53356 N, so theta = C / (3 mu Fz) =
    // 4.4463 (3.46). With u = theta tan(alpha): Fy = -3 mu Fz u (1 - u + u^2 /
    // 3) (3.11), Mz = mu Fz a u (1 - u)^3 (3.12, ISO's signs), the trail -Mz
    // / Fy, a / 3 at vanishing slip (3.14); Mz peaks at 27 mu Fz a / 256 at u =
    // 1/4; from u = 1 (3.9) the tyre slides, Fy = -mu Fz, Mz = 0.
    const BrushTyre tyre{BrushTyreParams{}};
    const Real theta = 53356.0 / 12000.0, a = 0.075;
    for (const Real s : {0.01, 0.05, 0.12, 0.2}) {
        INFO("tan(alpha) " << s);
        const Real u = theta * s;
        const TyreOutput o = tyre.evaluate(at(4000.0, 0.0, std::atan(s)));
        CHECK_THAT(o.Fy, WithinRel(-3.0 * 4000.0 * u * (1.0 - u + u * u / 3.0), 1e-13));
        CHECK_THAT(o.Mz, WithinRel(4000.0 * a * u * std::pow(1.0 - u, 3.0), 1e-12));
        CHECK_THAT(o.trail, WithinRel(-o.Mz / o.Fy, 1e-13));
        CHECK(o.Fy == o.Fy0);
    }
    CHECK_THAT(tyre.evaluate(at(4000.0, 0.0, 0.0)).trail, WithinRel(a / 3.0, 1e-15));
    CHECK_THAT(tyre.evaluate(at(4000.0, 0.0, std::atan(0.25 / theta))).Mz, WithinRel(4000.0 * a * 27.0 / 256.0, 1e-13));
    const TyreOutput sliding = tyre.evaluate(at(4000.0, 0.0, std::atan(1.1 / theta)));
    CHECK(sliding.Fy == -4000.0);
    CHECK(sliding.Mz == 0.0);
}

TEST_CASE("Simple tyres: the brush model is isotropic in the theoretical slip, and its stiffnesses are C",
          "[tyre][simple][brush]")
{
    // Pacejka's Figure 3.8: with one stiffness and one friction the
    // longitudinal force at kappa is the side force at tan(alpha) = kappa / (1
    // + kappa) (3.20), with its sign: to rounding, driving and braking. The
    // slopes at zero slip are C and -C (3.3, 3.27). The curve is odd but its
    // quadratic term goes with |s|, so a central difference of h gives F(h) /
    // h = C (1 - theta h + (theta h)^2 / 3), not C: with h = 1e-7, 4.4e-7
    // below it. That is asserted, to the rounding of 1e-16 x 4000 / 1e-7 =
    // 4e-6 N: allowed 1e-8 relative.
    const BrushTyre tyre{BrushTyreParams{}};
    for (const Real k : {0.03, -0.03, 0.2, -0.4}) {
        INFO("kappa " << k);
        const Real Fx = tyre.evaluate(at(4000.0, k, 0.0)).Fx;
        const Real Fy = tyre.evaluate(at(4000.0, 0.0, std::atan(std::abs(k) / (1.0 + k)))).Fy;
        CHECK_THAT(Fx, WithinRel(-Fy * (k > 0.0 ? 1.0 : -1.0), 1e-13));
    }
    const Real h = 1e-7, theta = 53356.0 / 12000.0;
    const Real slope = 53356.0 * (1.0 - theta * h + theta * theta * h * h / 3.0);
    CHECK_THAT((tyre.evaluate(at(4000.0, h, 0.0)).Fx - tyre.evaluate(at(4000.0, -h, 0.0)).Fx) / (2 * h),
               WithinRel(slope, 1e-8));
    CHECK_THAT((tyre.evaluate(at(4000.0, 0.0, std::atan(h))).Fy - tyre.evaluate(at(4000.0, 0.0, -std::atan(h))).Fy)
                   / (2 * h),
               WithinRel(-slope, 1e-8));
    CHECK(tyre.characteristics(4000.0, 0.0).Kxk == 53356.0);
    CHECK(tyre.characteristics(4000.0, 0.0).Kya == -53356.0);
}

TEST_CASE("Simple tyres: the brush model at combined slip is Pacejka's (3.48) to (3.50)", "[tyre][simple][brush]")
{
    // At kappa = 0.05 and tan(alpha) = 0.04 the theoretical slip is s =
    // (0.05, -0.04) / 1.05 (3.34, ISO's signs), |s| = 0.0640312 / 1.05; the
    // force mu Fz (1 - (1 - theta |s|)^3) (3.48) along s (3.49); the moment
    // -t Fy with the pure-slip trail at theta |s| (3.50). A locked wheel
    // slides, its force mu Fz against the sliding: Fx = -mu Fz cos(alpha), Fy
    // = -mu Fz sin(alpha) (Pacejka, below Figure 3.12).
    const BrushTyre tyre{BrushTyreParams{}};
    const Real theta = 53356.0 / 12000.0, a = 0.075;
    const Real n = std::hypot(0.05, 0.04);
    const Real u = theta * n / 1.05;
    const Real F = 4000.0 * (1.0 - std::pow(1.0 - u, 3.0));
    const Real t = a / 3.0 * std::pow(1.0 - u, 3.0) / (1.0 - u + u * u / 3.0);
    const TyreOutput o = tyre.evaluate(at(4000.0, 0.05, std::atan(0.04)));
    CHECK_THAT(o.Fx, WithinRel(F * 0.05 / n, 1e-13));
    CHECK_THAT(o.Fy, WithinRel(-F * 0.04 / n, 1e-13));
    CHECK_THAT(o.Mz, WithinRel(t * F * 0.04 / n, 1e-12));

    const TyreOutput locked = tyre.evaluate(at(4000.0, -1.0, 0.2));
    CHECK_THAT(locked.Fx, WithinRel(-4000.0 * std::cos(0.2), 1e-14));
    CHECK_THAT(locked.Fy, WithinRel(-4000.0 * std::sin(0.2), 1e-14));
    CHECK(locked.Mz == 0.0);
}

TEST_CASE("Simple tyres: the brush model's load, its contact length growing as the root of it",
          "[tyre][simple][brush]")
{
    // At 2000 N, a = 0.075 sqrt(0.5) and C = 53356 x 0.5, so theta = C / (3
    // mu Fz) is the nominal load's: the force over the load is the same curve,
    // the moment over Fz a too. To rounding.
    const BrushTyre tyre{BrushTyreParams{}};
    for (const Real s : {0.03, 0.15}) {
        const TyreOutput lo = tyre.evaluate(at(2000.0, 0.02, std::atan(s)));
        const TyreOutput hi = tyre.evaluate(at(4000.0, 0.02, std::atan(s)));
        CHECK_THAT(lo.Fy / 2000.0, WithinRel(hi.Fy / 4000.0, 1e-14));
        CHECK_THAT(lo.Fx / 2000.0, WithinRel(hi.Fx / 4000.0, 1e-14));
        CHECK_THAT(lo.Mz / (2000.0 * std::sqrt(0.5)), WithinRel(hi.Mz / 4000.0, 1e-13));
    }
}

// ============================================================================
// The bicycle model's reduction
// ============================================================================

TEST_CASE("Simple tyres: any model reduced to the simple formula keeps its stiffnesses and friction",
          "[tyre][simple]")
{
    // PacejkaTireParams::FromTyreModel, the bicycle model of a vehicle on a
    // simple tyre: the linear tyre's stiffnesses at its nominal load, and
    // their slope in the load increment, which for stiffnesses in proportion
    // to the load is the stiffness itself (central differences of 1e-3 on a
    // straight line: rounding only, 1e-16 x 86748 / 1e-3: allowed 1e-9); the
    // friction 1, flat in load.
    const LinearTyre tyre{LinearTyreParams{}};
    const PacejkaTireParams p = PacejkaTireParams::FromTyreModel(tyre);
    CHECK(p.longitudinal.K0 == 86748.0);
    CHECK(p.lateral.K0 == 53356.0);
    CHECK_THAT(p.longitudinal.K_Fz, WithinRel(86748.0, 1e-9));
    CHECK_THAT(p.lateral.K_Fz, WithinRel(53356.0, 1e-9));
    CHECK(p.lateral.mu0 == 1.0);
    CHECK_THAT(p.lateral.mu_Fz, WithinAbs(0.0, 1e-12));
    CHECK(p.lateral.Fz0 == 4000.0);
}
