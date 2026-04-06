#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/pacejka.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real eps = 1e-9;
}

// ============================================================================
// Basic Magic Formula properties
// ============================================================================

TEST_CASE("Pacejka: zero slip gives zero force", "[pacejka][basic]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    auto r = tire.compute(0.0, 0.0, Fz);

    REQUIRE_THAT(r.Fx, WithinAbs(0.0, 1e-6));
    REQUIRE_THAT(r.Fy, WithinAbs(0.0, 1e-6));
}

TEST_CASE("Pacejka: zero vertical load gives zero forces", "[pacejka][basic]")
{
    using namespace mbd;

    PacejkaTire tire;

    auto r = tire.compute(0.1, 0.05, 0.0);

    REQUIRE_THAT(r.Fx, WithinAbs(0.0, eps));
    REQUIRE_THAT(r.Fy, WithinAbs(0.0, eps));
}

TEST_CASE("Pacejka: negative vertical load gives zero forces", "[pacejka][basic]")
{
    using namespace mbd;

    PacejkaTire tire;

    auto r = tire.compute(0.1, 0.05, -100.0);

    REQUIRE_THAT(r.Fx, WithinAbs(0.0, eps));
    REQUIRE_THAT(r.Fy, WithinAbs(0.0, eps));
}

// ============================================================================
// Force sign convention
// ============================================================================

TEST_CASE("Pacejka: positive slip angle gives positive Fy", "[pacejka][signs]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;
    const Real alpha = 0.05; // ~3 degrees

    auto r = tire.compute(0.0, alpha, Fz);

    REQUIRE(r.Fy > 0.0);
    REQUIRE_THAT(r.Fx, WithinAbs(0.0, 1e-6)); // No longitudinal slip
}

TEST_CASE("Pacejka: negative slip angle gives negative Fy", "[pacejka][signs]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    auto r = tire.compute(0.0, -0.05, Fz);

    REQUIRE(r.Fy < 0.0);
}

TEST_CASE("Pacejka: positive slip ratio gives positive Fx (traction)",
          "[pacejka][signs]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    auto r = tire.compute(0.05, 0.0, Fz);

    REQUIRE(r.Fx > 0.0);
    REQUIRE_THAT(r.Fy, WithinAbs(0.0, 1e-6));
}

TEST_CASE("Pacejka: negative slip ratio gives negative Fx (braking)",
          "[pacejka][signs]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    auto r = tire.compute(-0.05, 0.0, Fz);

    REQUIRE(r.Fx < 0.0);
}

// ============================================================================
// Anti-symmetry
// ============================================================================

TEST_CASE("Pacejka: force is anti-symmetric in slip", "[pacejka][symmetry]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    // Lateral
    auto r_pos = tire.compute(0.0, 0.1, Fz);
    auto r_neg = tire.compute(0.0, -0.1, Fz);

    REQUIRE_THAT(r_pos.Fy, WithinAbs(-r_neg.Fy, 1e-6));

    // Longitudinal
    auto r_pos_x = tire.compute(0.1, 0.0, Fz);
    auto r_neg_x = tire.compute(-0.1, 0.0, Fz);

    REQUIRE_THAT(r_pos_x.Fx, WithinAbs(-r_neg_x.Fx, 1e-6));
}

// ============================================================================
// Cornering stiffness (initial slope)
// ============================================================================

TEST_CASE("Pacejka: cornering stiffness matches small-slip Fy slope",
          "[pacejka][stiffness]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    const Real K_alpha = tire.cornering_stiffness(Fz);

    // Finite difference at very small alpha
    const Real da = 1e-6;
    auto r = tire.compute(0.0, da, Fz);
    const Real K_alpha_fd = r.Fy / da;

    // Should match within 1%
    const Real rel_error = std::abs(K_alpha - K_alpha_fd) / K_alpha;
    REQUIRE(rel_error < 0.01);
}

TEST_CASE("Pacejka: longitudinal stiffness matches small-slip Fx slope",
          "[pacejka][stiffness]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    const Real K_kappa = tire.longitudinal_stiffness(Fz);

    const Real dk = 1e-6;
    auto r = tire.compute(dk, 0.0, Fz);
    const Real K_kappa_fd = r.Fx / dk;

    const Real rel_error = std::abs(K_kappa - K_kappa_fd) / K_kappa;
    REQUIRE(rel_error < 0.01);
}

// ============================================================================
// Peak force and saturation
// ============================================================================

TEST_CASE("Pacejka: lateral force saturates at approximately mu*Fz",
          "[pacejka][saturation]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    // Sweep alpha and find peak
    Real Fy_max = 0.0;
    for (Real alpha = 0.0; alpha < 0.5; alpha += 0.001) {
        auto r = tire.compute(0.0, alpha, Fz);
        Fy_max = std::max(Fy_max, r.Fy);
    }

    // Peak should be approximately mu * Fz
    const Real peak_expected = tire.peak_mu_lateral(Fz) * Fz;

    // Within 5% (C and E parameters affect exact peak)
    const Real rel_error = std::abs(Fy_max - peak_expected) / peak_expected;
    REQUIRE(rel_error < 0.05);
}

TEST_CASE("Pacejka: force increases with vertical load", "[pacejka][load]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real alpha = 0.05;

    auto r_low  = tire.compute(0.0, alpha, 2000.0);
    auto r_nom  = tire.compute(0.0, alpha, 4000.0);
    auto r_high = tire.compute(0.0, alpha, 6000.0);

    REQUIRE(r_low.Fy < r_nom.Fy);
    REQUIRE(r_nom.Fy < r_high.Fy);
}

// ============================================================================
// Load sensitivity (mu decreases with load)
// ============================================================================

TEST_CASE("Pacejka: friction coefficient decreases with load (load sensitivity)",
          "[pacejka][load_sensitivity]")
{
    using namespace mbd;

    PacejkaTire tire;

    const Real mu_low  = tire.peak_mu_lateral(2000.0);
    const Real mu_nom  = tire.peak_mu_lateral(4000.0);
    const Real mu_high = tire.peak_mu_lateral(6000.0);

    // mu_Fz is negative, so mu decreases with load
    REQUIRE(mu_low > mu_nom);
    REQUIRE(mu_nom > mu_high);
}

// ============================================================================
// Combined slip
// ============================================================================

TEST_CASE("Pacejka: combined slip reduces individual forces",
          "[pacejka][combined]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    // Pure lateral
    auto r_pure_lat = tire.compute(0.0, 0.1, Fz);

    // Combined: same lateral slip + some longitudinal slip
    auto r_combined = tire.compute(0.05, 0.1, Fz);

    // Combined Fy should be less than pure Fy (friction ellipse)
    REQUIRE(std::abs(r_combined.Fy) < std::abs(r_pure_lat.Fy));
    REQUIRE(r_combined.Gy < 1.0);

    // Pure longitudinal
    auto r_pure_lon = tire.compute(0.1, 0.0, Fz);

    // Combined: same longitudinal + some lateral
    auto r_combined2 = tire.compute(0.1, 0.05, Fz);

    // Combined Fx should be less than pure Fx
    REQUIRE(std::abs(r_combined2.Fx) < std::abs(r_pure_lon.Fx));
    REQUIRE(r_combined2.Gx < 1.0);
}

TEST_CASE("Pacejka: no combined slip reduction at zero other-slip",
          "[pacejka][combined]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    // kappa=0, alpha=0.05: Gy depends on kappa, so Gy(0) = 1.0
    auto r = tire.compute(0.0, 0.05, Fz);
    REQUIRE_THAT(r.Gy, WithinAbs(1.0, 1e-6));

    // kappa=0.05, alpha=0: Gx depends on alpha, so Gx(0) = 1.0
    auto r2 = tire.compute(0.05, 0.0, Fz);
    REQUIRE_THAT(r2.Gx, WithinAbs(1.0, 1e-6));
}

// ============================================================================
// Magic Formula direct evaluation
// ============================================================================

TEST_CASE("Pacejka: magic formula shape is correct (C=1 gives atan shape)",
          "[pacejka][formula]")
{
    using namespace mbd;

    // With C=1, E=0, the formula reduces to D * sin(atan(B*x)) = D * B*x / sqrt(1+(B*x)^2)
    MagicFormulaCoeffs c;
    c.mu0  = 1.0;
    c.mu_Fz = 0.0;
    c.C    = 1.0;
    c.K0   = 10000.0;
    c.K_Fz = 0.0;
    c.E0   = 0.0;
    c.E_Fz = 0.0;
    c.Fz0  = 4000.0;

    const Real Fz = 4000.0;
    const Real D = 1.0 * 4000.0; // mu * Fz
    const Real B = 10000.0 / (1.0 * D); // K / (C * D)

    for (Real x = 0.01; x < 0.3; x += 0.05) {
        Real Y = PacejkaTire::evaluate_magic_formula(x, Fz, c);
        Real Y_expected = D * std::sin(std::atan(B * x));
        REQUIRE_THAT(Y, WithinAbs(Y_expected, 1e-6));
    }
}

// ============================================================================
// Parameterization sanity
// ============================================================================

TEST_CASE("Pacejka: default tire produces reasonable forces at nominal load",
          "[pacejka][sanity]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    // At 5 deg slip angle (~0.087 rad), expect ~3000-4000 N lateral force
    auto r = tire.compute(0.0, 0.087, Fz);
    REQUIRE(r.Fy > 2000.0);
    REQUIRE(r.Fy < 5000.0);

    // At 10% slip ratio, expect ~3000-5000 N longitudinal force
    auto r2 = tire.compute(0.1, 0.0, Fz);
    REQUIRE(r2.Fx > 2000.0);
    REQUIRE(r2.Fx < 6000.0);

    // Cornering stiffness should be ~45000 N/rad (as set)
    REQUIRE_THAT(tire.cornering_stiffness(Fz), WithinAbs(45000.0, 1.0));

    // Longitudinal stiffness should be ~60000 N/unit_slip (as set)
    REQUIRE_THAT(tire.longitudinal_stiffness(Fz), WithinAbs(60000.0, 1.0));
}