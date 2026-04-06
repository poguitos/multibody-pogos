#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/optimization.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;
}

// ============================================================================
// Nelder-Mead on analytical test functions
// ============================================================================

TEST_CASE("NelderMead: minimizes 1D quadratic", "[optimization][nm]")
{
    using namespace mbd;

    // f(x) = (x - 3)^2, minimum at x = 3
    auto objective = [](const VecX& x) -> Real {
        return (x(0) - 3.0) * (x(0) - 3.0);
    };

    VecX x0(1);
    x0 << 0.0;

    NelderMeadConfig config;
    config.initial_step = 1.0;
    config.max_iterations = 200;

    auto result = nelder_mead_minimize(objective, x0, VecX(), VecX(), config);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.best_params(0), WithinAbs(3.0, 1e-6));
    REQUIRE_THAT(result.best_cost, WithinAbs(0.0, 1e-12));
}

TEST_CASE("NelderMead: minimizes 2D quadratic", "[optimization][nm]")
{
    using namespace mbd;

    // f(x,y) = (x - 1)^2 + (y + 2)^2, minimum at (1, -2)
    auto objective = [](const VecX& x) -> Real {
        return (x(0) - 1.0) * (x(0) - 1.0) + (x(1) + 2.0) * (x(1) + 2.0);
    };

    VecX x0(2);
    x0 << 0.0, 0.0;

    NelderMeadConfig config;
    config.initial_step = 1.0;
    config.max_iterations = 300;

    auto result = nelder_mead_minimize(objective, x0, VecX(), VecX(), config);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.best_params(0), WithinAbs(1.0, 1e-4));
    REQUIRE_THAT(result.best_params(1), WithinAbs(-2.0, 1e-4));
    REQUIRE_THAT(result.best_cost, WithinAbs(0.0, 1e-8));}

TEST_CASE("NelderMead: respects bounds", "[optimization][nm]")
{
    using namespace mbd;

    // f(x) = (x - 5)^2, but bounded to [0, 3], so minimum at x = 3
    auto objective = [](const VecX& x) -> Real {
        return (x(0) - 5.0) * (x(0) - 5.0);
    };

    VecX x0(1);
    x0 << 1.0;

    VecX lb(1), ub(1);
    lb << 0.0;
    ub << 3.0;

    NelderMeadConfig config;
    config.initial_step = 0.5;

    auto result = nelder_mead_minimize(objective, x0, lb, ub, config);

    REQUIRE(result.converged);
    REQUIRE_THAT(result.best_params(0), WithinAbs(3.0, 1e-6));
}

TEST_CASE("NelderMead: Rosenbrock function in 2D", "[optimization][nm]")
{
    using namespace mbd;

    // f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1, 1)
    auto rosenbrock = [](const VecX& x) -> Real {
        Real a = 1.0 - x(0);
        Real b = x(1) - x(0) * x(0);
        return a * a + 100.0 * b * b;
    };

    VecX x0(2);
    x0 << -1.0, 1.0;

    NelderMeadConfig config;
    config.initial_step = 0.5;
    config.max_iterations = 2000;
    config.tol_fun = 1e-10;
    config.tol_x   = 1e-10;

    auto result = nelder_mead_minimize(rosenbrock, x0, VecX(), VecX(), config);

    REQUIRE_THAT(result.best_params(0), WithinAbs(1.0, 1e-3));
    REQUIRE_THAT(result.best_params(1), WithinAbs(1.0, 1e-3));
    REQUIRE(result.best_cost < 1e-6);
}

// ============================================================================
// Cost function evaluation
// ============================================================================

TEST_CASE("Cost: camber range computed correctly", "[optimization][cost]")
{
    using namespace mbd;

    KinematicSweepResult sweep;
    sweep.points = {
        {-0.02, -0.01, 0.0, 0.0, true},
        { 0.00,  0.00, 0.0, 0.0, true},
        { 0.02,  0.02, 0.0, 0.0, true}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::CamberRange, 1.0)
    };

    Real cost = evaluate_suspension_cost(sweep, terms);
    // Range = 0.02 - (-0.01) = 0.03
    REQUIRE_THAT(cost, WithinAbs(0.03, 1e-10));
}

TEST_CASE("Cost: toe range computed correctly", "[optimization][cost]")
{
    using namespace mbd;

    KinematicSweepResult sweep;
    sweep.points = {
        {-0.02, 0.0, -0.005, 0.0, true},
        { 0.00, 0.0,  0.000, 0.0, true},
        { 0.02, 0.0,  0.003, 0.0, true}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::ToeRange, 1.0)
    };

    Real cost = evaluate_suspension_cost(sweep, terms);
    // Range = 0.003 - (-0.005) = 0.008
    REQUIRE_THAT(cost, WithinAbs(0.008, 1e-10));
}

TEST_CASE("Cost: unconverged points return max cost", "[optimization][cost]")
{
    using namespace mbd;

    KinematicSweepResult sweep;
    sweep.points = {
        {0.0, 0.0, 0.0, 0.0, true},
        {0.01, 0.0, 0.0, 0.0, false}  // Not converged
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::CamberRange, 1.0)
    };

    Real cost = evaluate_suspension_cost(sweep, terms);
    REQUIRE(cost > 1e30);
}

TEST_CASE("Cost: weighted multi-objective", "[optimization][cost]")
{
    using namespace mbd;

    KinematicSweepResult sweep;
    sweep.points = {
        {-0.02, -0.01, -0.005, 0.0, true},
        { 0.00,  0.00,  0.000, 0.0, true},
        { 0.02,  0.02,  0.003, 0.0, true}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::CamberRange, 2.0),   // weight 2
        CostTerm(CostTerm::Type::ToeRange, 10.0)       // weight 10
    };

    Real cost = evaluate_suspension_cost(sweep, terms);
    // 2 * 0.03 + 10 * 0.008 = 0.06 + 0.08 = 0.14
    REQUIRE_THAT(cost, WithinAbs(0.14, 1e-10));
}

// ============================================================================
// DWB optimization: reduce camber variation
// ============================================================================

TEST_CASE("DWB optimization: UCA height optimization reduces camber range",
          "[optimization][dwb]")
{
    using namespace mbd;

    DoubleWishboneParams base;

    // Optimize UCA pivot Y and UCA outer Y (arm height)
    DwbParameterMapping mapping;
    using PD = DwbParameterMapping::ParamDef;
    using PT = PD::Point;

    mapping.params = {
        {PT::UCA_PIVOT, 1, base.uca_pivot.y() - 0.05, base.uca_pivot.y() + 0.05},
        {PT::UCA_OUTER, 1, base.uca_outer.y() - 0.05, base.uca_outer.y() + 0.05}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::CamberRange, 1.0)
    };

    NelderMeadConfig nm_config;
    nm_config.initial_step = 0.003;
    nm_config.max_iterations = 100;
    nm_config.tol_fun = 1e-8;

    auto result = optimize_dwb(base, mapping, terms,
                               SweepConfig{-0.03, 0.03, 11}, nm_config);

    // Optimizer should have run
    REQUIRE(result.iterations > 0);

    // Final cost should be less than or equal to initial cost
    REQUIRE(result.final_cost <= result.initial_cost + 1e-10);

    // Both sweeps should have valid data
    REQUIRE(result.initial_sweep.points.size() == 11);
    REQUIRE(result.final_sweep.points.size() == 11);

    for (const auto& pt : result.final_sweep.points) {
        REQUIRE(pt.converged);
    }
}

TEST_CASE("DWB optimization: tie rod height optimization reduces bump steer",
          "[optimization][dwb]")
{
    using namespace mbd;

    DoubleWishboneParams base;

    // Optimize tie rod inner Y (height relative to LCA pivot)
    DwbParameterMapping mapping;
    using PD = DwbParameterMapping::ParamDef;
    using PT = PD::Point;

    mapping.params = {
        {PT::TIEROD_INNER, 1, base.tierod_inner.y() - 0.05, base.tierod_inner.y() + 0.05}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::ToeRange, 1.0)
    };

    NelderMeadConfig nm_config;
    nm_config.initial_step = 0.003;
    nm_config.max_iterations = 80;

    auto result = optimize_dwb(base, mapping, terms,
                               SweepConfig{-0.03, 0.03, 11}, nm_config);

    REQUIRE(result.iterations > 0);
    REQUIRE(result.final_cost <= result.initial_cost + 1e-10);

    // The optimized tie rod should produce less toe change
    Real initial_toe_range = 0.0;
    Real final_toe_range = 0.0;
    {
        Real tmin = 1e10, tmax = -1e10;
        for (const auto& pt : result.initial_sweep.points) {
            tmin = std::min(tmin, pt.toe);
            tmax = std::max(tmax, pt.toe);
        }
        initial_toe_range = tmax - tmin;
    }
    {
        Real tmin = 1e10, tmax = -1e10;
        for (const auto& pt : result.final_sweep.points) {
            tmin = std::min(tmin, pt.toe);
            tmax = std::max(tmax, pt.toe);
        }
        final_toe_range = tmax - tmin;
    }

    REQUIRE(final_toe_range <= initial_toe_range + 1e-8);
}

// ============================================================================
// DWB optimization: multi-objective (camber + toe)
// ============================================================================

TEST_CASE("DWB optimization: multi-objective reduces combined cost",
          "[optimization][dwb][multi]")
{
    using namespace mbd;

    DoubleWishboneParams base;

    // Optimize UCA outer Y + Z and tie rod inner Y (3 parameters)
    DwbParameterMapping mapping;
    using PD = DwbParameterMapping::ParamDef;
    using PT = PD::Point;

    mapping.params = {
        {PT::UCA_OUTER, 1, base.uca_outer.y() - 0.04, base.uca_outer.y() + 0.04},
        {PT::UCA_OUTER, 2, base.uca_outer.z() - 0.04, base.uca_outer.z() + 0.04},
        {PT::TIEROD_INNER, 1, base.tierod_inner.y() - 0.04, base.tierod_inner.y() + 0.04}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::CamberRange, 1.0),
        CostTerm(CostTerm::Type::ToeRange, 5.0)  // Weight toe higher
    };

    NelderMeadConfig nm_config;
    nm_config.initial_step = 0.002;
    nm_config.max_iterations = 150;

    auto result = optimize_dwb(base, mapping, terms,
                               SweepConfig{-0.025, 0.025, 11}, nm_config);

    REQUIRE(result.iterations > 0);
    REQUIRE(result.final_cost <= result.initial_cost + 1e-10);
}

// ============================================================================
// DWB optimization: target camber gain
// ============================================================================

TEST_CASE("DWB optimization: target specific camber gain",
          "[optimization][dwb][target]")
{
    using namespace mbd;

    DoubleWishboneParams base;

    // Target: -0.5 deg per 10mm bump = -0.5 * pi/180 / 0.01 = -0.873 rad/m
    const Real target_gain = -0.5 * deg / 0.01;

    DwbParameterMapping mapping;
    using PD = DwbParameterMapping::ParamDef;
    using PT = PD::Point;

    mapping.params = {
        {PT::UCA_PIVOT, 2, base.uca_pivot.z() - 0.05, base.uca_pivot.z() + 0.05},
        {PT::UCA_OUTER, 2, base.uca_outer.z() - 0.05, base.uca_outer.z() + 0.05}
    };

    std::vector<CostTerm> terms = {
        CostTerm(CostTerm::Type::TargetCamberGain, 1.0, target_gain)
    };

    NelderMeadConfig nm_config;
    nm_config.initial_step = 0.003;
    nm_config.max_iterations = 150;

    auto result = optimize_dwb(base, mapping, terms,
                               SweepConfig{-0.03, 0.03, 11}, nm_config);

    REQUIRE(result.iterations > 0);

    // The optimizer should get reasonably close to the target gain
    Real actual_gain = result.final_sweep.camber_gain();
    Real gain_error = std::abs(actual_gain - target_gain);
    Real initial_gain_error = std::abs(result.initial_sweep.camber_gain() - target_gain);

    // Should improve relative to initial
    REQUIRE(gain_error <= initial_gain_error + 0.01);
}

// ============================================================================
// Parameter extraction and application round-trips
// ============================================================================

TEST_CASE("DWB parameter mapping: extract and apply round-trips",
          "[optimization][mapping]")
{
    using namespace mbd;

    DoubleWishboneParams base;

    DwbParameterMapping mapping;
    using PD = DwbParameterMapping::ParamDef;
    using PT = PD::Point;

    mapping.params = {
        {PT::UCA_PIVOT, 1, 0.0, 1.0},
        {PT::UCA_OUTER, 2, 0.0, 1.0},
        {PT::TIEROD_INNER, 1, 0.0, 1.0}
    };

    VecX x0 = mapping.extract(base);
    REQUIRE(x0.size() == 3);
    REQUIRE_THAT(x0(0), WithinAbs(base.uca_pivot.y(), 1e-12));
    REQUIRE_THAT(x0(1), WithinAbs(base.uca_outer.z(), 1e-12));
    REQUIRE_THAT(x0(2), WithinAbs(base.tierod_inner.y(), 1e-12));

    // Modify and apply
    VecX x1 = x0;
    x1(0) += 0.01;
    x1(1) -= 0.02;
    x1(2) += 0.005;

    auto p1 = mapping.apply(base, x1);
    REQUIRE_THAT(p1.uca_pivot.y(), WithinAbs(base.uca_pivot.y() + 0.01, 1e-12));
    REQUIRE_THAT(p1.uca_outer.z(), WithinAbs(base.uca_outer.z() - 0.02, 1e-12));
    REQUIRE_THAT(p1.tierod_inner.y(), WithinAbs(base.tierod_inner.y() + 0.005, 1e-12));

    // Other points unchanged
    REQUIRE_THAT(p1.lca_pivot.y(), WithinAbs(base.lca_pivot.y(), 1e-12));
    REQUIRE_THAT(p1.lca_outer.z(), WithinAbs(base.lca_outer.z(), 1e-12));
}