#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>

#include "mbd/bicycle_model.hpp"

using Catch::Matchers::WithinAbs;

namespace
{
    constexpr mbd::Real deg = mbd::pi / 180.0;
    constexpr mbd::Real eps = 1e-9;
}

// ============================================================================
// Static loads
// ============================================================================

TEST_CASE("Bicycle: equal weight distribution with equal axle distances",
          "[bicycle][loads]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1000.0;
    p.front_axle_x = 1.5; // a
    p.rear_axle_x  = 1.5; // b

    BicycleModel bm(p);

    // 50/50 distribution
    REQUIRE_THAT(bm.front_axle_load(), WithinAbs(bm.rear_axle_load(), 0.1));
    REQUIRE_THAT(bm.front_axle_load(), WithinAbs(0.5 * p.mass * g_accel, 0.1));
}

TEST_CASE("Bicycle: front-heavy car has more front load",
          "[bicycle][loads]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1000.0;
    p.front_axle_x = 1.2; // CG closer to front
    p.rear_axle_x  = 1.8;

    BicycleModel bm(p);

    // W_f = m*g*b/L = 1000*9.81*1.8/3.0 = 5886 N
    // W_r = m*g*a/L = 1000*9.81*1.2/3.0 = 3924 N
    REQUIRE(bm.front_axle_load() > bm.rear_axle_load());
    REQUIRE_THAT(bm.front_axle_load() + bm.rear_axle_load(),
                 WithinAbs(p.mass * g_accel, 0.1));
}

// ============================================================================
// Understeer gradient
// ============================================================================

TEST_CASE("Bicycle: neutral steer with equal weight distribution and equal tires",
          "[bicycle][understeer]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1000.0;
    p.front_axle_x = 1.5;
    p.rear_axle_x  = 1.5;
    // Same tires front and rear (default)

    BicycleModel bm(p);

    // K_us = (m/L)*(b/C_f - a/C_r)
    // With a = b and identical tires at equal loads: C_f = C_r → K_us = 0
    REQUIRE_THAT(bm.understeer_gradient(), WithinAbs(0.0, 1e-6));
}

TEST_CASE("Bicycle: front-heavy car understeers with load-sensitive tires",
          "[bicycle][understeer]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1400.0;
    p.front_axle_x = 1.1;  // CG closer to front
    p.rear_axle_x  = 1.6;  // b > a → more weight on front

    // Use tires with significant load sensitivity
    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003; // Strong load sensitivity
    p.tire_front = tp;
    p.tire_rear  = tp;

    BicycleModel bm(p);

    // With load sensitivity, front tires are less efficient per unit load
    // → K_us > 0 → understeer
    REQUIRE(bm.understeer_gradient() > 0.0);
}

TEST_CASE("Bicycle: rear-heavy car oversteers with load-sensitive tires",
          "[bicycle][understeer]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1400.0;
    p.front_axle_x = 1.6;  // CG closer to rear
    p.rear_axle_x  = 1.1;

    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003;
    p.tire_front = tp;
    p.tire_rear  = tp;

    BicycleModel bm(p);

    // Rear heavier → K_us < 0 → oversteer
    REQUIRE(bm.understeer_gradient() < 0.0);
}

TEST_CASE("Bicycle: understeer gradient is antisymmetric in weight distribution",
          "[bicycle][understeer]")
{
    using namespace mbd;

    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003;

    BicycleModelParams p1;
    p1.mass = 1400.0;
    p1.front_axle_x = 1.1;
    p1.rear_axle_x  = 1.6;
    p1.tire_front = tp;
    p1.tire_rear  = tp;

    BicycleModelParams p2;
    p2.mass = 1400.0;
    p2.front_axle_x = 1.6;
    p2.rear_axle_x  = 1.1;
    p2.tire_front = tp;
    p2.tire_rear  = tp;

    BicycleModel bm1(p1);
    BicycleModel bm2(p2);

    // K_us should be opposite in sign
    REQUIRE_THAT(bm1.understeer_gradient(),
                 WithinAbs(-bm2.understeer_gradient(), 1e-6));
}

// ============================================================================
// Characteristic speed
// ============================================================================

TEST_CASE("Bicycle: characteristic speed exists for understeer vehicle",
          "[bicycle][characteristic]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1400.0;
    p.front_axle_x = 1.1;
    p.rear_axle_x  = 1.6;

    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003;
    p.tire_front = tp;
    p.tire_rear  = tp;

    BicycleModel bm(p);
    REQUIRE(bm.understeer_gradient() > 0.0);

    Real V_char = bm.characteristic_speed();
    REQUIRE(std::isfinite(V_char));
    REQUIRE(V_char > 0.0);

    // At characteristic speed, yaw rate gain = V / (2*L)
    Real gain_at_Vchar = bm.yaw_rate_gain(V_char);
    Real gain_low_speed = Real(1.0) / p.wheelbase(); // V→0 limit: V/(L + K*V²) → 1/L per unit V? No.

    // Actually yaw_rate_gain = V/(L + K*V²). At V_char: K*V² = L, so gain = V/(2L)
    // At very low V: gain ≈ V/L
    // So gain at V_char = V_char / (2*L), and gain at small V = V_small/L
    // The ratio is V_char / (2 * V_small)... this depends on V_small.
    // Better test: at V_char, gain should be half of what it would be without understeer
    Real gain_no_understeer = V_char / p.wheelbase(); // If K_us were 0
    REQUIRE_THAT(gain_at_Vchar, WithinAbs(gain_no_understeer * 0.5, 1e-4));
}

TEST_CASE("Bicycle: characteristic speed is infinite for neutral steer",
          "[bicycle][characteristic]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1000.0;
    p.front_axle_x = 1.5;
    p.rear_axle_x  = 1.5;

    BicycleModel bm(p);

    REQUIRE_THAT(bm.understeer_gradient(), WithinAbs(0.0, 1e-6));
    REQUIRE(std::isinf(bm.characteristic_speed()));
}

// ============================================================================
// Linear steering angle
// ============================================================================

TEST_CASE("Bicycle: linear steering angle at zero a_y is zero",
          "[bicycle][linear]")
{
    using namespace mbd;

    BicycleModel bm;
    REQUIRE_THAT(bm.linear_steering_angle(20.0, 0.0), WithinAbs(0.0, eps));
}

TEST_CASE("Bicycle: linear steering angle at low speed matches L/R",
          "[bicycle][linear]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1000.0;
    p.front_axle_x = 1.5;
    p.rear_axle_x  = 1.5;

    BicycleModel bm(p);

    // Neutral steer: delta = L/R = L*a_y/V²
    const Real V = 5.0;
    const Real a_y = 2.0; // ~0.2g
    const Real delta = bm.linear_steering_angle(V, a_y);

    const Real expected = p.wheelbase() * a_y / (V * V);
    REQUIRE_THAT(delta, WithinAbs(expected, 1e-6));
}

// ============================================================================
// Nonlinear steering angle
// ============================================================================

TEST_CASE("Bicycle: nonlinear matches linear at low lateral acceleration",
          "[bicycle][nonlinear]")
{
    using namespace mbd;

    BicycleModel bm;

    const Real V = 20.0;
    const Real a_y = 0.5; // Very low: ~0.05g

    const Real delta_lin = bm.linear_steering_angle(V, a_y);
    const Real delta_nl  = bm.nonlinear_steering_angle(V, a_y);

    // At low a_y, linear and nonlinear should agree within 5%
    const Real rel_err = std::abs(delta_nl - delta_lin) / std::abs(delta_lin);
    REQUIRE(rel_err < 0.05);
}

TEST_CASE("Bicycle: nonlinear steering angle increases faster than linear near limit",
          "[bicycle][nonlinear]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1400.0;
    p.front_axle_x = 1.1;
    p.rear_axle_x  = 1.6;

    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003;
    p.tire_front = tp;
    p.tire_rear  = tp;

    BicycleModel bm(p);

    const Real V = 20.0;
    const Real a_y_high = 0.7 * bm.max_lateral_acceleration(); // 70% of limit

    const Real delta_lin = bm.linear_steering_angle(V, a_y_high);
    const Real delta_nl  = bm.nonlinear_steering_angle(V, a_y_high);

    REQUIRE(std::isfinite(delta_nl));

    // For an understeering car near the limit, nonlinear delta > linear delta
    // because tire saturation requires more slip angle for the same force
    REQUIRE(delta_nl > delta_lin);
}

TEST_CASE("Bicycle: nonlinear returns NaN beyond tire limit",
          "[bicycle][nonlinear]")
{
    using namespace mbd;

    BicycleModel bm;
    const Real a_y_impossible = bm.max_lateral_acceleration() * 1.5;

    Real delta = bm.nonlinear_steering_angle(20.0, a_y_impossible);
    REQUIRE(std::isnan(delta));
}

// ============================================================================
// Cornering diagram
// ============================================================================

TEST_CASE("Bicycle: cornering diagram starts at zero", "[bicycle][diagram]")
{
    using namespace mbd;

    BicycleModel bm;
    auto diagram = bm.cornering_diagram(20.0, 5.0, 21);

    REQUIRE(diagram.size() == 21);
    REQUIRE_THAT(diagram[0].a_y, WithinAbs(0.0, eps));
    REQUIRE_THAT(diagram[0].delta_linear, WithinAbs(0.0, eps));
    REQUIRE_THAT(diagram[0].delta_nonlinear, WithinAbs(0.0, 1e-4));
    REQUIRE(diagram[0].valid);
}

TEST_CASE("Bicycle: cornering diagram linear and nonlinear diverge at high a_y",
          "[bicycle][diagram]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1400.0;
    p.front_axle_x = 1.1;
    p.rear_axle_x  = 1.6;

    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003;
    p.tire_front = tp;
    p.tire_rear  = tp;

    BicycleModel bm(p);
    auto diagram = bm.cornering_diagram(20.0, 0.0, 21);

    // Find a high-a_y valid point
    const BicycleModel::CorneringPoint* high_pt = nullptr;    for (int i = static_cast<int>(diagram.size()) - 1; i >= 0; --i) {
        if (diagram[i].valid && diagram[i].a_y > 3.0) {
            high_pt = &diagram[i];
            break;
        }
    }

    REQUIRE(high_pt != nullptr);

    // Nonlinear should differ from linear at high a_y
    const Real diff = std::abs(high_pt->delta_nonlinear - high_pt->delta_linear);
    REQUIRE(diff > 0.001); // More than 0.06 degrees difference
}

TEST_CASE("Bicycle: cornering diagram has invalid points near the limit",
          "[bicycle][diagram]")
{
    using namespace mbd;

    BicycleModel bm;

    // Force sweep past the limit by setting a high a_y_max
    const Real a_y_way_past = bm.max_lateral_acceleration() * 1.3;
    auto diagram = bm.cornering_diagram(20.0, a_y_way_past, 21);

    // Some later points should be invalid (beyond tire limit)
    bool found_invalid = false;
    for (const auto& pt : diagram) {
        if (!pt.valid) {
            found_invalid = true;
            break;
        }
    }
    REQUIRE(found_invalid);
}

// ============================================================================
// Maximum lateral acceleration
// ============================================================================

TEST_CASE("Bicycle: max lateral acceleration is approximately mu*g",
          "[bicycle][max_ay]")
{
    using namespace mbd;

    BicycleModel bm;

    const Real a_y_max = bm.max_lateral_acceleration();
    const Real mu = bm.tire_front.peak_mu_lateral(bm.front_tire_load());

    // Should be close to mu * g
    REQUIRE_THAT(a_y_max, WithinAbs(mu * g_accel, 0.5));
}

// ============================================================================
// Yaw rate gain
// ============================================================================

TEST_CASE("Bicycle: yaw rate gain decreases with speed for understeer vehicle",
          "[bicycle][yaw_gain]")
{
    using namespace mbd;

    BicycleModelParams p;
    p.mass = 1400.0;
    p.front_axle_x = 1.1;
    p.rear_axle_x  = 1.6;

    PacejkaTireParams tp = PacejkaTireParams::DefaultPassengerCar();
    tp.lateral.mu_Fz = -0.003;
    p.tire_front = tp;
    p.tire_rear  = tp;

    BicycleModel bm(p);
    REQUIRE(bm.understeer_gradient() > 0.0);

    // Gain should decrease with speed for understeer
    // gain = V / (L + K*V²) → d(gain)/dV at high V is negative
    // Both speeds must be above V_char for gain to decrease.
    // V_char for this config is ~30 m/s.
    Real gain_slow = bm.yaw_rate_gain(35.0);
    Real gain_fast = bm.yaw_rate_gain(70.0);
    
    // gain = V/(L + K*V²). At V=10: 10/(2.7 + K*100). At V=40: 40/(2.7 + K*1600).
    // For K > 0, the denominator grows faster than V → gain_fast < gain_slow
    // at sufficiently high speed.
    REQUIRE(gain_fast < gain_slow);
}

// ============================================================================
// Construct from VehicleParams
// ============================================================================

TEST_CASE("Bicycle: FromVehicle produces consistent model",
          "[bicycle][conversion]")
{
    using namespace mbd;

    VehicleParams vp;
    auto bp = BicycleModelParams::FromVehicle(vp);
    BicycleModel bm(bp);

    // Mass should match total vehicle mass
    REQUIRE_THAT(bp.mass, WithinAbs(vp.total_mass(), 0.1));

    // Axle distances should match
    REQUIRE_THAT(bp.front_axle_x, WithinAbs(vp.front_axle_x, eps));
    REQUIRE_THAT(bp.rear_axle_x, WithinAbs(vp.rear_axle_x, eps));

    // Loads should sum to total weight
    REQUIRE_THAT(bm.front_axle_load() + bm.rear_axle_load(),
                 WithinAbs(bp.mass * g_accel, 0.1));
}

// ============================================================================
// Pacejka inversion
// ============================================================================

TEST_CASE("Bicycle: Pacejka force inversion round-trips",
          "[bicycle][inversion]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;

    // Pick a moderate slip angle, compute force, then invert
    const Real alpha_in = 0.04; // ~2.3 deg
    auto r = tire.compute(0.0, alpha_in, Fz);
    const Real F_axle = 2.0 * r.Fy;

    Real alpha_out = BicycleModel::invert_axle_force(tire, F_axle, Fz);

    REQUIRE(std::isfinite(alpha_out));
    REQUIRE_THAT(alpha_out, WithinAbs(alpha_in, 1e-4));
}

TEST_CASE("Bicycle: Pacejka inversion at zero force gives zero slip",
          "[bicycle][inversion]")
{
    using namespace mbd;

    PacejkaTire tire;
    Real alpha = BicycleModel::invert_axle_force(tire, 0.0, 4000.0);

    REQUIRE(std::isfinite(alpha));
    REQUIRE_THAT(alpha, WithinAbs(0.0, 1e-6));
}

TEST_CASE("Bicycle: Pacejka inversion returns NaN beyond peak",
          "[bicycle][inversion]")
{
    using namespace mbd;

    PacejkaTire tire;
    const Real Fz = 4000.0;
    const Real mu = tire.peak_mu_lateral(Fz);
    const Real F_impossible = 2.0 * mu * Fz * 1.1; // 110% of peak

    Real alpha = BicycleModel::invert_axle_force(tire, F_impossible, Fz);

    REQUIRE(std::isnan(alpha));
}