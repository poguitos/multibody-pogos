#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "mbd/forces/aerodynamics.hpp"
#include "mbd/forces/road.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

// Aerodynamic forces. The force element reads only the chassis's world
// state, so most cases give it one directly; the rest simulate on the kernel
// (plan task 2.7).

using Catch::Matchers::WithinAbs;
using Catch::Matchers::ContainsSubstring;

namespace
{
    using mbd::Real;
    using mbd::Vec3;

    /// The force of an aero element on a chassis (body 1) at position p,
    /// moving at velocity v, not rotating.
    mbd::RigidBodyForces aero_force(const mbd::AerodynamicForce& aero, const Vec3& p, const Vec3& v)
    {
        std::vector<mbd::RigidBodyState> states(2);
        states[1] = mbd::RigidBodyState(p, mbd::Quat::Identity(), v, Vec3::Zero());
        std::vector<mbd::RigidBodyForces> forces(2);
        aero.apply(states, forces);
        return forces[1];
    }

    Real forward_speed(const mbd::kernel::Simulator& sim, mbd::BodyIndex chassis)
    {
        const auto& s = sim.states()[static_cast<std::size_t>(chassis)];
        return s.v_WB.dot(s.q_WB * Vec3::UnitX());
    }
}

// ============================================================================
// Static drag force
// ============================================================================

TEST_CASE("Aero: zero velocity produces zero force", "[aero][static]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 1.5;
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3::Zero());   // at rest

    REQUIRE_THAT(f.f_W.norm(), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(f.tau_W.norm(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Aero: drag force opposes horizontal velocity", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 0.0;
    AerodynamicForce aero(1, p);

    // Chassis moving forward at 30 m/s
    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(30.0, 0.0, 0.0));

    // Drag = 0.5 * 1.225 * 30^2 * 0.7 = 385.875 N, opposing +X
    const Real expected_drag = 0.5 * 1.225 * 30.0 * 30.0 * 0.7;
    REQUIRE_THAT(f.f_W.x(), WithinAbs(-expected_drag, 1.0));
    REQUIRE_THAT(f.f_W.y(), WithinAbs(0.0, 1.0));
    REQUIRE_THAT(f.f_W.z(), WithinAbs(0.0, 1.0));
}

TEST_CASE("Aero: drag scales as V^2", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(1, p);

    auto get_drag_at_speed = [&](Real V) -> Real {
        return -aero_force(aero, Vec3::Zero(), Vec3(V, 0.0, 0.0)).f_W.x();
    };

    Real F_10 = get_drag_at_speed(10.0);
    Real F_20 = get_drag_at_speed(20.0);
    Real F_40 = get_drag_at_speed(40.0);

    // F scales with V^2: F_20 should be 4x F_10, F_40 should be 16x F_10
    REQUIRE_THAT(F_20 / F_10, WithinAbs(4.0, 0.01));
    REQUIRE_THAT(F_40 / F_10, WithinAbs(16.0, 0.01));
}

TEST_CASE("Aero: downforce acts in -Y direction", "[aero][lift]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 0.0;
    p.ClA = 2.0;
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(50.0, 0.0, 0.0));  // 180 km/h

    // Downforce = 0.5 * 1.225 * 50^2 * 2.0 = 3062.5 N
    const Real expected_DF = 0.5 * 1.225 * 50.0 * 50.0 * 2.0;
    REQUIRE_THAT(f.f_W.z(), WithinAbs(-expected_DF, 5.0));

    // No drag (CdA = 0)
    REQUIRE_THAT(f.f_W.x(), WithinAbs(0.0, 5.0));
}

TEST_CASE("Aero: the constant model's drag acts along the chassis's x axis", "[aero][drag]")
{
    // Since task 7.10 (D71) the forces act in the chassis's axes: the
    // constant model is a map with one node, the same coefficients at every
    // yaw. A level chassis moving at (20, 15, 0) m/s, 25 m/s at a yaw of
    // atan(15 / 20): the drag is q CdA along -x, q of the whole 25 m/s, and
    // nothing sideways. (Before, the drag acted against the velocity.)
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(20.0, 15.0, 0.0));

    const Real q = 0.5 * 1.225 * 25.0 * 25.0;
    REQUIRE_THAT(f.f_W.x(), WithinAbs(-q * 1.0, 1e-12 * q));
    REQUIRE_THAT(f.f_W.y(), WithinAbs(0.0, 1e-12 * q));
    REQUIRE_THAT(aero.inputs(RigidBodyState(Vec3::Zero(), Quat::Identity(), Vec3(20.0, 15.0, 0.0), Vec3::Zero())).x[4],
                 WithinAbs(std::atan2(15.0, 20.0), 1e-15));
}

TEST_CASE("Aero: vertical velocity doesn't affect drag", "[aero][drag]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    AerodynamicForce aero(1, p);

    // Forward 30 m/s, vertical 10 m/s (jumping or falling)
    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(30.0, 0.0, 10.0));

    // Drag should be based on horizontal V = 30, not on full V
    const Real expected_drag = 0.5 * 1.225 * 30.0 * 30.0 * 1.0;
    REQUIRE_THAT(f.f_W.x(), WithinAbs(-expected_drag, 1.0));
    REQUIRE_THAT(f.f_W.z(), WithinAbs(0.0, 1.0));
}

// ============================================================================
// Center of pressure produces moment
// ============================================================================

TEST_CASE("Aero: CoP above CG produces a pitch moment from drag",
          "[aero][cop]")
{
    using namespace mbd;

    AeroParams p;
    p.CdA = 1.0;
    p.cop_offset_chassis = Vec3(0.0, 0.0, 0.5);  // CoP above CG (e.g., wing height)
    AerodynamicForce aero(1, p);

    const RigidBodyForces f = aero_force(aero, Vec3::Zero(), Vec3(30.0, 0.0, 0.0));

    // Drag F acts in -X at the CoP, 0.5 m above the CG. Its moment about the
    // CG is r x F = (0, 0, 0.5) x (-F, 0, 0) = (0, -0.5 F, 0): about Y, the
    // lateral axis, so a pitch moment. A positive rotation about Y turns +X
    // toward -Z (nose down, ISO 8855's positive pitch): drag high above the
    // CG lifts the nose, a negative pitch moment.
    const Real drag = 0.5 * 1.225 * 30.0 * 30.0 * 1.0;
    REQUIRE_THAT(f.tau_W.y(), WithinAbs(-0.5 * drag, 1e-9));
    REQUIRE_THAT(f.tau_W.x(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(f.tau_W.z(), WithinAbs(0.0, 1e-9));
}

// ============================================================================
// Ride-height-dependent downforce
// ============================================================================

TEST_CASE("Aero: ride-height sensitivity increases downforce at low h",
          "[aero][groundeffect]")
{
    using namespace mbd;

    AeroParams p;
    p.ClA = 1.0;
    p.h_ref = 0.10;
    p.dClA_dh = 5.0;  // strong ground effect
    AerodynamicForce aero(1, p);

    auto get_DF_at_h = [&](Real h, Real V) -> Real {
        return -aero_force(aero, Vec3(0.0, 0.0, h), Vec3(V, 0.0, 0.0)).f_W.z();
    };

    // At h = h_ref = 0.10: ClA_eff = 1.0 (baseline)
    // At h = 0.05 (below ref): ClA_eff = 1.0 + 5.0 * (0.10 - 0.05) = 1.25
    // At h = 0.15 (above ref): ClA_eff = 1.0 (no boost when above)

    Real DF_at_ref = get_DF_at_h(0.10, 50.0);
    Real DF_low    = get_DF_at_h(0.05, 50.0);
    Real DF_high   = get_DF_at_h(0.15, 50.0);

    // Low ride height: more downforce
    REQUIRE(DF_low > DF_at_ref);

    // High ride height: same as baseline
    REQUIRE_THAT(DF_high, WithinAbs(DF_at_ref, 1.0));

    // The increase should match: ClA goes from 1.0 to 1.25, 25% more
    const Real expected_ratio = 1.25 / 1.0;
    REQUIRE_THAT(DF_low / DF_at_ref, WithinAbs(expected_ratio, 0.01));
}

// ============================================================================
// Terminal velocity in free flight
// ============================================================================

TEST_CASE("Aero: terminal velocity from constant forward force", "[aero][terminal]")
{
    using namespace mbd;

    kernel::System sys;
    sys.model.gravity = Vec3::Zero();   // no gravity, to isolate aero
    const BodyIndex chassis = sys.model.add_body(
        0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
        Transform3::Identity(), RigidBodyInertia::from_solid_box(1500.0, Vec3(1.5, 0.8, 0.3)),
        "chassis");

    AeroParams p;
    p.CdA = 0.7;
    sys.force_elements.push_back(std::make_shared<AerodynamicForce>(chassis, p));

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    sim.initialize();

    // Constant forward force = 5000 N, along the chassis's X axis
    const Real F_const = 5000.0;
    sim.force_callback = [&](kernel::Simulator&, Real, VecX& tau) {
        tau(3) += F_const;
    };

    sim.run(180.0, 0.01);  // 3 minutes — drag rises slowly to terminal

    // Terminal velocity: F_drag = F_const → 0.5*rho*V^2*CdA = F_const
    // V_term = sqrt(2*F_const / (rho*CdA))
    const Real V_term_expected = std::sqrt(2.0 * F_const / (1.225 * 0.7));

    // Should be close to terminal (within 5%, since approach is asymptotic)
    REQUIRE_THAT(forward_speed(sim, chassis), WithinAbs(V_term_expected, V_term_expected * 0.05));
}

// ============================================================================
// Vehicle template integration
// ============================================================================

TEST_CASE("Aero: install_aerodynamics returns nullptr when both CdA and ClA are zero",
          "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.0;
    tmpl.chassis.ClA = 0.0;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero == nullptr);
}

TEST_CASE("Aero: install_aerodynamics adds force element", "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.7;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    const size_t n_before = sys.force_elements.size();
    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero != nullptr);
    REQUIRE(sys.force_elements.size() == n_before + 1);
}

TEST_CASE("Aero: drivetrain + aero coexist without exceptions",
          "[aero][template]")
{
    using namespace mbd;

    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.chassis.CdA = 0.7;
    tmpl.chassis.ClA = 0.0;

    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();

    Drivetrain dt(tmpl.drivetrain);
    dt.initialize(sim, vh);
    dt.connect(sim, vh);
    auto* aero = vh.install_aerodynamics(sys);
    REQUIRE(aero != nullptr);

    // Let the vehicle settle at idle (no throttle) for half a second first.
    // This avoids transient mismatches between drivetrain initial state and
    // chassis state.
    dt.throttle = 0.0;
    REQUIRE_NOTHROW(sim.run(0.5, 0.001));

    // Now apply moderate throttle and run.
    dt.throttle = 0.3;
    REQUIRE_NOTHROW(sim.run(2.0, 0.001));

    // Vehicle should be moving forward, at a sane speed.
    REQUIRE(forward_speed(sim, vh.chassis_body) > 0.5);
    REQUIRE(forward_speed(sim, vh.chassis_body) < 80.0);
}

TEST_CASE("Aero: downforce reduces ride height at speed", "[aero][downforce]")
{
    using namespace mbd;

    // Run two simulations: one without aero, one with aero. Compare ride heights
    // at the same simulation time, immediately after a short settle.
    auto run_scenario = [](Real ClA) -> Real {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.chassis.CdA = 0.0;
        tmpl.chassis.ClA = ClA;

        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);

        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);

        // Set the speed gently — give the system stable forward motion
        const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;
        sim.v(v_forward) = 30.0;
        sim.initialize();

        if (ClA > 0.0) {
            vh.install_aerodynamics(sys);
        }

        // Light speed-control to keep velocity ~constant at 30 m/s
        sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
            tau(v_forward) += 200.0 * (30.0 - forward_speed(s, vh.chassis_body));
        };

        // Short run to allow vertical equilibration
        sim.run(0.5, 0.0005);

        return sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.z();
    };

    const Real h_no_aero = run_scenario(0.0);
    const Real h_with_aero = run_scenario(3.0);

    INFO("h_no_aero = " << h_no_aero << ", h_with_aero = " << h_with_aero);

    // With downforce, chassis should be lower (smaller Z) at the same speed
    REQUIRE(h_with_aero < h_no_aero);

    const Real dh = h_no_aero - h_with_aero;
    REQUIRE(dh > 0.001);
    REQUIRE(dh < 0.5);
}

// ============================================================================
// The map (task 7.10, D71)
// ============================================================================

namespace
{
    using mbd::Vec6;

    /// A grid of 4 x 3 x 3 x 2 x 4 nodes, unevenly spaced: front and rear
    /// ride heights [m], pitch, roll, yaw [rad].
    mbd::AeroMap synthetic_grid()
    {
        mbd::AeroMap m;
        m.nodes[0] = {0.02, 0.05, 0.09, 0.14};
        m.nodes[1] = {0.04, 0.10, 0.16};
        m.nodes[2] = {-0.02, 0.0, 0.03};
        m.nodes[3] = {-0.03, 0.03};
        m.nodes[4] = {-0.2, -0.05, 0.05, 0.2};
        return m;
    }

    /// The node's variables for a flat index, the first variable fastest.
    std::array<Real, 5> node_of(const mbd::AeroMap& m, std::size_t index)
    {
        std::array<Real, 5> x{};
        for (std::size_t a = 0; a < 5; ++a) {
            x[a] = m.nodes[a][index % m.nodes[a].size()];
            index /= m.nodes[a].size();
        }
        return x;
    }

    /// A function multilinear in the five variables (each appears at most to
    /// the first power in each term): multilinear interpolation reproduces it
    /// everywhere between the nodes, to rounding.
    Vec6 synthetic(const std::array<Real, 5>& x)
    {
        const Real hf = x[0], hr = x[1], th = x[2], ph = x[3], b = x[4];
        Vec6 c;
        c << -0.9 + 0.8 * hf - 0.3 * hr + 2.0 * th + 0.4 * hf * b,
             0.05 - 2.4 * b + 0.6 * ph - 1.5 * hf * ph * b,
             -2.6 + 9.0 * hf + 6.0 * hr - 30.0 * hf * hr + 4.0 * th - 1.2 * th * b,
             0.3 * ph - 0.5 * b + 2.0 * hr * ph,
             -0.4 + 3.0 * hf - 2.5 * hr + 1.1 * th + 0.2 * th * ph,
             0.7 * b - 0.2 * ph * b + 1.3 * hf * hr * b;
        return c;
    }
}

TEST_CASE("Aero map: at its nodes the map gives its values exactly", "[aero][map]")
{
    // Arbitrary values (not a smooth function): at a node every weight is 1
    // or 0, so the coefficients are the node's, bitwise.
    using namespace mbd;
    AeroMap m = synthetic_grid();
    std::uint64_t seed = 12345;
    auto next = [&]() {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        return static_cast<Real>(seed >> 11) / 9007199254740992.0 - 0.5;
    };
    m.values.resize(4 * 3 * 3 * 2 * 4);
    for (Vec6& v : m.values) for (int k = 0; k < 6; ++k) v(k) = next();
    m.validate();
    int exact = 0;
    for (std::size_t i = 0; i < m.values.size(); ++i) exact += (m.at(node_of(m, i)) == m.values[i]) ? 1 : 0;
    CHECK(exact == static_cast<int>(m.values.size()));
}

TEST_CASE("Aero map: a synthetic map is reproduced exactly through the element", "[aero][map]")
{
    // The plan's test of task 7.10. The map holds a function multilinear in
    // its five variables at its 288 nodes; between them its interpolation is
    // that function. A chassis is placed and moved so that every variable
    // lies between nodes, the air moving too, and the element's force and
    // moment must be the function's coefficients times q, in the chassis's
    // axes, the moment carried from the reference point to the origin.
    //
    // The chassis turned by R = Rz(psi) Ry(theta) Rx(phi): its pitch is theta
    // and its roll phi by ISO 8855's definitions. The ride heights: the
    // points' heights, p_z + (R r)_z. The velocity relative to the air, along
    // the ground: V = U (cos(psi + b0), sin(psi + b0), 0); with x_B = R e_x =
    // (c_psi c_th, s_psi c_th, -s_th) and y_B = R e_y = (c_psi s_th s_ph -
    // s_psi c_ph, s_psi s_th s_ph + c_psi c_ph, c_th s_ph), expanding,
    //   V . x_B = U c_th cos(b0),  V . y_B = U (s_th s_ph cos(b0) + c_ph sin(b0)),
    // and the yaw is atan2 of the second over the first; q = rho U^2 / 2.
    // The chassis's own velocity is V plus the wind plus a vertical part
    // (which the element ignores), less omega x (R r_ref). Every operation is
    // rounding-exact to about 1e-16; allow 1e-12 of the largest component.
    using namespace mbd;
    auto map = std::make_shared<AeroMap>(synthetic_grid());
    map->values.resize(4 * 3 * 3 * 2 * 4);
    for (std::size_t i = 0; i < map->values.size(); ++i) map->values[i] = synthetic(node_of(*map, i));

    AeroParams p;
    p.map = map;
    p.reference_point = Vec3(0.1, 0.0, 0.2);
    p.front_point = Vec3(1.4, 0.3, -0.25);
    p.rear_point = Vec3(-1.5, -0.2, -0.25);
    const Vec3 wind(-3.0, 4.0, 0.0);
    p.wind = [wind](const Vec3&) { return wind; };

    const Real psi = 0.7, theta = 0.012, phi = -0.017, b0 = 0.08, U = 30.0;
    const Quat q_WB(Eigen::AngleAxis<Real>(psi, Vec3::UnitZ()) * Eigen::AngleAxis<Real>(theta, Vec3::UnitY())
                    * Eigen::AngleAxis<Real>(phi, Vec3::UnitX()));
    const Vec3 omega(0.1, -0.2, 0.3);

    for (const Real road_height : {0.0, 0.5}) {
        AeroParams pr = p;
        if (road_height != 0.0) pr.road = std::make_shared<FlatRoad>(road_height);
        AerodynamicForce aero(1, pr);
        const Vec3 p_WB(100.0, -40.0, road_height + 0.07 - (q_WB * p.front_point).z());
        const Vec3 V = U * Vec3(std::cos(psi + b0), std::sin(psi + b0), 0.0);
        const Vec3 r_ref = q_WB * p.reference_point;
        const Vec3 v_WB = V + wind + Vec3(0.0, 0.0, 0.5) - omega.cross(r_ref);
        std::vector<RigidBodyState> states(2);
        states[1] = RigidBodyState(p_WB, q_WB, v_WB, omega);
        std::vector<RigidBodyForces> forces(2);
        aero.apply(states, forces);

        std::array<Real, 5> x{};
        x[0] = 0.07;
        x[1] = p_WB.z() + (q_WB * p.rear_point).z() - road_height;
        x[2] = theta;
        x[3] = phi;
        x[4] = std::atan2(std::sin(theta) * std::sin(phi) * std::cos(b0) + std::cos(phi) * std::sin(b0),
                          std::cos(theta) * std::cos(b0));
        const Real q = 0.5 * 1.225 * U * U;
        const Vec6 c = synthetic(x);
        const Vec3 F = q_WB * (q * c.head<3>());
        const Vec3 M = q_WB * (q * c.tail<3>()) + r_ref.cross(F);
        INFO("road at " << road_height << " m: variables " << x[0] << ", " << x[1] << ", " << x[2] << ", " << x[3]
             << ", " << x[4] << "; force " << F.transpose() << " N; error " << (forces[1].f_W - F).norm() << " N, "
             << (forces[1].tau_W - M).norm() << " N m");
        CHECK(x[1] > 0.04);
        CHECK(x[1] < 0.16);
        const Real scale = std::max(F.cwiseAbs().maxCoeff(), M.cwiseAbs().maxCoeff());
        for (int k = 0; k < 3; ++k) {
            CHECK_THAT(forces[1].f_W(k), WithinAbs(F(k), 1e-12 * scale));
            CHECK_THAT(forces[1].tau_W(k), WithinAbs(M(k), 1e-12 * scale));
        }
    }
}

TEST_CASE("Aero map: beyond its first and last nodes the map is held", "[aero][map]")
{
    using namespace mbd;
    AeroMap m = synthetic_grid();
    m.values.resize(4 * 3 * 3 * 2 * 4);
    for (std::size_t i = 0; i < m.values.size(); ++i) m.values[i] = synthetic(node_of(m, i));
    const std::array<Real, 5> inside{0.07, 0.12, 0.01, -0.01, 0.1};
    for (std::size_t a = 0; a < 5; ++a) {
        std::array<Real, 5> above = inside, at_last = inside, below = inside, at_first = inside;
        above[a] = m.nodes[a].back() + 1.0;
        at_last[a] = m.nodes[a].back();
        below[a] = m.nodes[a].front() - 1.0;
        at_first[a] = m.nodes[a].front();
        CHECK(m.at(above) == m.at(at_last));
        CHECK(m.at(below) == m.at(at_first));
    }
}

TEST_CASE("Aero map: the constant model is a map", "[aero][map]")
{
    // The settings of the constant model (the sports car's in the regression
    // scenario: downforce growing by 3 m^2/m below 0.6 m, the centre of
    // pressure off the centre line) as a two-node map give the element's
    // forces without a map, at heights above and below h_ref, at any
    // attitude. The map interpolates over 10 m, so the two differ by
    // rounding of the weight, about 1e-16 x 10 m x 3 m^2/m relative to the
    // forces: allow 1e-12. And level, straight on, the old closed form: drag
    // q CdA back, downforce q (ClA + 3 (0.6 - h)), the moment cop x F.
    using namespace mbd;
    AeroParams p;
    p.CdA = 0.7;
    p.ClA = 1.2;
    p.h_ref = 0.6;
    p.dClA_dh = 3.0;
    p.cop_offset_chassis = Vec3(0.2, 0.03, 0.05);
    AerodynamicForce direct(1, p);
    AeroParams pm = p;
    pm.map = std::make_shared<AeroMap>(AeroMap::from(p));
    AerodynamicForce mapped(1, pm);
    const Quat tilted(Eigen::AngleAxis<Real>(0.3, Vec3::UnitZ()) * Eigen::AngleAxis<Real>(0.02, Vec3::UnitY()));
    for (const Real h : {0.45, 0.6, 0.8}) {
        for (const Quat& q_WB : {Quat::Identity(), tilted}) {
            std::vector<RigidBodyState> states(2);
            states[1] = RigidBodyState(Vec3(0.0, 0.0, h), q_WB, Vec3(25.0, 2.0, 0.1), Vec3(0.0, 0.1, 0.0));
            std::vector<RigidBodyForces> a(2), b(2);
            direct.apply(states, a);
            mapped.apply(states, b);
            const Real scale = a[1].f_W.norm();
            CHECK((a[1].f_W - b[1].f_W).norm() < 1e-12 * scale);
            CHECK((a[1].tau_W - b[1].tau_W).norm() < 1e-12 * scale);
        }
        const RigidBodyForces f = aero_force(mapped, Vec3(0.0, 0.0, h), Vec3(25.0, 0.0, 0.0));
        const Real q = 0.5 * 1.225 * 625.0;
        const Vec3 F(-q * 0.7, 0.0, -q * (1.2 + 3.0 * std::max(0.6 - h, 0.0)));
        CHECK((f.f_W - F).norm() < 1e-12 * F.norm());
        CHECK((f.tau_W - p.cop_offset_chassis.cross(F)).norm() < 1e-12 * F.norm());
    }
}

TEST_CASE("Aero map: refusals", "[aero][map]")
{
    using namespace mbd;
    auto error_of = [](const AeroParams& p) {
        try {
            AerodynamicForce bad(1, p);
        } catch (const MbdError& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    AeroParams p;
    auto m = std::make_shared<AeroMap>(synthetic_grid());
    m->values.assign(287, Vec6::Zero());          // one short of 4 x 3 x 3 x 2 x 4
    p.map = m;
    CHECK_THAT(error_of(p), ContainsSubstring("MBD-F060"));
    auto m2 = std::make_shared<AeroMap>(synthetic_grid());
    m2->nodes[4] = {0.1, 0.1};                    // not strictly increasing
    m2->values.assign(4 * 3 * 3 * 2 * 2, Vec6::Zero());
    p.map = m2;
    CHECK_THAT(error_of(p), ContainsSubstring("MBD-F060"));
    AeroParams q;
    q.dClA_dh = -1.0;
    CHECK_THAT(error_of(q), ContainsSubstring("MBD-F020"));
}
