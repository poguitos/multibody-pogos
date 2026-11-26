#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <Eigen/Geometry>
#include <cmath>

#include <memory>
#include <utility>
#include <vector>

#include "mbd/forces/tire.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/spatial/spatial.hpp"

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// The tyre on a wheel (task 7.4, D64): Pacejka's Appendix 3 tyre (MF 6.1)
// with the free radius, vertical stiffness and damping each test gives, so
// that the contact's numbers stay round: Fz = k_z depth exactly with c_z = 0.

namespace
{
    constexpr mbd::Real eps = 1e-9;

    void require_vec3_near(const mbd::Vec3& a, const mbd::Vec3& b, double tol)
    {
        REQUIRE_THAT(a.x(), WithinAbs(b.x(), tol));
        REQUIRE_THAT(a.y(), WithinAbs(b.y(), tol));
        REQUIRE_THAT(a.z(), WithinAbs(b.z(), tol));
    }

    /// A single wheel for testing the tyre force element, which reads only the
    /// wheel's world state. q holds the position and a rotation vector, q_dot
    /// the world velocity of the wheel centre (its last three entries, rotation
    /// rates, must stay zero); `spin` turns the wheel about its own axis, its
    /// body Y (task 7.3: the tyre takes the spin from the wheel body).
    struct FreeWheel {
        Eigen::Matrix<mbd::Real, 6, 1> q = Eigen::Matrix<mbd::Real, 6, 1>::Zero();
        Eigen::Matrix<mbd::Real, 6, 1> q_dot = Eigen::Matrix<mbd::Real, 6, 1>::Zero();
        mbd::Real spin{0.0};
        std::vector<mbd::RigidBodyState> states = std::vector<mbd::RigidBodyState>(2);
        std::vector<mbd::RigidBodyForces> forces = std::vector<mbd::RigidBodyForces>(2);
        std::shared_ptr<mbd::FullTireForce> tire;

        void compute_kinematics()
        {
            REQUIRE(q_dot.tail<3>().isZero());
            const mbd::Quat orientation = mbd::exp3(q.tail<3>());
            states[1] = mbd::RigidBodyState(q.head<3>(), orientation, q_dot.head<3>(),
                                            spin * (orientation * mbd::Vec3::UnitY()));
        }
        void clear_forces() { forces.assign(2, mbd::RigidBodyForces{}); }
        void apply_force_elements()
        {
            compute_kinematics();
            tire->apply(states, forces);
        }
    };

    /// The Appendix 3 tyre with this radius, stiffness and damping.
    mbd::MagicFormulaParameters tyre(mbd::Real R_free, mbd::Real k_z, mbd::Real c_z)
    {
        mbd::MagicFormulaParameters p = mbd::pacejka_appendix3_tyre();
        p.UNLOADED_RADIUS = R_free;
        p.VERTICAL_STIFFNESS = k_z;
        p.VERTICAL_DAMPING = c_z;
        return p;
    }

    /// A wheel with a FullTireForce. Returns {wheel, tire_ptr}.
    std::pair<FreeWheel, mbd::FullTireForce*> make_free_wheel(mbd::Real R_free, mbd::Real k_z, mbd::Real c_z,
                                                              bool lag)
    {
        using namespace mbd;
        FreeWheel w;
        w.tire = std::make_shared<FullTireForce>(1, tyre(R_free, k_z, c_z));
        w.tire->lag = lag;
        FullTireForce* ptr = w.tire.get();
        return {std::move(w), ptr};
    }

    /// A simulator to hand to rates(), which the tyre does not read.
    const mbd::kernel::Simulator& unused_simulator()
    {
        static mbd::kernel::System sys = [] {
            mbd::kernel::System s;
            s.model.add_body(0, std::make_shared<mbd::kernel::FreeJointModel>(), mbd::Transform3::Identity(),
                             mbd::Transform3::Identity(),
                             mbd::RigidBodyInertia::from_solid_box(1.0, mbd::Vec3(0.1, 0.1, 0.1)), "body");
            return s;
        }();
        static mbd::kernel::Simulator sim(sys);
        return sim;
    }

    /// The tyre's two rates at deflections (u, v), after apply() at them.
    mbd::Vec2 rates_at(FreeWheel& w, mbd::Real u, mbd::Real v)
    {
        w.tire->set_deflections(u, v);
        w.clear_forces();
        w.apply_force_elements();
        mbd::VecX x(2), x_dot(2);
        x << u, v;
        w.tire->rates(unused_simulator(), 0.0, x, x_dot);
        return {x_dot(0), x_dot(1)};
    }

    // The Appendix 3 tyre at Fz = FNOMIN = 4000 N (dfz = 0, dpi = 0, no
    // inclination), from its table: the slip stiffnesses (4.E15, 4.E25)
    // Kxk = Fz pKx1 = 4000 x 21.687 and Kya = pKy1 Fz sin(pKy4 atan(1 /
    // pKy2)), and the relaxation lengths, the paper's (19), (20): the
    // stiffnesses over the carcass's, cx0 = 358066 and cy0 = 102673 N/m.
    const mbd::Real Kxk_4000 = 4000.0 * 21.687;
    const mbd::Real Kya_4000 = -15.324 * 4000.0 * std::sin(2.0005 * std::atan(1.0 / 1.715));
    const mbd::Real sigma_kappa_4000 = Kxk_4000 / 358066.0;
    const mbd::Real sigma_alpha_4000 = -Kya_4000 / 102673.0;
}

// ============================================================================
// Vertical contact behavior
// ============================================================================

TEST_CASE("FullTireForce: no contact when wheel above ground",
          "[full_tire][vertical]")
{
    using namespace mbd;

    auto [sys, tire] = make_free_wheel(0.35, 200000.0, 500.0, true);

    // Wheel at z = 0.5 (well above R_free = 0.35 + ground at 0)
    sys.q << 0.0, 0.0, 0.5, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(0.0, eps));
    REQUIRE_THAT(tire->get_Fx(), WithinAbs(0.0, eps));
    REQUIRE_THAT(tire->get_Fy(), WithinAbs(0.0, eps));
    REQUIRE(sys.forces[1].f_W.isZero());
    REQUIRE(sys.forces[1].tau_W.isZero());
}

TEST_CASE("FullTireForce: vertical force at static deflection",
          "[full_tire][vertical]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(R_free, k_z, 500.0, true);

    // Wheel at R_free - 0.01 (10mm compression), at rest: Fz = k_z 0.01
    sys.q << 0.0, 0.0, R_free - 0.01, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(2000.0, 1e-6));
    REQUIRE_THAT(tire->get_deflection(), WithinAbs(0.01, 1e-12));
}

TEST_CASE("FullTireForce: the parameter set gives the radius, the stiffness and the damping",
          "[full_tire][vertical]")
{
    // The Appendix 3 tyre as it is: UNLOADED_RADIUS 0.3135 m,
    // VERTICAL_STIFFNESS 209651 N/m, VERTICAL_DAMPING 50 N s/m, VXLOW 1 m/s.
    using namespace mbd;
    const FullTireForce t(1, pacejka_appendix3_tyre());
    CHECK(t.R_free == 0.3135);
    CHECK(t.k_z == 209651.0);
    CHECK(t.c_z == 50.0);
    CHECK(t.V_low == 1.0);
    CHECK(t.lag);
    // A set without relaxation lengths cannot lag.
    MagicFormulaParameters p = pacejka_appendix3_tyre();
    p.LATERAL_STIFFNESS = 0.0;
    CHECK_THROWS(FullTireForce(1, p));
}

// ============================================================================
// The slips and the forces without lag: the steady-state tyre
// ============================================================================

TEST_CASE("FullTireForce: without lag the slips are the steady-state ones and the forces the Magic Formula's",
          "[full_tire][slip]")
{
    // A wheel 0.02 m into the ground (Fz = 200000 x 0.02 = 4000 N, c_z = 0),
    // moving at Vx = 20 and Vy = 1 m/s, spinning at 15 / R_eff: the slip
    // speeds are Vsx = 20 - 15 = 5 m/s and Vsy = 1 m/s, so kappa = -5 / 20 =
    // -0.25 and alpha = atan(1 / 20), positive (ISO: the contact moves to the
    // left). The forces are the Magic Formula's at those slips, upright
    // (gamma = 0), at Vcx = 20; they act at the contact point (0, 0, 0) along
    // the heading +X, to the left +Y, and up. To rounding.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, false);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    const Real R_eff = R_free - defl * 0.5;
    sys.spin = 15.0 / R_eff;
    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(4000.0, 1e-9));
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(-0.25, 1e-14));
    REQUIRE_THAT(tire->get_slip_angle(), WithinAbs(std::atan(0.05), 1e-15));
    REQUIRE_THAT(tire->last_gamma, WithinAbs(0.0, 1e-15));

    MagicFormulaInput in;
    in.Fz = 4000.0;
    in.kappa = -0.25;
    in.alpha = std::atan(0.05);
    in.Vcx = 20.0;
    const MagicFormulaOutput o = tire->tyre.evaluate(in);
    REQUIRE(o.Fx < -1000.0);   // braking
    REQUIRE(o.Fy < -500.0);    // against the leftward slide
    REQUIRE_THAT(tire->get_Fx(), WithinRel(o.Fx, 1e-12));
    REQUIRE_THAT(tire->get_Fy(), WithinRel(o.Fy, 1e-12));
    REQUIRE_THAT(tire->get_Mz(), WithinRel(o.Mz, 1e-12));
    require_vec3_near(sys.forces[1].f_W, Vec3(o.Fx, o.Fy, 4000.0), 1e-9);
}

TEST_CASE("FullTireForce: forward velocity with locked wheel produces slip ratio -1",
          "[full_tire][slip]")
{
    // Not spinning at 20 m/s: Vsx = 20, kappa = -20 / 20 = -1 exactly, the
    // force the Magic Formula's at -1: braking.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, false);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(-1.0, 1e-15));
    MagicFormulaInput in;
    in.Fz = 4000.0;
    in.kappa = -1.0;
    in.Vcx = 20.0;
    REQUIRE_THAT(tire->get_Fx(), WithinRel(tire->tyre.evaluate(in).Fx, 1e-12));
    REQUIRE(tire->get_Fx() < -0.5 * 1.0422 * 4000.0);   // braking, at least half the peak
}

TEST_CASE("FullTireForce: free-rolling wheel has zero slip ratio",
          "[full_tire][slip]")
{
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, Vx = 20.0;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, false);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << Vx, 0.0, 0.0, 0.0, 0.0, 0.0;

    // The wheel spinning at free rolling: omega = Vx / R_eff
    const Real R_eff = R_free - defl * 0.5;
    sys.spin = Vx / R_eff;
    sys.compute_kinematics();
    REQUIRE_THAT(tire->free_rolling_spin(sys.states), WithinAbs(Vx / R_eff, 1e-12));

    sys.clear_forces();
    sys.apply_force_elements();

    // Slip ratio zero but for rounding: (Vx / R_eff) R_eff - Vx, over Vx
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(0.0, 1e-14));
}

TEST_CASE("FullTireForce: rotated wheel produces correct force directions",
          "[full_tire][slip]")
{
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, false);

    // Wheel yawed 90 degrees about Z: its heading is world +Y and its left
    // world -X. It slides along world +X at 20 m/s, across its heading: Vx =
    // 0, Vy = -20, so without lag tan(alpha) = -20 / V_low = -20, alpha =
    // atan(-20). The cornering stiffness being negative, the side force is
    // positive, along the tyre's left, -X: against the slide. The slip ratio
    // is zero (no spin, no Vx).
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, pi / 2.0;
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_slip_angle(), WithinAbs(std::atan(-20.0), 1e-12));
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(0.0, 1e-12));
    REQUIRE(tire->get_Fy() > 2000.0);

    // The force: Fy along -X, Fx along +Y (the Magic Formula's shifts at zero
    // slip, small), k_z defl up.
    const Vec3& f = sys.forces[1].f_W;
    REQUIRE_THAT(f.x(), WithinAbs(-tire->get_Fy(), 1e-9));
    REQUIRE_THAT(f.y(), WithinAbs(tire->get_Fx(), 1e-9));
    REQUIRE(std::abs(f.y()) < 0.02 * std::abs(f.x()));
    REQUIRE_THAT(f.z(), WithinAbs(k_z * defl, 1e-6));
}

TEST_CASE("FullTireForce: a leaning wheel's inclination and camber thrust",
          "[full_tire][slip]")
{
    // The wheel turned by gamma = 0.1 rad about +X: its top leans to the
    // right (to -Y), its axis a = (0, cos g, sin g), the inclination asin(a .
    // z) = g, positive (ISO). Rolling freely at 10 m/s on its contact 0.02
    // into the ground (Fz 4000 N): kappa = alpha = 0, and the side force is
    // the Magic Formula's at gamma = 0.1, the camber thrust, to the right
    // (the Appendix 3 tyre's pKy6 is negative), the way it leans.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, g = 0.1;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, false);
    sys.q << 0.0, 0.0, R_free * std::cos(g) - defl, g, 0.0, 0.0;
    sys.q_dot << 10.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.spin = 10.0 / (R_free - 0.5 * defl);
    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->last_gamma, WithinAbs(g, 1e-15));
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(tire->get_slip_angle(), WithinAbs(0.0, 1e-14));
    MagicFormulaInput in;
    in.Fz = 4000.0;
    in.gamma = g;
    in.Vcx = 10.0;
    REQUIRE_THAT(tire->get_Fy(), WithinRel(tire->tyre.evaluate(in).Fy, 1e-12));
    REQUIRE(tire->get_Fy() < -200.0);
}

// ============================================================================
// The lag
// ============================================================================

TEST_CASE("FullTireForce: the deflections' rates, and their steady state is the steady-state tyre",
          "[full_tire][lag]")
{
    // The wheel of the steady-state test above (Fz 4000 N, Vx = 20, Vsx = 5,
    // Vsy = 1 m/s). The rates (8.110) are du/dt = -Vsx - |Vx| u / sigma_kappa
    // and dv/dt = Vsy - |Vx| v / sigma_alpha, the relaxation lengths those of
    // the table at 4000 N (above). At u = sigma_kappa (-0.25) and v =
    // sigma_alpha (0.05) both are zero, and the slips and forces are the
    // steady-state ones: to rounding.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    sys.spin = 15.0 / (R_free - defl * 0.5);

    const Vec2 r = rates_at(sys, 0.01, -0.003);
    REQUIRE_THAT(tire->last_sigma_kappa, WithinRel(sigma_kappa_4000, 1e-12));
    REQUIRE_THAT(tire->last_sigma_alpha, WithinRel(sigma_alpha_4000, 1e-12));
    REQUIRE_THAT(r(0), WithinRel(-5.0 - 20.0 * 0.01 / sigma_kappa_4000, 1e-12));
    REQUIRE_THAT(r(1), WithinRel(1.0 + 20.0 * 0.003 / sigma_alpha_4000, 1e-12));

    const Vec2 steady = rates_at(sys, -0.25 * sigma_kappa_4000, 0.05 * sigma_alpha_4000);
    REQUIRE_THAT(steady(0), WithinAbs(0.0, 1e-13));
    REQUIRE_THAT(steady(1), WithinAbs(0.0, 1e-13));
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(-0.25, 1e-14));
    REQUIRE_THAT(tire->get_slip_angle(), WithinAbs(std::atan(0.05), 1e-14));
    const Real Fx = tire->get_Fx(), Fy = tire->get_Fy();

    auto [ss, ss_tire] = make_free_wheel(R_free, k_z, 0.0, false);
    ss.q = sys.q;
    ss.q_dot = sys.q_dot;
    ss.spin = sys.spin;
    ss.clear_forces();
    ss.apply_force_elements();
    REQUIRE_THAT(Fx, WithinRel(ss_tire->get_Fx(), 1e-12));
    REQUIRE_THAT(Fy, WithinRel(ss_tire->get_Fy(), 1e-12));
}

TEST_CASE("FullTireForce: after a step in slip angle the deflection reaches 63 % in one relaxation length",
          "[full_tire][lag]")
{
    // Rolling freely at Vx = 20 m/s with Vy stepped to 0.2 m/s at t = 0
    // (tan alpha = 0.01) from v = 0: dv/dt = Vy - Vx v / sigma_alpha, so v =
    // sigma_alpha 0.01 (1 - exp(-s / sigma_alpha)) over the distance s = Vx
    // t: 1 - 1/e of its final value after one relaxation length, the length's
    // definition (Pacejka, section 7.2). The tyre's own rates, by RK4 with
    // 2 mm steps (Vx h / sigma about 3e-3: RK4's error 1e-12 per length);
    // compared with the exponential within 1e-9 of the final value.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, Vx = 20.0;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << Vx, 0.2, 0.0, 0.0, 0.0, 0.0;
    sys.spin = Vx / (R_free - defl * 0.5);

    const Real sigma = sigma_alpha_4000;
    const int n = 300;
    const Real h = sigma / Vx / n;   // one relaxation length in n steps
    Real v = 0.0;
    for (int i = 0; i < n; ++i) {
        const Real k1 = rates_at(sys, 0.0, v)(1);
        const Real k2 = rates_at(sys, 0.0, v + 0.5 * h * k1)(1);
        const Real k3 = rates_at(sys, 0.0, v + 0.5 * h * k2)(1);
        const Real k4 = rates_at(sys, 0.0, v + h * k3)(1);
        v += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    const Real final_v = sigma * 0.01;
    INFO("v after one relaxation length " << v << " of " << final_v << ": " << v / final_v);
    REQUIRE_THAT(v, WithinAbs(final_v * (1.0 - std::exp(-1.0)), 1e-9 * final_v));
}

TEST_CASE("FullTireForce: at rest the tyre holds like the carcass's spring, with the low-speed damping",
          "[full_tire][lag]")
{
    // At rest and locked (Vx = 0, no spin, Vsx = 0), a longitudinal
    // deflection u gives kappa' = u / sigma_kappa (8.111), and for a small
    // one Fx(u) - Fx(0) = Kxk u / sigma_kappa = cx u: the carcass's stiffness
    // of the table, cx0 = 358066 N/m at the nominal load (the paper's (19)).
    // For u = -1e-4 m, kappa' = -4e-4. The curve's slope there differs from
    // its slope at its centre, kappa = -SHx, by about (1 + E + C^2 / 2) (B
    // x)^2 at a distance x from the centre: x below 7e-4, B = Kxk / (Cx Dx)
    // = 13: 1e-4 at most, allowed 2e-4. Moving at Vsx = 0.01 m/s instead,
    // (8.127) adds -k_V_low0 Vsx / Kxk to the slip: a force of -770 x 0.01 =
    // -7.7 N, allowed 1e-3 of it. The same holds across: Fy(v) - Fy(0) = Kya
    // v / sigma_alpha = -cy0 v, cy0 = 102673; but the side force's curve is
    // centred 0.0018 rad away (-SHy = pHy1's opposite), with B = Kya / (Cy
    // Dy) = 11.7: (1 + E + C^2 / 2) (B x)^2 = 1.2 x (0.021)^2 = 5e-4,
    // allowed 1e-3.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    (void)rates_at(sys, 0.0, 0.0);
    const Real Fx0 = tire->get_Fx(), Fy0 = tire->get_Fy();
    (void)rates_at(sys, -1e-4, 0.0);
    REQUIRE_THAT(tire->get_Fx() - Fx0, WithinRel(-358066.0 * 1e-4, 2e-4));
    (void)rates_at(sys, 0.0, 1e-4);
    REQUIRE_THAT(tire->get_Fy() - Fy0, WithinRel(-102673.0 * 1e-4, 1e-3));

    sys.q_dot << 0.01, 0.0, 0.0, 0.0, 0.0, 0.0;
    (void)rates_at(sys, 0.0, 0.0);
    // k_V_low = 770 (1 + cos(pi 0.01 / 1)) / 2 at 0.01 m/s (8.128).
    const Real k_low = 0.5 * 770.0 * (1.0 + std::cos(pi * 0.01));
    REQUIRE_THAT(tire->get_slip_ratio(), WithinRel(-k_low * 0.01 / Kxk_4000, 1e-12));
    REQUIRE_THAT(tire->get_Fx() - Fx0, WithinRel(-7.7, 1e-3));
    // At V_low and above the damping is gone: kappa' = u / sigma only.
    sys.q_dot << 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    (void)rates_at(sys, 0.0, 0.0);
    REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(0.0, 1e-15));
}

TEST_CASE("FullTireForce: below V_low a deflection past full sliding is not let grow",
          "[full_tire][lag]")
{
    // Locked, sliding forward at 0.5 m/s (below V_low = 1): Vsx = 0.5. The
    // slip of full sliding is 3 Dx / Kxk = 3 x 1.0422 x 4000 / 86748 = 0.1442
    // (8.114). At u = -0.2 sigma_kappa, beyond it, the rate -0.5 - 0.5 u /
    // sigma = -0.4 would make |u| grow: it is held at 0 (8.112). At u = +0.2
    // sigma it shrinks, -0.6, and is kept; at 2 m/s nothing is held: -2 + 0.4.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << 0.5, 0.0, 0.0, 0.0, 0.0, 0.0;
    const Real s = sigma_kappa_4000;
    REQUIRE(0.2 > 3.0 * 1.0422 * 4000.0 / Kxk_4000);
    REQUIRE_THAT(rates_at(sys, -0.2 * s, 0.0)(0), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(rates_at(sys, 0.2 * s, 0.0)(0), WithinRel(-0.6, 1e-12));
    sys.q_dot << 2.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    REQUIRE_THAT(rates_at(sys, -0.2 * s, 0.0)(0), WithinRel(-1.6, 1e-12));
}

TEST_CASE("FullTireForce: off the road the deflections relax", "[full_tire][lag]")
{
    // In the air at 20 m/s: du/dt = -(|Vx| + V_low) u / sigma_kappa at the
    // nominal load (4000 N, the table's lengths above), and the same across.
    using namespace mbd;
    auto [sys, tire] = make_free_wheel(0.35, 200000.0, 0.0, true);
    sys.q << 0.0, 0.0, 0.5, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    const Vec2 r = rates_at(sys, 0.01, 0.02);
    REQUIRE_THAT(r(0), WithinRel(-21.0 * 0.01 / sigma_kappa_4000, 1e-12));
    REQUIRE_THAT(r(1), WithinRel(-21.0 * 0.02 / sigma_alpha_4000, 1e-12));
    REQUIRE(tire->get_Fx() == 0.0);
}

TEST_CASE("FullTireForce: the rolling resistance opposes the wheel's rolling, and a wheel at rest has none",
          "[full_tire][lag]")
{
    // Rolling freely forwards at 20 m/s (Omega R_eff = 20, above V_low) the
    // applied moment is the Magic Formula's My, negative about the wheel's
    // left (its spin axis): -R0 Fz0 (qsy1 + qsy3 |Vx / V0| + qsy4 (Vx /
    // V0)^4) (Fz / Fz0)^qsy7 at Fz = Fz0 and no Fx, R0 the set's free radius
    // 0.35. Rolling at Omega R_eff = 0.005 m/s, half of V_roll (0.01 m/s),
    // half of the formula's at that speed; at rest, none. The moment about
    // the axle is then -(R_free - defl) Fx + My.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    const Real R_eff = R_free - 0.5 * defl;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.spin = 20.0 / R_eff;
    // No deflection: Fx is the shifts' alone, and qsy2 is 0 anyway.
    (void)rates_at(sys, 0.0, 0.0);
    const Real v = 20.0 / 16.7;
    const Real My = -0.35 * 4000.0 * (0.00702 + 0.001515 * v + 8.514e-5 * std::pow(v, 4.0));
    REQUIRE_THAT(tire->last_My, WithinRel(My, 1e-12));
    REQUIRE_THAT(tire->last_axle_moment, WithinAbs(-(R_free - defl) * tire->get_Fx() + My, 1e-9));

    sys.q_dot << 0.005, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.spin = 0.005 / R_eff;
    (void)rates_at(sys, 0.0, 0.0);
    const Real v_slow = 0.005 / 16.7;
    const Real My_slow = -0.35 * 4000.0 * (0.00702 + 0.001515 * v_slow + 8.514e-5 * std::pow(v_slow, 4.0));
    REQUIRE_THAT(tire->last_My, WithinRel(0.5 * My_slow, 1e-12));

    sys.q_dot.setZero();
    sys.spin = 0.0;
    (void)rates_at(sys, 0.0, 0.0);
    REQUIRE(tire->last_My == 0.0);
}

TEST_CASE("FullTireForce: a simulator integrates the tyre's deflections", "[full_tire][lag]")
{
    // A simulator made from a system with the tyre attaches it (task 7.4):
    // two states, from zero; none without lag.
    using namespace mbd;
    for (const bool lag : {true, false}) {
        kernel::System sys;
        sys.model.add_body(0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
                           Transform3::Identity(), RigidBodyInertia::from_solid_box(40.0, Vec3(0.15, 0.15, 0.3)),
                           "wheel");
        auto tire = std::make_shared<FullTireForce>(1, pacejka_appendix3_tyre());
        tire->lag = lag;
        sys.force_elements.push_back(tire);
        kernel::Simulator sim(sys);
        REQUIRE(sim.state_elements().size() == 1);
        REQUIRE(sim.state_elements()[0] == tire.get());
        REQUIRE(sim.x.size() == (lag ? 2 : 0));
        if (lag) REQUIRE(sim.x.isZero());
    }
}

// ============================================================================
// Static settling with FullTireForce
// ============================================================================

TEST_CASE("FullTireForce: free wheel settles to static equilibrium under gravity",
          "[full_tire][settling]")
{
    using namespace mbd;

    const Real mass = 40.0;
    const Real R_free = 0.35;
    const Real k_z = 200000.0;
    const Real c_z = 2000.0;

    // A free wheel body on the kernel, with the tyre as its only force. A
    // wheel balanced on its edge is unstable: the Magic Formula's side force
    // and overturning couple at zero slip (its shifts pHy1, pVy1, qsx1, and
    // the residual torque qDz6) would tip it over (task 7.4). They are set to
    // 0 here, the wheel stays balanced, and the test is of the vertical
    // contact and its damping, with the lag running.
    MagicFormulaParameters balanced = tyre(R_free, k_z, c_z);
    balanced.PHX1 = balanced.PHX2 = balanced.PVX1 = balanced.PVX2 = 0.0;
    balanced.PHY1 = balanced.PHY2 = balanced.PVY1 = balanced.PVY2 = 0.0;
    balanced.QSX1 = balanced.QDZ6 = balanced.QDZ7 = 0.0;
    kernel::System sys;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    sys.model.add_body(0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
                       Transform3::Identity(),
                       RigidBodyInertia::from_solid_box(mass, Vec3(0.15, 0.15, R_free)), "wheel");
    sys.force_elements.push_back(std::make_shared<FullTireForce>(1, balanced));

    kernel::Simulator sim(sys);
    sim.method = kernel::Integrator::RK4;

    // Start slightly above equilibrium
    const Real z_eq = R_free - mass * g_accel / k_z;
    sim.q(2) = z_eq + 0.02;
    sim.initialize();

    sim.run(2.0, 0.001);

    // Should settle near equilibrium, without moving horizontally
    const RigidBodyState& wheel = sim.states()[1];
    REQUIRE_THAT(wheel.p_WB.z(), WithinAbs(z_eq, 0.002));
    REQUIRE_THAT(wheel.p_WB.x(), WithinAbs(0.0, 0.001));
    REQUIRE_THAT(wheel.p_WB.y(), WithinAbs(0.0, 0.001));

    // Velocities near zero
    for (int i = 0; i < 6; ++i) {
        REQUIRE_THAT(sim.v(i), WithinAbs(0.0, 0.05));
    }
}

// ============================================================================
// Telemetry accessors
// ============================================================================

TEST_CASE("FullTireForce: telemetry matches internal state",
          "[full_tire][telemetry]")
{
    using namespace mbd;
    auto [sys, tire] = make_free_wheel(0.35, 200000.0, 0.0, false);
    sys.q << 0.0, 0.0, 0.33, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE(tire->get_vertical_force() > 0.0);
    REQUIRE(tire->get_deflection() > 0.0);
    const MagicFormulaOutput& o = tire->last_output;
    REQUIRE(o.Fx == tire->get_Fx());
    REQUIRE(o.Fy == tire->get_Fy());
    REQUIRE(o.Mz == tire->get_Mz());
    REQUIRE_THAT(o.Fz, WithinAbs(tire->get_vertical_force(), eps));
}

// ============================================================================
// The contact from the wheel plane and the road (task 7.3, F8)
// ============================================================================

TEST_CASE("FullTireForce: a wheel turned about its axle keeps its contact point and its forces",
          "[full_tire][contact]")
{
    // The plan's "done when" for task 7.3, first part: a pitching upright no
    // longer moves the contact point. A wheel of free radius R = 0.35 m with
    // its centre at (1, 0.5, R - 0.02) on level ground, moving forward at
    // 20 m/s and spinning at 15 / R_eff, turned about its axle (its Y) by
    // beta. Its lowest point is the bottom of its circle, (1, 0.5, -0.02),
    // whatever beta: turning a circle in its own plane leaves it where it is.
    // So for every beta the contact point is (1, 0.5, 0), the depth 0.02, the
    // normal force k_z 0.02 = 4000 N, the slip ratio (15 - 20) / 20 = -0.25
    // (without lag) and the slip angle 0. A point fixed in the wheel at
    // (0, 0, -R), as the contact was before, would have moved forward or back
    // by R sin(beta): 0.10 m at 0.3 rad. The forces act at the contact point,
    // so the moment about the axle is that of Fx at the arm R - 0.02 down to
    // the road, (0, 0, -(R - 0.02)) x (Fx, 0, Fz) = (0, -(R - 0.02) Fx, 0),
    // plus the rolling resistance My about the axle (task 7.4). Tolerances:
    // rounding in turning the axis, 1e-12.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    const Real R_eff = R_free - 0.5 * defl;
    Real Fx_level = 0.0;
    for (const Real beta : {0.0, 0.3, -1.0, 2.5}) {
        INFO("beta " << beta);
        auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, false);
        sys.q << 1.0, 0.5, R_free - defl, 0.0, beta, 0.0;
        sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        sys.spin = 15.0 / R_eff;
        sys.clear_forces();
        sys.apply_force_elements();

        require_vec3_near(tire->last_contact_pos_W, Vec3(1.0, 0.5, 0.0), 1e-12);
        REQUIRE_THAT(tire->get_deflection(), WithinAbs(defl, 1e-12));
        REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(k_z * defl, 1e-6));
        REQUIRE_THAT(tire->get_slip_ratio(), WithinAbs(-0.25, 1e-12));
        REQUIRE_THAT(tire->get_slip_angle(), WithinAbs(0.0, 1e-12));
        REQUIRE(tire->get_Fx() < -1000.0);
        if (beta == 0.0) Fx_level = tire->get_Fx();
        REQUIRE_THAT(tire->get_Fx(), WithinAbs(Fx_level, 1e-9 * std::abs(Fx_level)));
        REQUIRE_THAT(tire->last_axle_moment,
                     WithinAbs(-(R_free - defl) * tire->get_Fx() + tire->last_My, 1e-9));
    }
}

TEST_CASE("FullTireForce: a leaning wheel's contact point moves along its plane",
          "[full_tire][contact]")
{
    // Camber: the wheel turned by gamma = 0.1 rad about X, its axis
    // a = (0, cos g, sin g). The lowest point of its circle is along the
    // part of -Z in its plane: -(z - (z.a) a) / |.| = (0, sin g, -cos g). With
    // the centre at height R cos g - 0.02 the lowest point is 0.02 below the
    // ground, at y + R sin g = 0.0349 m to the left of the centre, and so is
    // the contact point, on the ground. The heading is a x z = (cos g, 0, 0),
    // forward, and the lateral direction z x forward = Y.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, g = 0.1;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    sys.q << 0.0, 0.0, R_free * std::cos(g) - defl, g, 0.0, 0.0;
    sys.q_dot << 10.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.clear_forces();
    sys.apply_force_elements();

    require_vec3_near(tire->last_contact_pos_W, Vec3(0.0, R_free * std::sin(g), 0.0), 1e-12);
    REQUIRE_THAT(tire->get_deflection(), WithinAbs(defl, 1e-12));
    require_vec3_near(tire->last_forward_W, Vec3::UnitX(), 1e-15);
    require_vec3_near(tire->last_lateral_W, Vec3::UnitY(), 1e-15);
}

namespace
{
    /// A plane rising along X at the angle theta: h = x tan(theta), its normal
    /// (-sin theta, 0, cos theta).
    class Incline final : public mbd::Road {
    public:
        explicit Incline(mbd::Real theta) : theta_(theta) {}
        mbd::RoadPoint at(mbd::Real x, mbd::Real) const override
        {
            return {x * std::tan(theta_), mbd::Vec3(-std::sin(theta_), 0.0, std::cos(theta_)), 1.0};
        }
        const char* name() const override { return "incline"; }

    private:
        mbd::Real theta_;
    };
}

TEST_CASE("FullTireForce: on an incline the contact point is below the centre along the road's normal",
          "[full_tire][contact]")
{
    // A wheel standing upright (axis Y) on a plane rising at theta = 0.2 rad
    // along X, its centre R - 0.02 from the plane along the normal n: at
    // x = 2, z = 2 tan(theta) + (R - 0.02) / cos(theta). Its axis is normal to
    // n, so its lowest point is c - R n, 0.02 below the plane, and the contact
    // point c - (R - 0.02) n. The heading a x n = (cos theta, 0, sin theta):
    // up the slope. The normal force k_z 0.02 = 4000 N acts along n; at rest
    // with no deflection the forces in the road plane are the Magic Formula's
    // shifts at zero slip, Fx forward and Fy to the left.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, theta = 0.2;
    auto [sys, tire] = make_free_wheel(R_free, k_z, 0.0, true);
    tire->road = std::make_shared<Incline>(theta);
    const Vec3 n(-std::sin(theta), 0.0, std::cos(theta));
    const Vec3 forward(std::cos(theta), 0.0, std::sin(theta));
    const Vec3 centre(2.0, 0.0, 2.0 * std::tan(theta) + (R_free - defl) / std::cos(theta));
    sys.q << centre.x(), centre.y(), centre.z(), 0.0, 0.0, 0.0;
    sys.clear_forces();
    sys.apply_force_elements();

    require_vec3_near(tire->last_contact_pos_W, centre - (R_free - defl) * n, 1e-12);
    REQUIRE_THAT(tire->get_deflection(), WithinAbs(defl, 1e-12));
    require_vec3_near(tire->last_forward_W, forward, 1e-15);
    require_vec3_near(sys.forces[1].f_W,
                      k_z * defl * n + tire->get_Fx() * forward + tire->get_Fy() * Vec3::UnitY(), 1e-9);
}

TEST_CASE("FullTireForce: a tyre mounted on the other side of the car is mirrored", "[full_tire][slip]")
{
    // TNO's convention (task 7.4): the Appendix 3 tyre describes a left tyre
    // (TYRESIDE LEFT, the default). Mounted on the right it is mirrored. A
    // left wheel moving to the left (Vy = +1 m/s) and leaning to the right
    // (gamma = +0.05) and a right wheel in the mirror image (Vy = -1, gamma =
    // -0.05) then give mirror forces: the same Fx, opposite Fy, Mx and Mz,
    // to rounding. At zero slip the two of a car push against each other.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, g = 0.05;
    auto wheel = [&](bool right) {
        const Real s = right ? -1.0 : 1.0;
        auto [w, tire] = make_free_wheel(R_free, k_z, 0.0, false);
        tire->mount(right);
        REQUIRE(tire->mirrored == right);
        w.q << 0.0, 0.0, R_free * std::cos(g) - defl, s * g, 0.0, 0.0;
        w.q_dot << 20.0, s * 1.0, 0.0, 0.0, 0.0, 0.0;
        w.spin = 19.0 / (R_free - 0.5 * defl);
        w.clear_forces();
        w.apply_force_elements();
        return tire->last_output;
    };
    const MagicFormulaOutput left = wheel(false);
    const MagicFormulaOutput right = wheel(true);
    REQUIRE(std::abs(left.Fy) > 500.0);
    CHECK_THAT(right.Fx, WithinRel(left.Fx, 1e-12));
    CHECK_THAT(right.Fy, WithinRel(-left.Fy, 1e-12));
    CHECK_THAT(right.Mx, WithinRel(-left.Mx, 1e-12));
    CHECK_THAT(right.Mz, WithinRel(-left.Mz, 1e-12));

    // A right-side tyre's parameters, mounted on the right, are not mirrored.
    MagicFormulaParameters p = pacejka_appendix3_tyre();
    p.side = TyreSide::Right;
    FullTireForce t(1, p);
    t.mount(true);
    CHECK_FALSE(t.mirrored);
    t.mount(false);
    CHECK(t.mirrored);
}
