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

    /// A wheel with a FullTireForce. Returns {wheel, tire_ptr}.
    std::pair<FreeWheel, mbd::FullTireForce*>
    make_free_wheel(mbd::Real /*mass*/, mbd::Real R_free, mbd::Real k_z, mbd::Real c_z)
    {
        using namespace mbd;
        FreeWheel w;
        w.tire = std::make_shared<FullTireForce>(1, R_free, k_z, c_z,
                                                 PacejkaTireParams::DefaultPassengerCar());
        FullTireForce* ptr = w.tire.get();
        return {std::move(w), ptr};
    }
}

// ============================================================================
// Vertical contact behavior
// ============================================================================

TEST_CASE("FullTireForce: no contact when wheel above ground",
          "[full_tire][vertical]")
{
    using namespace mbd;

    auto [sys, tire] = make_free_wheel(40.0, 0.35, 200000.0, 500.0);

    // Wheel at z = 0.5 (well above R_free = 0.35 + ground at 0)
    sys.q << 0.0, 0.0, 0.5, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(0.0, eps));
    REQUIRE_THAT(tire->get_Fx(), WithinAbs(0.0, eps));
    REQUIRE_THAT(tire->get_Fy(), WithinAbs(0.0, eps));
}

TEST_CASE("FullTireForce: vertical force at static deflection",
          "[full_tire][vertical]")
{
    using namespace mbd;

    const Real mass = 40.0;
    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(mass, R_free, k_z, 500.0);

    // Wheel at R_free - 0.01 (10mm compression)
    sys.q << 0.0, 0.0, R_free - 0.01, 0.0, 0.0, 0.0;
    sys.q_dot.setZero();
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Fz = k_z * 0.01 = 2000 N
    REQUIRE_THAT(tire->get_vertical_force(), WithinAbs(2000.0, 1.0));
    REQUIRE_THAT(tire->get_deflection(), WithinAbs(0.01, 1e-6));
}

// ============================================================================
// Slip angle from lateral velocity
// ============================================================================

TEST_CASE("FullTireForce: lateral velocity produces slip angle and Fy",
          "[full_tire][lateral]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<mbd::FullTireForce*>(tire_ptr);
    // 20mm compression → Fz ≈ 4000 N
    const Real defl = 0.02;
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;

    // Forward 20 m/s, lateral 1 m/s; the wheel spinning at free rolling so
    // kappa ≈ 0 (otherwise combined slip kills Fy)
    sys.q_dot << 20.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    const Real R_eff = R_free - defl * 0.5;
    sys.spin = 20.0 / R_eff;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Should have contact
    REQUIRE(tire->get_vertical_force() > 100.0);

    // Slip angle should be nonzero
    REQUIRE(std::abs(tire->get_slip_angle()) > 0.01);

    // Fy should be nonzero
    REQUIRE(std::abs(tire->get_Fy()) > 100.0);

    // At this low slip angle, force should be approximately K_alpha * alpha
    const Real Fz_actual = tire->get_vertical_force();
    const Real alpha_actual = tire->get_slip_angle();
    const Real K_alpha = tire->pacejka.cornering_stiffness(Fz_actual);
    const Real Fy_linear = K_alpha * alpha_actual;

    // Should agree within 20% (we're in the linear region)
    const Real rel_error = std::abs(tire->get_Fy() - Fy_linear) /
                           std::max(std::abs(Fy_linear), Real(1.0));
    REQUIRE(rel_error < 0.2);
}
// ============================================================================
// Slip ratio from forward velocity (locked wheel)
// ============================================================================

TEST_CASE("FullTireForce: forward velocity with locked wheel produces slip ratio",
          "[full_tire][longitudinal]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<FullTireForce*>(tire_ptr);
    sys.spin = 0.0;

    const Real defl = 0.02;
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;

    // Moving forward at 20 m/s, wheel not spinning (locked wheel)
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Slip ratio should be strongly negative (locked wheel braking)
    REQUIRE(tire->get_slip_ratio() < -0.5);

    // Fx should be negative (braking force, opposing forward motion)
    REQUIRE(tire->get_Fx() < -100.0);
}

TEST_CASE("FullTireForce: free-rolling wheel has near-zero slip ratio",
          "[full_tire][longitudinal]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<mbd::FullTireForce*>(tire_ptr);

    const Real defl = 0.02;
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;

    const Real Vx = 20.0;
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

    // Fx should be near zero
    REQUIRE_THAT(tire->get_Fx(), WithinAbs(0.0, 50.0));
}

// ============================================================================
// Combined slip
// ============================================================================

TEST_CASE("FullTireForce: combined slip produces both Fx and Fy",
          "[full_tire][combined]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire_ptr] = make_free_wheel(40.0, R_free, k_z, 0.0);
    auto* tire = const_cast<FullTireForce*>(tire_ptr);

    const Real defl = 0.02;
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;

    // Forward at 20 m/s, lateral at 2 m/s, partial braking: the wheel turns
    // slower than free rolling
    sys.q_dot << 20.0, 2.0, 0.0, 0.0, 0.0, 0.0;
    const Real R_eff = R_free - defl * 0.5;
    sys.spin = 15.0 / R_eff;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Both Fx and Fy should be nonzero
    REQUIRE(std::abs(tire->get_Fx()) > 10.0);
    REQUIRE(std::abs(tire->get_Fy()) > 10.0);

    // Combined Fy should be less than pure lateral Fy at same alpha
    const Real alpha = tire->get_slip_angle();
    const Real Fz = tire->get_vertical_force();
    auto pure_result = tire->pacejka.compute(0.0, alpha, Fz);
    const Real Fy_pure_abs = std::abs(pure_result.Fy) + 1.0;
    REQUIRE(std::abs(tire->get_Fy()) < Fy_pure_abs);
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

    // A free wheel body on the kernel, with the tyre as its only force.
    kernel::System sys;
    sys.model.gravity = Vec3(0.0, 0.0, -g_accel);
    sys.model.add_body(0, std::make_shared<kernel::FreeJointModel>(), Transform3::Identity(),
                       Transform3::Identity(),
                       RigidBodyInertia::from_solid_box(mass, Vec3(0.15, 0.15, R_free)), "wheel");
    sys.force_elements.push_back(std::make_shared<FullTireForce>(
        1, R_free, k_z, c_z, PacejkaTireParams::DefaultPassengerCar()));

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
// Force directions
// ============================================================================

TEST_CASE("FullTireForce: rotated wheel produces correct force directions",
          "[full_tire][rotated]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);

    const Real defl = 0.02;

    // Wheel yawed 90 degrees about Z: body X now points along world +Y.
    // Tire forward direction in world = (0, 1, 0).
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, pi / 2.0;

    // We want the wheel to move along world +X (perpendicular to tire forward).
    // FreeWheel's q_dot(0:2) is the world velocity of the wheel centre, so
    // q_dot(0) = 20 gives 20 m/s along world X directly.
    sys.q_dot << 20.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // Tire moves entirely sideways: Vx ≈ 0, Vy ≈ 20 → large slip angle. The
    // wheel does not spin, and nothing moves it along its heading.
    REQUIRE(std::abs(tire->get_slip_angle()) > 0.5);
    REQUIRE(std::abs(tire->get_Fy()) > 500.0);

    // Tire forward is world +Y, lateral is world -X (Z x forward). The wheel
    // slides along +X, across its heading, so its slip ratio is zero (Vx = 0
    // and a free-rolling wheel) and its horizontal force is Fy alone, along
    // -X: against the sliding. Vertical force acts along world Z, k_z defl.
    const Vec3& f = sys.forces[1].f_W;
    REQUIRE(f.x() < -500.0);
    REQUIRE_THAT(f.y(), WithinAbs(0.0, 1e-9));
    REQUIRE_THAT(f.z(), WithinAbs(k_z * defl, 1e-6));
}

// ============================================================================
// Telemetry accessors
// ============================================================================

TEST_CASE("FullTireForce: telemetry matches internal state",
          "[full_tire][telemetry]")
{
    using namespace mbd;

    const Real R_free = 0.35;
    const Real k_z = 200000.0;

    auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);

    const Real defl = 0.02;
    sys.q << 0.0, 0.0, R_free - defl, 0.0, 0.0, 0.0;
    sys.q_dot << 20.0, 1.0, 0.0, 0.0, 0.0, 0.0;
    sys.compute_kinematics();

    sys.clear_forces();
    sys.apply_force_elements();

    // All telemetry should be populated
    REQUIRE(tire->get_vertical_force() > 0.0);
    REQUIRE(tire->get_deflection() > 0.0);

    // Result struct should match accessors
    const auto& r = tire->get_last_result();
    REQUIRE_THAT(r.Fx, WithinAbs(tire->get_Fx(), eps));
    REQUIRE_THAT(r.Fy, WithinAbs(tire->get_Fy(), eps));
    REQUIRE_THAT(r.kappa, WithinAbs(tire->get_slip_ratio(), eps));
    REQUIRE_THAT(r.alpha, WithinAbs(tire->get_slip_angle(), eps));
    REQUIRE_THAT(r.Fz, WithinAbs(tire->get_vertical_force(), eps));
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
    // and the slip angle 0. A point fixed in the wheel at (0, 0, -R), as the
    // contact was before, would have moved forward or back by R sin(beta):
    // 0.10 m at 0.3 rad. The forces act at the contact point, so the moment
    // about the axle is that of Fx at the arm R - 0.02 down to the road:
    // (0, 0, -(R - 0.02)) x (Fx, 0, Fz) = (0, -(R - 0.02) Fx, 0), its Y part
    // -(R - 0.02) Fx. Tolerances: rounding in turning the axis, 1e-12.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02;
    const Real R_eff = R_free - 0.5 * defl;
    Real Fx_level = 0.0;
    for (const Real beta : {0.0, 0.3, -1.0, 2.5}) {
        INFO("beta " << beta);
        auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);
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
        REQUIRE_THAT(tire->last_axle_moment, WithinAbs(-(R_free - defl) * tire->get_Fx(), 1e-9));
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
    auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);
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
    // up the slope. The normal force k_z 0.02 = 4000 N acts along n.
    using namespace mbd;
    const Real R_free = 0.35, k_z = 200000.0, defl = 0.02, theta = 0.2;
    auto [sys, tire] = make_free_wheel(40.0, R_free, k_z, 0.0);
    tire->road = std::make_shared<Incline>(theta);
    const Vec3 n(-std::sin(theta), 0.0, std::cos(theta));
    const Vec3 centre(2.0, 0.0, 2.0 * std::tan(theta) + (R_free - defl) / std::cos(theta));
    sys.q << centre.x(), centre.y(), centre.z(), 0.0, 0.0, 0.0;
    sys.clear_forces();
    sys.apply_force_elements();

    require_vec3_near(tire->last_contact_pos_W, centre - (R_free - defl) * n, 1e-12);
    REQUIRE_THAT(tire->get_deflection(), WithinAbs(defl, 1e-12));
    require_vec3_near(tire->last_forward_W, Vec3(std::cos(theta), 0.0, std::sin(theta)), 1e-15);
    require_vec3_near(sys.forces[1].f_W, k_z * defl * n, 1e-9);
}
