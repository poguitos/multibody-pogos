// Static equilibrium (plan task 3.5).
//
//   - a mass on a linear spring: one Newton step to the static deflection;
//   - a pendulum on a torsion spring: the angle where the spring holds
//     gravity, against the root of the scalar equation;
//   - a four-bar with a crank spring under gravity: the configuration of
//     least potential energy along the loop, found by brute force;
//   - a held coordinate stays where it is and the rest settles around it;
//   - an equilibrium balanced upright is found and reported unstable;
//   - a body hanging above the ground, where Newton has nothing to work
//     with, is brought down by dynamic relaxation;
//   - the detailed sedan (the plan's "done when"): under 20 iterations to
//     accelerations below 1e-6, with its tyre loads balancing its weight and
//     its centre of mass, and its free directions on the road reported.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::FourBar;

namespace {

bool has_code(const std::vector<std::string>& messages, const std::string& code)
{
    for (const auto& m : messages) {
        if (m.rfind(code, 0) == 0) return true;
    }
    return false;
}

/// A slender link of length l and mass m along X from its joint: centre of
/// mass at l/2 for a link, at l for a bob.
RigidBodyInertia bob(Real m, Real l)
{
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, Vec3(0.02, 0.02, 0.02));
    I.com_B = Vec3(l, 0.0, 0.0);
    return I;
}

/// A torsion spring on the coordinate of `body`'s joint.
std::shared_ptr<JointCoordinateForce> torsion_spring(const Model& model, int body, Real k, Real reference)
{
    JointCoordinateForceParams p;
    p.spring = Curve::linear(k);
    p.reference = reference;
    return std::make_shared<JointCoordinateForce>(model, body, p);
}

/// Root of a function increasing on [lo, hi], by bisection to the last bit.
template <class F>
Real root(F f, Real lo, Real hi)
{
    for (int k = 0; k < 200; ++k) {
        const Real mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0) hi = mid; else lo = mid;
    }
    return 0.5 * (lo + hi);
}

} // namespace

TEST_CASE("Statics: a mass on a spring settles at its static deflection in one step",
          "[kernel][statics]")
{
    // A body on a vertical slider, on a spring from the ground: k (L0 - z) = m g.
    // The force is linear in z, so the central differences give K exactly and
    // one Newton step lands on the solution.
    const Real m = 2.0, k = 1000.0, L0 = 0.5;
    System sys;
    const int b = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                     RigidBodyInertia::from_solid_box(m, Vec3(0.1, 0.1, 0.1)));
    SpringDamperParams p;
    p.free_length = L0;
    p.spring = Curve::linear(k);
    sys.force_elements.push_back(std::make_shared<SpringDamper>(0, b, Vec3::Zero(), Vec3::Zero(), p));
    Simulator sim(sys);
    sim.q(0) = L0;

    const StaticsReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    CHECK(r.iterations == 1);
    CHECK(std::abs(r.acceleration_before - g_accel) <= 1e-12);
    // One step, from a K whose only error is the roundoff of the central
    // difference: eps |k z| / (2 h) over k, 5.5e-11 relative with h = 1e-6
    // and z = 0.5. The step of 0.0196 m is then off by up to 1.1e-12 m, and
    // the acceleration by k / m times that, 5.4e-10.
    CHECK(std::abs(sim.q(0) - (L0 - m * g_accel / k)) <= 2e-12);
    CHECK(r.acceleration <= 1e-9);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(r.neutral_directions == 0);
    CHECK(r.unstable_directions == 0);
    CHECK(sim.v.isZero());
}

TEST_CASE("Statics: a pendulum settles where its torsion spring holds gravity", "[kernel][statics]")
{
    // A bob at distance l on a hinge about Z, gravity along -Y, angle th from
    // X: gravity's torque -m g l cos th, the spring's -k th. Equilibrium:
    // k th + m g l cos th = 0, increasing in th (k > m g l), so one root.
    const Real m = 1.0, l = 0.5, k = 20.0;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), bob(m, l));
    sys.joint_forces.push_back(torsion_spring(sys.model, b, k, 0.0));
    Simulator sim(sys);

    const StaticsReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    const Real th = root([&](Real x) { return k * x + m * g_accel * l * std::cos(x); }, -0.5 * pi, 0.0);
    // |a| <= tolerance leaves th off by at most tolerance x I / stiffness, with
    // I = m l^2 (plus the bob's own, negligible) and stiffness k - m g l sin th > k.
    const Real bound = 1e-6 * (m * l * l + 1e-3) / k;
    CHECK(std::abs(sim.q(0) - th) <= bound);
    CHECK(r.iterations <= 5);
    CHECK(r.unstable_directions == 0);
}

TEST_CASE("Statics: a four-bar settles at its least potential energy along the loop",
          "[kernel][statics]")
{
    // The four-bar under gravity (-Y) with a torsion spring on the crank, of
    // stiffness k about 1 rad. A closed loop: the equilibrium is where the
    // potential energy, gravity's plus the spring's, is least along the loop
    // q = closed(th). The search is a golden section over th; the bracket
    // [0.6, 1.4] holds the single minimum (checked below: the ends are
    // higher than the minimum found).
    FourBar fb;
    System sys;
    sys.model = fb.model;
    sys.constraints = fb.closure;
    const Real k = 50.0, th0 = 1.0;
    sys.joint_forces.push_back(torsion_spring(sys.model, fb.crank, k, th0));
    Simulator sim(sys);
    sim.q = FourBar::closed(1.2);

    const StaticsReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(r.unstable_directions == 0);

    Data data(sys.model);
    auto energy = [&](Real th) {
        forward_kinematics(sys.model, data, FourBar::closed(th), VecX::Zero(3));
        return potential_energy(sys.model, data) + 0.5 * k * (th - th0) * (th - th0);
    };
    Real lo = 0.6, hi = 1.4;
    const Real g = 0.5 * (std::sqrt(5.0) - 1.0);
    for (int n = 0; n < 200; ++n) {
        const Real x1 = hi - g * (hi - lo), x2 = lo + g * (hi - lo);
        if (energy(x1) < energy(x2)) hi = x2; else lo = x1;
    }
    const Real th_best = 0.5 * (lo + hi);
    REQUIRE(energy(0.6) > energy(th_best));
    REQUIRE(energy(1.4) > energy(th_best));
    // The golden section finds th to sqrt(eps) relative (the minimum is
    // quadratic), 2e-8; statics leaves th off by tolerance x inertia /
    // stiffness, below 1e-8 here. The other angles follow through the loop.
    CHECK(std::abs(sim.q(0) - th_best) <= 1e-7);
    CHECK((sim.q - FourBar::closed(sim.q(0))).cwiseAbs().maxCoeff() <= 1e-9);
}

TEST_CASE("Statics: a held coordinate stays and the rest settles around it", "[kernel][statics]")
{
    // A double pendulum under gravity (-Y), the first hinge held at 0.4 rad:
    // the second link hangs straight down, at absolute angle -pi/2, so its
    // relative angle is -pi/2 - 0.4.
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const auto hinge = std::make_shared<RevoluteJointModel>();
    const int b1 = sys.model.add_body(0, hinge, Transform3(), Transform3(), bob(1.0, 0.5), "upper");
    sys.model.add_body(b1, hinge, Transform3::FromTranslation(Vec3(0.5, 0.0, 0.0)), Transform3(), bob(1.0, 0.4), "lower");
    Simulator sim(sys);
    sim.q << 0.4, -1.4;

    StaticsOptions options;
    options.hold = {0};
    const StaticsReport r = static_equilibrium(sim, options);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    CHECK(sim.q(0) == 0.4);
    // tolerance x I / stiffness: I = m l^2 = 0.16, stiffness m g l = 3.9.
    CHECK(std::abs(sim.q(1) - (-0.5 * pi - 0.4)) <= 1e-7);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(r.unstable_directions == 0);
}

TEST_CASE("Statics: an equilibrium balanced upright is found and reported unstable",
          "[kernel][statics]")
{
    // A bob above its hinge, at th = pi/2, on a torsion spring too weak to
    // hold it (k = 1 < m g l = 4.9): an equilibrium, with stiffness
    // k - m g l < 0. Newton finds it from nearby; the report must say it is
    // unstable.
    const Real m = 1.0, l = 0.5, k = 1.0;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), bob(m, l));
    sys.joint_forces.push_back(torsion_spring(sys.model, b, k, 0.5 * pi));
    Simulator sim(sys);
    sim.q(0) = 0.5 * pi + 0.05;

    const StaticsReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    CHECK(std::abs(sim.q(0) - 0.5 * pi) <= 1e-7);
    CHECK(r.unstable_directions == 1);
    CHECK(has_code(r.warnings, "MBD-K072"));
}

TEST_CASE("Statics: dynamic relaxation brings a body down onto the ground", "[kernel][statics]")
{
    // A wheel on a vertical slider (world Y up), 5 cm above the ground it
    // will rest on through a tyre of stiffness k. Above the ground nothing
    // pushes back: the stiffness is zero where gravity acts, the Newton step
    // is zero, and dynamic relaxation must take over. On the ground,
    // k (R - y) = m g: y = R - m g / k.
    const Real m = 10.0, k = 1e5, R = 0.3;
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    // The slider's axis (Z of its joint frames) along world Y, with the body
    // frame kept aligned with the world.
    const Transform3 z_to_y(Quat(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX())), Vec3::Zero());
    const int b = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), z_to_y, z_to_y,
                                     RigidBodyInertia::from_solid_box(m, Vec3(0.2, 0.6, 0.6)));
    sys.force_elements.push_back(std::make_shared<TireContactForce>(b, R, k, 0.0));
    Simulator sim(sys);
    sim.q(0) = R + 0.05;
    sim.refresh();
    REQUIRE(std::abs(sim.states()[static_cast<std::size_t>(b)].p_WB.y() - (R + 0.05)) <= 1e-15);

    const StaticsReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    CHECK(r.relaxations >= 1);
    CHECK(r.relaxation_steps > 0);
    CHECK(has_code(r.notes, "MBD-K074"));
    // |a| <= 1e-6 leaves y off by at most m |a| / k = 1e-10.
    CHECK(std::abs(sim.q(0) - (R - m * g_accel / k)) <= 1e-10);
    CHECK(r.neutral_directions == 0);
}

TEST_CASE("Statics: the detailed sedan settles in under 20 iterations", "[kernel][statics][vehicle]")
{
    // The plan's "done when". The double-wishbone sedan from the approximate
    // start of set_vehicle_equilibrium (finding F9: accelerations of up to 73
    // there) to accelerations at rest below 1e-6.
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    System sys;
    const VehicleHandle vh = build_vehicle(sys, tmpl);
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();

    const StaticsReport r = static_equilibrium(sim);
    INFO(r.summary(sys.model));
    REQUIRE(r.converged);
    CHECK(r.iterations < 20);
    CHECK(r.acceleration <= 1e-6);
    CHECK(r.relaxation_steps == 0);
    CHECK(r.acceleration_before > 10.0);
    // On a flat road at rest nothing holds the car's position or heading:
    // forward, sideways and yaw are free; the other seven are held by the
    // springs and tyres.
    CHECK(r.degrees_of_freedom == 10);
    CHECK(r.neutral_directions == 3);
    CHECK(r.unstable_directions == 0);
    CHECK(has_code(r.notes, "MBD-K073"));

    // The whole car's balance, from its tyres alone. At rest the only
    // external forces are gravity at the centre of mass and the tyres'
    // vertical loads at their contact points, so the loads sum to the weight
    // and their moments about the centre of mass vanish. The accelerations
    // left (below 1e-6 per coordinate) bound the imbalance: each body's
    // acceleration is a sum over at most four joints with lever arms below
    // 2 m, under 1.2e-5 m/s^2, times 1,500 kg: 0.02 N, and 0.04 N m.
    sim.acceleration(sim.q, sim.v, sim.time);   // the tyres' loads at this state
    const Model& model = sys.model;
    Real mass = 0.0;
    for (int i = 1; i < model.nbodies(); ++i) mass += model.inertia[static_cast<std::size_t>(i)].mass;
    const Vec3 com = center_of_mass(model, sim.data());
    Real load = 0.0, pitch = 0.0, roll = 0.0;
    for (int c = 0; c < 4; ++c) {
        const FullTireForce* tire = vh.tire(c);
        load += tire->last_Fz;
        pitch += tire->last_Fz * (tire->last_contact_pos_W.x() - com.x());
        roll += tire->last_Fz * (tire->last_contact_pos_W.z() - com.z());
    }
    CHECK(std::abs(load - mass * g_accel) <= 0.05);
    CHECK(std::abs(pitch) <= 0.1);
    CHECK(std::abs(roll) <= 0.1);
}
