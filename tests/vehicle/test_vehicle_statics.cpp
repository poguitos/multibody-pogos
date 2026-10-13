// Static equilibrium of a full vehicle (plan task 3.5): the double-wishbone
// sedan settles in under 20 Newton iterations to accelerations below 1e-6.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <string>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using namespace mbd;

namespace {

VehicleTemplate sedan()
{
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    return tmpl;
}

/// How far the chassis origin moves in `duration` of simulation from the
/// simulator's state.
Real chassis_drift(kernel::Simulator& sim, const VehicleHandle& vh, Real duration)
{
    const int cq = sim.system.model.idx_q[static_cast<std::size_t>(vh.chassis_body)];
    const Vec3 start = sim.q.segment<3>(cq);
    sim.run(duration, 1e-3);
    return (sim.q.segment<3>(cq) - start).norm();
}

} // namespace

TEST_CASE("Statics: the double-wishbone sedan settles in under 20 iterations", "[vehicle][statics]")
{
    kernel::System sys;
    auto vh = build_vehicle(sys, sedan());
    const kernel::Model& model = sys.model;
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    const VecX q_start = sim.q;

    const kernel::StaticReport r = kernel::static_equilibrium(sim);
    INFO(r.summary(model));
    CHECK(r.converged);
    CHECK(r.iterations < 20);
    CHECK(r.residual < 1e-6);
    CHECK(r.relaxation_steps == 0);
    CHECK(r.warnings.empty());

    // Ten degrees of freedom, as validate() counts: three without stiffness
    // at rest, the chassis's position on the ground and its heading (the
    // tyres give no horizontal force without slip). Statics does not move
    // along them but for second-order effects: in the free joint's own
    // coordinates (its rigid motion is an exponential), a rise dy with a
    // pitch or roll dth moves the others by about dth dy / 2.
    CHECK(r.degrees_of_freedom == 10);
    CHECK(r.supported == 7);
    CHECK(r.neutral == 3);
    const int cv = model.idx_v[static_cast<std::size_t>(vh.chassis_body)];
    const VecX& d = r.position_change;   // free joint: rotation (roll X, yaw Y, pitch Z), then translation
    const Real second_order = 0.5 * (std::abs(d(cv)) + std::abs(d(cv + 2))) * std::abs(d(cv + 4));
    INFO("chassis changes: yaw " << d(cv + 1) << ", forward " << d(cv + 3) << ", lateral " << d(cv + 5)
         << "; second-order bound " << second_order);
    CHECK(std::abs(d(cv + 1)) <= second_order + 1e-9);
    CHECK(std::abs(d(cv + 3)) <= second_order + 1e-9);
    CHECK(std::abs(d(cv + 5)) <= second_order + 1e-9);

    // The tyres carry the weight. The vertical forces on the whole vehicle,
    // the tyres' and gravity's, sum to the mass times the vertical
    // accelerations, each below 1e-6 in its own coordinate; 1e-5 of the
    // weight allows for the lever arms that turn rotations into heights.
    Real mass = 0.0;
    for (int i = 1; i < model.nbodies(); ++i) mass += model.inertia[static_cast<std::size_t>(i)].mass;
    Real Fz = 0.0;
    for (int c = 0; c < 4; ++c) Fz += vh.tire(c)->get_vertical_force();
    INFO("tyre loads " << Fz << " N, weight " << mass * g_accel << " N");
    CHECK(std::abs(Fz - mass * g_accel) < 1e-5 * mass * g_accel);

    // At rest there, it stays there: a second of simulation moves the chassis
    // by less than a micrometre. With accelerations below 1e-6, a mode of
    // frequency w moves by at most 2e-6 / w^2, and the slowest supported
    // mode, the body's heave at about 1 Hz, gives 5e-8 m; along the neutral
    // directions there is no force. The same second from the builder's
    // starting state is printed for comparison.
    kernel::Simulator from_start(sys);
    from_start.q = q_start;
    from_start.initialize();
    const Real drift_start = chassis_drift(from_start, vh, 1.0);
    const Real drift = chassis_drift(sim, vh, 1.0);
    INFO("chassis drift over 1 s: from the start " << drift_start << " m, from equilibrium " << drift << " m");
    CHECK(drift < 1e-6);
}

TEST_CASE("Statics: the sedan started 1 cm too high settles to the same equilibrium", "[vehicle][statics]")
{
    // 1 cm keeps every tyre loaded (their static deflection is about 1.9 cm).
    // From higher, with the wheels in the air, this template has no rebound
    // stops and its tyre's contact point is fixed on the upright, so the
    // hanging rear suspension can fold over and the "contact point" leave the
    // ground: an equilibrium of the model, not of a car (journal: static
    // equilibrium; the builders' task 7.x).
    kernel::System sys;
    auto vh = build_vehicle(sys, sedan());
    const kernel::Model& model = sys.model;
    const int cq = model.idx_q[static_cast<std::size_t>(vh.chassis_body)];

    kernel::Simulator reference(sys);
    set_vehicle_equilibrium(reference, vh);
    reference.initialize();
    REQUIRE(kernel::static_equilibrium(reference).converged);

    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.q(cq + 1) += 0.01;   // Y, up
    sim.initialize();
    const kernel::StaticReport r = kernel::static_equilibrium(sim);
    INFO(r.summary(model));
    CHECK(r.converged);
    CHECK(r.residual < 1e-6);
    CHECK(std::abs(sim.q(cq + 1) - reference.q(cq + 1)) < 1e-6);
}
