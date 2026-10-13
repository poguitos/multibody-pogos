// Initial-condition assembly on a full vehicle (plan task 3.4): the
// double-wishbone sedan, its suspension coordinates disturbed, assembled with
// the chassis held.

#include <catch2/catch_test_macros.hpp>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

#include "support/rng.hpp"

using namespace mbd;

TEST_CASE("Assembly: the sedan's disturbed suspension is assembled with the chassis held",
          "[vehicle][assembly]")
{
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    kernel::System sys;
    auto vh = build_vehicle(sys, tmpl);
    const kernel::Model& model = sys.model;
    kernel::Simulator sim(sys);
    set_vehicle_equilibrium(sim, vh);
    sim.initialize();
    REQUIRE(sim.last_projection().converged);
    const VecX q_closed = sim.q;

    // Every coordinate but the chassis's moved by up to 0.02 (m or rad), and
    // velocities that ignore the linkages; the chassis drives forward
    // (body X) at 20 m/s.
    const int chassis = vh.chassis_body;
    const int cq = model.idx_q[chassis], cv = model.idx_v[chassis];
    auto is_chassis_velocity = [&](int k) { return k >= cv && k < cv + model.nvs[chassis]; };
    mbd_test::Rng rng(71);
    VecX dq = VecX::Zero(model.nv);
    sim.v.setZero();
    for (int k = 0; k < model.nv; ++k) {
        if (is_chassis_velocity(k)) continue;
        dq(k) = rng.range(-0.02, 0.02);
        sim.v(k) = rng.range(-0.5, 0.5);
    }
    sim.v(cv + 3) = 20.0;   // free joint: angular velocity, then linear
    kernel::integrate(model, q_closed, dq, 1.0, sim.q);
    const VecX given = sim.q, v_given = sim.v;

    kernel::AssemblyOptions options;
    options.hold_joint(model, chassis);
    const kernel::AssemblyReport r = kernel::assemble(sim, options);
    INFO(r.summary(model));
    CHECK(r.ok());
    CHECK(r.smallest_correction);
    CHECK(r.warnings.empty());
    CHECK(r.position_residual_given > 1e-3);
    CHECK(r.position_residual < 1e-10);
    CHECK(r.velocity_residual < 1e-10);
    CHECK(r.acceleration_residual < 1e-8);
    CHECK(r.degrees_of_freedom == 10);   // as validate() counts for the template vehicles

    // The chassis exactly as given.
    for (int i = cq; i < cq + model.nqs[chassis]; ++i) CHECK(sim.q(i) == given(i));
    for (int k = cv; k < cv + model.nvs[chassis]; ++k) CHECK(sim.v(k) == v_given(k));

    // No farther from the given positions than the configuration they were
    // disturbed from, which has the same chassis and closes every loop.
    kernel::Data data(model);
    const VecX w = kernel::crba(model, data, given).diagonal();
    auto distance2 = [&](const VecX& q) {
        VecX d;
        kernel::difference(model, given, q, d);
        return d.dot(w.asDiagonal() * d);
    };
    INFO("distance^2 from the given positions: assembled " << distance2(sim.q) << ", undisturbed "
         << distance2(q_closed) << "; iterations " << r.projection_iterations << " + "
         << r.refinement_iterations);
    CHECK(distance2(sim.q) <= distance2(q_closed));
}
