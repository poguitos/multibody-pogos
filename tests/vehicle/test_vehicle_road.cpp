// A vehicle on a road (plan task 7.2, D61): the tyres take the ground from
// the road they are put on.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <memory>

#include "mbd/forces/road.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using namespace mbd;

TEST_CASE("Vehicle on a road: over a ramp, the sedan ends higher by the ramp's rise", "[vehicle][road]")
{
    // The sedan at 8 m/s, held there by a push along its forward axis, for
    // 4 s: once on a road 5 cm up a ramp 2 m long starting 3 m ahead of its
    // centre, once on the level. On a level road of any height the
    // equilibrium is the same configuration lifted by that height, and both
    // cars start from the same state; so once both axles are up (the rear
    // axle, 1.35 m behind the centre, is up after (5 + 1.35) / 8 = 0.8 s) and
    // the transient has died away, the chassis heights differ by the rise.
    // The body's motions decay as e^(-zeta w t) with zeta w = c / (2 m) per
    // corner, the slower the rear's 1800 / (2 * 350) = 2.6 s^-1, so 3.2 s
    // later e^(-8.2) = 3e-4 of a disturbance of centimetres is left, about
    // 1e-5 m: allowed 1e-4 m.
    auto final_height = [](std::shared_ptr<const Road> road) {
        const auto tmpl = VehicleTemplate::DefaultSedan();
        kernel::System sys;
        auto vh = build_vehicle(sys, tmpl);
        vh.set_road(road);
        kernel::Simulator sim(sys);
        sim.method = kernel::Integrator::RK4;
        set_vehicle_equilibrium(sim, vh);
        const int v_forward = sys.model.idx_v[vh.chassis_body] + 3;   // along the chassis's X
        sim.v(v_forward) = 8.0;
        sim.initialize();
        sim.force_callback = [&](kernel::Simulator& s, Real, VecX& tau) {
            const auto& c = s.states()[static_cast<std::size_t>(vh.chassis_body)];
            tau(v_forward) += 500.0 * (8.0 - c.v_WB.dot(c.q_WB * Vec3::UnitX()));
        };
        sim.run(4.0, 1e-3);
        return sim.states()[static_cast<std::size_t>(vh.chassis_body)].p_WB.z();
    };
    const Real level = final_height(std::make_shared<FlatRoad>());
    const Real up = final_height(std::make_shared<RampRoad>(3.0, 2.0, 0.05));
    INFO("chassis height " << level << " m on the level, " << up << " m after the ramp");
    CHECK(std::abs(up - level - 0.05) < 1e-4);
}

TEST_CASE("Vehicle on a road: a null road is refused (MBD-F051)", "[vehicle][road]")
{
    kernel::System sys;
    auto vh = build_vehicle(sys, VehicleTemplate::DefaultSedan());
    try {
        vh.set_road(nullptr);
        FAIL("no error");
    } catch (const MbdError& e) {
        CHECK(std::string(e.what()).rfind("MBD-F051", 0) == 0);
    }
}
