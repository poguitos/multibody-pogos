// How-to examples (task H.6): loads and recorded channels. Quoted by
// docs/help/howto/outputs.md.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <memory>

#include "mbd/analysis/recorder.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/outputs.hpp"
#include "mbd/kernel/simulator.hpp"

using namespace mbd;
using namespace mbd::kernel;

TEST_CASE("Example: record a pendulum's angle, hinge force and energy", "[examples]")
{
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia bob = RigidBodyInertia::from_solid_box(1.0, Vec3(0.02, 0.02, 0.02));
    bob.com_B = Vec3(0.5, 0.0, 0.0);
    const int body = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), bob);
    Simulator sim(sys);
    const std::string path = (std::filesystem::temp_directory_path() / "mbd_example_pendulum.csv").string();
    // [record]
    Recorder rec;
    rec.add("angle", [&] { return sim.q(0); });
    rec.add("hinge_force", [&] {
        const Loads loads = compute_loads(sim);        // reactions, constraint forces, accelerations
        return loads.joint_reaction[static_cast<std::size_t>(body)].tail<3>().norm();
    });
    rec.add("energy", [&] {
        return kinetic_energy(sys.model, sim.data()) + potential_energy(sys.model, sim.data());
    });
    rec.sample(sim.time);
    for (int n = 0; n < 1000; ++n) {
        sim.step(1e-3);
        rec.sample(sim.time);
    }
    rec.write_csv(path);                               // time, then one column per channel
    // [record]
    REQUIRE(rec.samples() == 1001);
    CHECK(std::abs(rec.column("energy").back() - rec.column("energy").front()) <= 1e-8);
    CHECK(rec.column("hinge_force").front() > 0.0);
    std::filesystem::remove(path);
}
