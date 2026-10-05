// How-to examples (task H.6): assembly, statics, linearisation, contact,
// events and the step trace. Quoted by the pages of docs/help/howto/.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <filesystem>
#include <memory>
#include <sstream>

#include "mbd/forces/force_element.hpp"
#include "mbd/forces/plane_contact.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/assembly.hpp"
#include "mbd/kernel/diagnostics.hpp"
#include "mbd/kernel/linearization.hpp"
#include "mbd/kernel/statics.hpp"
#include "mbd/kernel/validate.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

using namespace mbd;
using namespace mbd::kernel;

namespace {

/// A four-bar (crank 0.3, coupler 0.8, rocker 0.6, ground 0.7) under gravity -Y.
System four_bar()
{
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const auto hinge = std::make_shared<RevoluteJointModel>();
    auto link = [](Real length) {
        RigidBodyInertia I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5 * length, 0.02, 0.02));
        I.com_B = Vec3(0.5 * length, 0.0, 0.0);
        return I;
    };
    const int crank = sys.model.add_body(0, hinge, Transform3(), Transform3(), link(0.3), "crank");
    const int coupler = sys.model.add_body(crank, hinge, Transform3::FromTranslation(Vec3(0.3, 0, 0)), Transform3(), link(0.8), "coupler");
    const int rocker = sys.model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(0.8, 0, 0)), Transform3(), link(0.6), "rocker");
    sys.constraints.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(0.7, 0, 0))},
                                               Marker{rocker, Transform3::FromTranslation(Vec3(0.6, 0, 0))}));
    return sys;
}

} // namespace

TEST_CASE("Example: assemble a loop from approximate angles", "[examples]")
{
    System sys = four_bar();
    std::ostringstream out;
    // [assemble]
    Simulator sim(sys);
    sim.q << 1.0, -0.5, -2.0;          // the crank at 1 rad, the rest roughly
    sim.v << 2.0, 0.0, 0.0;            // the crank turning at 2 rad/s
    AssemblySpec spec;
    spec.hold(sys.model, 1);           // keep the crank's angle and rate as given
    const AssemblyReport report = assemble(sim, spec);
    out << report.summary(sys.model);  // residuals before and after, what moved most
    // [assemble]
    REQUIRE(report.ok());
    CHECK(sim.q(0) == 1.0);
    CHECK(sim.v(0) == 2.0);
    CHECK(report.position.residual <= 1e-10);
}

TEST_CASE("Example: settle a vehicle on its springs and tyres", "[examples]")
{
    std::ostringstream out;
    // [statics]
    auto tmpl = VehicleTemplate::DefaultSedan();
    tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
    tmpl.rear_axle.suspension_type = SuspensionType::DoubleWishbone;
    System sys;
    const VehicleHandle car = build_vehicle(sys, tmpl);
    Simulator sim(sys);
    set_vehicle_equilibrium(sim, car);     // an estimate of the resting state
    sim.initialize();
    const StaticsReport rest = static_equilibrium(sim);
    out << rest.summary(sys.model);        // iterations, accelerations, free directions
    // [statics]
    REQUIRE(rest.converged);
    CHECK(rest.acceleration <= 1e-6);
}

TEST_CASE("Example: natural frequencies and modes of a quarter car", "[examples]")
{
    std::ostringstream out;
    // [modes]
    // Wheel (40 kg) and body (250 kg) on vertical sliders: suspension spring and
    // damper between them, a tyre spring under the wheel.
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    const Transform3 up = Transform3::FromRotation(Mat3(Eigen::AngleAxisd(-0.5 * pi, Vec3::UnitX()).toRotationMatrix()));
    const auto slider = std::make_shared<PrismaticJointModel>();
    const int wheel = sys.model.add_body(0, slider, up, up, RigidBodyInertia::from_solid_box(40.0, Vec3(0.15, 0.15, 0.15)), "wheel");
    const int body = sys.model.add_body(0, slider, up, up, RigidBodyInertia::from_solid_box(250.0, Vec3(0.5, 0.2, 0.4)), "body");
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(wheel, body, Vec3::Zero(), Vec3::Zero(), 2e4, 1500.0, 0.3));
    sys.force_elements.push_back(std::make_shared<TireContactForce>(wheel, 0.35, 2e5, 0.0));

    Simulator sim(sys);
    sim.q << 0.34, 0.6;
    static_equilibrium(sim);                 // linearise about the resting state
    const Linearization lin = linearize(sim);
    for (const Mode& mode : lin.modes) {
        out << mode.natural_frequency_hz << " Hz, damping ratio " << mode.damping_ratio << '\n';
    }
    // [modes]
    REQUIRE(lin.modes.size() == 2);
    CHECK(std::abs(lin.undamped[0].frequency_hz - 1.3564) <= 1e-4);   // body bounce
    CHECK(std::abs(lin.undamped[1].frequency_hz - 11.811) <= 1e-3);   // wheel hop
}

TEST_CASE("Example: a box on the ground with friction", "[examples]")
{
    System sys;
    const int box = sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                       RigidBodyInertia::from_solid_box(5.0, Vec3(0.2, 0.15, 0.1)), "box");
    // [contact]
    PlaneContactParams p;
    p.stiffness = 1e5;          // [N/m] per point: a penalty spring
    p.damping = 700.0;          // [N s/m], reached 0.1 mm deep
    p.friction = 0.6;
    p.slip_speed = 1e-3;        // [m/s]: below it friction acts as a damper
    std::vector<ContactSphere> corners;
    for (const Real x : {-0.2, 0.2}) {
        for (const Real y : {-0.15, 0.15}) corners.push_back({Vec3(x, y, -0.1), 0.0});   // points: radius 0
    }
    sys.force_elements.push_back(std::make_shared<PlaneContact>(box, corners, p));   // against z = 0
    // [contact]
    Simulator sim(sys);
    sim.q(2) = 0.12;            // dropped from 2 cm
    sim.v(3) = 1.0;             // sliding at 1 m/s along its X
    sim.run(1.0, 2e-4);
    // mu g = 5.9 m/s^2 stops 1 m/s in 0.17 s: at rest after a second.
    CHECK(sim.v.norm() <= 0.01);
    CHECK(std::abs(sim.q(2) - (0.1 - 5.0 * g_accel / (4.0 * 1e5))) <= 1e-5);
}

TEST_CASE("Example: a ball bouncing on a floor, by events", "[examples]")
{
    System sys;
    sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia::from_solid_box(0.5, Vec3(0.05, 0.05, 0.05)), "ball");
    Simulator sim(sys);
    sim.q(0) = 1.1;                     // the centre 1 m above contact (radius 0.1)
    std::ostringstream out;
    // [events]
    Event impact;
    impact.name = "impact";
    impact.function = [](const Simulator& s) { return s.q(0) - 0.1; };   // height above contact
    impact.direction = -1;                                                // falling through zero only
    impact.action = [](Simulator& s) { s.v(0) = -0.8 * s.v(0); };         // rebound, restitution 0.8
    sim.events.push_back(impact);
    sim.run(2.0, 0.01);
    for (const EventRecord& e : sim.event_log()) {
        out << sim.events[e.event].name << " at " << e.time << " s\n";   // to 1e-10 s
    }
    // [events]
    REQUIRE(!sim.event_log().empty());
    CHECK(std::abs(sim.event_log()[0].time - std::sqrt(2.0 / g_accel)) <= 1e-9);
}

TEST_CASE("Example: trace a run and diagnose it", "[examples]")
{
    System sys = four_bar();
    std::ostringstream out;
    Simulator sim(sys);
    sim.q << 1.0, -0.5, -2.0;
    AssemblySpec spec;
    spec.hold(sys.model, 1);
    assemble(sim, spec);
    const std::string path = (std::filesystem::temp_directory_path() / "mbd_example_trace.csv").string();
    // [trace]
    out << validate(sys, sim.q).summary();        // anything wrong with the model?
    sim.trace = true;                             // a row per step
    sim.run(1.0, 1e-3);
    write_trace_csv(sim.trace_rows(), path);      // to look at, or to diagnose later
    const TraceDiagnosis d = diagnose(read_trace_csv(path));
    out << d.summary() << dump_state(sim);        // what went wrong, and where it ended
    // [trace]
    CHECK(d.warnings.empty());
    std::filesystem::remove(path);
}
