// Debugging aids (documentation track H, task H.4).
//
//   - describe() gives the counts, the bodies with their joints, the
//     constraints and force elements with their bodies;
//   - dump_state() gives the state and the constraint residuals;
//   - a model with two drivers fighting over one coordinate is diagnosed
//     from its step trace alone, read back from the CSV file, and told apart
//     from the same model with drivers that agree (harmless redundancy).

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/describe.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using mbd_test::FourBar;

namespace {

/// A CSV file with a header row, as columns by name.
std::map<std::string, std::vector<Real>> read_csv(const std::string& path)
{
    std::ifstream in(path);
    std::string line;
    std::getline(in, line);
    std::vector<std::string> names;
    {
        std::stringstream header(line);
        std::string name;
        while (std::getline(header, name, ',')) names.push_back(name);
    }
    std::map<std::string, std::vector<Real>> columns;
    while (std::getline(in, line)) {
        std::stringstream row(line);
        std::string cell;
        for (std::size_t k = 0; k < names.size() && std::getline(row, cell, ','); ++k) {
            columns[names[k]].push_back(std::stod(cell));
        }
    }
    return columns;
}

std::vector<std::string>& captured()
{
    static std::vector<std::string> messages;
    return messages;
}

/// A slider along Z, no gravity, driven to 0.1 by its joint driver and held
/// at `held` by a point constraint along Z as well.
System two_drivers(Real held)
{
    System sys;
    sys.model.gravity = Vec3::Zero();
    const int slider = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                          RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1)), "slider");
    sys.constraints.push_back(std::make_shared<JointDriver>(sys.model, slider, TimeFunction::constant(0.1)));
    sys.constraints.push_back(std::make_shared<Dot2>(Marker{0, Transform3()}, 2, Marker{slider, Transform3()},
                                                     TimeFunction::constant(held)));
    return sys;
}

/// Ten steps of the model with the trace on, written to CSV and read back.
std::map<std::string, std::vector<Real>> traced_run(System& sys, const char* file)
{
    Simulator sim(sys);
    sim.trace_steps = true;
    captured().clear();
    DiagnosticSink saved = diagnostic_sink();
    diagnostic_sink() = [](const std::string& m) { captured().push_back(m); };
    sim.run(0.01, 1e-3);
    diagnostic_sink() = saved;
    const std::string path = (std::filesystem::temp_directory_path() / file).string();
    sim.write_trace(path);
    auto columns = read_csv(path);
    std::filesystem::remove(path);
    return columns;
}

} // namespace

TEST_CASE("Debugging: describe() lists the counts, bodies, joints, constraints and forces",
          "[kernel][debugging]")
{
    FourBar fb;
    fb.sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, fb.coupler, Vec3(0.35, 0.9, 0.0), Vec3(0.4, 0.0, 0.0), 300.0, 0.0, 0.2));
    // At an assembled configuration (the neutral one, all links along X, is
    // neither assembled nor of full rank).
    const std::string s = describe(fb.sys, FourBar::configuration(1.0));
    INFO(s);
    CHECK_THAT(s, ContainsSubstring("3 bodies, 3 coordinates, 3 velocities; 1 constraints (5 equations, rank 2)"));
    CHECK_THAT(s, ContainsSubstring("degrees of freedom 1"));
    CHECK_THAT(s, ContainsSubstring("validate(): 0 errors, 0 warnings, 1 notes"));
    CHECK_THAT(s, ContainsSubstring("body 1 (crank): revolute joint to body 0 (ground); coordinates 0, velocities 0; mass 1 kg"));
    CHECK_THAT(s, ContainsSubstring("body 3 (rocker): revolute joint to body 2 (coupler)"));
    CHECK_THAT(s, ContainsSubstring("5 equations, on body 0 (ground), body 3 (rocker)\n"));   // each body once
    CHECK_THAT(s, ContainsSubstring("spring-damper: on body 0 (ground), body 2 (coupler)\n"));
    CHECK_THAT(s, ContainsSubstring("Joint forces: none"));
}

TEST_CASE("Debugging: dump_state() gives the state and the constraint residuals", "[kernel][debugging]")
{
    FourBar fb;
    Simulator sim(fb.sys);
    sim.q = FourBar::configuration(1.0);
    sim.refresh();
    const std::string s = dump_state(sim);
    INFO(s);
    CHECK_THAT(s, ContainsSubstring("State at t = 0 s"));
    CHECK_THAT(s, ContainsSubstring("body 2 (coupler): ("));
    CHECK_THAT(s, ContainsSubstring("Constraints: |phi| = "));
}

TEST_CASE("Debugging: two drivers fighting over a coordinate are diagnosed from the step trace alone",
          "[kernel][debugging]")
{
    // The broken model: driven to 0.1 and held at 0.3. Its trace says, in
    // every row: two equations of rank 1 (one is redundant), a projection
    // that never converges and uses all 20 iterations, and a residual of
    // exactly 0.2, the distance between the two targets: the projection
    // meets one equation and the other is off by the conflict. Nothing but
    // the CSV file is read.
    {
        System sys = two_drivers(0.3);
        const auto trace = traced_run(sys, "mbd_trace_conflict.csv");
        REQUIRE(trace.at("time").size() == 10);
        for (std::size_t k = 0; k < 10; ++k) {
            CHECK(trace.at("equations")[k] == 2.0);
            CHECK(trace.at("rank")[k] == 1.0);
            CHECK(trace.at("projection_converged")[k] == 0.0);
            CHECK(trace.at("projection_iterations")[k] == 20.0);
            CHECK(std::abs(trace.at("position_residual")[k] - 0.2) < 1e-12);
        }
        CHECK(captured().size() == 2);   // K050 (redundancy) and K051 (projection), once each
        // The projection's warning names the two equations that conflict
        // (task 5.2, D44): the driver's and the point constraint's.
        bool named = false;
        for (const auto& m : captured()) {
            if (m.find("MBD-K051") == std::string::npos) continue;
            INFO(m);
            CHECK(m.find("cannot all hold together") != std::string::npos);
            CHECK(m.find("constraint 0 (joint driver)") != std::string::npos);
            CHECK(m.find("constraint 1 (dot-2)") != std::string::npos);
            named = true;
        }
        CHECK(named);
    }
    // The same model with the targets agreeing: still two equations of rank
    // 1 (redundant, and harmless), but the projection converges and the
    // residual vanishes. The residual column tells the two apart.
    {
        System sys = two_drivers(0.1);
        const auto trace = traced_run(sys, "mbd_trace_agree.csv");
        REQUIRE(trace.at("time").size() == 10);
        for (std::size_t k = 0; k < 10; ++k) {
            CHECK(trace.at("rank")[k] == 1.0);
            CHECK(trace.at("projection_converged")[k] == 1.0);
            CHECK(trace.at("position_residual")[k] < 1e-12);
        }
    }
}
