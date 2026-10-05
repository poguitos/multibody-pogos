// Debugging aids (plan task H.4).
//
//   - describe() names every body, joint, constraint and element, and counts
//     the degrees of freedom;
//   - dump_state() gives the state and each constraint's residual;
//   - the trace reads back from CSV exactly, and its energy balance stays at
//     zero for a healthy run, damping included;
//   - broken models diagnosed from their traces alone (the plan's "done
//     when"): a loop that cannot close, a damper with the wrong sign, a step
//     too long for a stiff spring; a healthy run has nothing to report.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/diagnostics.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/four_bar.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using mbd_test::FourBar;

namespace {

bool has_code(const std::vector<std::string>& messages, const std::string& code)
{
    for (const auto& m : messages) {
        if (m.rfind(code, 0) == 0) return true;
    }
    return false;
}

/// Write the trace to a file, read it back, and diagnose what was read: the
/// diagnosis sees the file and nothing else.
TraceDiagnosis diagnose_from_file(const Simulator& sim, const std::string& name)
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / ("mbd_trace_" + name + ".csv");
    write_trace_csv(sim.trace_rows(), path.string());
    const std::vector<TraceRow> read = read_trace_csv(path.string());
    std::filesystem::remove(path);
    REQUIRE(read.size() == sim.trace_rows().size());
    return diagnose(read);
}

/// A pendulum (bob 0.5 m below a hinge about Z, gravity -Y) with a damper on
/// its hinge of coefficient c: negative c is a damper with the wrong sign.
System damped_pendulum(Real c)
{
    System sys;
    sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.02, 0.02, 0.02));
    I.com_B = Vec3(0.0, -0.5, 0.0);
    const int b = sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), I, "bob");
    JointCoordinateForceParams p;
    p.damper = Curve::linear(c);
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, b, p));
    return sys;
}

std::vector<std::string>& silenced()
{
    static std::vector<std::string> w;
    return w;
}

} // namespace

TEST_CASE("Diagnostics: describe names the parts and counts the freedom", "[kernel][diagnostics]")
{
    FourBar fb;
    System sys;
    sys.model = fb.model;
    sys.constraints = fb.closure;
    const std::string text = describe(sys, fb.q);
    INFO(text);
    CHECK_THAT(text, ContainsSubstring("3 bodies besides the ground, 3 coordinates, 3 velocities"));
    CHECK_THAT(text, ContainsSubstring("body 1 (crank): revolute joint on body 0 (ground); q 0, v 0; mass 1 kg"));
    CHECK_THAT(text, ContainsSubstring("body 3 (rocker): revolute joint on body 2 (coupler)"));
    CHECK_THAT(text, ContainsSubstring("constraint 0 (revolute closure), 5 equation(s), on body 0 (ground), body 3 (rocker)\n"));
    // Closed at the given configuration, the loop leaves one degree of freedom.
    CHECK_THAT(text, ContainsSubstring("3 velocities less 2 independent constraint equations = 1"));
}

TEST_CASE("Diagnostics: the state dump gives each body and each constraint's residual",
          "[kernel][diagnostics]")
{
    FourBar fb;
    System sys;
    sys.model = fb.model;
    sys.constraints = fb.closure;
    Simulator sim(sys);
    sim.q = fb.q;
    sim.initialize();
    const std::string text = dump_state(sim);
    INFO(text);
    CHECK_THAT(text, ContainsSubstring("time 0 s"));
    CHECK_THAT(text, ContainsSubstring("body 2 (coupler): q ("));
    CHECK_THAT(text, ContainsSubstring("constraint 0 (revolute closure): largest |phi|"));
    CHECK_THAT(text, ContainsSubstring("last projection:"));
}

TEST_CASE("Diagnostics: the trace reads back exactly and balances its energy", "[kernel][diagnostics]")
{
    // A damped pendulum loses energy to its damper; the damper's work is in
    // the trace's work column, so the balance kinetic + potential - work
    // stays at its first value, to RK4's accuracy: at 1 ms on a pendulum of
    // about 4.4 rad/s, an error of order (w dt)^4 = 4e-10 of the energy per
    // radian of swing, some 1e-8 J over 5 s; bound 1e-6 J.
    System sys = damped_pendulum(0.3);
    Simulator sim(sys);
    sim.q(0) = 0.5;
    sim.trace = true;
    sim.run(5.0, 1e-3);
    const std::vector<TraceRow>& rows = sim.trace_rows();
    REQUIRE(rows.size() == 5001);
    CHECK(rows.front().dt == 0.0);
    CHECK(rows.back().work < 0.0);   // the damper takes energy out
    Real worst = 0.0;
    for (const TraceRow& r : rows) worst = std::max(worst, std::abs(r.balance));
    CHECK(worst <= 1e-6);

    const std::filesystem::path path = std::filesystem::temp_directory_path() / "mbd_trace_roundtrip.csv";
    write_trace_csv(rows, path.string());
    const std::vector<TraceRow> read = read_trace_csv(path.string());
    std::filesystem::remove(path);
    REQUIRE(read.size() == rows.size());
    for (std::size_t k = 0; k < rows.size(); k += 997) {
        CHECK(read[k].time == rows[k].time);
        CHECK(read[k].kinetic == rows[k].kinetic);
        CHECK(read[k].work == rows[k].work);
        CHECK(read[k].balance == rows[k].balance);
    }
    CHECK(diagnose(read).clean());
    CHECK_THROWS_WITH(read_trace_csv(path.string()), ContainsSubstring("MBD-A030"));
}

TEST_CASE("Diagnostics: broken models are diagnosed from their traces alone", "[kernel][diagnostics]")
{
    const DiagnosticSink previous = diagnostic_sink();
    diagnostic_sink() = [](const std::string& m) { silenced().push_back(m); };

    SECTION("a loop that cannot close")
    {
        // The four-bar with its ground pivot 2 m away: the links reach 1.7 m.
        FourBar fb;
        System sys;
        sys.model = fb.model;
        sys.constraints.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(2.0, 0.0, 0.0))},
                                                   Marker{fb.rocker, Transform3::FromTranslation(Vec3(FourBar::c, 0.0, 0.0))}));
        Simulator sim(sys);
        sim.q = fb.q;
        sim.trace = true;
        sim.initialize();
        sim.run(0.5, 1e-3);
        const TraceDiagnosis d = diagnose_from_file(sim, "loop");
        INFO(d.summary());
        CHECK(has_code(d.warnings, "MBD-K100"));
    }

    SECTION("a damper with the wrong sign")
    {
        System sys = damped_pendulum(-0.3);
        Simulator sim(sys);
        sim.q(0) = 0.2;
        sim.trace = true;
        sim.run(5.0, 1e-3);
        const TraceDiagnosis d = diagnose_from_file(sim, "damper");
        INFO(d.summary());
        CHECK(has_code(d.notes, "MBD-K102"));
        CHECK_FALSE(has_code(d.warnings, "MBD-K101"));   // the integration is fine
    }

    SECTION("a step too long for a stiff spring")
    {
        // 1 kg on 1e6 N/m: w = 1000 rad/s, and RK4 at 2.5 ms (w dt = 2.5, just
        // inside its stability limit of 2.8) damps the motion away.
        System sys;
        sys.model.gravity = Vec3::Zero();
        const int b = sys.model.add_body(0, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                         RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05)));
        sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0, b, Vec3::Zero(), Vec3::Zero(), 1e6, 0.0, 0.5));
        Simulator sim(sys);
        sim.q(0) = 0.51;
        sim.trace = true;
        sim.run(0.5, 2.5e-3);
        const TraceDiagnosis d = diagnose_from_file(sim, "stiff");
        INFO(d.summary());
        CHECK(has_code(d.warnings, "MBD-K101"));

        // The same with a step that resolves it: nothing to report.
        System sys2 = sys;
        Simulator fine(sys2);
        fine.q(0) = 0.51;
        fine.trace = true;
        fine.run(0.05, 1e-5);
        CHECK(diagnose_from_file(fine, "stiff_fine").clean());
    }

    diagnostic_sink() = previous;
}
