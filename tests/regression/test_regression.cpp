// The regression harness (plan task 5.5, D49): each scenario of
// regression/scenarios.cpp against its stored reference
// (regression/data/<name>.csv), every channel within its tolerance.
//
//   - The fast scenarios run on every commit (CI, every test executable);
//     the long ones are hidden ([long]) and run nightly.
//   - A change of physics by one part in a million must fail the
//     comparison: the harness catches what it is for.
//   - The hidden [sensitivity] case measures, for every scenario, how far a
//     relative change of 1e-12 and of 1e-6 in one physical parameter moves
//     its channels, against the tolerance: the evidence for where the
//     tolerance sits (README.md).
//
// Environment: MBD_REGRESSION_UPDATE=1 writes the references instead of
// comparing (README.md says when that is right); MBD_REGRESSION_REPORT=<file>
// appends each scenario's worst channel to <file> (CI turns those lines into
// annotations, readable without logging in).

#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>
#include <string>

#include "regression/regression.hpp"

using namespace mbd_regression;

namespace {

/// An environment variable, or "" (std::getenv, which MSVC deprecates).
std::string env(const char* name)
{
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, name) != 0 || value == nullptr) return {};
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv(name);
    return value ? std::string(value) : std::string();
#endif
}

const Scenario& scenario(const std::string& name)
{
    for (const auto& s : scenarios())
        if (s.name == name) return s;
    throw std::runtime_error("no scenario " + name);
}

void check(const std::string& name)
{
    const Scenario& s = scenario(name);
    const Record result = s.run(1.0);
    if (env("MBD_REGRESSION_UPDATE") == "1") {
        write_reference(s, result);
        WARN("wrote " << reference_path(s));
        return;
    }
    const Record reference = read_reference(s);
    const Comparison c = compare(s, result, reference);
    const std::string report = env("MBD_REGRESSION_REPORT");
    if (!report.empty()) {
        std::ofstream out(report, std::ios::app);
        if (const auto* w = c.worst()) {
            out << s.name << ',' << w->name << ',' << w->ratio() << ',' << w->difference << ',' << w->tolerance << '\n';
        } else {
            out << s.name << ',' << c.mismatch << ",,,\n";
        }
    }
    INFO(s.name << ": " << s.description << "\n" << c.summary());
    CHECK(c.passed());
}

} // namespace

TEST_CASE("Regression: the four-bar", "[regression]") { check("four-bar"); }
TEST_CASE("Regression: the heavy top", "[regression]") { check("heavy-top"); }
TEST_CASE("Regression: the bushed pendulum", "[regression]") { check("bushed-pendulum"); }
TEST_CASE("Regression: the HMMWV with joints", "[regression]") { check("hmmwv-joints"); }
TEST_CASE("Regression: the HMMWV with bushings", "[regression]") { check("hmmwv-bushings"); }
TEST_CASE("Regression: the sedan driving", "[regression]") { check("sedan-drive"); }
TEST_CASE("Regression: the sedan's modes at rest", "[regression]") { check("sedan-modes"); }
TEST_CASE("Regression: a double wishbone's kinematics", "[regression]") { check("wishbone-sweep"); }
TEST_CASE("Regression: a lap of an oval", "[regression]") { check("lap-oval"); }

TEST_CASE("Regression: the HMMWV with joints for 5 s", "[.][regression][long]") { check("hmmwv-joints-5s"); }
TEST_CASE("Regression: the HMMWV with bushings for 5 s", "[.][regression][long]") { check("hmmwv-bushings-5s"); }
TEST_CASE("Regression: the sedan's slalom", "[.][regression][long]") { check("sedan-slalom"); }

TEST_CASE("Regression: a change of physics by one part in a million is caught", "[regression]")
{
    // Gravity (or, for the analyses, the friction and a hardpoint) one part
    // in a million different: every channel's response is about 1e-6 of its
    // magnitude or more, a hundred times the tolerance of 1e-8 (the
    // sensitivity probe below measures it for every scenario). Each must
    // fail its comparison, and by a margin.
    for (const char* name : {"four-bar", "hmmwv-joints", "sedan-drive", "wishbone-sweep", "lap-oval"}) {
        const Scenario& s = scenario(name);
        const Comparison c = compare(s, s.run(1.0 + 1e-6), read_reference(s));
        INFO(s.name << ", " << s.perturbed << " changed by 1e-6:\n" << c.summary());
        REQUIRE(c.worst() != nullptr);
        CHECK_FALSE(c.passed());
        CHECK(c.worst()->ratio() > 10.0);
    }
}

TEST_CASE("Regression: how far small changes move each scenario", "[.][regression][sensitivity]")
{
    // For every scenario, the largest change of any channel over its
    // tolerance when one physical parameter changes by 1e-12 (about what
    // rounding differences between compilers do: each operation's rounding
    // is 1e-16, and many add up) and by 1e-6 (the smallest physics change the
    // harness is meant to catch). Run by hand with -s: the table is the
    // evidence for the tolerance (README.md).
    std::printf("%-20s %-36s %12s %12s\n", "scenario", "perturbed", "1e-12", "1e-6");
    for (const auto& s : scenarios()) {
        const Record base = s.run(1.0);
        const Comparison tiny = compare(s, s.run(1.0 + 1e-12), base);
        const Comparison small = compare(s, s.run(1.0 + 1e-6), base);
        const Real r12 = tiny.worst() ? tiny.worst()->ratio() : 0.0;
        const Real r6 = small.worst() ? small.worst()->ratio() : 0.0;
        std::printf("%-20s %-36s %12.3g %12.3g\n", s.name.c_str(), s.perturbed.c_str(), r12, r6);
        CHECK(r12 < 0.1);
        CHECK(r6 > 10.0);
    }
}
