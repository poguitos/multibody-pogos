#pragma once

// The regression harness (plan task 5.5, D49): scenarios that run the
// library deterministically and record named channels, and their reference
// results, stored in tests/regression/data/ with the tolerance each channel is
// held to. A reference is this library's own earlier result: the harness
// does not say the result is right (the other tests do), it says it has not
// changed. See tests/regression/README.md for when and how references are
// written again.

#include <functional>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"

namespace mbd_regression {

using mbd::Real;

/// Rows of an abscissa (time, travel, distance or an index) and the values of
/// named channels, each name with its unit.
struct Record {
    std::string abscissa;
    std::vector<std::string> channels;
    std::vector<std::vector<Real>> rows;   // abscissa, then one value per channel

    void add(Real x, const std::vector<Real>& values);
};

/// One scenario. run(scale) runs it with one physical parameter, named in
/// `perturbed`, multiplied by scale; the reference is run(1). Each channel is
/// held to relative_tolerance times its largest magnitude in the reference.
struct Scenario {
    std::string name;          // the reference file's stem
    std::string description;
    std::string perturbed;
    Real relative_tolerance{1e-8};
    bool long_run{false};      // the nightly suite's
    std::function<Record(Real scale)> run;
};

/// Every scenario, fast and long.
const std::vector<Scenario>& scenarios();

/// The reference file of a scenario: tests/regression/data/<name>.csv.
std::string reference_path(const Scenario& s);

/// Writes `r` as the reference of `s` (17 significant digits, a header
/// saying what it is and how it was made).
void write_reference(const Scenario& s, const Record& r);

/// Reads a reference; throws std::runtime_error naming the file if it is
/// missing or malformed.
Record read_reference(const Scenario& s);

/// A comparison of a result with its reference, channel by channel.
struct Comparison {
    struct Channel {
        std::string name;
        Real difference{0.0};   // the largest over the rows
        Real tolerance{0.0};
        Real ratio() const { return tolerance > 0.0 ? difference / tolerance : (difference > 0.0 ? 1e300 : 0.0); }
    };
    std::string mismatch;       // non-empty if the channels or rows do not match
    std::vector<Channel> channels;

    bool passed() const;
    const Channel* worst() const;
    std::string summary() const;
};

Comparison compare(const Scenario& s, const Record& result, const Record& reference);

} // namespace mbd_regression
