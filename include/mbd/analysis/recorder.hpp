#pragma once

// A recorder with named channels (plan task 3.3): each channel is a function
// read at every sample, so it can record anything the program can compute (a
// body's position, a joint reaction, a spring's force, an energy).
//
//   Recorder rec;
//   rec.add("chassis_z", [&] { return sim.states()[chassis].p_WB.z(); });
//   for (...) { sim.step(dt); rec.sample(sim.time); }
//   rec.write_csv("run.csv");

#include <functional>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"

namespace mbd {

class Recorder {
public:
    using Channel = std::function<Real()>;

    /// Add a channel. Names are unique and not empty; channels are added
    /// before the first sample.
    void add(const std::string& name, Channel channel);

    /// Read every channel and store the values with the time.
    void sample(Real time);

    std::size_t samples() const { return times_.size(); }
    const std::vector<std::string>& names() const { return names_; }
    const std::vector<Real>& times() const { return times_; }

    /// The values of one channel, one per sample.
    const std::vector<Real>& column(const std::string& name) const;

    /// One row per sample: time, then the channels, with a header row of
    /// names, written with 17 significant digits so values read back exactly.
    void write_csv(const std::string& path) const;

private:
    std::vector<std::string> names_;
    std::vector<Channel> channels_;
    std::vector<Real> times_;
    std::vector<std::vector<Real>> columns_;
};

} // namespace mbd
