#include "mbd/analysis/recorder.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <utility>

namespace mbd {

void Recorder::add(const std::string& name, Channel channel)
{
    MBD_THROW_IF(name.empty() || !channel || std::find(names_.begin(), names_.end(), name) != names_.end(),
                 "MBD-A020: Recorder::add: channel \"" + name
                     + "\" needs a name not used before, and a function");
    MBD_THROW_IF(!times_.empty(),
                 "MBD-A021: Recorder::add: channel \"" + name
                     + "\" added after sampling began; add every channel first");
    names_.push_back(name);
    channels_.push_back(std::move(channel));
    columns_.emplace_back();
}

void Recorder::sample(Real time)
{
    times_.push_back(time);
    for (std::size_t k = 0; k < channels_.size(); ++k) columns_[k].push_back(channels_[k]());
}

const std::vector<Real>& Recorder::column(const std::string& name) const
{
    const auto it = std::find(names_.begin(), names_.end(), name);
    MBD_THROW_IF(it == names_.end(), "MBD-A022: Recorder::column: no channel named \"" + name + "\"");
    return columns_[static_cast<std::size_t>(it - names_.begin())];
}

void Recorder::write_csv(const std::string& path) const
{
    std::ofstream out(path);
    MBD_THROW_IF(!out, "MBD-A023: Recorder::write_csv: cannot open \"" + path + "\" for writing");
    out << "time";
    for (const auto& n : names_) out << ',' << n;
    out << '\n' << std::setprecision(17);
    for (std::size_t s = 0; s < times_.size(); ++s) {
        out << times_[s];
        for (const auto& c : columns_) out << ',' << c[s];
        out << '\n';
    }
    MBD_THROW_IF(!out, "MBD-A023: Recorder::write_csv: writing \"" + path + "\" failed");
}

} // namespace mbd
