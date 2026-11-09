// The regression harness's files and comparison (plan task 5.5, D49).

#include "regression/regression.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace mbd_regression {

void Record::add(Real x, const std::vector<Real>& values)
{
    std::vector<Real> row{x};
    row.insert(row.end(), values.begin(), values.end());
    rows.push_back(std::move(row));
}

std::string reference_path(const Scenario& s)
{
    return (std::filesystem::path(__FILE__).parent_path() / "data" / (s.name + ".csv")).string();   // tests/regression/
}

namespace {

std::vector<std::string> split(const std::string& line)
{
    std::vector<std::string> out;
    std::stringstream in(line);
    std::string cell;
    while (std::getline(in, cell, ',')) out.push_back(cell);
    return out;
}

} // namespace

void write_reference(const Scenario& s, const Record& r)
{
    std::ofstream out(reference_path(s));
    if (!out) throw std::runtime_error("cannot write " + reference_path(s));
    out << "# Regression reference (plan task 5.5): " << s.name << "\n"
        << "# " << s.description << "\n"
        << "# Each channel is held to " << s.relative_tolerance
        << " times its largest magnitude here. Written by test_regression with\n"
        << "# MBD_REGRESSION_UPDATE=1; see tests/regression/README.md before writing it again.\n";
    out << r.abscissa;
    for (const auto& c : r.channels) out << ',' << c;
    out << '\n' << std::setprecision(17);
    for (const auto& row : r.rows) {
        for (std::size_t k = 0; k < row.size(); ++k) out << (k ? "," : "") << row[k];
        out << '\n';
    }
    if (!out) throw std::runtime_error("cannot write " + reference_path(s));
}

Record read_reference(const Scenario& s)
{
    const std::string path = reference_path(s);
    std::ifstream in(path);
    if (!in) throw std::runtime_error("no reference " + path + " (write it with MBD_REGRESSION_UPDATE=1)");
    Record r;
    std::string line;
    bool header = true;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        const auto cells = split(line);
        if (header) {
            if (cells.size() < 2) throw std::runtime_error(path + ": a header with no channels");
            r.abscissa = cells[0];
            r.channels.assign(cells.begin() + 1, cells.end());
            header = false;
            continue;
        }
        if (cells.size() != r.channels.size() + 1)
            throw std::runtime_error(path + ": a row of " + std::to_string(cells.size()) + " values, not "
                                     + std::to_string(r.channels.size() + 1));
        std::vector<Real> row;
        for (const auto& c : cells) row.push_back(std::stod(c));
        r.rows.push_back(std::move(row));
    }
    if (header || r.rows.empty()) throw std::runtime_error(path + ": no rows");
    return r;
}

bool Comparison::passed() const
{
    if (!mismatch.empty()) return false;
    return std::all_of(channels.begin(), channels.end(),
                       [](const Channel& c) { return c.difference <= c.tolerance; });
}

const Comparison::Channel* Comparison::worst() const
{
    const Channel* w = nullptr;
    for (const auto& c : channels)
        if (!w || c.ratio() > w->ratio()) w = &c;
    return w;
}

std::string Comparison::summary() const
{
    if (!mismatch.empty()) return mismatch;
    std::ostringstream o;
    o << std::setprecision(3);
    for (const auto& c : channels)
        o << "  " << c.name << ": " << c.difference << " (tolerance " << c.tolerance << ", " << c.ratio()
          << " of it)\n";
    return o.str();
}

Comparison compare(const Scenario& s, const Record& result, const Record& reference)
{
    Comparison out;
    if (result.abscissa != reference.abscissa || result.channels != reference.channels) {
        out.mismatch = "the channels differ from the reference's";
        return out;
    }
    if (result.rows.size() != reference.rows.size()) {
        out.mismatch = std::to_string(result.rows.size()) + " rows against the reference's "
                       + std::to_string(reference.rows.size());
        return out;
    }
    for (std::size_t k = 0; k < result.rows.size(); ++k) {
        const Real x = result.rows[k][0], x_ref = reference.rows[k][0];
        if (std::abs(x - x_ref) > 1e-9 * (1.0 + std::abs(x_ref))) {
            out.mismatch = "row " + std::to_string(k) + " is at " + std::to_string(x) + ", the reference's at "
                           + std::to_string(x_ref);
            return out;
        }
    }
    for (std::size_t c = 0; c < reference.channels.size(); ++c) {
        Comparison::Channel ch;
        ch.name = reference.channels[c];
        Real scale = 0.0;
        for (const auto& row : reference.rows) scale = std::max(scale, std::abs(row[c + 1]));
        ch.tolerance = s.relative_tolerance * (scale > 0.0 ? scale : 1.0);
        for (std::size_t k = 0; k < reference.rows.size(); ++k) {
            const Real d = std::abs(result.rows[k][c + 1] - reference.rows[k][c + 1]);
            ch.difference = std::isnan(d) ? 1e300 : std::max(ch.difference, d);
        }
        out.channels.push_back(ch);
    }
    return out;
}

} // namespace mbd_regression
