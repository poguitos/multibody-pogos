// The Magic Formula tyre (plan task 7.4) against an independent
// implementation: MFeval 4.3.1 (Marco Furlan, BSD-3), whose forces and
// moments for the three test tyres are in tests/forces/data/mfeval/*.csv,
// written by a MATLAB script kept with MFeval outside the repository (each
// CSV's header says how; the journal entry of 2026-11-25 quotes the script).
// MFeval was run without limit checks, low-speed reductions, alpha* or turn
// slip: the steady-state formulas only, as here.
//
// The tolerances. The two codes evaluate the same formulas, so they differ
// by rounding (1e-16 relative per operation, some hundred of them: 1e-14),
// by the CSV's 12 significant digits (at most 5e-12 relative), and by one
// difference of form: MFeval multiplies the trail by cos(alpha') = Vcx / (Vc
// + 1e-6), here cos(alpha) = Vcx / Vc (D63), which differ by 1e-6 / Vc
// relative: 6e-8 at 16.7 m/s, 5e-7 at the sweep's slowest point, 2 m/s. That
// reaches Mz through t Fy, never more than Mz's largest magnitude in a file.
// So t and Mz are allowed 1e-6 of their largest magnitude in the file, twice
// the cos(alpha') difference at 2 m/s; every other channel 1e-10, twenty
// times the CSV's rounding. A formula read differently moves a channel by
// its own size: a camber term by 1e-3 to 1e-1 of it, a scaling factor by 5
// to 30 %. Seen on 25 November 2026: t 2.1e-7 and Mz 1.1e-7, both at 2 m/s;
// the others at most 3.5e-12.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "mbd/forces/magic_formula.hpp"
#include "mbd/io/tir_file.hpp"

using namespace mbd;

namespace
{
    std::filesystem::path data_dir() { return std::filesystem::path(__FILE__).parent_path() / "data"; }   // tests/forces/data

    /// A CSV of MFeval's: '#' lines are comments, the first other line names the columns.
    struct Table {
        std::vector<std::string> names;
        std::vector<std::vector<Real>> rows;
        std::size_t column(const std::string& name) const
        {
            const auto it = std::find(names.begin(), names.end(), name);
            REQUIRE(it != names.end());
            return static_cast<std::size_t>(it - names.begin());
        }
    };

    Table read_csv(const std::filesystem::path& path)
    {
        std::ifstream in(path);
        REQUIRE(in.good());
        Table t;
        std::string line;
        while (std::getline(in, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            std::string cell;
            if (t.names.empty()) {
                while (std::getline(ss, cell, ',')) t.names.push_back(cell);
                continue;
            }
            std::vector<Real> row;
            while (std::getline(ss, cell, ',')) row.push_back(std::stod(cell));
            REQUIRE(row.size() == t.names.size());
            t.rows.push_back(row);
        }
        return t;
    }

    /// Our value of a channel MFeval's CSV has.
    Real channel(const MagicFormulaOutput& o, const std::string& name)
    {
        static const std::map<std::string, Real MagicFormulaOutput::*> members{
            {"Fx", &MagicFormulaOutput::Fx},       {"Fy", &MagicFormulaOutput::Fy},
            {"Mx", &MagicFormulaOutput::Mx},       {"My", &MagicFormulaOutput::My},
            {"Mz", &MagicFormulaOutput::Mz},       {"t", &MagicFormulaOutput::trail},
            {"Mzr", &MagicFormulaOutput::Mzr},     {"mux", &MagicFormulaOutput::mux},
            {"muy", &MagicFormulaOutput::muy},     {"Kxk", &MagicFormulaOutput::Kxk},
            {"Kya", &MagicFormulaOutput::Kya},     {"sigma_kappa", &MagicFormulaOutput::sigma_kappa},
            {"sigma_alpha", &MagicFormulaOutput::sigma_alpha}};
        return o.*members.at(name);
    }

    /// Every point of a tyre's CSV, every channel within its tolerance of its
    /// largest magnitude in the file (the derivation is at the top).
    void compare(const std::string& tyre, MagicFormulaVersion version)
    {
        const MagicFormulaTyre ours(io::read_tir((data_dir() / (tyre + ".tir")).string()).parameters);
        REQUIRE(ours.parameters().version == version);
        const Table table = read_csv(data_dir() / "mfeval" / (tyre + ".csv"));
        REQUIRE(table.rows.size() > 500);
        const std::size_t iFz = table.column("Fz"), ik = table.column("kappa"), ia = table.column("alpha"),
                          ig = table.column("gamma"), iV = table.column("Vx"), ip = table.column("p");

        std::vector<MagicFormulaOutput> outputs;
        for (const auto& r : table.rows) {
            MagicFormulaInput in;
            in.Fz = r[iFz];
            in.kappa = r[ik];
            in.alpha = r[ia];
            in.gamma = r[ig];
            in.Vcx = r[iV];
            in.pressure = r[ip] > 0.0 ? r[ip] : -1.0;   // 0: the file's INFLPRES
            outputs.push_back(ours.evaluate(in));
        }

        for (std::size_t c = ip + 1; c < table.names.size(); ++c) {
            const std::string& name = table.names[c];
            const Real tolerance = name == "t" || name == "Mz" ? 1e-6 : 1e-10;
            Real scale = 0.0, worst = 0.0;
            std::size_t worst_row = 0;
            for (std::size_t i = 0; i < table.rows.size(); ++i) {
                scale = std::max(scale, std::abs(table.rows[i][c]));
                const Real d = std::abs(channel(outputs[i], name) - table.rows[i][c]);
                if (d > worst) {
                    worst = d;
                    worst_row = i;
                }
            }
            const auto& r = table.rows[worst_row];
            INFO(tyre << " " << name << ": largest difference " << worst << " of largest magnitude " << scale << " ("
                      << worst / scale << "), at Fz " << r[iFz] << ", kappa " << r[ik] << ", alpha " << r[ia]
                      << ", gamma " << r[ig] << ", Vx " << r[iV] << ": ours " << channel(outputs[worst_row], name)
                      << ", MFeval's " << r[c]);
            CHECK(scale > 0.0);
            CHECK(worst <= tolerance * scale);
        }
    }
}

TEST_CASE("Magic Formula: MF 6.1 matches MFeval, Appendix 3 tyre", "[tyre][mf][mfeval]")
{
    compare("pacejka_a3_205_60R15", MagicFormulaVersion::MF61);
}

TEST_CASE("Magic Formula: MF 6.1 matches MFeval, Appendix 3 tyre with scaling factors", "[tyre][mf][mfeval]")
{
    compare("pacejka_a3_205_60R15_scaled", MagicFormulaVersion::MF61);
}

TEST_CASE("Magic Formula: MF 5.2 matches MFeval, synthetic tyre with scaling factors", "[tyre][mf][mfeval]")
{
    compare("mf52_test_tyre", MagicFormulaVersion::MF52);
}
