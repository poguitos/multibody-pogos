#include "mbd/io/tir_file.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>

namespace mbd::io {

namespace {

std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

std::string upper(std::string s)
{
    for (char& c : s) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

/// The line without its comment: '$' outside quotes begins one.
std::string strip_comment(const std::string& line)
{
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        if (line[i] == '\'' || line[i] == '"') quoted = !quoted;
        if (line[i] == '$' && !quoted) return line.substr(0, i);
    }
    return line;
}

/// A string value without its quotes, lower case.
std::string unquote(const std::string& value)
{
    std::string v = trim(value);
    if (v.size() >= 2 && (v.front() == '\'' || v.front() == '"')) v = v.substr(1, v.size() - 2);
    for (char& c : v) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return trim(v);
}

using Member = Real MagicFormulaParameters::*;

const std::map<std::string, Member>& members()
{
    static const std::map<std::string, Member> table = {
#define MBD_MF_ENTRY(section, name, value) {#name, &MagicFormulaParameters::name},
        MBD_MAGIC_FORMULA_PARAMETERS(MBD_MF_ENTRY)
#undef MBD_MF_ENTRY
    };
    return table;
}

} // namespace

TirFile parse_tir(const std::string& text, const std::string& name)
{
    TirFile file;
    MagicFormulaParameters& p = file.parameters;
    std::set<std::string> found;
    std::string section;
    std::istringstream lines(text);
    std::string raw;
    int line_number = 0;
    bool fittyp = false;
    std::optional<MagicFormulaVersion> format_version;
    while (std::getline(lines, raw)) {
        ++line_number;
        const std::string where = name + ", line " + std::to_string(line_number);
        if (!trim(raw).empty() && trim(raw).front() == '!') continue;
        const std::string line = trim(strip_comment(raw));
        if (line.empty()) continue;
        if (line.front() == '[') {
            const auto close = line.find(']');
            MBD_THROW_IF(close == std::string::npos, "MBD-I024: " + where + ": a section header without ']'");
            section = upper(trim(line.substr(1, close - 1)));
            continue;
        }
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;   // a table's row
        const std::string key = upper(trim(line.substr(0, eq)));
        const std::string value = trim(line.substr(eq + 1));

        if (section == "UNITS") {
            static const std::map<std::string, std::set<std::string>> si = {
                {"LENGTH", {"meter", "metre", "m"}},
                {"FORCE", {"newton", "n"}},
                {"ANGLE", {"radians", "radian", "rad"}},
                {"MASS", {"kg", "kilogram"}},
                {"TIME", {"second", "sec", "s"}},
                {"PRESSURE", {"pascal", "pa"}},
            };
            const auto it = si.find(key);
            const std::string unit = unquote(value);
            MBD_THROW_IF(it != si.end() && it->second.count(unit) == 0,
                         "MBD-I021: " + where + ": " + key + " is in '" + unit
                             + "'; only SI units are read (meter, newton, radians, kg, second, pascal)");
            continue;
        }

        if (key == "FITTYP") {
            const int f = std::atoi(value.c_str());
            if (f == 6 || f == 21) {
                p.version = MagicFormulaVersion::MF52;
            } else if (f == 61 || f == 62) {
                p.version = MagicFormulaVersion::MF61;
            } else {
                MBD_THROW_IF(true, "MBD-I022: " + where + ": FITTYP " + value
                          + " is not a Magic Formula this tyre evaluates (6 or 21: MF 5.2; 61 or 62: MF 6.1)");
            }
            p.FITTYP = f;
            fittyp = true;
            continue;
        }

        if (key == "TYRESIDE") {
            // The side the parameters describe; a tyre on the other is mirrored.
            const std::string side = unquote(value);
            MBD_THROW_IF(side != "left" && side != "right",
                         "MBD-I024: " + where + ": TYRESIDE = '" + value + "' is not 'LEFT' or 'RIGHT'");
            p.side = side == "right" ? TyreSide::Right : TyreSide::Left;
            continue;
        }

        if (key == "PROPERTY_FILE_FORMAT") {
            // Adams's name for the formulas, in files without FITTYP.
            const std::string format = unquote(value);
            if (format == "pac2002" || format == "mf_05") {
                format_version = MagicFormulaVersion::MF52;
            } else if (format == "mf_61" || format == "mf_62") {
                format_version = MagicFormulaVersion::MF61;
            }
            file.ignored.push_back(section + "." + key);
            continue;
        }

        const auto m = members().find(key);
        if (m == members().end()) {
            file.ignored.push_back(section + "." + key);
            continue;
        }
        char* end = nullptr;
        const std::string number = trim(value);
        const double v = std::strtod(number.c_str(), &end);
        MBD_THROW_IF(end == number.c_str() || !trim(std::string(end)).empty() || !std::isfinite(v),
                     "MBD-I024: " + where + ": " + key + " = '" + value + "' is not a number");
        p.*(m->second) = v;
        found.insert(key);
    }
    for (const char* required : {"FNOMIN", "UNLOADED_RADIUS"}) {
        MBD_THROW_IF(found.count(required) == 0, "MBD-I023: " + name + ": " + required + " is missing");
    }
    if (!fittyp && format_version) {
        p.version = *format_version;
        p.FITTYP = *format_version == MagicFormulaVersion::MF52 ? 6 : 61;
        fittyp = true;
    }
    MBD_THROW_IF(!fittyp, "MBD-I023: " + name + ": FITTYP is missing (nor is PROPERTY_FILE_FORMAT 'PAC2002', "
                                                "'MF_05', 'MF_61' or 'MF_62')");
    return file;
}

TirFile read_tir(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    MBD_THROW_IF(!in, "MBD-I020: read_tir: cannot open \"" + path + "\"");
    std::ostringstream text;
    text << in.rdbuf();
    return parse_tir(text.str(), path);
}

} // namespace mbd::io
