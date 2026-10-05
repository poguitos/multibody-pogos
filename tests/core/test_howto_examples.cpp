// The how-to pages (task H.6): every code block a page quotes is the region
// of an example in tests/examples/ that is compiled and run, so the pages
// cannot go stale. A page marks its quote with
//     <!-- example: tests/examples/<file>.cpp#<name> -->
// followed by a ```cpp block; the source marks the region with two lines
// "// [<name>]". The block must equal the region, less its common
// indentation.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

fs::path repository_root()
{
    return fs::path(__FILE__).parent_path().parent_path().parent_path().lexically_normal();
}

std::vector<std::string> read_lines(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    return lines;
}

std::string trim(const std::string& s)
{
    const auto b = s.find_first_not_of(" \t");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t");
    return s.substr(b, e - b + 1);
}

// The lines between the two "// [name]" markers, less their common
// indentation; empty if the markers are not there exactly twice.
std::vector<std::string> region(const fs::path& source, const std::string& name, int& markers)
{
    const std::vector<std::string> lines = read_lines(source);
    std::vector<std::size_t> at;
    for (std::size_t k = 0; k < lines.size(); ++k) {
        if (trim(lines[k]) == "// [" + name + "]") at.push_back(k);
    }
    markers = static_cast<int>(at.size());
    if (at.size() != 2) return {};
    std::vector<std::string> body(lines.begin() + static_cast<std::ptrdiff_t>(at[0]) + 1,
                                  lines.begin() + static_cast<std::ptrdiff_t>(at[1]));
    std::size_t indent = std::string::npos;
    for (const auto& l : body) {
        if (trim(l).empty()) continue;
        indent = std::min(indent, l.find_first_not_of(' '));
    }
    for (auto& l : body) l = l.size() >= indent && indent != std::string::npos ? l.substr(indent) : trim(l);
    return body;
}

} // namespace

TEST_CASE("How-to pages: every quoted example is the code that runs", "[core][howto]")
{
    const fs::path root = repository_root();
    const fs::path pages = root / "docs" / "help" / "howto";
    REQUIRE(fs::exists(pages));

    const std::regex marker(R"(^<!-- example: (\S+?)#(\S+?) -->$)");
    int quotes = 0;
    for (const auto& entry : fs::directory_iterator(pages)) {
        if (entry.path().extension() != ".md" || entry.path().filename() == "README.md") continue;
        const std::vector<std::string> lines = read_lines(entry.path());
        int on_page = 0;
        for (std::size_t k = 0; k < lines.size(); ++k) {
            std::smatch m;
            if (!std::regex_match(lines[k], m, marker)) continue;
            ++on_page;
            ++quotes;
            const std::string where = entry.path().filename().string() + ", line " + std::to_string(k + 1);
            INFO(where << ": " << m[1] << "#" << m[2]);

            int markers = 0;
            const fs::path source = root / m[1].str();
            REQUIRE(fs::exists(source));
            const std::vector<std::string> expected = region(source, m[2].str(), markers);
            REQUIRE(markers == 2);

            // The block: the first ```cpp after the marker, up to its closing ```.
            std::size_t open = k + 1;
            while (open < lines.size() && trim(lines[open]).empty()) ++open;
            REQUIRE(open < lines.size());
            REQUIRE(lines[open] == "```cpp");
            std::vector<std::string> quoted;
            std::size_t close = open + 1;
            while (close < lines.size() && lines[close] != "```") quoted.push_back(lines[close++]);
            REQUIRE(close < lines.size());

            CHECK(quoted == expected);
        }
        INFO(entry.path().filename().string() << " has no example");
        CHECK(on_page > 0);
    }
    CHECK(quotes >= 9);
}
