// The how-to guides quote their examples from the tests that run them (task
// H.6): every code block marked <!-- example: name --> in docs/help/howto/
// must equal, line for line once a common indentation is removed, the code
// between "// [example: name]" and "// [end example]" in tests/howto/, and
// every example in the tests must be quoted somewhere. A guide whose code is
// changed without its test, or a test changed without its guide, fails here.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// The repository root, from this file's path: tests/howto/<this file>.
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

/// The lines with their common indentation removed, and trailing blanks.
std::vector<std::string> dedent(std::vector<std::string> lines)
{
    std::size_t indent = std::string::npos;
    for (const std::string& l : lines) {
        const std::size_t first = l.find_first_not_of(' ');
        if (first != std::string::npos) indent = std::min(indent, first);
    }
    if (indent == std::string::npos) indent = 0;   // all blank
    for (std::string& l : lines) {
        l = l.size() >= indent ? l.substr(indent) : std::string();
        while (!l.empty() && l.back() == ' ') l.pop_back();
    }
    return lines;
}

std::vector<fs::path> files_in(const fs::path& dir, const std::string& extension)
{
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == extension) files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

std::string joined(const std::vector<std::string>& lines)
{
    std::ostringstream o;
    for (const std::string& l : lines) o << l << '\n';
    return o.str();
}

} // namespace

TEST_CASE("How-to guides: every example is quoted exactly from the test that runs it", "[howto]")
{
    const fs::path root = repository_root();
    const std::regex begin(R"(^\s*// \[example: ([a-z0-9-]+)\]\s*$)");
    const std::regex end(R"(^\s*// \[end example\]\s*$)");
    const std::regex marker(R"(^<!-- example: ([a-z0-9-]+) -->\s*$)");

    // The examples in the tests.
    std::map<std::string, std::vector<std::string>> examples;
    for (const fs::path& file : files_in(root / "tests" / "howto", ".cpp")) {
        const auto lines = read_lines(file);
        for (std::size_t i = 0; i < lines.size(); ++i) {
            std::smatch m;
            if (!std::regex_match(lines[i], m, begin)) continue;
            std::vector<std::string> body;
            std::size_t j = i + 1;
            while (j < lines.size() && !std::regex_match(lines[j], end)) body.push_back(lines[j++]);
            INFO(file.filename().string() << ": example " << m[1]);
            REQUIRE(j < lines.size());   // closed
            CHECK(examples.count(m[1]) == 0);   // names unique
            examples[m[1]] = dedent(body);
        }
    }
    REQUIRE(!examples.empty());

    // The quotations in the guides.
    std::set<std::string> quoted;
    const auto pages = files_in(root / "docs" / "help" / "howto", ".md");
    REQUIRE(!pages.empty());
    for (const fs::path& page : pages) {
        const auto lines = read_lines(page);
        for (std::size_t i = 0; i < lines.size(); ++i) {
            std::smatch m;
            if (!std::regex_match(lines[i], m, marker)) continue;
            const std::string name = m[1];
            INFO(page.filename().string() << ", line " << i + 1 << ": example " << name);
            REQUIRE(i + 1 < lines.size());
            REQUIRE(lines[i + 1] == "```cpp");
            std::vector<std::string> block;
            std::size_t j = i + 2;
            while (j < lines.size() && lines[j] != "```") block.push_back(lines[j++]);
            REQUIRE(j < lines.size());
            REQUIRE(examples.count(name) == 1);
            const auto expected = examples.at(name);
            INFO("the test's code:\n" << joined(expected) << "the guide's:\n" << joined(dedent(block)));
            CHECK(dedent(block) == expected);
            quoted.insert(name);
        }
    }
    for (const auto& [name, code] : examples) {
        INFO("example " << name << " is in the tests but no guide quotes it");
        CHECK(quoted.count(name) == 1);
    }
}
