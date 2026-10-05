// The message catalogue (plan task H.2): every error and warning in the
// sources carries a code (MBD-<letter><three digits>), every code used has an
// entry in docs/help/messages.md, and every entry there is used.

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// The repository root, from this file's path: tests/core/<this file>.
fs::path repository_root()
{
    return fs::path(__FILE__).parent_path().parent_path().parent_path().lexically_normal();
}

std::string read_file(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    std::ostringstream text;
    text << in.rdbuf();
    return text.str();
}

std::vector<fs::path> source_files(const fs::path& root)
{
    std::vector<fs::path> files;
    for (const char* dir : {"include", "src"}) {
        for (const auto& entry : fs::recursive_directory_iterator(root / dir)) {
            const auto ext = entry.path().extension();
            if (entry.is_regular_file() && (ext == ".hpp" || ext == ".cpp")) {
                files.push_back(entry.path());
            }
        }
    }
    return files;
}

std::string join(const std::set<std::string>& items)
{
    std::string out;
    for (const auto& s : items) out += (out.empty() ? "" : ", ") + s;
    return out;
}

} // namespace

TEST_CASE("Messages: every code is documented and every documented code is used",
          "[core][messages]")
{
    const fs::path root = repository_root();
    const fs::path catalogue = root / "docs" / "help" / "messages.md";
    REQUIRE(fs::exists(catalogue));

    std::set<std::string> documented;
    {
        const std::string text = read_file(catalogue);
        const std::regex heading(R"(\n### (MBD-[A-Z][0-9]{3})\r?\n)");
        for (auto it = std::sregex_iterator(text.begin(), text.end(), heading);
             it != std::sregex_iterator(); ++it) {
            documented.insert((*it)[1]);
        }
    }

    std::set<std::string> used;
    const std::regex code(R"(MBD-[A-Z][0-9]{3})");
    for (const auto& file : source_files(root)) {
        const std::string text = read_file(file);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), code);
             it != std::sregex_iterator(); ++it) {
            used.insert(it->str());
        }
    }

    std::set<std::string> undocumented, unused;
    for (const auto& c : used) if (!documented.count(c)) undocumented.insert(c);
    for (const auto& c : documented) if (!used.count(c)) unused.insert(c);

    INFO("Used in the sources but missing from docs/help/messages.md: " << join(undocumented));
    CHECK(undocumented.empty());
    INFO("Documented but used nowhere: " << join(unused));
    CHECK(unused.empty());
    CHECK(used.size() > 50);   // the scan found the sources at all
}

TEST_CASE("Messages: every error and warning in the sources carries a code", "[core][messages]")
{
    // Each statement that throws, warns or adds to a validate() report must
    // contain a code before its terminating semicolon.
    const std::regex site(
        R"(MBD_THROW_IF\(|throw MbdError\(|report_warning\(|\.(errors|warnings|notes)\.push_back\()");
    std::vector<std::string> missing;
    int sites = 0;
    for (const auto& file : source_files(repository_root())) {
        const std::string text = read_file(file);
        for (auto it = std::sregex_iterator(text.begin(), text.end(), site);
             it != std::sregex_iterator(); ++it) {
            const auto start = static_cast<std::size_t>(it->position());
            const std::size_t line_start = text.rfind('\n', start) + 1;
            const std::string line_head = text.substr(line_start, start - line_start);
            // Comments, the macro's definition and report_warning's own
            // definition are not sites.
            if (line_head.find("//") != std::string::npos
                || line_head.find("#define") != std::string::npos
                || line_head.find("void ") != std::string::npos) {
                continue;
            }
            ++sites;
            const std::size_t end = text.find(';', start);
            if (text.substr(start, end - start).find("MBD-") == std::string::npos) {
                const auto line = 1 + std::count(text.begin(), text.begin() + static_cast<std::ptrdiff_t>(start), '\n');
                missing.push_back(file.filename().string() + ":" + std::to_string(line));
            }
        }
    }
    std::string list;
    for (const auto& m : missing) list += m + " ";
    INFO("Without a code: " << list);
    CHECK(missing.empty());
    CHECK(sites > 50);
}
