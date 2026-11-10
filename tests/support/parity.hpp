#pragma once

// The numbers a C++ test shares with its reproduction through Python (plan
// task 6.3, D55). When the environment variable MBD_PARITY_FILE names a file,
// parity(name, value) appends a line "name value" for each number, the value
// in hexadecimal floating point so that it reads back exactly; a vector gives
// one line per entry (a matrix column by column), a complex number the lines
// "name.re" and "name.im". Without the variable it does nothing, so the test
// runs as before. python/tests/test_parity.py runs one test case with the
// variable set, keeps the same names in the same order from Python, and
// requires every number to be identical.

#include <complex>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <type_traits>

#include <Eigen/Core>

namespace mbd_test {

namespace parity_detail {

/// The file named by MBD_PARITY_FILE, or "" (std::getenv, which MSVC deprecates).
inline std::string file()
{
#ifdef _MSC_VER
    char* value = nullptr;
    std::size_t size = 0;
    if (_dupenv_s(&value, &size, "MBD_PARITY_FILE") != 0 || value == nullptr) return {};
    std::string out(value);
    std::free(value);
    return out;
#else
    const char* value = std::getenv("MBD_PARITY_FILE");
    return value ? std::string(value) : std::string();
#endif
}

inline void write(const std::string& name, double value)
{
    const std::string path = file();
    if (path.empty()) return;
    char text[64];
    std::snprintf(text, sizeof text, "%a", value);
    std::ofstream(path, std::ios::app) << name << ' ' << text << '\n';
}

} // namespace parity_detail

/// A number: a real, an integer or a flag (as 0 or 1).
template <class T>
    requires std::is_arithmetic_v<T>
void parity(const std::string& name, T value)
{
    parity_detail::write(name, static_cast<double>(value));
}

inline void parity(const std::string& name, const std::complex<double>& value)
{
    parity_detail::write(name + ".re", value.real());
    parity_detail::write(name + ".im", value.imag());
}

/// Every entry of a vector or matrix, column by column.
template <class Derived>
void parity(const std::string& name, const Eigen::DenseBase<Derived>& values)
{
    const typename Derived::PlainObject plain = values.derived();
    for (Eigen::Index j = 0; j < plain.cols(); ++j) {
        for (Eigen::Index i = 0; i < plain.rows(); ++i) parity(name, plain(i, j));
    }
}

} // namespace mbd_test
