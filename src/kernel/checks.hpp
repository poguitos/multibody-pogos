#pragma once

// Argument checks of the kernel's entry points (plan task 2.9). They stay on
// in every build: without them a vector of the wrong size, or a Data made for
// another model, is read out of bounds in a release build. A check costs a
// comparison; its message is built only when it fails.

#include <cstddef>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd::kernel::checks {

[[noreturn]] void size_error(const char* function, const char* argument, Index size,
                             const char* dimension, int expected);
[[noreturn]] void data_error(const char* function, const Model& model, const Data& data);
[[noreturn]] void body_error(const char* function, int body, int bodies);

/// x must have `expected` entries, the model's `dimension` ("nq" or "nv").
inline void size(const char* function, const char* argument, const VecX& x,
                 const char* dimension, int expected)
{
    if (x.size() != expected) size_error(function, argument, x.size(), dimension, expected);
}

inline void q(const char* function, const Model& model, const VecX& x, const char* argument = "q")
{
    size(function, argument, x, "nq", model.nq);
}

inline void v(const char* function, const Model& model, const VecX& x, const char* argument = "v")
{
    size(function, argument, x, "nv", model.nv);
}

/// `data` must have been made for `model`.
inline void data(const char* function, const Model& model, const Data& d)
{
    if (d.oMi.size() != static_cast<std::size_t>(model.nbodies()) || d.zero_v.size() != model.nv) {
        data_error(function, model, d);
    }
}

/// Body `i` must exist among `bodies`.
inline void body(const char* function, int i, std::size_t bodies)
{
    if (i < 0 || static_cast<std::size_t>(i) >= bodies) body_error(function, i, static_cast<int>(bodies));
}

} // namespace mbd::kernel::checks
