#pragma once

// Core types, units and error utilities for the multibody solver.
//
// Logging lives in mbd/core/logging.hpp, so that this header (included by
// everything) does not pull in the logging library.

#include <cassert>
#include <cstdio>
#include <stdexcept>
#include <string>

#include "mbd/core/math.hpp"

namespace mbd {

//------------------------------------------------------------------------------
// Indices / identifiers
//------------------------------------------------------------------------------

// We start simple: just integer indices. Later we can wrap them in strong types.
using BodyIndex       = int;
using JointIndex      = int;
using ConstraintIndex = int;

/// Sentinel value for "no parent" (used by the ground body).
inline constexpr BodyIndex kNoParent = -1;

/// The ground body always occupies index 0 in a kernel::Model.
inline constexpr BodyIndex kGroundIndex = 0;
//------------------------------------------------------------------------------
// Units and constants (global conventions)
//------------------------------------------------------------------------------
//
// All physical quantities in the solver are in SI units:
// - Length: meters
// - Mass:   kilograms
// - Time:   seconds
// - Angles: radians
//
// These helpers just make angle conversions explicit.

inline constexpr Real pi = 3.14159265358979323846;

inline Real deg2rad(Real deg)
{
    return deg * pi / Real(180.0);
}

inline Real rad2deg(Real rad)
{
    return rad * Real(180.0) / pi;
}

//------------------------------------------------------------------------------
// Error type
//------------------------------------------------------------------------------

struct MbdError : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

//------------------------------------------------------------------------------
// Diagnostics
//------------------------------------------------------------------------------
//
// Warnings raised by the solver (a redundant constraint set, a projection that
// did not converge) go through one replaceable sink. The default writes to
// stderr; init_logging() in mbd/core/logging.hpp redirects it to the logger.

using DiagnosticSink = void (*)(const std::string& message);

inline DiagnosticSink& diagnostic_sink()
{
    static DiagnosticSink sink = [](const std::string& message) {
        std::fprintf(stderr, "[mbd warning] %s\n", message.c_str());
    };
    return sink;
}

inline void report_warning(const std::string& message)
{
    diagnostic_sink()(message);
}

//------------------------------------------------------------------------------
// Assertions and throwing helpers
//------------------------------------------------------------------------------

#ifndef NDEBUG
  #define MBD_ASSERT(expr) assert(expr)
#else
  #define MBD_ASSERT(expr) ((void)0)
#endif

#define MBD_THROW_IF(cond, msg) \
    do { if (cond) throw ::mbd::MbdError(msg); } while(false)

} // namespace mbd
