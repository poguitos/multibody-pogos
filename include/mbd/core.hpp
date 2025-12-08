#pragma once

// Core types, units, logging and error utilities for the multibody solver.

#include <stdexcept>
#include <string>
#include <memory>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "mbd/math.hpp"

namespace mbd {

//------------------------------------------------------------------------------
// Indices / identifiers
//------------------------------------------------------------------------------

// We start simple: just integer indices. Later we can wrap them in strong types.
using BodyIndex       = int;
using JointIndex      = int;
using ConstraintIndex = int;

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
// Logging utilities (thin wrapper around spdlog)
//------------------------------------------------------------------------------

enum class LogLevel {
    trace,
    debug,
    info,
    warn,
    err,
    critical,
    off
};

inline spdlog::level::level_enum to_spdlog_level(LogLevel lvl)
{
    using L = spdlog::level::level_enum;
    switch (lvl) {
        case LogLevel::trace:    return L::trace;
        case LogLevel::debug:    return L::debug;
        case LogLevel::info:     return L::info;
        case LogLevel::warn:     return L::warn;
        case LogLevel::err:      return L::err;
        case LogLevel::critical: return L::critical;
        case LogLevel::off:      return L::off;
    }
    return L::info;
}

// Get or create the project-wide logger named "mbd".
inline std::shared_ptr<spdlog::logger> get_logger()
{
    auto logger = spdlog::get("mbd");
    if (!logger) {
        logger = spdlog::stdout_color_mt("mbd");
    }
    return logger;
}

// Initialize logging: call once at program / test start.
inline void init_logging(LogLevel level = LogLevel::info)
{
    auto logger = get_logger();
    spdlog::set_default_logger(logger);
    spdlog::set_level(to_spdlog_level(level));
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");
}

//------------------------------------------------------------------------------
// Assertions and throwing helpers
//------------------------------------------------------------------------------

#ifndef NDEBUG
  #include <cassert>
  #define MBD_ASSERT(expr) assert(expr)
#else
  #define MBD_ASSERT(expr) ((void)0)
#endif

#define MBD_THROW_IF(cond, msg) \
    do { if (cond) throw ::mbd::MbdError(msg); } while(false)

} // namespace mbd
