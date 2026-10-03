#pragma once

// Logging utilities (thin wrapper around spdlog).
//
// Kept apart from mbd/core/core.hpp: only code that actually logs should pay
// for including the logging library.

#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include "mbd/core/core.hpp"

namespace mbd {

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
// Solver warnings (see report_warning in core.hpp) are routed to the logger.
inline void init_logging(LogLevel level = LogLevel::info)
{
    auto logger = get_logger();
    spdlog::set_default_logger(logger);
    spdlog::set_level(to_spdlog_level(level));
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    diagnostic_sink() = [](const std::string& message) {
        get_logger()->warn(message);
    };
}

} // namespace mbd
