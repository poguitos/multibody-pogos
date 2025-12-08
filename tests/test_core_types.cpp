#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mbd/core.hpp>

using Catch::Approx;

TEST_CASE("BodyIndex behaves as an integer", "[core]")
{
    mbd::BodyIndex i = 3;
    REQUIRE(i == 3);
}

TEST_CASE("deg2rad and rad2deg are approximate inverses", "[core]")
{
    mbd::Real deg = 45.0;
    mbd::Real rad = mbd::deg2rad(deg);
    mbd::Real deg_back = mbd::rad2deg(rad);

    REQUIRE(deg_back == Approx(deg).margin(1e-12));
}

TEST_CASE("MbdError can be thrown and caught", "[core]")
{
    try {
        throw mbd::MbdError("test error");
    } catch (const mbd::MbdError& e) {
        REQUIRE(std::string(e.what()) == "test error");
        return;
    }
    FAIL("MbdError was not caught");
}

TEST_CASE("Logger initialization returns a non-null logger", "[core]")
{
    REQUIRE_NOTHROW(mbd::init_logging(mbd::LogLevel::debug));

    auto logger = mbd::get_logger();
    REQUIRE(logger != nullptr);

    // We don't assert on output: just ensure it doesn't crash.
    REQUIRE_NOTHROW(logger->info("Test log message from test_core_types"));
}
