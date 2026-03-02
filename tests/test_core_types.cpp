#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mbd/core.hpp>

using Catch::Matchers::WithinAbs;

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

    REQUIRE_THAT(deg_back, WithinAbs(deg, 1e-12));
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

    REQUIRE_NOTHROW(logger->info("Test log message from test_core_types"));
}

TEST_CASE("Ground index and no-parent sentinel are distinct", "[core]")
{
    REQUIRE(mbd::kGroundIndex == 0);
    REQUIRE(mbd::kNoParent == -1);
    REQUIRE(mbd::kGroundIndex != mbd::kNoParent);
}