#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>   // <-- for Approx
#include "core.hpp"
#include <Eigen/Dense>

using Catch::Approx;

TEST_CASE("compute_mean works for simple vectors", "[core]") {
    Eigen::VectorXd v(3);
    v << 1.0, 2.0, 3.0;

    double m = compute_mean(v);
    REQUIRE(m == Approx(2.0));
}

TEST_CASE("compute_mean handles empty vector", "[core]") {
    Eigen::VectorXd v; // size 0
    double m = compute_mean(v);
    REQUIRE(m == Approx(0.0));
}

TEST_CASE("make_greeting returns expected string", "[core]") {
    auto msg = make_greeting("World");
    REQUIRE(msg == "Hello, World from multibody_core!");
}
