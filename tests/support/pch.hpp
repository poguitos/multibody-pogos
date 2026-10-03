#pragma once

// Precompiled header shared by all test executables (see tests/CMakeLists.txt).
//
// It holds the test framework, Eigen and the whole engine. Nothing here may be
// required for correctness: every test file includes what it uses and must
// also compile with -DMBD_TEST_PCH=OFF.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/logging.hpp"
#include "mbd/core/math.hpp"

#include "mbd/model/rigid_body.hpp"
#include "mbd/model/joint.hpp"
#include "mbd/model/system.hpp"
#include "mbd/model/constraint.hpp"

#include "mbd/algorithms/dynamics_og.hpp"
#include "mbd/algorithms/dynamics.hpp"
#include "mbd/solvers/solver.hpp"
#include "mbd/integrators/simulator.hpp"

#include "mbd/forces/force_element.hpp"
#include "mbd/forces/pacejka.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/forces/aerodynamics.hpp"
#include "mbd/forces/anti_roll_bar.hpp"

#include "mbd/vehicle/suspension/double_wishbone.hpp"
#include "mbd/vehicle/suspension/mcpherson.hpp"
#include "mbd/vehicle/suspension/multilink.hpp"
#include "mbd/vehicle/vehicle.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

#include "mbd/analysis/position_kinematics.hpp"
#include "mbd/analysis/bicycle_model.hpp"
#include "mbd/analysis/track.hpp"
#include "mbd/analysis/lap_vehicle.hpp"
#include "mbd/analysis/lap_speed_profile.hpp"
#include "mbd/analysis/optimization.hpp"
