#pragma once

// Precompiled header shared by all test executables (see tests/CMakeLists.txt).
//
// It holds the test framework, Eigen and the whole engine. Nothing here may be
// required for correctness: every test file includes what it uses and must
// also compile with -DMBD_PCH=OFF.

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

#include "mbd/spatial/spatial.hpp"
#include "mbd/kernel/joint_model.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/validate.hpp"

#include "mbd/model/rigid_body.hpp"

#include "mbd/forces/force_element.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/user_force.hpp"
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
