#pragma once

// Precompiled header of the library (see src/CMakeLists.txt).
//
// It holds the standard library, Eigen and the stable lower layers that
// almost every source file includes: the core types, spatial algebra and the
// kernel. The force, vehicle and analysis headers are left out, so that
// editing one of them recompiles only the files that include it. Nothing here
// may be required for correctness: every source file includes what it uses
// and must also compile with -DMBD_PCH=OFF.

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/spatial/spatial.hpp"
#include "mbd/model/rigid_body.hpp"
#include "mbd/kernel/joint_model.hpp"
#include "mbd/kernel/model.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/forces.hpp"
#include "mbd/kernel/simulator.hpp"
