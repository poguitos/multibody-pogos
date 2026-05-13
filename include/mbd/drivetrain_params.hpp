#pragma once

// Lightweight parameter structs for the drivetrain.
// Separated from drivetrain.hpp to break include cycles with vehicle_template.hpp.

#include "mbd/core.hpp"

#include <vector>

namespace mbd {

// ============================================================================
// Drive layout
// ============================================================================

enum class DriveLayout { RWD, FWD, AWD };

// ============================================================================
// Engine parameters
// ============================================================================

struct EngineParams {
    Real max_torque{400.0};
    Real idle_rpm{1000.0};
    Real peak_torque_rpm{4500.0};
    Real redline_rpm{7000.0};
    Real idle_torque_fraction{0.4};
    Real redline_torque_fraction{0.7};
    Real inertia{0.15};
};

// ============================================================================
// Gearbox parameters
// ============================================================================

struct GearboxParams {
    std::vector<Real> ratios{{3.5, 2.3, 1.7, 1.3, 1.0, 0.8}};
    Real final_drive{3.5};
    Real efficiency{0.92};
    Real shift_up_rpm{6500.0};
    Real shift_down_rpm{2000.0};
};

// ============================================================================
// Brake parameters
// ============================================================================

struct BrakeParams {
    Real max_torque{3000.0};
    Real front_bias{0.65};
};

// ============================================================================
// Full drivetrain parameters
// ============================================================================

struct DrivetrainParams {
    EngineParams engine;
    GearboxParams gearbox;
    BrakeParams brakes;
    DriveLayout layout{DriveLayout::RWD};
    Real front_torque_split{0.4};
};

} // namespace mbd