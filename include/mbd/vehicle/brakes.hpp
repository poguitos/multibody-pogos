#pragma once

// The brakes (plan task 7.9, decision D70): from the pedal to a pressure to a
// torque at each wheel, with the axles' bias, a proportioning valve, wheel
// lock, and an optional anti-lock controller. A state element: each wheel's
// line pressure is a state, lagging its target with the system's response
// time,
//
//     p_i' = (target_i - p_i) / tau,
//
// the target the master cylinder's pressure (pedal times max_pressure), the
// rear's through the proportioning valve (above its knee the rear's rises by
// rear_slope of the master's), and zero while the anti-lock controller
// releases the wheel. The brake's torque available is the axle's gain times
// the pressure (gain = piston area x 2 pads x pad friction x effective
// radius); the bias is the front's share of it at equal pressures.
//
// The brake is dry friction, as the template's drivetrain's (D62): it
// opposes the wheel's spin relative to its upright in the direction it had
// at the step's start, kept through the step; a wheel that comes to rest
// while the brake can hold it is locked (its spin set to rest at each step's
// end, the brake holding the other torques), and released when they exceed
// the brake. The torque reacts on the upright (outboard brakes).
//
// The anti-lock controller (abs), at each step's end and above abs_min_speed:
// a wheel whose braking slip (the tyre's Vsx / |Vx|, positive when the wheel
// turns slower than it rolls) rises past abs_release_slip is released (its
// target pressure set to zero) until the slip falls below abs_apply_slip.

#include <array>
#include <functional>
#include <limits>

#include "mbd/core/core.hpp"
#include "mbd/forces/tire.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"

namespace mbd {

struct VehicleHandle;
struct VehicleModel;

struct BrakeSystemParams {
    Real max_pressure{1.2e7};     ///< The master cylinder's at full pedal [Pa]
    Real response_time{0.05};     ///< tau [s]
    Real front_gain{1.9e-4};      ///< Each front brake's torque per pressure [N m/Pa]
    Real rear_gain{0.9e-4};       ///< Each rear one's
    /// The proportioning valve: the rear's pressure follows the master's up
    /// to the knee, then rises by rear_slope of it. Infinite: none.
    Real knee_pressure{std::numeric_limits<Real>::infinity()};
    Real rear_slope{0.3};
    bool abs{false};
    Real abs_release_slip{0.15};
    Real abs_apply_slip{0.08};
    Real abs_min_speed{2.0};      ///< [m/s] below it the controller leaves the brakes alone
    Real lock_time{0.005};        ///< As the drivetrain's: a locked wheel's damping, its inertia over this
};

class BrakeSystem : public kernel::StateElement {
public:
    struct Corner {
        FullTireForce* tire{nullptr};
        BodyIndex wheel{0};
        BodyIndex upright{0};
    };

    BrakeSystemParams params;
    Real pedal{0.0};   ///< 0..1
    /// Other torques on each wheel about its axle besides the tyre's (a
    /// drive's), for the decision to lock and release; none if not set.
    std::function<Real(int corner)> drive_torque;

    // Telemetry
    std::array<bool, 4> locked{{false, false, false, false}};
    std::array<bool, 4> releasing{{false, false, false, false}};   ///< The anti-lock controller's state
    mutable std::array<Real, 4> torque_applied{{0.0, 0.0, 0.0, 0.0}};

    explicit BrakeSystem(BrakeSystemParams p);

    void connect(kernel::Simulator& sim, const VehicleHandle& vh);
    void connect(kernel::Simulator& sim, const VehicleModel& vm);
    void connect(kernel::Simulator& sim, const std::array<Corner, 4>& corners);

    /// The target pressure of wheel `corner` for the pedal [Pa] (no
    /// anti-lock release).
    Real target_pressure(int corner) const;
    /// The torque available at wheel `corner` at pressure p [N m].
    Real torque_at(int corner, Real p) const { return (corner < 2 ? params.front_gain : params.rear_gain) * p; }

    const char* name() const override { return "brakes"; }
    int size() const override { return 4; }
    void initial_states(kernel::VecRef x) const override { x.setZero(); }
    void begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x) override;
    void rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot) override;
    void end_step(const kernel::Simulator& sim, Real t, kernel::VecRef x) override;

    void apply_torques(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const;

private:
    Real spin(const std::vector<RigidBodyState>& states, int c) const;
    Real free_torque(int c) const;

    std::array<Corner, 4> corners_{};
    std::array<Real, 4> inertia_{{0.0, 0.0, 0.0, 0.0}};
    std::array<Real, 4> pressure_{{0.0, 0.0, 0.0, 0.0}};
    std::array<Real, 4> direction_{{0.0, 0.0, 0.0, 0.0}};
    std::array<Real, 4> omega_last_{{0.0, 0.0, 0.0, 0.0}};
    Real last_time_{0.0};
    kernel::Simulator* sim_{nullptr};
};

} // namespace mbd
