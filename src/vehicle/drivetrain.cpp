#include "mbd/vehicle/drivetrain.hpp"

namespace mbd {

Real Drivetrain::torque_curve(Real rpm, const EngineParams& ep)
{
    if (rpm < ep.idle_rpm) return ep.idle_torque_fraction;
    if (rpm > ep.redline_rpm) return Real(0.0); // Rev limiter

    if (rpm <= ep.peak_torque_rpm) {
        const Real t = (rpm - ep.idle_rpm) / (ep.peak_torque_rpm - ep.idle_rpm);
        return ep.idle_torque_fraction + t * (Real(1.0) - ep.idle_torque_fraction);
    } else {
        const Real t = (rpm - ep.peak_torque_rpm) / (ep.redline_rpm - ep.peak_torque_rpm);
        return Real(1.0) + t * (ep.redline_torque_fraction - Real(1.0));
    }
}

Real Drivetrain::compute_engine_torque(Real rpm, Real throttle_input,
                                       const EngineParams& ep)
{
    const Real clamped_throttle = std::clamp(throttle_input, Real(0.0), Real(1.0));
    return ep.max_torque * clamped_throttle * torque_curve(rpm, ep);
}

Real Drivetrain::total_ratio() const
{
    int idx = std::clamp(current_gear, 1, static_cast<int>(params.gearbox.ratios.size())) - 1;
    return params.gearbox.ratios[idx] * params.gearbox.final_drive;
}

Real Drivetrain::omega_to_rpm(Real omega_wheel_avg) const
{
    const Real omega_engine = std::abs(omega_wheel_avg) * total_ratio();
    return omega_engine * Real(60.0) / (Real(2.0) * pi);
}

void Drivetrain::initialize(const kernel::Simulator& sim, const VehicleModel& vm)
{
    const Real R = vm.params.tire_free_radius;
    const Real I = disc_inertia(vm.params.wheel_mass, R);
    wheel_inertia = {{I, I, I, I}};
    initialize_from_speed(sim.states(), vm.chassis_body, {{R, R, R, R}});
}

void Drivetrain::initialize(const kernel::Simulator& sim, const VehicleHandle& vh)
{
    const auto& front = vh.tmpl.front_axle;
    const auto& rear  = vh.tmpl.rear_axle;
    const Real R_f = front.tire_free_radius;
    const Real R_r = rear.tire_free_radius;
    const Real I_f = disc_inertia(front.wheel_mass, R_f);
    const Real I_r = disc_inertia(rear.wheel_mass, R_r);
    wheel_inertia = {{I_f, I_f, I_r, I_r}};
    initialize_from_speed(sim.states(), vh.chassis_body, {{R_f, R_f, R_r, R_r}});
}

void Drivetrain::initial_states(kernel::VecRef x) const
{
    for (int c = 0; c < 4; ++c) x(c) = wheel_omega[c];
}

void Drivetrain::begin_evaluation(const kernel::Simulator& /*sim*/, Real /*t*/, kernel::ConstVecRef x)
{
    for (int c = 0; c < 4; ++c) {
        tires_[c]->omega_wheel = x(c);
        tires_[c]->auto_free_roll = false;
    }
}

void Drivetrain::update_torques(const std::array<Real, 4>& omega)
{
    const auto& ep = params.engine;
    const auto& gp = params.gearbox;
    const auto& bp = params.brakes;

    engine_rpm = std::max(omega_to_rpm(driven_wheel_speed(omega)), ep.idle_rpm);
    engine_torque_out = compute_engine_torque(engine_rpm, throttle, ep);

    // Drive torque, shared among the driven wheels.
    const Real T_at_wheels = engine_torque_out * total_ratio() * gp.efficiency;
    drive_torque.fill(0.0);
    switch (params.layout) {
        case DriveLayout::RWD:
            drive_torque[2] = T_at_wheels * Real(0.5); // RL
            drive_torque[3] = T_at_wheels * Real(0.5); // RR
            break;
        case DriveLayout::FWD:
            drive_torque[0] = T_at_wheels * Real(0.5); // FL
            drive_torque[1] = T_at_wheels * Real(0.5); // FR
            break;
        case DriveLayout::AWD: {
            const Real T_front = T_at_wheels * params.front_torque_split;
            const Real T_rear  = T_at_wheels * (Real(1.0) - params.front_torque_split);
            drive_torque[0] = T_front * Real(0.5);
            drive_torque[1] = T_front * Real(0.5);
            drive_torque[2] = T_rear  * Real(0.5);
            drive_torque[3] = T_rear  * Real(0.5);
            break;
        }
    }

    // Brake torque. bp.max_torque is the sum over the four wheels at full
    // pedal.
    const Real pedal = std::clamp(brake, Real(0.0), Real(1.0));
    const Real T_front_wheel = pedal * bp.max_torque * bp.front_bias * Real(0.5);
    const Real T_rear_wheel  = pedal * bp.max_torque * (Real(1.0) - bp.front_bias) * Real(0.5);
    brake_torque_out[0] = T_front_wheel;
    brake_torque_out[1] = T_front_wheel;
    brake_torque_out[2] = T_rear_wheel;
    brake_torque_out[3] = T_rear_wheel;
}

Real Drivetrain::free_torque(int c) const
{
    return drive_torque[c] - tires_[c]->get_Fx() * tires_[c]->get_rolling_radius();
}

void Drivetrain::rates(const kernel::Simulator& /*sim*/, Real /*t*/, kernel::ConstVecRef x, kernel::VecRef x_dot)
{
    const std::array<Real, 4> omega{{x(0), x(1), x(2), x(3)}};
    update_torques(omega);
    for (int c = 0; c < 4; ++c) {
        if (wheel_locked[c]) {
            x_dot(c) = 0.0;
            continue;
        }
        // The brake opposes the rotation in the direction it had at the start
        // of the step, kept for the whole step: if the spin crosses zero
        // within it, end_step decides whether the brake holds the wheel. A
        // sign taken at each evaluation would flip between the stages and
        // make the rates discontinuous within a step.
        x_dot(c) = (free_torque(c) - brake_direction_[c] * brake_torque_out[c]) / compute_effective_inertia(c);
    }
}

void Drivetrain::end_step(const kernel::Simulator& /*sim*/, Real t, kernel::VecRef x)
{
    std::array<Real, 4> omega{{x(0), x(1), x(2), x(3)}};
    update_torques(omega);

    // Dry friction: a braked wheel that reached rest within the step (its
    // spin passed zero, or it started there), or that the brake will stop
    // within the next step, is locked if the brake can hold it; a locked
    // wheel is released when the other torques exceed the brake. The torques
    // are those of the step's last evaluation. Locking a wheel the brake is
    // about to stop saves the step in which the spin would pass zero and
    // end, unlocked, slightly the other way.
    const Real dt = t - last_step_time_;
    last_step_time_ = t;
    for (int c = 0; c < 4; ++c) {
        const Real T_brake = brake_torque_out[c];
        const Real T_free = free_torque(c);
        if (T_brake <= 0.0) {
            wheel_locked[c] = false;
        } else if (wheel_locked[c]) {
            if (std::abs(T_free) > T_brake) {
                wheel_locked[c] = false;
            } else {
                // Held at exactly zero: an implicit method's Newton
                // corrections, solved to a tolerance, can leave a locked
                // spin a little off it.
                omega[c] = 0.0;
                x(c) = 0.0;
            }
        } else {
            const Real before = omega_last_step_[c];
            const Real direction = omega[c] > 0.0 ? 1.0 : (omega[c] < 0.0 ? -1.0 : 0.0);
            const Real rate = (T_free - direction * T_brake) / compute_effective_inertia(c);
            const bool stops_next = rate * omega[c] < 0.0 && std::abs(omega[c]) <= std::abs(rate) * dt;
            const bool reached_rest = before == 0.0 || before * omega[c] <= 0.0 || stops_next;
            if (reached_rest && std::abs(T_free) <= T_brake) {
                wheel_locked[c] = true;
                omega[c] = 0.0;
                x(c) = 0.0;
            }
        }
        // The brake's direction for the next step: against the spin, or,
        // from rest, against the torque that turns the wheel.
        const Real turning = omega[c] != 0.0 ? omega[c] : T_free;
        brake_direction_[c] = wheel_locked[c] ? 0.0 : (turning > 0.0 ? 1.0 : (turning < 0.0 ? -1.0 : 0.0));
    }

    // Gear: shift up or down by engine speed.
    const auto& ep = params.engine;
    const auto& gp = params.gearbox;
    const Real omega_driven = driven_wheel_speed(omega);
    engine_rpm = std::max(omega_to_rpm(omega_driven), ep.idle_rpm);
    if (engine_rpm > gp.shift_up_rpm && current_gear < num_gears()) {
        ++current_gear;
    } else if (engine_rpm < gp.shift_down_rpm && current_gear > 1) {
        --current_gear;
    }
    engine_rpm = std::max(omega_to_rpm(omega_driven), ep.idle_rpm);

    wheel_omega = omega;
    omega_last_step_ = omega;
}

bool Drivetrain::is_driven(int corner) const
{
    switch (params.layout) {
        case DriveLayout::RWD: return corner >= 2;
        case DriveLayout::FWD: return corner < 2;
        case DriveLayout::AWD: return true;
    }
    return false;
}

Real Drivetrain::compute_effective_inertia(int corner) const
{
    const Real I_wheel = wheel_inertia[corner];

    if (!is_driven(corner)) return I_wheel;

    int n_driven = 0;
    for (int c = 0; c < 4; ++c) {
        if (is_driven(c)) ++n_driven;
    }

    const Real ratio = total_ratio();
    const Real I_reflected = params.engine.inertia * ratio * ratio /
                             static_cast<Real>(n_driven);

    return I_wheel + I_reflected;
}

void Drivetrain::connect(kernel::Simulator& sim, const TireSet& tires)
{
    tires_ = tires;
    omega_last_step_ = wheel_omega;
    last_step_time_ = sim.time;
    wheel_locked.fill(false);
    for (int c = 0; c < 4; ++c) {
        brake_direction_[c] = wheel_omega[c] > 0.0 ? 1.0 : (wheel_omega[c] < 0.0 ? -1.0 : 0.0);
    }
    sim.attach(*this);
}

void Drivetrain::connect(kernel::Simulator& sim, const VehicleModel& vm)
{
    connect(sim, vm.tires);
}

void Drivetrain::connect(kernel::Simulator& sim, const VehicleHandle& vh)
{
    connect(sim, tires_of(vh));
}

Drivetrain::TireSet Drivetrain::tires_of(const VehicleHandle& vh)
{
    return {{vh.corners[0].tire, vh.corners[1].tire,
             vh.corners[2].tire, vh.corners[3].tire}};
}

Real Drivetrain::driven_wheel_speed(const std::array<Real, 4>& omega) const
{
    Real sum = Real(0.0);
    int n_driven = 0;
    for (int c = 0; c < 4; ++c) {
        if (is_driven(c)) {
            sum += omega[c];
            ++n_driven;
        }
    }
    return (n_driven > 0) ? sum / n_driven : Real(0.0);
}

void Drivetrain::initialize_from_speed(const std::vector<RigidBodyState>& states, BodyIndex chassis,
                                       const std::array<Real, 4>& free_radius)
{
    const auto& chassis_state = states[static_cast<std::size_t>(chassis)];
    const Vec3 fwd_W = chassis_state.q_WB * Vec3::UnitX();
    const Real Vx = chassis_state.v_WB.dot(fwd_W);

    // Free rolling on a tyre that is about 3 % smaller under load.
    for (int c = 0; c < 4; ++c) {
        wheel_omega[c] = Vx / (free_radius[c] * Real(0.97));
    }

    // Select appropriate gear for current speed
    const Real omega_init = driven_wheel_speed();
    if (Vx < Real(0.5)) {
        current_gear = 1;
    } else {
        const Real rpm_target = (params.engine.peak_torque_rpm +
                                 params.engine.idle_rpm) * Real(0.5);
        for (int g = num_gears(); g >= 1; --g) {
            current_gear = g;
            if (omega_to_rpm(omega_init) >= rpm_target) break;
        }
    }

    engine_rpm = std::max(omega_to_rpm(omega_init), params.engine.idle_rpm);
}

} // namespace mbd
