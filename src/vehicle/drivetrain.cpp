#include "mbd/vehicle/drivetrain.hpp"

namespace mbd {

namespace {

/// The drivetrain's torques as a force element of the system: what connect()
/// adds, so that they act in every evaluation with the other forces.
class DrivetrainTorques final : public ForceElement {
public:
    DrivetrainTorques(Drivetrain& drivetrain, std::vector<BodyIndex> bodies)
        : drivetrain_(drivetrain), bodies_(std::move(bodies))
    {}

    const char* name() const override { return "drivetrain"; }
    std::vector<BodyIndex> bodies() const override { return bodies_; }

    void apply(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const override
    {
        drivetrain_.apply_torques(states, forces);
    }

private:
    Drivetrain& drivetrain_;
    std::vector<BodyIndex> bodies_;
};

Real sign(Real x)
{
    return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0);
}

} // namespace

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

Drivetrain::Corners Drivetrain::corners_of(const VehicleModel& vm)
{
    Corners corners;
    for (std::size_t c = 0; c < 4; ++c) corners[c] = {vm.tires[c], vm.wheel_bodies[c], vm.carrier_bodies[c]};
    return corners;
}

Drivetrain::Corners Drivetrain::corners_of(const VehicleHandle& vh)
{
    Corners corners;
    for (std::size_t c = 0; c < 4; ++c) {
        const CornerHandle& h = vh.corners[c];
        corners[c] = {h.tire, h.wheel_body, h.upright_body};
    }
    return corners;
}

void Drivetrain::set_corners(const kernel::Simulator& sim, BodyIndex chassis, const Corners& corners)
{
    const kernel::Model& model = sim.system.model;
    for (const Corner& c : corners) {
        MBD_THROW_IF(c.tire == nullptr || c.wheel <= 0 || c.upright <= 0 || c.wheel >= model.nbodies()
                         || c.upright >= model.nbodies() || model.parent[c.wheel] != c.upright
                         || model.nvs[c.wheel] != 1,
                     "MBD-F052: Drivetrain: each corner needs its tyre, and a wheel on a spin joint of one "
                     "coordinate on the corner's upright");
    }
    chassis_ = chassis;
    corners_ = corners;
    for (std::size_t c = 0; c < 4; ++c) {
        wheel_inertia[c] = model.inertia[corners[c].wheel].I_com_B(1, 1);
    }
}

void Drivetrain::initialize(kernel::Simulator& sim, const VehicleModel& vm)
{
    set_corners(sim, vm.chassis_body, corners_of(vm));
    roll_wheels(sim, vm);
    initialize_from_spins(sim);
}

void Drivetrain::initialize(kernel::Simulator& sim, const VehicleHandle& vh)
{
    set_corners(sim, vh.chassis_body, corners_of(vh));
    roll_wheels(sim, vh);
    initialize_from_spins(sim);
}

void Drivetrain::initialize_from_spins(const kernel::Simulator& sim)
{
    wheel_omega = spins(sim.states());
    choose_gear(wheel_omega);
    engine_rpm = std::max(omega_to_rpm(driven_wheel_speed()), params.engine.idle_rpm);
}

std::array<Real, 4> Drivetrain::spins(const std::vector<RigidBodyState>& states) const
{
    std::array<Real, 4> omega{};
    for (std::size_t c = 0; c < 4; ++c) {
        const RigidBodyState& w = states[static_cast<std::size_t>(corners_[c].wheel)];
        const RigidBodyState& u = states[static_cast<std::size_t>(corners_[c].upright)];
        omega[c] = (w.w_WB - u.w_WB).dot(w.q_WB * Vec3::UnitY());
    }
    return omega;
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
    return drive_torque[c] + corners_[static_cast<std::size_t>(c)].tire->last_axle_moment;
}

void Drivetrain::apply_torques(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces)
{
    const std::array<Real, 4> omega = spins(states);
    update_torques(omega);
    for (std::size_t c = 0; c < 4; ++c) {
        const Corner& corner = corners_[c];
        const int k = static_cast<int>(c);
        // The brake opposes the rotation in the direction it had at the start
        // of the step, kept for the whole step: if the spin crosses zero
        // within it, end_step decides whether the brake holds the wheel. A
        // sign taken at each evaluation would flip between the stages and
        // make the rates discontinuous within a step. A locked wheel's brake
        // holds it against the other torques, and damps the spin left at the
        // lock, within what the brake can give.
        Real T_brake = -brake_direction_[c] * brake_torque_out[c];
        if (wheel_locked[c]) {
            const Real hold = -free_torque(k) - wheel_inertia[c] / lock_time * omega[c];
            T_brake = std::clamp(hold, -brake_torque_out[c], brake_torque_out[c]);
        }
        brake_torque_applied[c] = T_brake;

        const RigidBodyState& w = states[static_cast<std::size_t>(corner.wheel)];
        const Vec3 axis = w.q_WB * Vec3::UnitY();
        forces[static_cast<std::size_t>(corner.wheel)].tau_W += (drive_torque[c] + T_brake) * axis;
        forces[static_cast<std::size_t>(chassis_)].tau_W -= drive_torque[c] * axis;
        forces[static_cast<std::size_t>(corner.upright)].tau_W -= T_brake * axis;
    }
}

void Drivetrain::end_step(const kernel::Simulator& sim, Real t, kernel::VecRef /*x*/)
{
    MBD_THROW_IF(&sim != sim_, "MBD-F052: Drivetrain: end_step from a simulator it is not connected to");
    std::array<Real, 4> omega = spins(sim.states());
    update_torques(omega);

    // Dry friction: a braked wheel that reached rest within the step (its
    // spin passed zero, or it started there), or that the brake will stop
    // within the next step, is locked if the brake can hold it; a locked
    // wheel is released when the other torques exceed the brake. The torques
    // are those of the step's last evaluation. Locking a wheel the brake is
    // about to stop saves the step in which the spin would pass zero and
    // end, unlocked, slightly the other way.
    // A locked wheel turns with its upright: its spin joint is set to rest.
    const Real dt = t - last_step_time_;
    last_step_time_ = t;
    bool held = false;
    for (int c = 0; c < 4; ++c) {
        const Real T_brake = brake_torque_out[c];
        const Real T_free = free_torque(c);
        if (T_brake <= 0.0) {
            wheel_locked[c] = false;
        } else if (wheel_locked[c]) {
            if (std::abs(T_free) > T_brake) wheel_locked[c] = false;
        } else {
            const Real before = omega_last_step_[c];
            const Real rate = (T_free - sign(omega[c]) * T_brake) / wheel_inertia[c];
            const bool stops_next = rate * omega[c] < 0.0 && std::abs(omega[c]) <= std::abs(rate) * dt;
            const bool reached_rest = before == 0.0 || before * omega[c] <= 0.0 || stops_next;
            if (reached_rest && std::abs(T_free) <= T_brake) wheel_locked[c] = true;
        }
        if (wheel_locked[c] && omega[c] != 0.0) {
            sim_->v(sim_->system.model.idx_v[static_cast<std::size_t>(corners_[static_cast<std::size_t>(c)].wheel)]) = 0.0;
            omega[c] = 0.0;
            held = true;
        }
        // The brake's direction for the next step: against the spin, or,
        // from rest, against the torque that turns the wheel.
        const Real turning = omega[c] != 0.0 ? omega[c] : T_free;
        brake_direction_[c] = wheel_locked[c] ? 0.0 : sign(turning);
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
    if (held) sim_->refresh();
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

void Drivetrain::connect(kernel::Simulator& sim, BodyIndex chassis, const Corners& corners)
{
    set_corners(sim, chassis, corners);
    std::vector<BodyIndex> bodies{chassis};
    for (const Corner& c : corners) {
        bodies.push_back(c.upright);
        bodies.push_back(c.wheel);
    }
    sim.attach(*this);
    sim.system.force_elements.push_back(std::make_shared<DrivetrainTorques>(*this, std::move(bodies)));
    sim_ = &sim;
    omega_last_step_ = wheel_omega;
    last_step_time_ = sim.time;
    wheel_locked.fill(false);
    for (int c = 0; c < 4; ++c) brake_direction_[c] = sign(wheel_omega[c]);
}

void Drivetrain::connect(kernel::Simulator& sim, const VehicleModel& vm)
{
    connect(sim, vm.chassis_body, corners_of(vm));
}

void Drivetrain::connect(kernel::Simulator& sim, const VehicleHandle& vh)
{
    connect(sim, vh.chassis_body, corners_of(vh));
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

void Drivetrain::choose_gear(const std::array<Real, 4>& omega)
{
    // Speed at the driven wheels, from their spins at the tyres' free radius.
    Real radius = 0.0;
    int n_driven = 0;
    for (int c = 0; c < 4; ++c) {
        if (is_driven(c)) {
            radius += corners_[static_cast<std::size_t>(c)].tire->R_free;
            ++n_driven;
        }
    }
    const Real omega_driven = driven_wheel_speed(omega);
    const Real V = n_driven > 0 ? omega_driven * radius / n_driven : Real(0.0);
    if (V < Real(0.5)) {
        current_gear = 1;
        return;
    }
    const Real rpm_target = (params.engine.peak_torque_rpm + params.engine.idle_rpm) * Real(0.5);
    for (int g = num_gears(); g >= 1; --g) {
        current_gear = g;
        if (omega_to_rpm(omega_driven) >= rpm_target) break;
    }
}

} // namespace mbd
