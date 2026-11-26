#include "mbd/vehicle/brakes.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "mbd/vehicle/vehicle.hpp"
#include "mbd/vehicle/vehicle_template.hpp"

namespace mbd {

namespace {

class BrakeTorques final : public ForceElement {
public:
    BrakeTorques(const BrakeSystem& b, std::vector<BodyIndex> bodies) : b_(b), bodies_(std::move(bodies)) {}
    const char* name() const override { return "brakes"; }
    std::vector<BodyIndex> bodies() const override { return bodies_; }
    void apply(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const override
    {
        b_.apply_torques(states, forces);
    }

private:
    const BrakeSystem& b_;
    std::vector<BodyIndex> bodies_;
};

Real sign(Real x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

} // namespace

BrakeSystem::BrakeSystem(BrakeSystemParams p) : params(std::move(p))
{
    const BrakeSystemParams& q = params;
    MBD_THROW_IF(!(q.max_pressure >= 0.0) || !(q.response_time > 0.0) || !(q.front_gain >= 0.0)
                     || !(q.rear_gain >= 0.0) || !(q.knee_pressure >= 0.0) || !(q.rear_slope >= 0.0)
                     || !(q.lock_time > 0.0) || !(q.abs_release_slip > q.abs_apply_slip) || !(q.abs_apply_slip >= 0.0),
                 "MBD-V007: BrakeSystem: a pressure, gain, slope or time that is negative (a response or lock time "
                 "that is not positive), or an anti-lock release slip not above its apply slip");
}

Real BrakeSystem::target_pressure(int corner) const
{
    const Real p = std::clamp(pedal, Real(0.0), Real(1.0)) * params.max_pressure;
    if (corner < 2 || p <= params.knee_pressure) return p;
    return params.knee_pressure + params.rear_slope * (p - params.knee_pressure);
}

Real BrakeSystem::spin(const std::vector<RigidBodyState>& states, int c) const
{
    const Corner& k = corners_[static_cast<std::size_t>(c)];
    const RigidBodyState& w = states[static_cast<std::size_t>(k.wheel)];
    const RigidBodyState& u = states[static_cast<std::size_t>(k.upright)];
    return (w.w_WB - u.w_WB).dot(w.q_WB * Vec3::UnitY());
}

Real BrakeSystem::free_torque(int c) const
{
    const Real drive = drive_torque ? drive_torque(c) : Real(0.0);
    return drive + corners_[static_cast<std::size_t>(c)].tire->last_axle_moment;
}

void BrakeSystem::begin_evaluation(const kernel::Simulator&, Real, kernel::ConstVecRef x)
{
    for (int c = 0; c < 4; ++c) pressure_[static_cast<std::size_t>(c)] = std::max(x(c), Real(0.0));
}

void BrakeSystem::rates(const kernel::Simulator&, Real, kernel::ConstVecRef x, kernel::VecRef x_dot)
{
    for (int c = 0; c < 4; ++c) {
        const Real target = releasing[static_cast<std::size_t>(c)] ? Real(0.0) : target_pressure(c);
        x_dot(c) = (target - x(c)) / params.response_time;
    }
}

void BrakeSystem::apply_torques(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const
{
    for (int c = 0; c < 4; ++c) {
        const std::size_t k = static_cast<std::size_t>(c);
        const Real available = torque_at(c, pressure_[k]);
        // Against the spin's direction at the step's start; a locked wheel's
        // brake holds the other torques, and damps the spin its upright's
        // turning leaves, within what the brake can give (as the drivetrain's).
        Real T = -direction_[k] * available;
        if (locked[k]) {
            const Real hold = -free_torque(c) - inertia_[k] / params.lock_time * spin(states, c);
            T = std::clamp(hold, -available, available);
        }
        torque_applied[k] = T;
        const Corner& corner = corners_[k];
        const Vec3 axis = states[static_cast<std::size_t>(corner.wheel)].q_WB * Vec3::UnitY();
        forces[static_cast<std::size_t>(corner.wheel)].tau_W += T * axis;
        forces[static_cast<std::size_t>(corner.upright)].tau_W -= T * axis;
    }
}

void BrakeSystem::end_step(const kernel::Simulator& sim, Real t, kernel::VecRef x)
{
    MBD_THROW_IF(&sim != sim_, "MBD-V007: BrakeSystem: end_step from a simulator it is not connected to");
    const Real dt = t - last_time_;
    last_time_ = t;
    bool held = false;
    for (int c = 0; c < 4; ++c) {
        const std::size_t k = static_cast<std::size_t>(c);
        Real omega = spin(sim.states(), c);
        const Real T_brake = torque_at(c, std::max(x(c), Real(0.0)));
        const Real T_free = free_torque(c);
        // Dry friction: lock a wheel that reached rest within the step (or
        // will within the next) if the brake can hold it; release a locked
        // one when the other torques exceed the brake.
        if (T_brake <= 0.0) {
            locked[k] = false;
        } else if (locked[k]) {
            if (std::abs(T_free) > T_brake) locked[k] = false;
        } else {
            const Real rate = (T_free - sign(omega) * T_brake) / inertia_[k];
            const bool stops_next = rate * omega < 0.0 && std::abs(omega) <= std::abs(rate) * dt;
            const bool reached_rest = omega_last_[k] == 0.0 || omega_last_[k] * omega <= 0.0 || stops_next;
            if (reached_rest && std::abs(T_free) <= T_brake) locked[k] = true;
        }
        // The anti-lock controller: by the tyre's longitudinal slip.
        const FullTireForce& tire = *corners_[k].tire;
        if (params.abs && std::abs(tire.last_Vx) > params.abs_min_speed) {
            const Real slip = tire.last_Vsx / std::abs(tire.last_Vx);   // braking: the contact faster than the tread
            if (!releasing[k] && slip > params.abs_release_slip) releasing[k] = true;
            if (releasing[k] && slip < params.abs_apply_slip) releasing[k] = false;
            if (releasing[k]) locked[k] = false;
        } else {
            releasing[k] = false;
        }
        if (locked[k] && omega != 0.0) {
            sim_->v(sim_->system.model.idx_v[static_cast<std::size_t>(corners_[k].wheel)]) = 0.0;
            omega = 0.0;
            held = true;
        }
        const Real turning = omega != 0.0 ? omega : T_free;
        direction_[k] = locked[k] ? 0.0 : sign(turning);
        omega_last_[k] = omega;
    }
    if (held) sim_->refresh();
}

void BrakeSystem::connect(kernel::Simulator& sim, const std::array<Corner, 4>& corners)
{
    const kernel::Model& model = sim.system.model;
    std::vector<BodyIndex> bodies;
    for (std::size_t c = 0; c < 4; ++c) {
        const Corner& k = corners[c];
        MBD_THROW_IF(k.tire == nullptr || k.wheel <= 0 || k.wheel >= model.nbodies() || model.parent[k.wheel] != k.upright
                         || model.nvs[k.wheel] != 1,
                     "MBD-V007: BrakeSystem: each corner needs its tyre, and a wheel on a spin joint on its upright");
        inertia_[c] = model.inertia[static_cast<std::size_t>(k.wheel)].I_com_B(1, 1);
        bodies.push_back(k.wheel);
        bodies.push_back(k.upright);
    }
    corners_ = corners;
    sim.attach(*this);
    sim.system.force_elements.push_back(std::make_shared<BrakeTorques>(*this, std::move(bodies)));
    sim_ = &sim;
    last_time_ = sim.time;
    for (int c = 0; c < 4; ++c) {
        const Real w = spin(sim.states(), c);
        omega_last_[static_cast<std::size_t>(c)] = w;
        direction_[static_cast<std::size_t>(c)] = sign(w);
    }
}

void BrakeSystem::connect(kernel::Simulator& sim, const VehicleHandle& vh)
{
    std::array<Corner, 4> c{};
    for (std::size_t i = 0; i < 4; ++i) c[i] = {vh.corners[i].tire, vh.corners[i].wheel_body, vh.corners[i].upright_body};
    connect(sim, c);
}

void BrakeSystem::connect(kernel::Simulator& sim, const VehicleModel& vm)
{
    std::array<Corner, 4> c{};
    for (std::size_t i = 0; i < 4; ++i) c[i] = {vm.tires[i], vm.wheel_bodies[i], vm.carrier_bodies[i]};
    connect(sim, c);
}

} // namespace mbd
