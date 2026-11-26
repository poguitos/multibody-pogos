#include "mbd/vehicle/powertrain.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>

#include "mbd/vehicle/vehicle_template.hpp"

namespace mbd {

// ============================================================================
// The engine's map
// ============================================================================

Real EngineMap::torque(Real w, Real throttle) const
{
    if (w > redline_speed) return drag.value(w);   // fuel cut
    const Real u = std::clamp(throttle, Real(0.0), Real(1.0));
    Real full = full_load.value(w), dr = drag.value(w);
    if (max_power > 0.0 && std::abs(w) * std::abs(full) > max_power) {
        const Real scale = max_power / (std::abs(w) * std::abs(full));
        full *= scale;
        dr *= scale;
    }
    Real T = u * full + (1.0 - u) * dr;
    if (idle_speed > 0.0 && w < idle_speed) T += std::min(idle_gain * (idle_speed - w), std::max(full - T, Real(0.0)));
    return T;
}

EngineMap EngineMap::from(const EngineParams& p)
{
    const Real rpm = 2.0 * pi / 60.0;
    EngineMap m;
    m.idle_speed = p.idle_rpm * rpm;
    m.redline_speed = p.redline_rpm * rpm;
    m.inertia = p.inertia;
    m.full_load = Curve::table({0.0, m.idle_speed, p.peak_torque_rpm * rpm, m.redline_speed},
                               {p.idle_torque_fraction * p.max_torque, p.idle_torque_fraction * p.max_torque,
                                p.max_torque, p.redline_torque_fraction * p.max_torque});
    m.drag = Curve::table({0.0, m.redline_speed}, {-0.05 * p.max_torque, -0.12 * p.max_torque});
    return m;
}

EngineMap EngineMap::motor(Real peak_torque, Real base_speed, Real top_speed, Real inertia, Real regen)
{
    EngineMap m;
    m.full_load = Curve::linear(0.0, peak_torque);
    m.drag = Curve::linear(0.0, -regen * peak_torque);
    m.max_power = peak_torque * base_speed;
    m.inertia = inertia;
    m.idle_speed = 0.0;
    m.redline_speed = top_speed;
    return m;
}

// ============================================================================
// The powertrain
// ============================================================================

namespace {

class PowertrainTorques final : public ForceElement {
public:
    PowertrainTorques(const Powertrain& p, std::vector<BodyIndex> bodies) : p_(p), bodies_(std::move(bodies)) {}
    const char* name() const override { return "powertrain"; }
    std::vector<BodyIndex> bodies() const override { return bodies_; }
    void apply(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const override
    {
        p_.apply_torques(states, forces);
    }

private:
    const Powertrain& p_;
    std::vector<BodyIndex> bodies_;
};

/// The driven axles (0 front, 1 rear) of a layout.
std::vector<int> driven_axles(DriveLayout layout)
{
    switch (layout) {
        case DriveLayout::FWD: return {0};
        case DriveLayout::RWD: return {1};
        case DriveLayout::AWD: return {0, 1};
    }
    return {};
}

} // namespace

struct Powertrain::Derived {
    Real w_e{0.0}, w_in{0.0}, w_t{0.0};
    Real T_engine{0.0}, T_pump{0.0}, T_turbine{0.0}, T_gearbox{0.0};
    std::array<Real, 2> w_c{{0.0, 0.0}}, w_s{{0.0, 0.0}}, T_axle{{0.0, 0.0}}, wc_dot{{0.0, 0.0}}, ws_dot{{0.0, 0.0}};
    std::array<Real, 4> T_shaft{{0.0, 0.0, 0.0, 0.0}}, twist_rate{{0.0, 0.0, 0.0, 0.0}}, z_rate{{0.0, 0.0, 0.0, 0.0}};
    Real we_dot{0.0};
};

Powertrain::Powertrain(PowertrainParams p) : params(std::move(p))
{
    const PowertrainParams& q = params;
    bool ok = !q.gear_ratios.empty() && q.final_drive > 0.0 && q.efficiency > 0.0 && q.efficiency <= 1.0
           && q.carrier_inertia > 0.0 && q.side_gear_inertia > 0.0 && q.spider_inertia > 0.0
           && q.shaft_stiffness > 0.0 && q.shaft_damping >= 0.0 && q.shift_time > 0.0 && q.front_share >= 0.0
           && q.front_share <= 1.0 && q.engine.inertia > 0.0;
    for (Real r : q.gear_ratios) ok = ok && r > 0.0;
    if (q.coupling == Coupling::TorqueConverter) {
        ok = ok && q.converter_capacity > 0.0 && q.converter_stall_ratio >= 1.0 && q.converter_coupling_point > 0.0
          && q.converter_coupling_point < 1.0;
    }
    if (q.coupling == Coupling::Clutch) ok = ok && q.clutch_capacity > 0.0 && q.clutch_slip_speed > 0.0;
    MBD_THROW_IF(!ok, "MBD-V006: Powertrain: a ratio, inertia, stiffness, capacity or time that is not positive, an "
                      "efficiency outside (0, 1], a front share outside [0, 1], or a converter's stall ratio below 1 "
                      "or coupling point outside (0, 1)");

    if (q.coupling != Coupling::Direct) i_engine_ = static_cast<int>(n_++);
    const bool locked = q.differential == Differential::Locked;
    const bool iso = q.isolator.stiffness > 0.0;
    for (int a : driven_axles(q.layout)) {
        i_axle_[static_cast<std::size_t>(a)] = static_cast<int>(n_);
        n_ += 1 + (locked ? 0 : 1) + 2 + (iso ? 2 : 0);
    }
    x_ = VecX::Zero(static_cast<Index>(n_));
}

bool Powertrain::driven(int corner) const
{
    return i_axle_[static_cast<std::size_t>(corner / 2)] >= 0;
}

Real Powertrain::total_ratio() const
{
    const int g = std::clamp(gear, 1, static_cast<int>(params.gear_ratios.size()));
    return params.gear_ratios[static_cast<std::size_t>(g - 1)] * params.final_drive;
}

int Powertrain::spider_index(int axle) const
{
    const int i = i_axle_[static_cast<std::size_t>(axle)];
    return i < 0 || params.differential == Differential::Locked ? -1 : i + 1;
}

int Powertrain::twist_index(int corner) const
{
    const int i = i_axle_[static_cast<std::size_t>(corner / 2)];
    if (i < 0) return -1;
    return i + 1 + (params.differential == Differential::Locked ? 0 : 1) + corner % 2;
}

Real Powertrain::reflected(int axle) const
{
    // The gearbox's input side, and a directly coupled motor, turn at g times
    // the carriers' speed: their inertia appears at the carriers times g^2,
    // shared between the axles as the torque is.
    const PowertrainParams& p = params;
    const Real g = total_ratio();
    const Real share = p.layout == DriveLayout::AWD ? (axle == 0 ? p.front_share : 1.0 - p.front_share) : 1.0;
    const Real J = p.input_inertia + (p.coupling == Coupling::Direct ? p.engine.inertia : Real(0.0));
    return J * g * g * share;
}

Real Powertrain::shift_factor(Real t) const
{
    if (shift_start_ < 0.0) return 1.0;
    const Real u = (t - shift_start_) / params.shift_time;
    return u >= 1.0 ? Real(1.0) : std::abs(1.0 - 2.0 * std::max(u, Real(0.0)));
}

Real Powertrain::spin(const std::vector<RigidBodyState>& states, int corner) const
{
    const Corner& c = corners_[static_cast<std::size_t>(corner)];
    const RigidBodyState& w = states[static_cast<std::size_t>(c.wheel)];
    const RigidBodyState& u = states[static_cast<std::size_t>(c.upright)];
    return (w.w_WB - u.w_WB).dot(w.q_WB * Vec3::UnitY());
}

void Powertrain::evaluate(const std::vector<RigidBodyState>& states, kernel::ConstVecRef x, Real t, Derived& d) const
{
    (void)states;
    const PowertrainParams& p = params;
    const std::vector<int> axles = driven_axles(p.layout);
    const bool locked = p.differential == Differential::Locked;
    const bool iso = p.isolator.stiffness > 0.0;
    const Real g = total_ratio();
    const Real b = p.layout == DriveLayout::AWD ? p.front_share : Real(1.0);

    for (int a : axles) {
        const std::size_t k = static_cast<std::size_t>(a);
        d.w_c[k] = x(i_axle_[k]);
        d.w_s[k] = locked ? Real(0.0) : x(i_axle_[k] + 1);
    }
    d.w_in = p.layout == DriveLayout::AWD ? b * d.w_c[0] + (1.0 - b) * d.w_c[1] : d.w_c[static_cast<std::size_t>(axles[0])];
    d.w_t = g * d.w_in;

    // The coupling.
    switch (p.coupling) {
        case Coupling::TorqueConverter: {
            d.w_e = x(i_engine_);
            d.T_pump = d.w_e * (d.w_e - d.w_t) / (p.converter_capacity * p.converter_capacity);
            const Real sr = d.w_e > 0.0 ? d.w_t / d.w_e : Real(1.0);
            const Real sr_c = p.converter_coupling_point;
            const Real tr = sr < sr_c ? p.converter_stall_ratio + (1.0 - p.converter_stall_ratio) * std::max(sr, Real(0.0)) / sr_c
                                      : Real(1.0);
            d.T_turbine = tr * d.T_pump;
            d.T_engine = p.engine.torque(d.w_e, throttle);
            d.we_dot = (d.T_engine - d.T_pump) / p.engine.inertia;
            break;
        }
        case Coupling::Clutch:
            d.w_e = x(i_engine_);
            d.T_pump = std::clamp(clutch, Real(0.0), Real(1.0)) * p.clutch_capacity
                     * std::tanh((d.w_e - d.w_t) / p.clutch_slip_speed);
            d.T_turbine = d.T_pump;
            d.T_engine = p.engine.torque(d.w_e, throttle);
            d.we_dot = (d.T_engine - d.T_pump) / p.engine.inertia;
            break;
        case Coupling::Direct:
            d.w_e = d.w_t;
            d.T_engine = p.engine.torque(d.w_t, throttle);
            d.T_turbine = d.T_engine;
            break;
    }
    d.T_gearbox = shift_factor(t) * p.efficiency * g * d.T_turbine;
    if (p.layout == DriveLayout::AWD) {
        const Real T_vc = p.centre_viscous * (d.w_c[0] - d.w_c[1]);
        d.T_axle = {{b * d.T_gearbox - T_vc, (1.0 - b) * d.T_gearbox + T_vc}};
    } else {
        d.T_axle[static_cast<std::size_t>(axles[0])] = d.T_gearbox;
    }

    // Each driven axle: its half-shafts, its carrier and spider.
    for (int a : axles) {
        const std::size_t k = static_cast<std::size_t>(a);
        std::array<Real, 2> T{};
        for (int side = 0; side < 2; ++side) {
            const int c = 2 * a + side;
            const std::size_t ci = static_cast<std::size_t>(c);
            const Real w_side = d.w_c[k] + (side == 0 ? 1.0 : -1.0) * d.w_s[k];
            const int it = twist_index(c);
            d.twist_rate[ci] = w_side - wheel_rate_[ci];
            Real torque = p.shaft_stiffness * x(it) + p.shaft_damping * d.twist_rate[ci];
            if (iso) {
                const Real z = x(it + 2);
                torque += p.isolator.stiffness * z;
                d.z_rate[ci] = p.isolator.rate(z, d.twist_rate[ci]);
            }
            d.T_shaft[ci] = torque;
            T[static_cast<std::size_t>(side)] = torque;
        }
        d.wc_dot[k] = (d.T_axle[k] - T[0] - T[1]) / (p.carrier_inertia + 2.0 * p.side_gear_inertia + reflected(a));
        if (!locked) {
            Real tau = 0.0;   // resisting the side gears' relative turn, w_L - w_R = 2 w_s
            if (p.differential == Differential::LimitedSlip) {
                const Real rel = 2.0 * d.w_s[k];
                tau = p.lsd_viscous * rel
                    + (p.lsd_preload + p.lsd_ramp * std::abs(d.T_axle[k])) * std::tanh(rel / p.lsd_slip_speed);
            }
            d.ws_dot[k] = (-T[0] + T[1] - 2.0 * tau) / (2.0 * p.side_gear_inertia + p.spider_inertia);
        }
    }
}

void Powertrain::initial_states(kernel::VecRef x) const
{
    x.setZero();
    if (i_engine_ >= 0) x(i_engine_) = params.engine.idle_speed;
}

void Powertrain::begin_evaluation(const kernel::Simulator&, Real t, kernel::ConstVecRef x)
{
    x_ = x;
    t_ = t;
}

void Powertrain::apply_torques(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const
{
    for (int c = 0; c < 4; ++c) wheel_rate_[static_cast<std::size_t>(c)] = driven(c) ? spin(states, c) : 0.0;
    Derived d;
    evaluate(states, x_, t_, d);
    engine_torque = d.T_engine;
    turbine_torque = d.T_turbine;
    gearbox_torque = d.T_gearbox;
    shaft_torque = d.T_shaft;

    // Each half-shaft's torque on its wheel, about the wheel's axle, and its
    // reaction on the differential's housing.
    for (int c = 0; c < 4; ++c) {
        if (!driven(c)) continue;
        const Corner& k = corners_[static_cast<std::size_t>(c)];
        const Vec3 axis = states[static_cast<std::size_t>(k.wheel)].q_WB * Vec3::UnitY();
        const Vec3 T = d.T_shaft[static_cast<std::size_t>(c)] * axis;
        forces[static_cast<std::size_t>(k.wheel)].tau_W += T;
        forces[static_cast<std::size_t>(k.housing)].tau_W -= T;
    }
    // The hidden parts (engine, carriers, side gears) turn about the
    // chassis's lateral axis: the rate of their angular momentum is drawn
    // from the chassis.
    const PowertrainParams& p = params;
    Real L_dot = 0.0;
    if (i_engine_ >= 0) L_dot += p.engine.inertia * d.we_dot;
    for (int a : driven_axles(p.layout)) {
        const std::size_t k = static_cast<std::size_t>(a);
        L_dot += (p.carrier_inertia + 2.0 * p.side_gear_inertia + reflected(a)) * d.wc_dot[k];
    }
    const Vec3 lateral = states[static_cast<std::size_t>(chassis_)].q_WB * Vec3::UnitY();
    forces[static_cast<std::size_t>(chassis_)].tau_W -= L_dot * lateral;
    hidden_momentum_rate_ = L_dot * lateral;
}

void Powertrain::rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot)
{
    Derived d;
    evaluate(sim.states(), x, t, d);
    if (i_engine_ >= 0) x_dot(i_engine_) = d.we_dot;
    const bool locked = params.differential == Differential::Locked;
    const bool iso = params.isolator.stiffness > 0.0;
    for (int a : driven_axles(params.layout)) {
        const std::size_t k = static_cast<std::size_t>(a);
        x_dot(i_axle_[k]) = d.wc_dot[k];
        if (!locked) x_dot(i_axle_[k] + 1) = d.ws_dot[k];
        for (int side = 0; side < 2; ++side) {
            const int c = 2 * a + side;
            x_dot(twist_index(c)) = d.twist_rate[static_cast<std::size_t>(c)];
            if (iso) x_dot(twist_index(c) + 2) = d.z_rate[static_cast<std::size_t>(c)];
        }
    }
}

void Powertrain::end_step(const kernel::Simulator&, Real t, kernel::VecRef x)
{
    if (shift_start_ >= 0.0) {
        const Real u = (t - shift_start_) / params.shift_time;
        if (u >= 0.5 && gear != shift_to_) gear = shift_to_;   // the ratio changes midway, the torque at zero
        if (u >= 1.0) shift_start_ = -1.0;
        return;
    }
    if (!params.automatic) return;
    Derived d;
    evaluate({}, x, t, d);
    const int n = static_cast<int>(params.gear_ratios.size());
    if (d.w_e > params.shift_up_speed && gear < n) {
        shift_start_ = t;
        shift_to_ = gear + 1;
    } else if (d.w_e < params.shift_down_speed && gear > 1) {
        shift_start_ = t;
        shift_to_ = gear - 1;
    }
}

void Powertrain::connect(kernel::Simulator& sim, BodyIndex chassis, const std::array<Corner, 4>& corners)
{
    chassis_ = chassis;
    corners_ = corners;
    std::vector<BodyIndex> bodies{chassis};
    for (const Corner& c : corners) {
        bodies.push_back(c.wheel);
        bodies.push_back(c.housing);
    }
    sim.attach(*this);
    sim.system.force_elements.push_back(std::make_shared<PowertrainTorques>(*this, std::move(bodies)));
    sim_ = &sim;

    // The states: the driveline at the wheels' spins, the engine at idle or
    // the speed they give, if higher; untwisted shafts.
    std::size_t me = 0;
    while (sim.state_elements()[me] != this) ++me;
    auto x = sim.x.segment(sim.state_offset(me), static_cast<Index>(n_));
    x.setZero();
    for (int c = 0; c < 4; ++c) wheel_rate_[static_cast<std::size_t>(c)] = driven(c) ? spin(sim.states(), c) : 0.0;
    for (int a : driven_axles(params.layout)) {
        const std::size_t k = static_cast<std::size_t>(a);
        const Real L = wheel_rate_[2 * k], R = wheel_rate_[2 * k + 1];
        x(i_axle_[k]) = 0.5 * (L + R);
        if (params.differential != Differential::Locked) x(i_axle_[k] + 1) = 0.5 * (L - R);
    }
    Derived d;
    gear = 1;
    evaluate(sim.states(), x, sim.time, d);
    if (i_engine_ >= 0) x(i_engine_) = std::max(params.engine.idle_speed, d.w_t);
    x_ = x;
    t_ = sim.time;
}

void Powertrain::connect(kernel::Simulator& sim, const VehicleHandle& vh)
{
    std::array<Corner, 4> corners{};
    for (std::size_t c = 0; c < 4; ++c) {
        const CornerHandle& h = vh.corners[c];
        corners[c] = {h.wheel_body, h.upright_body,
                      h.type == SuspensionType::SolidAxle ? h.upright_body : vh.chassis_body};
    }
    connect(sim, vh.chassis_body, corners);
}

} // namespace mbd
