#pragma once

// The powertrain (plan task 7.8, decision D69): an engine with its own speed
// and inertia, a torque converter or clutch (or an electric motor coupled
// directly), a gearbox with shift time and hysteresis, a centre
// differential for four-wheel drive, each driven axle's differential (open,
// locked or limited-slip) and compliant half-shafts, one of them optionally
// a hysteretic isolator (Bouc-Wen). A state element: its parts' speeds and
// the half-shafts' twists are states, integrated with the bodies; it acts on
// the bodies through the half-shafts' torques on the wheels, their
// reactions on the differential's housing (the chassis, or a solid axle),
// and the reaction of its hidden parts' angular momentum on the chassis.
//
// The states, in order: the engine's speed w_e (not with a direct coupling);
// then, for each driven axle (front first), its carrier's speed w_c, its
// spider's w_s (not when locked: the side gears turn at w_c +- w_s), the
// half-shafts' twists a_L, a_R, and with an isolator the Bouc-Wen variables
// z_L, z_R. The half-shafts' torques T = k a + c a' + k_h z, a' the side
// gear's speed less the wheel's spin.
//
// The equations (each part's inertia J, the gearbox's total ratio g = gear
// times final drive, its efficiency eta, the shift's torque factor s):
//   I_e w_e' = T_engine(w_e, throttle) - T_pump,
//   the gearbox's input speed w_t = g w_in, w_in the axles' carriers' speed
//     (with a centre differential of front share b: b w_cf + (1 - b) w_cr),
//   its output torque T_g = s eta g T_turbine, to the carriers (b T_g and
//     (1 - b) T_g),
//   (J_c + 2 J_g) w_c' = T_axle - T_L - T_R,
//   (2 J_g + J_s) w_s' = -T_L + T_R + T_lock,
// the side gears of inertia J_g each, the spider's J_s. The coupling:
//   torque converter: T_pump = w_e (w_e - w_t) / K^2 (K the capacity factor
//     at stall, so T_pump = (w_e / K)^2 there), T_turbine = TR(SR) T_pump,
//     the torque ratio falling linearly from TR_0 at the stall to 1 at the
//     coupling point SR_c, SR = w_t / w_e;
//   clutch: T_pump = T_turbine = engagement T_max tanh((w_e - w_t) / w_slip);
//   direct (an electric motor): the motor's torque at w_t, its inertia
//     carried by the carriers as I_m g^2 (shared as the torque is).

#include <array>
#include <memory>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/hysteresis.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/state_element.hpp"
#include "mbd/vehicle/drivetrain_params.hpp"

namespace mbd {

struct VehicleHandle;

/// An engine's or a motor's torque map: at full and at no throttle against
/// its speed [N m against rad/s], blended linearly in the throttle; zero
/// throttle gives the drag (engine braking, negative). Above the redline the
/// fuel is cut (the drag alone); below idle an idle controller adds up to
/// the full-load torque to hold idle_speed.
struct EngineMap {
    Curve full_load;
    Curve drag;
    Real inertia{0.2};             ///< [kg m^2]
    Real idle_speed{104.72};       ///< [rad/s] (1000 rpm); zero for a motor
    Real redline_speed{733.04};    ///< [rad/s] (7000 rpm)
    Real idle_gain{5.0};           ///< The idle controller's torque per rad/s below idle [N m s]
    /// A power limit [W]: where the full-load torque exceeds max_power /
    /// |w| (a motor's constant-power range), it and the drag are scaled down
    /// to it; zero: none.
    Real max_power{0.0};

    /// The torque at speed w and throttle in [0, 1].
    Real torque(Real w, Real throttle) const;

    /// The template's engine as a map: its piecewise full-load curve through
    /// idle, peak and redline, a drag of 5 % of the peak torque at idle
    /// rising to 12 % at the redline.
    static EngineMap from(const EngineParams& p);

    /// An electric motor: its peak torque up to its base speed, then
    /// constant power to its top speed; regenerative drag of `regen` times
    /// the full-load torque at zero throttle.
    static EngineMap motor(Real peak_torque, Real base_speed, Real top_speed, Real inertia, Real regen = 0.0);
};

enum class Coupling { TorqueConverter, Clutch, Direct };
enum class Differential { Open, Locked, LimitedSlip };

struct PowertrainParams {
    EngineMap engine;
    Coupling coupling{Coupling::TorqueConverter};
    Real converter_capacity{12.0};       ///< K at stall [rad/s / sqrt(N m)]
    Real converter_stall_ratio{2.0};     ///< TR_0
    Real converter_coupling_point{0.85}; ///< SR_c
    Real clutch_capacity{600.0};         ///< T_max [N m]
    Real clutch_slip_speed{2.0};         ///< w_slip [rad/s]

    std::vector<Real> gear_ratios{{3.5, 2.3, 1.7, 1.3, 1.0, 0.8}};
    Real final_drive{3.5};
    Real efficiency{0.92};
    bool automatic{true};                ///< Shifts by the engine's speed; otherwise `gear` is the driver's
    Real shift_up_speed{680.7};          ///< [rad/s] (6500 rpm)
    Real shift_down_speed{209.4};        ///< [rad/s] (2000 rpm): the gap is the hysteresis
    Real shift_time{0.3};                ///< [s]: the torque falls to zero and back, the ratio changing midway

    DriveLayout layout{DriveLayout::RWD};
    Real front_share{0.4};               ///< The centre differential's torque to the front (AWD)
    Real centre_viscous{0.0};            ///< A viscous coupling across it [N m s]

    Differential differential{Differential::Open};
    Real lsd_viscous{0.0};               ///< Limited slip: viscous [N m s] ...
    Real lsd_preload{0.0};               ///< ... and a clutch pack's preload [N m] ...
    Real lsd_ramp{0.0};                  ///< ... plus this share of the axle's input torque
    Real lsd_slip_speed{1.0};            ///< Its clutch's smoothing speed [rad/s]

    /// The gearbox's input side (the turbine or the clutch's disc, the
    /// input shaft) [kg m^2]: carried by the carriers as J_in g^2 (shared as
    /// the torque is). With it a torque converter's or clutch's coupling,
    /// brought through the ratio to the carriers, stays within an explicit
    /// method's reach.
    Real input_inertia{0.05};
    Real carrier_inertia{0.05};          ///< J_c [kg m^2]
    Real side_gear_inertia{0.005};       ///< J_g
    Real spider_inertia{0.002};          ///< J_s
    Real shaft_stiffness{1.0e4};         ///< k [N m/rad]
    Real shaft_damping{20.0};            ///< c [N m s/rad]
    /// A hysteretic isolator in each half-shaft (D42): Bouc-Wen on the twist
    /// (stiffness zero: none).
    BoucWen isolator;
};

class Powertrain : public kernel::StateElement {
public:
    struct Corner {
        BodyIndex wheel{0};     ///< The wheel, on its spin joint
        BodyIndex upright{0};   ///< The body it turns on
        BodyIndex housing{0};   ///< Where the half-shaft's torque reacts: the chassis, or a solid axle
    };

    PowertrainParams params;
    Real throttle{0.0};          ///< 0..1
    Real clutch{1.0};            ///< Engagement 0..1 (Coupling::Clutch)
    int gear{1};                 ///< 1-indexed; chosen by the gearbox when automatic

    // Telemetry at the last evaluation.
    mutable Real engine_torque{0.0};
    mutable Real turbine_torque{0.0};
    mutable Real gearbox_torque{0.0};   ///< At its output, the shift's factor included [N m]
    mutable std::array<Real, 4> shaft_torque{{0.0, 0.0, 0.0, 0.0}};   ///< On each wheel (FL, FR, RL, RR) [N m]

    explicit Powertrain(PowertrainParams p);

    /// Attach to a template-built vehicle's simulator: the force element that
    /// applies the torques goes into its system, the powertrain is attached
    /// as a state element; the engine starts at idle (or at the speed the
    /// wheels give, if higher), the driveline at the wheels' spins, the
    /// half-shafts untwisted. The powertrain must outlive the simulator.
    void connect(kernel::Simulator& sim, const VehicleHandle& vh);
    /// The same with the bodies given: the chassis, and the four corners.
    void connect(kernel::Simulator& sim, BodyIndex chassis, const std::array<Corner, 4>& corners);

    bool driven(int corner) const;
    Real total_ratio() const;
    bool shifting() const { return shift_start_ >= 0.0; }

    // --- As a state element ---
    const char* name() const override { return "powertrain"; }
    int size() const override { return static_cast<int>(n_); }
    void initial_states(kernel::VecRef x) const override;
    void begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x) override;
    void rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot) override;
    void end_step(const kernel::Simulator& sim, Real t, kernel::VecRef x) override;

    /// The torques at the bodies' states and the states of the last
    /// begin_evaluation: each half-shaft's on its wheel and its reaction on
    /// the housing, and the hidden parts' angular momentum's reaction on the
    /// chassis. Called by the force element connect() adds.
    void apply_torques(const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) const;

    /// Each state's index in the element's states (-1: absent).
    int engine_index() const { return i_engine_; }
    int carrier_index(int axle) const { return i_axle_[static_cast<std::size_t>(axle)]; }
    int spider_index(int axle) const;
    int twist_index(int corner) const;

private:
    struct Derived;   // the speeds and torques of one evaluation
    void evaluate(const std::vector<RigidBodyState>& states, kernel::ConstVecRef x, Real t, Derived& d) const;
    Real spin(const std::vector<RigidBodyState>& states, int corner) const;
    Real shift_factor(Real t) const;
    Real reflected(int axle) const;

    BodyIndex chassis_{0};
    std::array<Corner, 4> corners_{};
    std::size_t n_{0};
    int i_engine_{-1};
    std::array<int, 2> i_axle_{{-1, -1}};    // each driven axle's first state (its carrier)
    VecX x_;                                  // the states of the last begin_evaluation
    Real t_{0.0};
    mutable std::array<Real, 4> wheel_rate_{{0.0, 0.0, 0.0, 0.0}};   // spins at the last apply_torques
    mutable Vec3 hidden_momentum_rate_{Vec3::Zero()};
    Real shift_start_{-1.0};
    int shift_to_{0};
    kernel::Simulator* sim_{nullptr};
};

} // namespace mbd
