#pragma once

// Tyre forces on a wheel body.
//
// The wheel (plan task 7.3, D62) is a body whose Y axis is its spin axis, its
// origin at the axle's centre. Where it meets the road is found from the
// wheel plane and the road, not fixed in the body: the wheel's lowest point is
// the point of its circle (the free radius about the centre, in the plane
// normal to the spin axis) furthest along the road's normal, below the
// centre; the contact point is that point brought onto the road's tangent
// plane. A wheel that pitches about its axle, as an upright does under
// braking, does not move its contact; a wheel that leans (camber) moves it
// along its plane.
//
// The road (mbd/forces/road.hpp, task 7.2) is the tyre's only source of the
// ground: at the lowest point's (x, y) it gives the height h, the normal n and
// the friction. The penetration is the lowest point's depth below the road's
// tangent plane there, (h - z) n_z.
//
// TireContactForce is the vertical part alone (quarter cars, ride);
// FullTireForce adds the Magic Formula's forces and moments, lagged (task 7.4).

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/magic_formula.hpp"
#include "mbd/forces/road.hpp"
#include "mbd/kernel/state_element.hpp"

#include <memory>

namespace mbd {

/// Where a wheel meets the road, and the road's frame there (task 7.3).
struct WheelContact {
    Vec3 point;        ///< On the road's tangent plane, below the lowest point along the normal [m]
    Vec3 lowest;       ///< The wheel's lowest point: centre + free radius towards the road, in the wheel plane [m]
    Vec3 normal;       ///< The road's unit normal there
    Vec3 forward;      ///< The tyre's heading: the wheel plane's line in the road plane, a x n normalized
    Vec3 lateral;      ///< n x forward: to the left of the heading
    Real penetration;  ///< The lowest point's depth below the road's tangent plane [m]; positive in contact
    Real friction;     ///< The road's friction there (a scale on the tyre's own)
};

/// The contact of a wheel with the road: its centre `centre` and unit spin
/// axis `spin_axis` in the world, its free radius, and `steer`, an angle by
/// which the wheel plane is turned about the road's normal (positive to the
/// left; the simple corners' steering, which turns the tyre and not a body).
///
/// The road is asked twice: under the centre, for the normal that gives the
/// direction from the centre to the lowest point (in the wheel plane, against
/// the normal); then at the lowest point, for the height, the normal and the
/// friction there. On a plane the two normals agree and the point is exact.
/// Throws MBD-F051 if the road is null.
WheelContact wheel_contact(const std::shared_ptr<const Road>& road, const Vec3& centre,
                           const Vec3& spin_axis, Real free_radius, Real steer = 0.0);

class TireContactForce : public ForceElement {
public:
    BodyIndex wheel_body_idx;  ///< The wheel: origin at the axle centre, Y its spin axis

    Real k_z;        ///< Vertical stiffness [N/m]
    Real c_z;        ///< Vertical damping [N·s/m]
    Real R_free;     ///< Unloaded tire radius [m]
    std::shared_ptr<const Road> road;  ///< The ground (never null)

    /// On `on_road`; a level road at height 0 if none is given.
    TireContactForce(BodyIndex wheel_idx,
                     Real free_radius,
                     Real stiffness,
                     Real damping,
                     std::shared_ptr<const Road> on_road = nullptr);

    const char* name() const override { return "tyre contact"; }
    std::vector<BodyIndex> bodies() const override { return {wheel_body_idx}; }

    /// The normal force k_z depth - c_z (rate of approach), pushing only,
    /// along the road's normal at the contact point.
    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// Current tire deflection (positive = compressed). Returns 0 if no contact.
    Real get_deflection(const std::vector<RigidBodyState>& states) const;

    /// Current vertical force. Returns 0 if no contact.
    Real get_vertical_force(const std::vector<RigidBodyState>& states) const;
};

// ============================================================================
// Full tyre force element: the vertical contact and the Magic Formula, lagged
// ============================================================================
//
// The tyre of tasks 7.4 and 7.5 (D63, D64, D65). Its model
// (mbd/forces/tyre_model.hpp) gives everything: the Magic Formula of a
// parameter set (mbd/forces/magic_formula.hpp, read from a .tir file by
// mbd/io/tir_file.hpp), or a simple tyre (mbd/forces/simple_tyres.hpp). Its
// body() gives the free radius, the vertical stiffness and damping of the
// contact above and the low speed; its evaluate() the forces in the road
// plane and the moments. They act at the contact point, in the tyre's W axes
// there: x the heading, y to its left, z the road's normal (ISO).
//
// The wheel (ISO 8855, docs/conventions.md): its origin at the axle centre,
// its Y the spin axis (to the left on a car), a forward roll a positive
// rotation about it. Its spin Omega is its angular velocity about that axis;
// the contact point's velocity is the wheel's without that spin. With Vx and
// Vy that velocity along the heading and to the left, and R_eff = R_free -
// depth / 2 the rolling radius, the slip speeds are Vsx = Vx - Omega R_eff
// and Vsy = Vy. The inclination is gamma = asin(a . n), a the spin axis and n
// the road's normal: positive when the wheel's top leans to the right of its
// heading.
//
// The slips lag behind the motion (Pacejka, 3rd ed., section 8.6, the
// semi-non-linear model): two states, the tread's deflections u and v,
//     du/dt = -Vsx - |Vx| u / sigma_kappa,    kappa' = u / sigma_kappa,
//     dv/dt =  Vsy - |Vx| v / sigma_alpha,    tan(alpha') = v / sigma_alpha,
// sigma the Magic Formula's relaxation lengths at the load (at least
// sigma_min). In a steady state kappa' = -Vsx / |Vx| and tan(alpha') = Vsy
// / |Vx|, the steady-state slips; at rest the deflections follow the
// contact's motion and the tyre holds like a spring, of stiffness K / sigma.
// Below V_low (the file's VXLOW) two terms keep this sane (8.112 to 8.114,
// 8.127, 8.128): a damping k_V_low, from k_V_low0 at rest to 0 at V_low,
// added to the slips as -k_V_low Vs / K; and a deflection that would grow
// past limit_factor times the slip of full sliding (3 D / K) is held. The
// Magic Formula's shifts fade out below VXLOW too (magic_formula.hpp), and
// the rolling resistance moment acts in full once the wheel rolls at V_roll,
// in proportion below it. Without lag (lag = false) the slips are the
// steady-state ones over max(|Vx|, V_low).
//
// The tyre is a state element too (mbd/kernel/state_element.hpp): a
// simulator made from a system that holds it integrates its two states.

class FullTireForce : public ForceElement, public kernel::StateElement {
public:
    BodyIndex wheel_body_idx;  ///< The wheel: origin at the axle centre, Y its spin axis

    std::shared_ptr<const TyreModel> tyre;   ///< The forces and moments (never null)

    Real R_free;               ///< Free radius [m]: the model's (UNLOADED_RADIUS)
    Real k_z;                  ///< Vertical stiffness [N/m]: the model's (VERTICAL_STIFFNESS)
    Real c_z;                  ///< Vertical damping [N s/m]: the model's (VERTICAL_DAMPING)
    std::shared_ptr<const Road> road;  ///< The ground (never null): height, normal, friction

    /// The angle by which the wheel plane is turned about the road's normal
    /// [rad], positive to the left: how a simple corner, whose wheel turns on
    /// an upright that does not steer, steers its tyre. Linkage corners steer
    /// their uprights and leave it at zero.
    Real steer_angle{0.0};

    /// True when the tyre is mounted on the other side of the vehicle than
    /// its parameters describe (their TYRESIDE): the formula is then evaluated
    /// mirrored, as TNO's MF-Tyre does: the slip angle and the inclination
    /// negated going in, the side force, the overturning and the aligning
    /// moments negated coming out. A car's left and right tyres from one
    /// file then pull against each other, not together.
    bool mirrored{false};

    /// Mount the tyre on the vehicle's right side (true) or its left: sets
    /// `mirrored`. The vehicle builders call it for every corner.
    void mount(bool right_side) { mirrored = right_side != (tyre->body().side == TyreSide::Right); }

    // --- The lag (set before a simulator is made) ---
    bool lag{true};            ///< Two deflection states; without, the steady-state slips
    Real V_low;                ///< [m/s] Below it, the low-speed damping and limit: the model's (VXLOW)
    Real k_V_low0{770.0};      ///< [N s/m] The low-speed damping at rest (Pacejka's Table 8.4)
    Real sigma_min{0.02};      ///< [m] The shortest relaxation length (Pacejka's Table 8.4)
    Real limit_factor{1.0};    ///< A of (8.112): the deflection is held past A times full sliding
    /// [m/s] The rolling resistance acts in full once the wheel rolls at
    /// this speed (Omega R_eff), and in proportion below it: dry friction,
    /// smoothed, so that a wheel at rest has none.
    Real V_roll{0.01};

    // --- Cached at each apply() ---
    mutable TyreOutput last_output;   ///< The model's outputs (zero off the road)
    mutable Real last_kappa{0.0};       ///< The longitudinal slip the Magic Formula took
    mutable Real last_alpha{0.0};       ///< The slip angle it took [rad], ISO: positive to the left
    mutable Real last_gamma{0.0};       ///< The inclination [rad]
    mutable Real last_Fz{0.0};
    mutable Real last_deflection{0.0};
    mutable Vec3 last_contact_pos_W{Vec3::Zero()};  ///< The contact point (on the road)
    mutable Vec3 last_forward_W{Vec3::UnitX()};
    mutable Vec3 last_lateral_W{Vec3::UnitY()};
    mutable Real last_Vx{0.0};          ///< The contact's speed along the heading [m/s]
    mutable Real last_Vsx{0.0};         ///< Longitudinal slip speed Vx - Omega R_eff [m/s]
    mutable Real last_Vsy{0.0};         ///< Lateral slip speed [m/s]
    mutable Real last_R_eff{0.0};       ///< Rolling radius used for slip [m]
    mutable Real last_friction{1.0};    ///< The road's friction at the contact [-]
    mutable Real last_spin{0.0};        ///< The wheel's spin about its axis [rad/s]
    mutable Real last_sigma_kappa{0.0}; ///< The relaxation lengths in use [m]
    mutable Real last_sigma_alpha{0.0};
    mutable Real last_My{0.0};          ///< The rolling resistance moment applied [N m]
    /// The moment of the tyre's forces and moments about the wheel's spin
    /// axis, through the centre [N m]: positive turns the wheel forward. Zero
    /// off the road.
    mutable Real last_axle_moment{0.0};

    /// The Magic Formula tyre of `parameters` on `wheel_idx`, on `on_road` (a
    /// level road at height 0 if none is given). Throws MBD-F011 for a free
    /// radius that is not positive or a negative stiffness or damping,
    /// MBD-F053 for a set the Magic Formula cannot evaluate, MBD-F054 for one
    /// without relaxation lengths.
    FullTireForce(BodyIndex wheel_idx,
                  const MagicFormulaParameters& parameters,
                  std::shared_ptr<const Road> on_road = nullptr);

    /// The tyre `model` on `wheel_idx` (task 7.5), on `on_road`. Throws
    /// MBD-F011 for a model that is null or whose body is not valid,
    /// MBD-F054 for one without relaxation lengths.
    FullTireForce(BodyIndex wheel_idx,
                  std::shared_ptr<const TyreModel> model,
                  std::shared_ptr<const Road> on_road = nullptr);

    const char* name() const override { return "tyre"; }
    std::vector<BodyIndex> bodies() const override { return {wheel_body_idx}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    // --- The lag, as a state element ---
    int size() const override { return lag ? 2 : 0; }
    void initial_states(kernel::VecRef x) const override { x.setZero(); }
    void begin_evaluation(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x) override;
    void rates(const kernel::Simulator& sim, Real t, kernel::ConstVecRef x, kernel::VecRef x_dot) override;

    /// The deflections the next apply() takes [m], as begin_evaluation hands
    /// them: for calling apply() without a simulator.
    void set_deflections(Real u, Real v)
    {
        u_ = u;
        v_ = v;
    }
    Real deflection_u() const { return u_; }   ///< Longitudinal [m]
    Real deflection_v() const { return v_; }   ///< Lateral [m]

    /// The spin at which the wheel would roll without longitudinal slip in
    /// the state `states` [rad/s]: Vx / R_eff, the contact point's speed
    /// along the heading over the rolling radius at the present deflection.
    Real free_rolling_spin(const std::vector<RigidBodyState>& states) const;

    // --- Telemetry accessors ---

    Real get_vertical_force() const { return last_Fz; }
    Real get_deflection() const { return last_deflection; }
    Real get_slip_ratio() const { return last_kappa; }
    Real get_slip_angle() const { return last_alpha; }
    Real get_Fx() const { return last_output.Fx; }
    Real get_Fy() const { return last_output.Fy; }
    Real get_Mz() const { return last_output.Mz; }

    /// Rolling radius: the radius that converts wheel spin into slip [m],
    /// R_free - deflection / 2.
    Real get_rolling_radius() const { return R_free - last_deflection * 0.5; }

private:
    Real u_{0.0}, v_{0.0};             // the deflections of this evaluation
    Real sigma_kappa_nominal_{0.0};    // the relaxation lengths at FNOMIN: off the road
    Real sigma_alpha_nominal_{0.0};
    mutable bool in_contact_{false};
    mutable Real kappa_sliding_{0.0};  // 3 D / K: the slips of full sliding
    mutable Real alpha_sliding_{0.0};
};

} // namespace mbd
