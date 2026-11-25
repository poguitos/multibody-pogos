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
// FullTireForce adds the Pacejka slip forces.

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/pacejka.hpp"
#include "mbd/forces/road.hpp"

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
// Full tire force element: vertical contact + Pacejka slip forces
// ============================================================================
//
// Combines vertical spring-damper contact with Pacejka Magic Formula
// lateral and longitudinal forces computed from slip kinematics.
//
// The wheel body (ISO 8855, docs/conventions.md): its origin at the axle
// centre, its Y the spin axis (to the left on a car), a forward roll a
// positive rotation about it. The tyre's heading is the wheel plane's line in
// the road plane, its lateral direction to the left; a positive slip angle
// gives a positive lateral force, to the left.
//
// The slips (task 7.3): the wheel's spin Omega is its angular velocity about
// its own axis, as the body turns; the contact point's velocity is the
// wheel's without that spin, at the contact point. Then
//     kappa = (Omega R_eff - Vx) / max(|Vx|, V_low),
//     alpha = -atan(Vy / max(|Vx|, V_low)),
// with Vx and Vy that velocity along the heading and to the left, and R_eff
// the rolling radius. The forces act at the contact point, so they turn the
// wheel about its axle with the arm from the centre to the road.

class FullTireForce : public ForceElement {
public:
    BodyIndex wheel_body_idx;  ///< The wheel: origin at the axle centre, Y its spin axis

    // Tire geometry
    Real R_free;          ///< Unloaded radius [m]
    Real R_loaded_approx; ///< Approximate loaded radius for effective radius calc

    // Vertical contact
    Real k_z;             ///< Vertical stiffness [N/m]
    Real c_z;             ///< Vertical damping [Ns/m]
    std::shared_ptr<const Road> road;  ///< The ground (never null): height, normal, friction

    // Pacejka model
    PacejkaTire pacejka;

    // Low-speed threshold to avoid singularity in slip computation
    Real V_low{1.0};     ///< [m/s] — below this speed, slip is clamped

    /// The angle by which the wheel plane is turned about the road's normal
    /// [rad], positive to the left: how a simple corner, whose wheel turns on
    /// an upright that does not steer, steers its tyre. Linkage corners steer
    /// their uprights and leave it at zero.
    Real steer_angle{0.0};

    // --- Cached output (updated each apply() call) ---
    mutable TireForceResult last_result;
    mutable Real last_Fz{0.0};
    mutable Real last_deflection{0.0};
    mutable Vec3 last_contact_pos_W{Vec3::Zero()};  ///< The contact point (on the road)
    mutable Vec3 last_forward_W{Vec3::UnitX()};
    mutable Vec3 last_lateral_W{Vec3::UnitY()};
    mutable Real last_V_abs{1.0};   ///< Speed used to normalize slip [m/s]
    mutable Real last_R_eff{0.0};   ///< Rolling radius used for slip [m]
    mutable Real last_friction{1.0};  ///< The road's friction at the contact [-]
    mutable Real last_spin{0.0};    ///< The wheel's spin about its axis [rad/s]
    /// The moment of the tyre's force about the wheel's spin axis, through
    /// the centre [N m]: positive turns the wheel forward. Zero off the road.
    mutable Real last_axle_moment{0.0};

    FullTireForce(BodyIndex wheel_idx,
                  Real free_radius,
                  Real vert_stiffness,
                  Real vert_damping,
                  const PacejkaTireParams& tire_params,
                  std::shared_ptr<const Road> on_road = nullptr);

    const char* name() const override { return "tyre"; }
    std::vector<BodyIndex> bodies() const override { return {wheel_body_idx}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// The spin at which the wheel would roll without longitudinal slip in
    /// the state `states` [rad/s]: Vx / R_eff, the contact point's speed
    /// along the heading over the rolling radius at the present deflection.
    Real free_rolling_spin(const std::vector<RigidBodyState>& states) const;

    // --- Telemetry accessors ---

    Real get_vertical_force() const { return last_Fz; }
    Real get_deflection() const { return last_deflection; }
    Real get_slip_ratio() const { return last_result.kappa; }
    Real get_slip_angle() const { return last_result.alpha; }
    Real get_Fx() const { return last_result.Fx; }
    Real get_Fy() const { return last_result.Fy; }
    const TireForceResult& get_last_result() const { return last_result; }

    /// Rolling radius: the radius that converts wheel spin into slip [m],
    /// R_free - deflection / 2.
    Real get_rolling_radius() const { return R_free - last_deflection * 0.5; }

    /// How fast the longitudinal force grows with wheel spin, dFx/d(omega)
    /// [N s/rad], at the state of the last apply() call.
    ///
    ///   Fx depends on omega through the slip ratio,
    ///   kappa = (omega * R_eff - Vx) / V_abs, so
    ///   dFx/d(omega) = dFx/d(kappa) * R_eff / V_abs.
    ///
    /// Zero when the tyre is off the ground. Past the friction peak, where
    /// the force falls as slip grows, zero is returned instead of a negative
    /// value: callers use this to treat the stiff, stable part of the tyre
    /// implicitly, and must not be handed a destabilizing term.
    Real longitudinal_force_slope() const;
};

} // namespace mbd
