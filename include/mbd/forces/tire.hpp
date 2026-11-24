#pragma once

// Tire vertical contact force model.
//
// Monitors a contact point on a wheel body. When the contact point is below
// the road, applies a spring-damper force along the road's normal there. The
// tire only pushes (lifts off when no contact).
//
// The road (mbd/forces/road.hpp, task 7.2) is the tyre's only source of the
// ground: at the contact point's (x, y) it gives the height h, the normal n
// and the friction. The penetration is the contact point's depth below the
// road's tangent plane there, (h - z) n_z.
//
// This is L2.3.1: the simplest tire model — vertical forces only,
// no slip, no lateral dynamics. Suitable for quarter-car validation,
// ride comfort analysis, and as the foundation for more advanced models.

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/pacejka.hpp"
#include "mbd/forces/road.hpp"

#include <memory>

namespace mbd {

class TireContactForce : public ForceElement {
public:
    BodyIndex wheel_body_idx;

    /// Contact point in the wheel body frame: (0, 0, -R_free), the bottom of
    /// a wheel whose origin is at the axle centre and whose Z is up.
    Vec3 contact_point_B;

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
// Wheel body frame convention (ISO 8855, docs/conventions.md):
//   X: forward (travel direction)
//   Y: to the left (the spin axis)
//   Z: up
//   Contact point at (0, 0, -R_free) in body frame.
// The tyre's lateral direction is the ground normal crossed with its heading,
// to the left; a positive slip angle gives a positive lateral force, to the
// left.

class FullTireForce : public ForceElement {
public:
    BodyIndex wheel_body_idx;

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

    // Wheel spin rate (set externally, e.g. by drivetrain model)
    // For free-rolling, set to Vx / R_eff after each step.
    Real omega_wheel{0.0};

    /// When true, omega_wheel is computed automatically from forward velocity
    /// (free-rolling assumption: zero drive/brake torque). When false, the
    /// externally-set omega_wheel value is used.
    bool auto_free_roll{true};
    /// Steering angle applied to this tire [rad], about the wheel's Z axis.
    /// Positive = the heading turns to the left (counter-clockwise from above).
    Real steer_angle{0.0};

    // --- Cached output (updated each apply() call) ---
    mutable TireForceResult last_result;
    mutable Real last_Fz{0.0};
    mutable Real last_deflection{0.0};
    mutable Vec3 last_contact_pos_W{Vec3::Zero()};
    mutable Vec3 last_forward_W{Vec3::UnitX()};
    mutable Vec3 last_lateral_W{Vec3::UnitY()};
    mutable Real last_V_abs{1.0};   ///< Speed used to normalize slip [m/s]
    mutable Real last_R_eff{0.0};   ///< Rolling radius used for slip [m]
    mutable Real last_friction{1.0};  ///< The road's friction at the contact [-]

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

    // --- Telemetry accessors ---

    Real get_vertical_force() const { return last_Fz; }
    Real get_deflection() const { return last_deflection; }
    Real get_slip_ratio() const { return last_result.kappa; }
    Real get_slip_angle() const { return last_result.alpha; }
    Real get_Fx() const { return last_result.Fx; }
    Real get_Fy() const { return last_result.Fy; }
    const TireForceResult& get_last_result() const { return last_result; }

    /// Rolling radius: the lever arm of the longitudinal force about the
    /// wheel axis, and the radius that converts wheel spin into slip [m].
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
