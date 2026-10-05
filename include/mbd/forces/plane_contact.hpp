#pragma once

// Contact of points and spheres against a plane (plan task 3.8).
//
// A body carries contact points, each a sphere of radius r about a point of
// the body (r = 0 for a point). The plane is the XY plane of a frame fixed on
// a body (the ground by default), its normal the frame's Z axis, pointing to
// the side the spheres stay on. For each sphere, with h the height of its
// centre above the plane,
//
//     depth d = r - h,   contact point x = centre - r n,
//
// and when d > 0 the plane pushes back with a normal force
//
//     F_n = k d^e + c s(d / d_c) d_dot,      F_n >= 0,
//
// a penalty spring of stiffness k and exponent e (1 linear, 1.5 for Hertz's
// sphere) and a damper whose coefficient grows smoothly from zero at first
// touch to c at the depth d_c (s(x) = 3x^2 - 2x^3 up to 1), so that the force
// has no jump when contact begins. d_dot is the rate of penetration, from the
// velocities of the two material points at x. The plane never pulls.
//
// Friction is Coulomb's, regularised:
//
//     F_t = -mu F_n tanh(|u| / v_s) u / |u|,
//
// u the tangential slip velocity at x. Below the slip speed v_s friction
// grows linearly with u, a viscous approximation of sticking: a body that
// would stick creeps instead, at a speed of the order of v_s.
//
// Each law returns its force with its derivatives (decision D24).

#include <vector>

#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

struct PlaneContactParams {
    Real stiffness{1e5};          ///< k [N / m^e]
    Real exponent{1.0};           ///< e, at least 1
    Real damping{0.0};            ///< c [N s / m], reached at damping_depth
    Real damping_depth{1e-4};     ///< d_c [m]
    Real friction{0.0};           ///< mu
    Real slip_speed{1e-3};        ///< v_s [m/s]
};

/// A sphere of radius `radius` about the point `center_B` of the body (in its
/// frame); radius 0 is a point.
struct ContactSphere {
    Vec3 center_B{Vec3::Zero()};
    Real radius{0.0};
};

/// The state of one contact, for outputs and tests.
struct ContactState {
    bool touching{false};
    Real depth{0.0};            ///< r - h; positive in contact
    Real depth_rate{0.0};       ///< d(depth)/dt
    Vec3 point_W{Vec3::Zero()}; ///< centre - r n
    Vec3 normal_W{Vec3::Zero()};
    Real normal_force{0.0};
    Vec3 friction_W{Vec3::Zero()};  ///< Friction force on the body
    Real slip{0.0};             ///< |u|
};

class PlaneContact : public ForceElement {
public:
    /// The spheres of `body` against the XY plane of `plane_frame`, a frame on
    /// `plane_body` given in its body frame (0 and the identity: the world's
    /// XY plane, normal +Z).
    PlaneContact(BodyIndex body, std::vector<ContactSphere> spheres, PlaneContactParams params,
                 BodyIndex plane_body = 0, const Transform3& plane_frame = Transform3());

    const char* name() const override { return "plane contact"; }
    std::vector<BodyIndex> bodies() const override { return {body_, plane_body_}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// The normal force at depth d and penetration rate d_dot, with its
    /// derivatives; zero out of contact.
    LawValue normal_law(Real depth, Real depth_rate) const;

    /// The friction force's magnitude at slip speed |u| under normal force
    /// F_n, with its derivative with respect to |u| (in d_rate).
    LawValue friction_law(Real slip, Real normal_force) const;

    /// Energy stored by the penalty spring at depth d: k d^(e+1) / (e+1).
    Real potential_energy(Real depth) const;

    /// The state of sphere i at the given body states.
    ContactState contact(const std::vector<RigidBodyState>& states, std::size_t i) const;

    std::size_t size() const { return spheres_.size(); }
    const PlaneContactParams& params() const { return params_; }

private:
    BodyIndex body_;
    std::vector<ContactSphere> spheres_;
    PlaneContactParams params_;
    BodyIndex plane_body_;
    Transform3 plane_frame_;
};

} // namespace mbd
