#pragma once

// Aerodynamic forces and moments on the vehicle chassis (plan task 7.10,
// decision D71).
//
// The forces and moments come from a map (AeroMap) of six coefficients over
// five variables: the front and rear ride heights, the pitch, the roll and
// the aerodynamic yaw angle. In the chassis's axes (ISO 8855: x forward, y
// left, z up) the force is q (A_x, A_y, A_z) and the moment about the
// reference point q (B_x, B_y, B_z), where
//
//     q = rho V^2 / 2,
//
// V the reference point's velocity relative to the air (the wind's taken
// away), its component along the ground's normal removed. A drag area CdA
// appears as A_x = -CdA, a downforce area ClA as A_z = -ClA. The variables:
//
//   - ride heights: the heights of two points fixed in the chassis (front,
//     rear) above the road under them, or, without a road, along the ground
//     normal n above the plane through the world's origin;
//   - pitch theta = asin(-x_B . n), positive nose down; roll phi =
//     atan2(y_B . n, z_B . n), positive left side up (ISO 8855's angles of
//     the chassis to the ground plane);
//   - yaw beta = atan2(V . y_B, V . x_B): the angle of the velocity relative
//     to the air from the chassis's x axis, positive when the air comes from
//     the left (ISO 8855's sideslip angle, with the air's velocity).
//
// Between the map's nodes the coefficients are multilinear; beyond its first
// and last node on an axis they are held. An axis with one node is one the
// coefficients do not depend on.
//
// Without a map, the element uses the constant model it had before: a drag
// area, a downforce area that grows by dClA_dh below the ride
// height h_ref, both acting at a centre of pressure; AeroMap::from gives it
// as a map. That model's drag now acts along the chassis's x axis and its
// downforce along its z axis, rather than along the velocity and the ground
// normal (D71).

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"

namespace mbd {

class Road;
struct AeroParams;

/// The map's five variables, in this order.
enum class AeroVariable { FrontHeight = 0, RearHeight = 1, Pitch = 2, Roll = 3, Yaw = 4 };

/// Six coefficients over the five variables (front and rear ride height [m],
/// pitch, roll, yaw [rad]): (A_x, A_y, A_z) [m^2], the force over q in the
/// chassis's axes, then (B_x, B_y, B_z) [m^3], the moment over q about the
/// reference point.
struct AeroMap {
    /// Each variable's nodes, strictly increasing; one node: no dependence.
    std::array<std::vector<Real>, 5> nodes{{{0.0}, {0.0}, {0.0}, {0.0}, {0.0}}};
    /// One set of coefficients per node of the grid, the first variable's
    /// index changing fastest: index = i0 + n0 (i1 + n1 (i2 + n2 (i3 + n3 i4))).
    std::vector<Vec6> values;

    /// The same coefficients everywhere.
    static AeroMap constant(const Vec6& coefficients);
    /// The constant model's settings as a map: one node, or, when dClA_dh is
    /// not zero, two on the front ride height (10 m below h_ref, and h_ref).
    static AeroMap from(const AeroParams& p);

    /// The coefficients at x (front height, rear height, pitch, roll, yaw):
    /// multilinear between nodes, held beyond the first and last.
    Vec6 at(const std::array<Real, 5>& x) const;

    /// MBD-F060 unless every variable has finite, strictly increasing nodes
    /// and there is one finite set of coefficients per node of the grid.
    void validate() const;
};

struct AeroParams {
    // The constant model, used when no map is given.
    Real CdA{0.7};      ///< Drag area [m^2]. Typical sedan: 0.6-0.8. Sports: 0.4-0.6. Race: 0.8-1.5
    Real ClA{0.0};      ///< Downforce area [m^2]. Positive = downforce. Sedan: ~0. Race: 1-4.
    /// Centre of pressure, from the chassis's origin, chassis axes [m].
    Vec3 cop_offset_chassis{Vec3::Zero()};
    /// Ride-height-dependent downforce: ClA + dClA_dh (h_ref - h) below
    /// h_ref, h the front ride height. dClA_dh = 0: none.
    Real h_ref{0.10};       ///< [m]
    Real dClA_dh{0.0};      ///< [m^2 / m], not negative

    /// When set, the forces and moments come from it (the constant model's
    /// settings are not read).
    std::shared_ptr<const AeroMap> map;

    Real air_density{1.225};  ///< [kg/m^3] at sea level
    /// The ground's unit normal, up, in world axes.
    Vec3 ground_normal_W{Vec3::UnitZ()};
    /// Where the velocity is taken and the map's moments are about, chassis
    /// axes [m].
    Vec3 reference_point{Vec3::Zero()};
    /// The points whose heights are the ride heights, chassis axes [m].
    Vec3 front_point{Vec3::Zero()};
    Vec3 rear_point{Vec3::Zero()};
    /// The road the ride heights are measured from (vertically, world Z);
    /// none: the plane through the world's origin, along ground_normal_W.
    std::shared_ptr<const Road> road;
    /// The air's velocity at a point, world axes [m/s]; none: still air.
    std::function<Vec3(const Vec3& point_W)> wind;

    /// The constant model's coefficients at front ride height h.
    Vec6 constant_coefficients(Real h) const;
};

class AerodynamicForce : public ForceElement {
public:
    BodyIndex chassis_idx;
    AeroParams params;

    AerodynamicForce(BodyIndex chassis, const AeroParams& p);

    const char* name() const override { return "aerodynamics"; }
    std::vector<BodyIndex> bodies() const override { return {chassis_idx}; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override;

    /// What the element reads from a chassis state.
    struct Inputs {
        std::array<Real, 5> x{};   ///< front height, rear height, pitch, roll, yaw
        Real q{0.0};               ///< dynamic pressure [Pa]
    };
    Inputs inputs(const RigidBodyState& chassis) const;
};

} // namespace mbd
