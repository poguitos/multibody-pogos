#pragma once

// The HMMWV benchmark of Haug, Negrut, Serban and Solis (1999) (plan task
// 5.4): the fourteen-body model of the Army's high mobility multipurpose
// wheeled vehicle, driven at 10 mph over a half-cylinder bump 0.1 m across,
// and its stiff variant with bushings.
//
// From the paper (section 2 and its stiff example):
// - Fourteen bodies: the chassis, a steering rack, and at each corner a lower
//   and an upper control arm, each on a revolute joint to the chassis, and a
//   wheel spindle body on spherical joints to both arms. A tie rod, a distance
//   constraint, holds each spindle to the rack (front) or the chassis (rear).
//   The steering is an input, here held still: 10 degrees of freedom.
// - Tyres as translational spring-dampers: 296,325 N/m and 3,502 N s/m.
// - The stiff variant: each upper arm on a spherical joint instead, its two
//   extra rotations held by two spring-dampers of 2e7 N/m and 2e6 N s/m in
//   complementary directions, as bushings: 18 degrees of freedom.
// - Driven at 10 mph; the front wheels meet the bump at about 0.5 s, the rear
//   ones at about 1.2 s; 5 s in all; the chassis settles near 0.71 m.
//
// The paper gives no geometry, masses or suspension laws. They are taken
// from Project Chrono's HMMWV (BSD-3-Clause, projectchrono.org), a
// descendant of the same research group's model: data/vehicle/hmmwv
// (HMMWV_Chassis.json, HMMWV_DoubleWishboneFront.json and ...Rear.json,
// HMMWV_Wheel.json, HMMWV_RackPinion.json, tyre radius and mass from
// HMMWV_TMeasyTire.json and HMMWV_RigidTire.json) in Chrono 10.0.0. Its
// chassis centre of mass sits 0.709 m above the ground at the design
// position, the paper's 0.71 m. Where this model departs from Chrono's:
// - Each spindle body is Chrono's upright, spindle, wheel and tyre in one
//   (the paper's model has no wheel spin), their inertias added about the
//   common centre of mass with every part's axes taken as the vehicle's
//   (Chrono's upright is tilted by its kingpin lean; the arms' moments fit
//   the vehicle's axes as they are).
// - The control arms' moments of inertia, (0.4, 0.4, 0.8938) and (0.03,
//   0.03, 0.06276) kg m^2 in Chrono, break the triangle inequality (no body
//   has them; validate() says MBD-M014): taken as flat arms, Izz = Ixx + Iyy.
// - The springs follow Chrono's table, with free lengths that rest the arms
//   at their design angles (kFreeLength): Chrono's own lift the vehicle onto
//   its rebound stops. Chrono's stops (0.15 m and 0.30 / 0.35 m) are left
//   out; the shocks are the JSON files' linear dampers, without their
//   hydraulic stops. The springs and shocks stay inside those limits over
//   the runs (journal: Phase 5, the HMMWV benchmark).
// - The tie rods are massless (distance constraints, as in the paper).
// - A tyre is one spring-damper from the wheel centre towards the nearest
//   point of the terrain, flat ground or the bump, along the normal there,
//   pushing only; it carries no friction, so the vehicle coasts at its
//   initial speed but for what the bump's slope takes from it.

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/user_force.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/statics.hpp"

namespace mbd_test {

/// The road: flat ground at z = 0 and a half cylinder of radius
/// bump_radius lying across it, its axis along Y at x = bump_x.
struct HmmwvTerrain {
    mbd::Real bump_x{std::numeric_limits<mbd::Real>::infinity()};
    mbd::Real bump_radius{0.05};

    /// The distance from point c (above the ground) to the bump's surface;
    /// infinite without a bump.
    mbd::Real distance_to_bump(const mbd::Vec3& c) const
    {
        if (!std::isfinite(bump_x)) return std::numeric_limits<mbd::Real>::infinity();
        return std::hypot(c.x() - bump_x, c.z()) - bump_radius;
    }

    /// The distance from point c to the terrain, and the terrain's unit
    /// normal at the nearest point.
    mbd::Real distance(const mbd::Vec3& c, mbd::Vec3& normal) const
    {
        const mbd::Real bump = distance_to_bump(c);
        if (bump < c.z()) {
            normal = mbd::Vec3(c.x() - bump_x, 0.0, c.z()) / (bump + bump_radius);
            return bump;
        }
        normal = mbd::Vec3::UnitZ();
        return c.z();
    }
};

struct Hmmwv {
    using Real = mbd::Real;
    using Vec3 = mbd::Vec3;
    using Mat3 = mbd::Mat3;

    // --- Data ------------------------------------------------------------------

    static constexpr Real kSpeed = 4.4704;           ///< 10 mph [m/s]
    static constexpr Real kTyreRadius = 0.4699;      ///< Unloaded (HMMWV_TMeasyTire.json) [m]
    static constexpr Real kTyreStiffness = 296325.0; ///< The paper [N/m]
    static constexpr Real kTyreDamping = 3502.0;     ///< The paper [N s/m]
    static constexpr Real kBushingStiffness = 2.0e7; ///< The paper [N/m]
    static constexpr Real kBushingDamping = 2.0e6;   ///< The paper [N s/m]
    static constexpr Real kBushingLength = 0.1;      ///< The bushings' spring-dampers, free length [m]
    static constexpr Real kAxle = 1.688965;          ///< Axles at x = +-kAxle (HMMWV_Vehicle.json) [m]
    static constexpr Real kDesignHeight = kTyreRadius + 0.026;   ///< Chassis frame at design: wheel centres at kTyreRadius
    /// The springs' free lengths, front and rear [m]: those at which the
    /// vehicle rests on flat ground with its suspension at the design
    /// position (every arm's joint at zero), as the paper's model rests at
    /// its design height. Chrono's (0.339 and 0.382 m) lift the chassis
    /// 0.085 m and put the springs past their rebound stops.
    static constexpr std::array<Real, 2> kFreeLength{0.294734288, 0.332951200};

    /// One axle's double wishbone, left side, in the suspension frame (x
    /// forward, y left, z up; origin at the axle's centre), from Chrono's
    /// HMMWV_DoubleWishbone{Front,Rear}.json.
    struct Axle {
        Vec3 spindle, upright;                       // centres of mass; the wheel centre is the spindle's
        Vec3 uca_com, uca_front, uca_back, uca_upright;
        Vec3 lca_com, lca_front, lca_back, lca_upright;
        Vec3 tierod_chassis, tierod_upright;
        Vec3 spring_chassis, spring_arm, shock_chassis, shock_arm;
        Real spring_free_length;                     // Chrono's; not used (see kFreeLength)
        std::vector<Real> spring_force;              // at compressions -0.2, -0.18, ..., 0.2 m
        Real shock_damping;                          // [N s/m]
    };

    static Axle front_axle()
    {
        return Axle{{-0.040, 0.910, -0.026}, {-0.040, 0.751, -0.026},
                    {-0.106, 0.589, 0.218}, {-0.048, 0.446, 0.245}, {-0.268, 0.478, 0.196}, {-0.053, 0.716, 0.215},
                    {0.000, 0.547, -0.059}, {0.223, 0.307, 0.000}, {-0.223, 0.307, 0.000}, {-0.036, 0.787, -0.118},
                    {-0.250, 0.448, 0.054}, {-0.176, 0.821, -0.016},
                    {0.104, 0.510, 0.197}, {0.097, 0.543, -0.047}, {0.104, 0.498, 0.323}, {0.097, 0.543, -0.047},
                    0.339,
                    {-322095.536, -240521.166, -174535.686, -122406.996, -82402.997, -52791.592, -31840.681,
                     -17818.165, -8991.945, -3629.923, 0.0, 3629.923, 8991.945, 17818.165, 31840.681, 52791.592,
                     82402.997, 122406.996, 174535.686, 240521.166, 322095.536},
                    19015.5692};
    }

    static Axle rear_axle()
    {
        return Axle{{0.036, 0.910, -0.026}, {0.036, 0.751, -0.026},
                    {0.124, 0.589, 0.221}, {0.349, 0.462, 0.228}, {0.077, 0.462, 0.224}, {0.036, 0.716, 0.216},
                    {0.000, 0.547, -0.059}, {0.223, 0.307, 0.000}, {-0.223, 0.307, 0.000}, {0.036, 0.787, -0.118},
                    {0.223, 0.416, 0.059}, {0.170, 0.821, -0.009},
                    {-0.104, 0.502, 0.256}, {-0.097, 0.543, -0.047}, {-0.104, 0.498, 0.323}, {-0.097, 0.544, -0.038},
                    0.382,
                    {-711719.272, -531468.245, -385663.250, -270476.949, -182082.006, -116651.084, -70356.846,
                     -39371.956, -19869.076, -8020.869, 0.0, 8020.869, 19869.076, 39371.956, 70356.846, 116651.084,
                     182082.006, 270476.949, 385663.250, 531468.245, 711719.272},
                    36725.2201};
    }

    /// A part of a body: mass, centre of mass and principal moments about it,
    /// in the vehicle's axes.
    struct Part {
        Real mass;
        Vec3 com;
        Vec3 moments;
    };

    /// The parts together, in a body frame at `origin` with axes R (columns,
    /// in the vehicle's axes).
    static mbd::RigidBodyInertia combine(const std::vector<Part>& parts, const Vec3& origin, const Mat3& R)
    {
        Real m = 0.0;
        Vec3 c = Vec3::Zero();
        for (const Part& p : parts) {
            m += p.mass;
            c += p.mass * p.com;
        }
        c /= m;
        Mat3 I = Mat3::Zero();
        for (const Part& p : parts) {
            const Vec3 d = p.com - c;
            I += Mat3(p.moments.asDiagonal()) + p.mass * (d.squaredNorm() * Mat3::Identity() - d * d.transpose());
        }
        mbd::RigidBodyInertia out;
        out.mass = m;
        out.com_B = R.transpose() * (c - origin);
        out.I_com_B = R.transpose() * I * R;
        return out;
    }

    /// Axes with Z along `axis` (unit), X horizontal.
    static Mat3 axes_along(const Vec3& axis)
    {
        const Vec3 x = Vec3::UnitZ().cross(axis).normalized();
        Mat3 R;
        R.col(0) = x;
        R.col(1) = axis.cross(x);
        R.col(2) = axis;
        return R;
    }

    // --- The model -------------------------------------------------------------

    struct Corner {
        bool front{true}, left{true};
        int lca{0}, uca{0}, spindle{0};
        Vec3 wheel_centre_B;      ///< In the spindle's frame
        /// The spring's and the shock's points on the chassis (its frame) and
        /// on the lower arm (the arm's frame).
        Vec3 spring_chassis, spring_arm_B, shock_chassis, shock_arm_B;
    };

    mbd::kernel::System sys;
    int chassis{0}, rack{0};
    std::array<Corner, 4> corners{};   // front left, front right, rear left, rear right
    std::shared_ptr<HmmwvTerrain> terrain = std::make_shared<HmmwvTerrain>();
    bool bushed{false};

    /// free_length: the springs' free lengths, front and rear [m]; the
    /// defaults are kFreeLength (see there).
    explicit Hmmwv(bool with_bushings = false, std::array<Real, 2> free_length = kFreeLength)
        : bushed(with_bushings)
    {
        using namespace mbd;
        using namespace mbd::kernel;
        Model& model = sys.model;
        model.gravity = Vec3(0.0, 0.0, -g_accel);

        // Chassis (HMMWV_Chassis.json): its frame is Chrono's chassis frame,
        // origin between the axles at the wheel centres' height + 0.026 m.
        chassis = model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                                 combine({{2086.52, Vec3(0.056, 0.0, 0.213), Vec3(1078.52, 2955.66, 3570.20)}},
                                         Vec3::Zero(), Mat3::Identity()),
                                 "chassis");

        // Steering rack (HMMWV_RackPinion.json): across the front inner tie-rod
        // points, sliding along Y, held still by a driver (the steering input).
        const Axle fa = front_axle();
        const Vec3 rack_centre(kAxle + fa.tierod_chassis.x(), 0.0, fa.tierod_chassis.z());
        const Mat3 R_rack = axes_along(Vec3::UnitY());
        rack = model.add_body(chassis, std::make_shared<PrismaticJointModel>(), Transform3(R_rack, rack_centre),
                              Transform3(), combine({{9.072, rack_centre, Vec3(1.0, 1.0, 1.0)}}, rack_centre, R_rack),
                              "steering rack");
        sys.constraints.push_back(std::make_shared<JointDriver>(model, rack, TimeFunction()));

        const auto revolute = std::make_shared<RevoluteJointModel>();
        const auto spherical = std::make_shared<SphericalJointModel>();
        std::size_t k = 0;
        for (const bool front : {true, false}) {
            const Axle a = front ? front_axle() : rear_axle();
            for (const bool left : {true, false}) {
                Corner& c = corners[k++];
                c.front = front;
                c.left = left;
                const std::string name = std::string(front ? "front " : "rear ") + (left ? "left " : "right ");
                // A point of the axle's data in the chassis frame, mirrored to the right.
                auto P = [&](const Vec3& p) { return Vec3((front ? kAxle : -kAxle) + p.x(), left ? p.y() : -p.y(), p.z()); };

                // Lower arm, on a revolute joint about its two chassis points.
                const Vec3 lca_o = P(a.lca_front);
                const Mat3 R_lca = axes_along((P(a.lca_front) - P(a.lca_back)).normalized());
                c.lca = model.add_body(chassis, revolute, Transform3(R_lca, lca_o), Transform3(),
                                       combine({{23.965, P(a.lca_com), Vec3(0.4, 0.4, 0.8)}}, lca_o, R_lca),
                                       name + "lower arm");

                // Upper arm: a revolute joint, or for the stiff variant a spherical
                // joint at its front point with two bushing spring-dampers at its back.
                const Vec3 uca_o = P(a.uca_front);
                const Mat3 R_uca = axes_along((P(a.uca_front) - P(a.uca_back)).normalized());
                c.uca = model.add_body(chassis, bushed ? std::shared_ptr<const JointModel>(spherical) : revolute,
                                       Transform3(R_uca, uca_o), Transform3(),
                                       combine({{5.813, P(a.uca_com), Vec3(0.03, 0.03, 0.06)}}, uca_o, R_uca),
                                       name + "upper arm");

                // Spindle body: upright, spindle, wheel and tyre, on a spherical
                // joint at the lower ball joint, its axes the vehicle's.
                const Vec3 s_o = P(a.lca_upright);
                c.spindle = model.add_body(
                    c.lca, spherical, Transform3(Mat3(R_lca.transpose()), R_lca.transpose() * (s_o - lca_o)),
                    Transform3(),
                    combine({{19.450, P(a.upright), Vec3(0.1656, 0.1934, 0.04367)},
                             {14.705, P(a.spindle), Vec3(0.04117, 0.07352, 0.04117)},
                             {18.8, P(a.spindle), Vec3(0.4634, 0.6243, 0.4634)},
                             {37.6, P(a.spindle), Vec3(3.84, 6.69, 3.84)}},
                            s_o, Mat3::Identity()),
                    name + "spindle");
                c.wheel_centre_B = P(a.spindle) - s_o;

                // Upper ball joint and tie rod.
                sys.constraints.push_back(spherical_closure(
                    Marker{c.spindle, Transform3::FromTranslation(P(a.uca_upright) - s_o)},
                    Marker{c.uca, Transform3::FromTranslation(R_uca.transpose() * (P(a.uca_upright) - uca_o))}));
                const int inner = front ? rack : chassis;
                const Vec3 inner_B = front ? Vec3(R_rack.transpose() * (P(a.tierod_chassis) - rack_centre))
                                           : P(a.tierod_chassis);
                sys.constraints.push_back(std::make_shared<Distance>(
                    Marker{c.spindle, Transform3::FromTranslation(P(a.tierod_upright) - s_o)},
                    Marker{inner, Transform3::FromTranslation(inner_B)},
                    (P(a.tierod_upright) - P(a.tierod_chassis)).norm()));

                // Spring and shock, chassis to lower arm.
                SpringDamperParams spring;
                spring.free_length = free_length[front ? 0 : 1];
                std::vector<Real> x;
                for (int i = -10; i <= 10; ++i) x.push_back(0.02 * i);
                spring.spring = Curve::table(x, a.spring_force);
                c.spring_chassis = P(a.spring_chassis);
                c.spring_arm_B = R_lca.transpose() * (P(a.spring_arm) - lca_o);
                c.shock_chassis = P(a.shock_chassis);
                c.shock_arm_B = R_lca.transpose() * (P(a.shock_arm) - lca_o);
                sys.force_elements.push_back(
                    std::make_shared<SpringDamper>(chassis, c.lca, c.spring_chassis, c.spring_arm_B, spring));
                SpringDamperParams shock;
                shock.damper = Curve::linear(a.shock_damping);
                sys.force_elements.push_back(
                    std::make_shared<SpringDamper>(chassis, c.lca, c.shock_chassis, c.shock_arm_B, shock));

                // The stiff variant's bushings: at the upper arm's back point,
                // two spring-dampers across the arm's axis, perpendicular to each
                // other, from anchors on the chassis kBushingLength away.
                if (bushed) {
                    SpringDamperParams b;
                    b.free_length = kBushingLength;
                    b.spring = Curve::linear(kBushingStiffness);
                    b.damper = Curve::linear(kBushingDamping);
                    const Vec3 back = P(a.uca_back);
                    for (int i = 0; i < 2; ++i) {
                        sys.force_elements.push_back(std::make_shared<SpringDamper>(
                            chassis, c.uca, Vec3(back + kBushingLength * R_uca.col(i)),
                            R_uca.transpose() * (back - uca_o), b));
                    }
                }

                // Tyre: a spring-damper from the wheel centre to the terrain.
                const int body = c.spindle;
                const Vec3 wc = c.wheel_centre_B;
                const std::shared_ptr<const HmmwvTerrain> road = terrain;
                sys.force_elements.push_back(std::make_shared<UserForce>(
                    name + "tyre", std::vector<BodyIndex>{body},
                    [body, wc, road](const std::vector<RigidBodyState>& states, std::vector<RigidBodyForces>& forces) {
                        const TyreContact t = tyre_contact(*road, states[static_cast<std::size_t>(body)], wc);
                        if (t.deflection <= 0.0 || t.force <= 0.0) return;
                        RigidBodyForces& out = forces[static_cast<std::size_t>(body)];
                        out.f_W += t.force * t.n;
                        out.tau_W += t.r.cross(t.force * t.n);
                    }));
            }
        }
    }

    // --- The tyres ------------------------------------------------------------

    /// A tyre against the road: the wheel centre's offset from the spindle
    /// body's origin (world axes), the road's normal at the nearest point,
    /// the deflection (kTyreRadius less the distance), its rate, the force
    /// k deflection + c rate, applied along n when both the deflection and
    /// the force are positive, and the distances to the flat ground and to
    /// the bump.
    struct TyreContact {
        Vec3 r, n;
        Real deflection{0.0}, rate{0.0}, force{0.0}, flat{0.0}, bump{0.0};
    };

    static TyreContact tyre_contact(const HmmwvTerrain& road, const mbd::RigidBodyState& s, const Vec3& wheel_centre_B)
    {
        TyreContact t;
        t.r = s.q_WB * wheel_centre_B;
        const Vec3 centre = s.p_WB + t.r;
        t.flat = centre.z();
        t.bump = road.distance_to_bump(centre);
        t.deflection = kTyreRadius - road.distance(centre, t.n);
        t.rate = -t.n.dot(s.v_WB + s.w_WB.cross(t.r));
        t.force = kTyreStiffness * t.deflection + kTyreDamping * t.rate;
        return t;
    }

    TyreContact tyre(const mbd::kernel::Simulator& sim, std::size_t corner) const
    {
        const Corner& c = corners[corner];
        return tyre_contact(*terrain, sim.states()[static_cast<std::size_t>(c.spindle)], c.wheel_centre_B);
    }

    /// Events where a tyre's force changes abruptly, so that the integrators
    /// step to them instead of across them: the nearest surface changing
    /// between the ground and the bump (the force turns, its rate jumps:
    /// meeting the bump at 10 mph, by about 2 m/s, so 7 kN through the
    /// damper), contact beginning or ending, and the spring and damper's sum
    /// crossing zero (where the law has its kink).
    void add_events(mbd::kernel::Simulator& sim) const
    {
        for (std::size_t k = 0; k < corners.size(); ++k) {
            const std::string name = std::string(corners[k].front ? "front " : "rear ")
                                     + (corners[k].left ? "left" : "right") + " tyre: ";
            auto event = [this, k](Real TyreContact::*field) {
                return [this, k, field](const mbd::kernel::Simulator& s) { return tyre(s, k).*field; };
            };
            sim.events.push_back({name + "deflection", event(&TyreContact::deflection), {}, 0});
            sim.events.push_back({name + "force", event(&TyreContact::force), {}, 0});
            sim.events.push_back({name + "surface",
                                  [this, k](const mbd::kernel::Simulator& s) {
                                      const TyreContact t = tyre(s, k);
                                      return std::isfinite(t.bump) ? t.flat - t.bump : -1.0;
                                  },
                                  {}, 0});
        }
    }

    // --- Running it ------------------------------------------------------------

    /// The design position, at rest: every joint at zero, the chassis frame at
    /// kDesignHeight with its centre between the axles over x = 0.
    void design(mbd::kernel::Simulator& sim) const
    {
        sim.q = sys.model.neutral_configuration();
        sim.q(sys.model.idx_q[static_cast<std::size_t>(chassis)] + 2) = kDesignHeight;
        sim.v.setZero();
        sim.time = 0.0;
        sim.refresh();
    }

    /// Settles the vehicle on flat ground (static equilibrium from the design
    /// position), puts the bump where the front tyres turn to it at t_front,
    /// and sets every body moving forward at 10 mph; with_events adds the
    /// tyres' events (add_events).
    mbd::kernel::StaticReport start(mbd::kernel::Simulator& sim, Real t_front = 0.5, bool with_events = true) const
    {
        terrain->bump_x = std::numeric_limits<Real>::infinity();
        design(sim);
        mbd::kernel::StaticOptions options;
        options.tolerance = 1e-8;   // the bushed variant's rounding floor is about 3e-9
        const mbd::kernel::StaticReport report = mbd::kernel::static_equilibrium(sim, options);

        // The front wheel centres' height z and position at rest. A tyre's
        // force turns to the bump when the bump is nearer than the ground:
        // when the centre is z + r from the bump's axis. (Its unloaded circle
        // touches the bump earlier, at R + r, 0.047 m and 10 ms before; the
        // one spring of this tyre model follows the nearest surface only.)
        const Vec3 c = wheel_centre(sim, 0);
        const Real r = terrain->bump_radius;
        terrain->bump_x = c.x() + kSpeed * t_front + std::sqrt((c.z() + r) * (c.z() + r) - c.z() * c.z());

        // The chassis's free-joint velocity is (angular, linear) in its own
        // axes; the other joints stay still, so every body moves with it.
        const auto& m = sys.model;
        const std::size_t ic = static_cast<std::size_t>(chassis);
        const mbd::Quat q_c(sim.q.segment<4>(m.idx_q[ic] + 3));
        sim.v.setZero();
        sim.v.segment<3>(m.idx_v[ic] + 3) = q_c.conjugate() * Vec3(kSpeed, 0.0, 0.0);
        sim.time = 0.0;
        sim.initialize();
        if (with_events) add_events(sim);
        return report;
    }

    Vec3 wheel_centre(const mbd::kernel::Simulator& sim, std::size_t corner) const
    {
        const auto& s = sim.states()[static_cast<std::size_t>(corners[corner].spindle)];
        return s.p_WB + s.q_WB * corners[corner].wheel_centre_B;
    }

    /// The chassis's centre of mass (the paper's z1 is its height).
    Vec3 chassis_com(const mbd::kernel::Simulator& sim) const
    {
        const auto& s = sim.states()[static_cast<std::size_t>(chassis)];
        return s.p_WB + s.q_WB * sys.model.inertia[static_cast<std::size_t>(chassis)].com_B;
    }

    /// The chassis's pitch [rad], nose down positive (rotation about Y).
    Real chassis_pitch(const mbd::kernel::Simulator& sim) const
    {
        const auto& s = sim.states()[static_cast<std::size_t>(chassis)];
        const Vec3 x = s.q_WB * Vec3::UnitX();
        return std::atan2(-x.z(), x.x());
    }

    /// The length of a corner's spring, or of its shock.
    Real spring_length(const mbd::kernel::Simulator& sim, std::size_t corner, bool shock = false) const
    {
        const Corner& c = corners[corner];
        const auto& sc = sim.states()[static_cast<std::size_t>(chassis)];
        const auto& sa = sim.states()[static_cast<std::size_t>(c.lca)];
        return ((sc.p_WB + sc.q_WB * (shock ? c.shock_chassis : c.spring_chassis))
                - (sa.p_WB + sa.q_WB * (shock ? c.shock_arm_B : c.spring_arm_B)))
            .norm();
    }
};

} // namespace mbd_test
