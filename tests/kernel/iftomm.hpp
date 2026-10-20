#pragma once

// The IFToMM benchmark problems of group A (plan task 5.2): the
// mechanisms of the Library of Computational Benchmark Problems of the IFToMM
// Technical Committee for Multibody Dynamics (www.iftomm-multibody.org), with
// the references they are checked against. The problems' parameters are as
// the library states them, read here from its reimplementation for OpenSim
// (Tagliapietra et al. 2015, MBS-BOS) and, for Andrews' mechanism, from the
// Test Set for IVP Solvers (Mazzia and Magherini), which also publishes its
// reference solution. No code is taken from either: each mechanism is
// built here from its geometry.

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

#include "support/elliptic.hpp"

namespace mbd_test::iftomm {

using mbd::Mat3;
using mbd::Quat;
using mbd::Real;
using mbd::RigidBodyInertia;
using mbd::Transform3;
using mbd::Vec3;
using mbd::VecX;

inline Transform3 at(Real x, Real y, Real z = 0.0)
{
    return Transform3::FromTranslation(Vec3(x, y, z));
}

/// A slender uniform bar of length L along its body's X axis from the
/// origin, its centre of mass at (L/2, 0, 0) (or (-L/2, 0, 0) when it runs
/// along -X): m L^2 / 12 about Y and Z, and a moment about its axis a
/// millionth of that, so that the inertia is that of a real body.
inline RigidBodyInertia bar(Real m, Real L, Real direction = 1.0)
{
    const Real I = m * L * L / 12.0;
    return RigidBodyInertia(m, Vec3(0.5 * L * direction, 0.0, 0.0), Vec3(1e-6 * I, I, I).asDiagonal());
}

/// A body turning in the XY plane about Z: mass m, its centre of mass at c
/// (body frame), moment i about Z at the centre of mass (the same about X
/// and Y, which the planar motion does not use).
inline RigidBodyInertia planar(Real m, const Vec3& c, Real i)
{
    return RigidBodyInertia(m, c, Vec3(i, i, i).asDiagonal());
}

// --- A01: the simple pendulum ---------------------------------------------------

/// A point mass m = 1 kg on a massless rod of L = 1 m, hinged at the origin
/// about Z, gravity 9.81 m/s^2 along -Y, released at rest with the rod
/// horizontal. Its angle from the downward vertical obeys
/// theta'' = -(g / L) sin(theta) from theta0 = pi / 2, so
///
///   sin(theta / 2) = k sn(K(k) - w0 t | k^2),  k = sin(theta0 / 2),  w0 = sqrt(g / L),
///
/// and the mass is at (L sin(theta), -L cos(theta)).
struct Pendulum {
    static constexpr Real m = 1.0, L = 1.0, g = 9.81;
    mbd::kernel::System sys;

    /// On a revolute joint (a tree), or as a free body held at L from the
    /// origin by a distance constraint (a loop closed by a constraint).
    explicit Pendulum(bool as_constraint = false)
    {
        using namespace mbd::kernel;
        sys.model.gravity = Vec3(0.0, -g, 0.0);
        if (as_constraint) {
            // Its centre of mass at the body origin, where the constraint
            // acts: the body's turning never couples to its swing, so the
            // small sphere's inertia (it must have one) leaves the point
            // mass's motion exact.
            const Real i = 0.4 * m * 1e-4;   // a sphere of 1 cm radius
            sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                               RigidBodyInertia(m, Vec3::Zero(), Vec3(i, i, i).asDiagonal()), "mass");
            sys.constraints.push_back(std::make_shared<Distance>(Marker{0, Transform3()}, Marker{1, Transform3()}, L));
        } else {
            sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(),
                               RigidBodyInertia(m, Vec3(L, 0.0, 0.0), Mat3::Zero()), "mass");
        }
        constraint_ = as_constraint;
    }

    /// The released state: the mass at (L, 0, 0), at rest.
    void release(mbd::kernel::Simulator& sim) const
    {
        sim.q = sys.model.neutral_configuration();
        if (constraint_) sim.q(0) = L;   // free joint: translation first
        sim.v.setZero();
    }

    /// Where the mass is.
    Vec3 position(const mbd::kernel::Simulator& sim) const
    {
        if (constraint_) return sim.q.head<3>();
        return Vec3(L * std::cos(sim.q(0)), L * std::sin(sim.q(0)), 0.0);
    }

    /// The closed form at time t.
    static Vec3 exact(Real t)
    {
        const Real k = std::sin(0.25 * mbd::pi);
        const Real w0 = std::sqrt(g / L);
        Real sn = 0.0, cn = 0.0, dn = 0.0;
        jacobi(elliptic_K(k) - w0 * t, k * k, sn, cn, dn);
        const Real s = k * sn;                                 // sin(theta / 2)
        const Real c = std::sqrt(std::max(0.0, 1.0 - s * s));   // cos(theta / 2) >= 0: |theta| <= pi / 2
        const Real sin_theta = 2.0 * s * c, cos_theta = 1.0 - 2.0 * s * s;
        return Vec3(L * sin_theta, -L * cos_theta, 0.0);
    }

private:
    bool constraint_{false};
};

// --- A02: the N-four-bar mechanism ----------------------------------------------

/// N four-bar linkages in a row in the XY plane, sharing their vertical
/// bars: N + 1 vertical bars of L = 1 m hinged on the ground at (i L, 0), their
/// tops joined by N horizontal couplers of L; every bar 1 kg, uniform;
/// gravity 9.81 m/s^2 along -Y. A tree (bar 1 on the ground, then coupler,
/// bar, coupler, ...), each vertical bar after the first closed at its
/// bottom to its ground point. Started with the vertical bars upright,
/// turning at -1 rad/s (clockwise), the couplers level.
///
/// The bars are all equal, so the linkage moves as one parallelogram: every
/// vertical bar at the same angle theta from the X axis, the couplers level.
/// Its kinetic energy is J theta'^2 / 2, J = (N + 1) m L^2 / 3 + N m L^2 (the
/// vertical bars turning about their ground points, the couplers
/// translating with their tops), and its potential energy C sin(theta),
/// C = m g L ((N + 1) / 2 + N). With psi = theta + pi / 2, the angle from
/// hanging, J psi'' = -C sin(psi): a pendulum started upside down (psi0 =
/// pi) with psi0' = -1, so it turns over and over. Its energy E = J / 2 + 2 C
/// above the hanging position exceeds the 2 C needed to pass the top, and
/// phi = psi / 2 obeys phi'^2 = lambda^2 (1 - m sin^2 phi), lambda^2 =
/// E / (2 J), m = 2 C / E < 1, so
///
///   phi = am(K(m) - lambda t | m),   theta = 2 phi - pi / 2.
///
/// Every half turn the bars pass through horizontal, where each four-bar is
/// singular (its constraint equations lose rank): the benchmark's
/// difficulty.
struct NFourBar {
    static constexpr Real m = 1.0, L = 1.0, g = 9.81;
    int N;
    mbd::kernel::System sys;
    std::vector<int> vertical, coupler;

    explicit NFourBar(int n) : N(n)
    {
        using namespace mbd::kernel;
        sys.model.gravity = Vec3(0.0, -g, 0.0);
        const auto hinge = std::make_shared<RevoluteJointModel>();
        // Bar 1 from its ground hinge up its X axis; every later vertical bar
        // from its top (the body origin, on the coupler's end) down its -X
        // axis, so that its joint angle is also theta's change.
        vertical.push_back(sys.model.add_body(0, hinge, Transform3(), Transform3(), bar(m, L), "bar 1"));
        for (int i = 1; i <= N; ++i) {
            const int top = vertical.back();
            const Transform3 at_top = i == 1 ? at(L, 0.0) : Transform3();
            coupler.push_back(sys.model.add_body(top, hinge, at_top, Transform3(), bar(m, L),
                                                 "coupler " + std::to_string(i)));
            vertical.push_back(sys.model.add_body(coupler.back(), hinge, at(L, 0.0), Transform3(), bar(m, L, -1.0),
                                                  "bar " + std::to_string(i + 1)));
            sys.constraints.push_back(std::make_shared<PointCoincidence>(
                Marker{vertical.back(), at(-L, 0.0)}, Marker{0, at(i * L, 0.0)}));
        }
    }

    /// Upright, turning at -1 rad/s: bar 1 at pi / 2, each coupler at -pi / 2
    /// from the bar below it (level), each later bar at +pi / 2 from its
    /// coupler; rates -1, +1, -1, ...
    void release(mbd::kernel::Simulator& sim) const
    {
        sim.q = VecX::Zero(sys.model.nq);
        sim.v = VecX::Zero(sys.model.nv);
        for (int k = 0; k < sys.model.nv; ++k) {
            const Real sign = k % 2 == 0 ? 1.0 : -1.0;
            sim.q(k) = sign * 0.5 * mbd::pi;
            sim.v(k) = -sign;
        }
    }

    /// The vertical bars' angle from X: bar 1's joint angle.
    static Real theta(const mbd::kernel::Simulator& sim) { return sim.q(0); }

    /// The closed form at time t.
    Real exact(Real t) const
    {
        const Real J = (N + 1) * m * L * L / 3.0 + N * m * L * L;
        const Real C = m * g * L * (0.5 * (N + 1) + N);
        const Real E = 0.5 * J + 2.0 * C;
        const Real k2 = 2.0 * C / E;
        const Real lambda = std::sqrt(E / (2.0 * J));
        Real dn = 0.0;
        const Real phi = jacobi_amplitude(elliptic_K(std::sqrt(k2)) - lambda * t, k2, dn);
        return 2.0 * phi - 0.5 * mbd::pi;
    }
};

// --- A03: Andrews' squeezing mechanism ---------------------------------------------

/// Seven bodies in the XY plane on revolute joints about Z, no gravity
/// (Schiehlen 1990; Hairer and Wanner, Solving ODEs II, VII.7), with the
/// parameters and initial angles of the Test Set for IVP Solvers. The
/// joint coordinates are the problem's angles: beta (body 1 about O),
/// Theta (body 2 on body 1 at F), gamma (body 3 about B), Phi (body 4 on
/// body 5 at G), delta (body 5 about A), Omega (body 6 on body 7 at H),
/// epsilon (body 7 about A). Three loops meet at point E of body 3. A spring
/// of c0 = 4530 N/m and rest length l0 pulls point D of body 3 towards C, and
/// a torque of 0.033 N m drives body 1. Released at rest, it is checked at
/// t = 0.03 s against the test set's reference (PSIDE at tolerances 1e-14,
/// 16 digits).
struct Andrews {
    // The test set's parameters [kg, kg m^2, m, N/m, N m].
    static constexpr Real m1 = .04325, m2 = .00365, m3 = .02373, m4 = .00706, m5 = .07050, m6 = .00706, m7 = .05498;
    static constexpr Real xa = -.06934, ya = -.00227, xb = -0.03635, yb = .03273, xc = .014, yc = .072, c0 = 4530;
    static constexpr Real i1 = 2.194e-6, i2 = 4.410e-7, i3 = 5.255e-6, i4 = 5.667e-7, i5 = 1.169e-5, i6 = 5.667e-7,
                          i7 = 1.912e-5;
    static constexpr Real d = 28e-3, da = 115e-4, e = 2e-2, ea = 1421e-5, rr = 7e-3, ra = 92e-5, l0 = 7785e-5;
    static constexpr Real ss = 35e-3, sa = 1874e-5, sb = 1043e-5, sc = 18e-3, sd = 2e-2, ta = 2308e-5, tb = 916e-5;
    static constexpr Real u = 4e-2, ua = 1228e-5, ub = 449e-5, zf = 2e-2, zt = 4e-2, fa = 1421e-5, mom = 33e-3;

    mbd::kernel::System sys;
    int b1{0}, b2{0}, b3{0}, b4{0}, b5{0}, b6{0}, b7{0};

    Andrews()
    {
        using namespace mbd::kernel;
        sys.model.gravity = Vec3::Zero();
        const auto hinge = std::make_shared<RevoluteJointModel>();
        const Transform3 I3;
        // Each body's centre of mass at its distance from the body's hinge
        // (the direction matters only for bodies 2, 4 and 6, whose hinges move).
        b1 = sys.model.add_body(0, hinge, I3, I3, planar(m1, Vec3(ra, 0, 0), i1), "1 (OF)");
        b2 = sys.model.add_body(b1, hinge, at(rr, 0), I3, planar(m2, Vec3(-da, 0, 0), i2), "2 (FE)");
        b3 = sys.model.add_body(0, hinge, at(xb, yb), I3, planar(m3, Vec3(sa, sb, 0), i3), "3 (BDE)");
        b5 = sys.model.add_body(0, hinge, at(xa, ya), I3, planar(m5, Vec3(ta, tb, 0), i5), "5 (AG)");
        b4 = sys.model.add_body(b5, hinge, at(zt, 0), I3, planar(m4, Vec3(0, -(e - ea), 0), i4), "4 (GE)");
        b7 = sys.model.add_body(0, hinge, at(xa, ya), I3, planar(m7, Vec3(ua, ub, 0), i7), "7 (AH)");
        b6 = sys.model.add_body(b7, hinge, at(0, -u), I3, planar(m6, Vec3(zf - fa, 0, 0), i6), "6 (HE)");
        // Point E of bodies 2, 4 and 6 on point E of body 3.
        const Marker E3{b3, at(0, -ss)};
        sys.constraints.push_back(std::make_shared<PointCoincidence>(Marker{b2, at(-d, 0)}, E3));
        sys.constraints.push_back(std::make_shared<PointCoincidence>(Marker{b4, at(0, -e)}, E3));
        sys.constraints.push_back(std::make_shared<PointCoincidence>(Marker{b6, at(zf, 0)}, E3));
        // The spring from D on body 3 to C, and the drive.
        sys.force_elements.push_back(std::make_shared<mbd::LinearSpringDamper>(b3, 0, Vec3(sd, -sc, 0),
                                                                               Vec3(xc, yc, 0), c0, 0.0, l0));
        JointCoordinateForceParams drive;
        drive.preload = mom;
        sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(sys.model, b1, drive));
    }

    /// The test set's initial angles, at rest; in the order of the bodies'
    /// joints here (beta, Theta, gamma, delta, Phi, epsilon, Omega).
    void release(mbd::kernel::Simulator& sim) const
    {
        sim.q = VecX::Zero(7);
        sim.q << -0.0617138900142764496358948458001, 0.0, 0.455279819163070380255912382449,
            0.487364979543842550225598953530, 0.222668390165885884674473185609, 1.23054744454982119249735015568,
            -0.222668390165885884674473185609;
        sim.v = VecX::Zero(7);
    }

    /// The test set's reference at t = 0.03 s, in the same order.
    static VecX reference()
    {
        VecX r(7);
        r << 0.1581077119629904e+2, -0.1575637105984298e+2, 0.4082224013073101e-1, 0.5244099658805304e+0,
            -0.5347301163226948e+0, 0.1048080741042263e+1, 0.5347301163226948e+0;
        return r;
    }
};


// --- A04: Bricard's mechanism ------------------------------------------------------

/// Six revolute joints in a closed loop of six bars of 1 m and 1 kg along the
/// edges of a unit cube, each joint's axis square to both its bars:
/// O(0, 0, 0) -x- P1(1, 0, 0) -(-y)- P2(1, -1, 0) -z- P3(1, -1, 1) -(-x)- P4(0, -1, 1)
/// -y- P5(0, 0, 1), and the ground from P5 back to O, with the axes along
/// y, z, x, y, z at O to P4 and x at P5 (the orthogonal Bricard linkage);
/// gravity 9.81 m/s^2 along -Y. Six joints closed in a loop leave six minus
/// six degrees of freedom by Grubler's count, but this geometry moves with
/// one: one of the closure's equations is redundant, and stays so along the
/// motion. Here a tree of five revolutes closed at P5 by a revolute closure
/// (five equations, rank four). Every body's frame is the world frame at the
/// start, so each joint's two frames are the same transform. Each bar a
/// box of 1 x 0.1 x 0.1 m.
struct Bricard {
    static constexpr Real m = 1.0, L = 1.0, g = 9.81, side = 0.1;
    mbd::kernel::System sys;
    std::vector<int> bars;

    Bricard()
    {
        using namespace mbd::kernel;
        sys.model.gravity = Vec3(0.0, -g, 0.0);
        const auto hinge = std::make_shared<RevoluteJointModel>();
        const Vec3 corner[6] = {Vec3(0, 0, 0), Vec3(1, 0, 0), Vec3(1, -1, 0), Vec3(1, -1, 1), Vec3(0, -1, 1),
                                Vec3(0, 0, 1)};
        const int axis[6] = {1, 2, 0, 1, 2, 0};   // y, z, x, y, z, x
        int parent = 0;
        for (int i = 0; i < 5; ++i) {
            const Transform3 joint(along(axis[i]), corner[i]);
            const Vec3 centre = 0.5 * (corner[i] + corner[i + 1]);
            const Vec3 direction = (corner[i + 1] - corner[i]).cwiseAbs();
            parent = sys.model.add_body(parent, hinge, joint, joint, box(centre, direction),
                                        "bar " + std::to_string(i + 1));
            bars.push_back(parent);
        }
        const Transform3 last(along(axis[5]), corner[5]);
        sys.constraints.push_back(revolute_closure(Marker{bars.back(), last}, Marker{0, last}));
    }

    /// A joint frame whose Z axis is world axis k.
    static Quat along(int k)
    {
        if (k == 0) return Quat(Eigen::AngleAxisd(0.5 * mbd::pi, Vec3::UnitY()));    // Z onto X
        if (k == 1) return Quat(Eigen::AngleAxisd(-0.5 * mbd::pi, Vec3::UnitX()));   // Z onto Y
        return Quat::Identity();
    }

    /// The closed form of the motion: with u = tan(phi / 2), joints 1, 3 and 5
    /// turn by phi and joints 2 and 4 by psi, tan(psi / 2) = w, the root near
    /// zero of u w^2 + (u + 1)^2 w + u = 0 (the factor common to all six
    /// closure equations of the chain written with half-angle tangents; its
    /// discriminant (u + 1)^4 - 4 u^2 = (u^2 + 1)(u^2 + 4 u + 1)). The roots'
    /// product is 1, so the small one is 1 over the large one, written without
    /// the cancellation of -(u + 1)^2 + sqrt(...) when u is large (phi near
    /// pi, which the swing reaches).
    static Real psi(Real phi)
    {
        const Real u = std::tan(0.5 * phi);
        const Real w = -2.0 * u / ((u + 1.0) * (u + 1.0) + std::sqrt((u * u + 1.0) * (u * u + 4.0 * u + 1.0)));
        return 2.0 * std::atan(w);
    }

    /// A bar of 1 x 0.1 x 0.1 m centred at c, its long side along the axis
    /// whose entry of `direction` is 1.
    static RigidBodyInertia box(const Vec3& c, const Vec3& direction)
    {
        const Vec3 half = 0.5 * (side * Vec3::Ones() + (L - side) * direction);
        RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, half);
        I.com_B = c;
        return I;
    }
};


// --- A05: the flyball governor ------------------------------------------------------

/// A spindle turning about the vertical Z axis (gravity 9.81 m/s^2 along -Z),
/// started at 2 pi rad/s; a collar sliding on it; two rods hinged at the
/// spindle's top on either side of it, each carrying a ball at its end; and
/// a spring-damper from the collar to each rod's middle (k = 8e5 N/m, c = 4e4
/// N s/m, free length 0.5 m), stiff and strongly damped. As the
/// reimplementation for OpenSim (MBS-BOS) describes the IFToMM problem:
/// density 3000 kg/m^3; the spindle 1 x 0.01 x 0.01 m (0.3 kg), its top 1 m
/// above the ground hinge; the collar a cube of 0.1 m (3 kg) starting 0.5 m
/// up; the rods 1 x 0.01 x 0.01 m (0.3 kg) hinged 0.05 m either side of the
/// axis, about axes along Y, starting 120 degrees from upward (30 degrees
/// below horizontal, outwards); balls of 5 kg, points at the rods' ends,
/// on fixed joints; the springs from points 0.05 m either side of the
/// collar's centre, at their free length at the start. Nothing drives the
/// spindle, so the angular momentum about Z is kept exactly; the dampers only
/// take energy out. They act on the springs' length alone, and the slow
/// motion (collar and rods together, a period near 40 s) hardly changes it,
/// so the governor does not settle within tens of seconds.
struct Flyball {
    static constexpr Real g = 9.81, density = 3000.0, ball = 5.0, k = 8e5, c = 4e4, free_length = 0.5;
    static constexpr Real spin0 = 2.0 * mbd::pi, hinge_offset = 0.05, rod_length = 1.0;
    static constexpr Real rod_mass = 1.0 * 0.01 * 0.01 * density, collar_mass = 0.1 * 0.1 * 0.1 * density;
    mbd::kernel::System sys;
    int spindle{0}, collar{0}, rod[2] = {0, 0}, balls[2] = {0, 0};

    Flyball()
    {
        using namespace mbd::kernel;
        sys.model.gravity = Vec3(0.0, 0.0, -g);
        const auto hinge = std::make_shared<RevoluteJointModel>();
        // The spindle's frame at the ground hinge, Z up; the collar's at its
        // centre; each rod's at its hinge with Z along the rod.
        spindle = sys.model.add_body(0, hinge, Transform3(), Transform3(),
                                     slender(rod_mass, Vec3(0.0, 0.0, 0.5)), "spindle");
        collar = sys.model.add_body(spindle, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                                    RigidBodyInertia::from_solid_box(collar_mass, Vec3(0.05, 0.05, 0.05)), "collar");
        for (int side = 0; side < 2; ++side) {
            const Real s = side == 0 ? 1.0 : -1.0;
            // Hinge axis along +Y for the rod on +X, -Y for the rod on -X, so
            // that a positive angle swings either rod outwards from upright
            // (a turn about +Y takes Z towards +X).
            const Quat z_onto_axis(Eigen::AngleAxisd(-s * 0.5 * mbd::pi, Vec3::UnitX()));
            const Transform3 at_top(z_onto_axis, Vec3(s * hinge_offset, 0.0, rod_length));
            // The rod's frame has Z along the rod, and its joint frame the same
            // turn as the spindle's, so that the rod points up at angle 0.
            const Transform3 rod_frame(z_onto_axis, Vec3::Zero());
            rod[side] = sys.model.add_body(spindle, hinge, at_top, rod_frame,
                                           slender(rod_mass, Vec3(0.0, 0.0, 0.5 * rod_length)),
                                           side == 0 ? "rod +x" : "rod -x");
            balls[side] = sys.model.add_body(rod[side], std::make_shared<FixedJointModel>(),
                                             Transform3::FromTranslation(Vec3(0.0, 0.0, rod_length)), Transform3(),
                                             RigidBodyInertia(ball, Vec3::Zero(), Mat3::Zero()),
                                             side == 0 ? "ball +x" : "ball -x");
            sys.force_elements.push_back(std::make_shared<mbd::LinearSpringDamper>(
                collar, rod[side], Vec3(s * hinge_offset, 0.0, 0.0), Vec3(0.0, 0.0, 0.5 * rod_length), k, c,
                free_length));
        }
    }

    /// A slender rod of 1 x 0.01 x 0.01 m along Z, its centre of mass at c.
    static RigidBodyInertia slender(Real m, const Vec3& c)
    {
        RigidBodyInertia I = RigidBodyInertia::from_solid_box(m, Vec3(0.005, 0.005, 0.5));
        I.com_B = c;
        return I;
    }

    /// The start: spinning at 2 pi rad/s, the collar 0.5 m up, the rods at
    /// 120 degrees from upward, at rest relative to the spindle.
    void release(mbd::kernel::Simulator& sim) const
    {
        sim.q = VecX::Zero(sys.model.nq);
        sim.v = VecX::Zero(sys.model.nv);
        sim.q(1) = 0.5;
        sim.q(2) = sim.q(3) = 2.0 * mbd::pi / 3.0;
        sim.v(0) = spin0;
    }
};

} // namespace mbd_test::iftomm
