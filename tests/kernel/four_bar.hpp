#pragma once

// A planar four-bar for the kernel tests, with its closed-form geometry.

#include <cmath>
#include <memory>

#include <Eigen/Dense>

#include "mbd/core/core.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd_test {

/// A planar four-bar in the XY plane, hinges along Z, gravity along -Y:
/// crank AB, coupler BC, rocker CD, ground AD (a crank-rocker: 0.3 + 0.8 <
/// 0.6 + 0.7). A tree of three revolute joints, closed at D. Optionally a
/// pendulum hangs from the coupler, outside the loop.
struct FourBar {
    using Real = mbd::Real;
    using VecX = mbd::VecX;
    static constexpr Real a = 0.3, b = 0.8, c = 0.6, d = 0.7;
    mbd::kernel::System sys;
    int crank{0}, coupler{0}, rocker{0};

    explicit FourBar(bool with_pendulum = false)
    {
        using namespace mbd;
        using namespace mbd::kernel;
        sys.model.gravity = Vec3(0.0, -g_accel, 0.0);
        const auto hinge = std::make_shared<RevoluteJointModel>();
        const Transform3 I3;
        crank   = sys.model.add_body(0, hinge, I3, I3, link(a, 1.0), "crank");
        coupler = sys.model.add_body(crank, hinge, Transform3::FromTranslation(Vec3(a, 0, 0)), I3,
                                     link(b, 2.0), "coupler");
        rocker  = sys.model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(b, 0, 0)), I3,
                                     link(c, 1.5), "rocker");
        if (with_pendulum) {
            sys.model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(0.4, 0, 0)), I3,
                               link(0.5, 0.7), "pendulum");
        }
        sys.constraints.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(d, 0, 0))},
                                                   Marker{rocker, Transform3::FromTranslation(Vec3(c, 0, 0))}));
    }

    /// A slender link along its X axis, its centre of mass halfway.
    static mbd::RigidBodyInertia link(Real length, Real mass)
    {
        mbd::RigidBodyInertia I = mbd::RigidBodyInertia::from_solid_box(mass, mbd::Vec3(0.5 * length, 0.02, 0.02));
        I.com_B = mbd::Vec3(0.5 * length, 0.0, 0.0);
        return I;
    }

    /// The closed loop at crank angle th1: C is where the circles about B
    /// (radius b) and D (radius c) meet, on the upper side.
    static Eigen::Vector3d configuration(Real th1)
    {
        const Eigen::Vector2d B(a * std::cos(th1), a * std::sin(th1)), D(d, 0.0);
        const Real dist = (D - B).norm();
        const Real along = (b * b - c * c + dist * dist) / (2.0 * dist);
        const Real h = std::sqrt(b * b - along * along);
        const Eigen::Vector2d e = (D - B) / dist;
        const Eigen::Vector2d C = B + along * e + h * Eigen::Vector2d(-e.y(), e.x());
        const Real phi2 = std::atan2(C.y() - B.y(), C.x() - B.x());
        const Real phi3 = std::atan2(D.y() - C.y(), D.x() - C.x());
        return {th1, phi2 - th1, phi3 - phi2};
    }

    /// The gap of the loop at D in the plane, a e(f1) + b e(f2) + c e(f3) -
    /// (d, 0) with the absolute angles f1 = q0, f2 = f1 + q1, f3 = f2 + q2, and
    /// its Jacobian with respect to q. Written from the geometry, apart from
    /// the kernel's constraints.
    static Eigen::Vector2d gap(const VecX& q, Eigen::Matrix<Real, 2, 3>* J = nullptr)
    {
        const Real f1 = q(0), f2 = f1 + q(1), f3 = f2 + q(2);
        auto e = [](Real f) { return Eigen::Vector2d(std::cos(f), std::sin(f)); };
        auto n = [](Real f) { return Eigen::Vector2d(-std::sin(f), std::cos(f)); };
        if (J) {
            J->col(2) = c * n(f3);
            J->col(1) = b * n(f2) + J->col(2);
            J->col(0) = a * n(f1) + J->col(1);
        }
        return a * e(f1) + b * e(f2) + c * e(f3) - Eigen::Vector2d(d, 0.0);
    }

    /// The joint rates that keep the loop closed per unit crank rate, at q:
    /// J (1, x1, x2) = 0.
    static Eigen::Vector3d motion(const VecX& q)
    {
        Eigen::Matrix<Real, 2, 3> J;
        gap(q, &J);
        const Eigen::Matrix2d J12 = J.rightCols<2>();
        const Eigen::Vector2d rest = -(J12.inverse() * J.col(0));
        return {1.0, rest(0), rest(1)};
    }
};

} // namespace mbd_test
