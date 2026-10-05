#pragma once

// A planar four-bar for the kernel tests: crank AB, coupler BC, rocker CD,
// ground AD, in the XY plane with hinges along Z. A tree of three revolute
// joints (crank on the ground at A, coupler on the crank at B, rocker on the
// coupler at C), closed at D by a revolute closure: 5 equations, of which 2
// are independent in the plane, leaving one degree of freedom.

#include <cmath>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/model.hpp"

namespace mbd_test {

struct FourBar {
    static constexpr mbd::Real a = 0.3, b = 0.8, c = 0.6, d = 0.7;
    mbd::kernel::Model model;
    std::vector<std::shared_ptr<const mbd::kernel::ConstraintModel>> closure;
    mbd::VecX q;   ///< Assembled at a crank angle of 1 rad
    int crank{0}, coupler{0}, rocker{0};

    FourBar()
    {
        using namespace mbd;
        using namespace mbd::kernel;
        model.gravity = Vec3(0.0, -g_accel, 0.0);
        const auto hinge = std::make_shared<RevoluteJointModel>();
        auto link = [](Real length, Real mass) {
            RigidBodyInertia I = RigidBodyInertia::from_solid_box(mass, Vec3(0.5 * length, 0.02, 0.02));
            I.com_B = Vec3(0.5 * length, 0.0, 0.0);
            return I;
        };
        const Transform3 I3;
        crank   = model.add_body(0, hinge, I3, I3, link(a, 1.0), "crank");
        coupler = model.add_body(crank, hinge, Transform3::FromTranslation(Vec3(a, 0, 0)),
                                 I3, link(b, 2.0), "coupler");
        rocker  = model.add_body(coupler, hinge, Transform3::FromTranslation(Vec3(b, 0, 0)),
                                 I3, link(c, 1.5), "rocker");
        closure.push_back(revolute_closure(Marker{0, Transform3::FromTranslation(Vec3(d, 0, 0))},
                                           Marker{rocker, Transform3::FromTranslation(Vec3(c, 0, 0))}));
        q = closed(1.0);
    }

    /// The joint coordinates that close the loop at crank angle th1, on the
    /// branch with C above the line BD: C is where the circles about B
    /// (radius b) and D (radius c) meet.
    static mbd::VecX closed(mbd::Real th1)
    {
        using mbd::Real;
        const Eigen::Vector2d B(a * std::cos(th1), a * std::sin(th1)), D(d, 0.0);
        const Real dist = (D - B).norm();
        const Real along = (b * b - c * c + dist * dist) / (2.0 * dist);
        const Real h = std::sqrt(b * b - along * along);
        const Eigen::Vector2d e = (D - B) / dist;
        const Eigen::Vector2d C = B + along * e + h * Eigen::Vector2d(-e.y(), e.x());
        const Real phi2 = std::atan2(C.y() - B.y(), C.x() - B.x());
        const Real phi3 = std::atan2(D.y() - C.y(), D.x() - C.x());
        mbd::VecX out(3);
        out << th1, phi2 - th1, phi3 - phi2;
        return out;
    }
};

} // namespace mbd_test
