#pragma once

// The bushed benchmark (plan task 4.4): a double pendulum whose hinges are
// stiff bushings instead of joints. Two 1 kg links, 1 m long along X, each on
// a free joint to the ground; a bushing at each hinge holds the links together
// with stiffness k in translation and k_r in rotation about X and Z, and
// nothing about the hinge axis Y, damped at a ratio of about 0.2 on the fast
// modes. Released level and at rest under gravity (-Z), the pendulum falls
// and swings in the XZ plane: a motion near 5 rad/s, beside translational
// modes near sqrt(2 k / 1 kg), 1.4e4 rad/s at k = 1e8 N/m, 14 at a step of 1
// ms, far beyond an explicit method's reach.

#include <cmath>
#include <memory>

#include "mbd/core/core.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd_test {

struct BushedPendulum {
    mbd::kernel::System sys;
    int link1{0}, link2{0};

    explicit BushedPendulum(mbd::Real k = 1e8, mbd::Real k_r = 1e6)
    {
        using namespace mbd;
        auto link = [] {
            RigidBodyInertia I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.5, 0.025, 0.025));
            I.com_B = Vec3(0.5, 0.0, 0.0);
            return I;
        };
        const auto free = std::make_shared<kernel::FreeJointModel>();
        link1 = sys.model.add_body(0, free, Transform3(), Transform3(), link(), "link 1");
        link2 = sys.model.add_body(0, free, Transform3(), Transform3(), link(), "link 2");

        // Damping ratio 0.2 on each fast mode: c = 2 zeta sqrt(k m), with the
        // link's mass (1 kg) in translation and its moment of inertia about
        // the hinge for bending and torsion (m L^2 / 3 = 1/3 kg m^2).
        const Real zeta = 0.2;
        BushingParams p;
        for (std::size_t i = 0; i < 3; ++i) {
            p.stiffness[i] = Curve::linear(k);
            p.damping[i] = Curve::linear(2.0 * zeta * std::sqrt(k * 1.0));
        }
        for (const std::size_t i : {std::size_t(0), std::size_t(2)}) {   // about X and Z; Y is the hinge
            p.rotational_stiffness[i] = Curve::linear(k_r);
            p.rotational_damping[i] = Curve::linear(2.0 * zeta * std::sqrt(k_r / 3.0));
        }
        sys.force_elements.push_back(std::make_shared<Bushing>(0, Transform3(), link1, Transform3(), p));
        sys.force_elements.push_back(std::make_shared<Bushing>(
            link1, Transform3::FromTranslation(Vec3(1.0, 0.0, 0.0)), link2, Transform3(), p));
    }

    /// Both links level along X, at rest.
    void release(mbd::kernel::Simulator& sim) const
    {
        sim.q = sys.model.neutral_configuration();
        sim.q(sys.model.idx_q[static_cast<std::size_t>(link2)]) = 1.0;
        sim.v.setZero();
        sim.refresh();
    }

    /// The free end of link 2, in the world.
    mbd::Vec3 tip(const mbd::kernel::Simulator& sim) const
    {
        const auto& s = sim.states()[static_cast<std::size_t>(link2)];
        return s.p_WB + s.q_WB * mbd::Vec3(1.0, 0.0, 0.0);
    }
};

} // namespace mbd_test
