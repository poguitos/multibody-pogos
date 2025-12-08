#pragma once

// Basic rigid-body types: inertia and state representation.
#include <Eigen/Eigenvalues> 
#include <mbd/core.hpp>   // includes math + logging + basic types

namespace mbd {

//------------------------------------------------------------------------------
// Rigid body inertia (about a body-fixed frame)
//------------------------------------------------------------------------------

struct RigidBodyInertia
{
    Real mass{0.0};      // total mass [kg]
    Vec3 com_B{Vec3::Zero()};  // center of mass position expressed in body frame B [m]
    Mat3 I_com_B{Mat3::Zero()}; // inertia tensor about COM, expressed in B [kg m^2]

    RigidBodyInertia() = default;

    RigidBodyInertia(Real m, const Vec3& com, const Mat3& I_com)
        : mass(m)
        , com_B(com)
        , I_com_B(I_com)
    {}

    // Check basic physical sanity of the inertia.
    bool is_physically_valid(Real tol = Real(1e-10)) const
    {
        if (mass <= Real(0.0)) {
            return false;
        }

        // Symmetry of inertia matrix
        if (!I_com_B.isApprox(I_com_B.transpose(), tol)) {
            return false;
        }

        // Principal moments non-negative
        Eigen::SelfAdjointEigenSolver<Mat3> es(I_com_B);
        if (es.info() != Eigen::Success) {
            return false;
        }
        auto vals = es.eigenvalues();
        if (vals.minCoeff() < -tol) { // allow tiny negative due to numerics
            return false;
        }
        return true;
    }

    // Factory: solid box with half extents hx, hy, hz about COM.
    // Full side lengths are 2*hx, 2*hy, 2*hz.
    static RigidBodyInertia from_solid_box(Real mass,
                                           const Vec3& half_extents_B)
    {
        MBD_THROW_IF(mass <= Real(0.0), "RigidBodyInertia::from_solid_box: mass must be > 0");

        const Real hx = half_extents_B.x();
        const Real hy = half_extents_B.y();
        const Real hz = half_extents_B.z();

        MBD_THROW_IF(hx <= Real(0.0) || hy <= Real(0.0) || hz <= Real(0.0),
                     "RigidBodyInertia::from_solid_box: half extents must be > 0");

        // For a box of size (2hx, 2hy, 2hz) about COM:
        // Ixx = (1/3) * m * (hy^2 + hz^2), etc.
        const Real Ixx = (mass / Real(3.0)) * (hy * hy + hz * hz);
        const Real Iyy = (mass / Real(3.0)) * (hx * hx + hz * hz);
        const Real Izz = (mass / Real(3.0)) * (hx * hx + hy * hy);

        Mat3 I_com = Mat3::Zero();
        I_com.diagonal() << Ixx, Iyy, Izz;

        return RigidBodyInertia(mass, Vec3::Zero(), I_com);
    }
};

//------------------------------------------------------------------------------
// Rigid body state (pose and velocities)
//------------------------------------------------------------------------------
//
// World frame: W
// Body frame:  B
//
// p_WB : position of body frame origin B expressed in W
// q_WB : orientation of B w.r.t W (mapping from B to W)
// v_WB : linear velocity of B-origin expressed in W
// w_WB : angular velocity of body frame B expressed in W
//

struct RigidBodyState
{
    Vec3 p_WB{Vec3::Zero()};
    Quat q_WB{Quat::Identity()};
    Vec3 v_WB{Vec3::Zero()};
    Vec3 w_WB{Vec3::Zero()};

    RigidBodyState() = default;

    RigidBodyState(const Vec3& p, const Quat& q,
                   const Vec3& v, const Vec3& w)
        : p_WB(p)
        , q_WB(normalize_quat(q))
        , v_WB(v)
        , w_WB(w)
    {}

    // Convenience factory: body at rest at a given pose.
    static RigidBodyState at_rest(const Vec3& p_WB,
                                  const Quat& q_WB = Quat::Identity())
    {
        return RigidBodyState(p_WB,
                              normalize_quat(q_WB),
                              Vec3::Zero(),
                              Vec3::Zero());
    }
};

} // namespace mbd
