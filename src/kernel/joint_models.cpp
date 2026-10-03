// Joint models of the kinematics kernel. Derivations in docs/kernel.md.

#include "mbd/kernel/joint_model.hpp"

#include <cmath>

#include "mbd/spatial/spatial.hpp"

namespace mbd::kernel {

namespace {

/// Quaternion stored as (x, y, z, w) at the start of `q`.
Quat quat_from(ConstVecRef q, Index start)
{
    return Quat(q(start + 3), q(start), q(start + 1), q(start + 2));
}

/// d/dt of a quaternion (x, y, z, w) turning at body angular velocity w_B:
/// q_dot = q * (w_B, 0) / 2. Linear in q, so valid for a quaternion that is
/// not exactly unit, as inside a Runge-Kutta step.
void quat_rate(ConstVecRef q, Index start, const Vec3& w_B, VecRef qd)
{
    const Vec3 u  = q.segment<3>(start);
    const Real qw = q(start + 3);
    qd.segment<3>(start) = Real(0.5) * (qw * w_B + u.cross(w_B));
    qd(start + 3)        = Real(-0.5) * u.dot(w_B);
}

void quat_normalize(VecRef q, Index start)
{
    const Real n = q.segment<4>(start).norm();
    if (n < Real(1e-12)) {
        q.segment<4>(start) << 0.0, 0.0, 0.0, 1.0;
    } else {
        q.segment<4>(start) /= n;
    }
}

} // namespace

// --- Defaults ------------------------------------------------------------------

void JointModel::q_dot(ConstVecRef /*q*/, ConstVecRef v, VecRef qd) const
{
    qd = v;
}

void JointModel::integrate(ConstVecRef q, ConstVecRef dv, VecRef q_out) const
{
    q_out = q + dv;
}

void JointModel::difference(ConstVecRef q0, ConstVecRef q1, VecRef dv) const
{
    dv = q1 - q0;
}

void JointModel::normalize(VecRef /*q*/) const {}

void JointModel::neutral(VecRef q) const
{
    q.setZero();
}

// --- Revolute ------------------------------------------------------------------

void RevoluteJointModel::calc(ConstVecRef q, ConstVecRef /*v*/, JointData& d) const
{
    d.X_J = Transform3(Quat(Eigen::AngleAxisd(q(0), Vec3::UnitZ())), Vec3::Zero());
    d.S.setZero(6, 1);
    d.S(2, 0) = 1.0;
    d.c.setZero();
}

// --- Prismatic -----------------------------------------------------------------

void PrismaticJointModel::calc(ConstVecRef q, ConstVecRef /*v*/, JointData& d) const
{
    d.X_J = Transform3(Quat::Identity(), Vec3(0.0, 0.0, q(0)));
    d.S.setZero(6, 1);
    d.S(5, 0) = 1.0;
    d.c.setZero();
}

// --- Fixed ---------------------------------------------------------------------

void FixedJointModel::calc(ConstVecRef /*q*/, ConstVecRef /*v*/, JointData& d) const
{
    d.X_J = Transform3::Identity();
    d.S.setZero(6, 0);
    d.c.setZero();
}

// --- Universal -----------------------------------------------------------------
//
// R_J = Rz(q0) Rx(q1). In the child-side frame the angular velocity is
// Rx(q1)^T e_z q0_dot + e_x q1_dot, so S = [(0, s1, c1) | e_x] (angular rows),
// and c = dS/dt v = q0_dot q1_dot (0, c1, -s1).

void UniversalJointModel::calc(ConstVecRef q, ConstVecRef v, JointData& d) const
{
    d.X_J = Transform3(Quat(Eigen::AngleAxisd(q(0), Vec3::UnitZ()))
                           * Quat(Eigen::AngleAxisd(q(1), Vec3::UnitX())),
                       Vec3::Zero());
    const Real s1 = std::sin(q(1));
    const Real c1 = std::cos(q(1));
    d.S.setZero(6, 2);
    d.S(1, 0) = s1;
    d.S(2, 0) = c1;
    d.S(0, 1) = 1.0;
    d.c.setZero();
    d.c(1) =  c1 * v(0) * v(1);
    d.c(2) = -s1 * v(0) * v(1);
}

// --- Cylindrical ---------------------------------------------------------------

void CylindricalJointModel::calc(ConstVecRef q, ConstVecRef /*v*/, JointData& d) const
{
    d.X_J = Transform3(Quat(Eigen::AngleAxisd(q(0), Vec3::UnitZ())), Vec3(0.0, 0.0, q(1)));
    d.S.setZero(6, 2);
    d.S(2, 0) = 1.0;
    d.S(5, 1) = 1.0;
    d.c.setZero();
}

// --- Planar --------------------------------------------------------------------
//
// X_J = (Rz(q2), (q0, q1, 0)). The velocity of the child-side origin is
// (q0_dot, q1_dot, 0) in the parent-side frame, Rz(q2)^T times that in the
// child-side frame.

void PlanarJointModel::calc(ConstVecRef q, ConstVecRef v, JointData& d) const
{
    d.X_J = Transform3(Quat(Eigen::AngleAxisd(q(2), Vec3::UnitZ())), Vec3(q(0), q(1), 0.0));
    const Real s = std::sin(q(2));
    const Real c = std::cos(q(2));
    d.S.setZero(6, 3);
    d.S(3, 0) =  c;
    d.S(4, 0) = -s;
    d.S(3, 1) =  s;
    d.S(4, 1) =  c;
    d.S(2, 2) = 1.0;
    d.c.setZero();
    d.c(3) = (-s * v(0) + c * v(1)) * v(2);
    d.c(4) = (-c * v(0) - s * v(1)) * v(2);
}

// --- Spherical -----------------------------------------------------------------

void SphericalJointModel::calc(ConstVecRef q, ConstVecRef /*v*/, JointData& d) const
{
    d.X_J = Transform3(quat_from(q, 0), Vec3::Zero());
    d.S.setZero(6, 3);
    d.S.topRows<3>().setIdentity();
    d.c.setZero();
}

void SphericalJointModel::q_dot(ConstVecRef q, ConstVecRef v, VecRef qd) const
{
    quat_rate(q, 0, v.head<3>(), qd);
}

void SphericalJointModel::integrate(ConstVecRef q, ConstVecRef dv, VecRef q_out) const
{
    const Quat Q = (quat_from(q, 0).normalized() * exp3(dv.head<3>())).normalized();
    q_out = Q.coeffs();
}

void SphericalJointModel::difference(ConstVecRef q0, ConstVecRef q1, VecRef dv) const
{
    dv = log3(quat_from(q0, 0).normalized().conjugate() * quat_from(q1, 0).normalized());
}

void SphericalJointModel::normalize(VecRef q) const
{
    quat_normalize(q, 0);
}

void SphericalJointModel::neutral(VecRef q) const
{
    q << 0.0, 0.0, 0.0, 1.0;
}

// --- Free ----------------------------------------------------------------------
//
// v holds the velocity of the child-side frame, in that frame, so S is the
// identity and c is zero. The translation is in the parent-side frame:
// t_dot = R_J v_linear.

void FreeJointModel::calc(ConstVecRef q, ConstVecRef /*v*/, JointData& d) const
{
    d.X_J = Transform3(quat_from(q, 3), q.head<3>());
    d.S.setIdentity(6, 6);
    d.c.setZero();
}

void FreeJointModel::q_dot(ConstVecRef q, ConstVecRef v, VecRef qd) const
{
    const Quat R = quat_from(q, 3).normalized();
    qd.head<3>() = R * v.tail<3>();
    quat_rate(q, 3, v.head<3>(), qd);
}

void FreeJointModel::integrate(ConstVecRef q, ConstVecRef dv, VecRef q_out) const
{
    // The joint frame moves at constant body-fixed velocity: X_J exp(dv).
    const Transform3 X = Transform3(quat_from(q, 3), q.head<3>()) * exp6(dv.head<6>());
    q_out.head<3>()     = X.p;
    q_out.segment<4>(3) = X.q.coeffs();
}

void FreeJointModel::difference(ConstVecRef q0, ConstVecRef q1, VecRef dv) const
{
    const Transform3 X0(quat_from(q0, 3), q0.head<3>());
    const Transform3 X1(quat_from(q1, 3), q1.head<3>());
    dv = log6(X0.inverse() * X1);
}

void FreeJointModel::normalize(VecRef q) const
{
    quat_normalize(q, 3);
}

void FreeJointModel::neutral(VecRef q) const
{
    q << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0;
}

} // namespace mbd::kernel
