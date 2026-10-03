#pragma once

// Spatial (six-dimensional) vector algebra, after R. Featherstone, "Rigid
// Body Dynamics Algorithms" (Springer, 2008). Conventions (docs/kernel.md):
//
//   - Motion vectors (velocities, accelerations) and force vectors (forces
//     with their moment) are Vec6, ordered [angular; linear].
//   - A motion vector "in frame A" holds the angular velocity and the
//     velocity of the point at A's origin, both in A's axes. A force vector
//     "in frame A" holds the moment about A's origin and the force, in A's
//     axes.
//   - X_AB (a Transform3) maps coordinates from frame B to frame A, as
//     everywhere else in the code: rotation R = X_AB.q, translation p = X_AB.p
//     (the origin of B, in A).
//
// motion_act(X_AB, m_B) re-expresses in A a motion vector given in B;
// motion_act_inv(X_AB, m_A) goes back. force_act / force_act_inv likewise.
// The 6x6 matrix forms are
//
//     motion_matrix(X_AB) = [ R       0 ]      force_matrix(X_AB) = [ R  [p]x R ]
//                           [ [p]x R  R ]                           [ 0    R    ]
//
// and force_matrix = motion_matrix^-T, so that power f . m is the same in
// every frame.

#include "mbd/core/math.hpp"
#include "mbd/model/rigid_body.hpp"

namespace mbd {

//------------------------------------------------------------------------------
// Change of frame
//------------------------------------------------------------------------------

/// Motion vector given in B, expressed in A.
inline Vec6 motion_act(const Transform3& X_AB, const Vec6& m_B)
{
    const Vec3 w = X_AB.q * m_B.head<3>();
    const Vec3 v = X_AB.q * m_B.tail<3>() + X_AB.p.cross(w);
    Vec6 m_A;
    m_A << w, v;
    return m_A;
}

/// Motion vector given in A, expressed in B.
inline Vec6 motion_act_inv(const Transform3& X_AB, const Vec6& m_A)
{
    const Quat q_BA = X_AB.q.conjugate();
    const Vec3 w_A  = m_A.head<3>();
    Vec6 m_B;
    m_B << q_BA * w_A, q_BA * (m_A.tail<3>() - X_AB.p.cross(w_A));
    return m_B;
}

/// Force vector given in B, expressed in A.
inline Vec6 force_act(const Transform3& X_AB, const Vec6& f_B)
{
    const Vec3 f = X_AB.q * f_B.tail<3>();
    const Vec3 n = X_AB.q * f_B.head<3>() + X_AB.p.cross(f);
    Vec6 f_A;
    f_A << n, f;
    return f_A;
}

/// Force vector given in A, expressed in B.
inline Vec6 force_act_inv(const Transform3& X_AB, const Vec6& f_A)
{
    const Quat q_BA = X_AB.q.conjugate();
    const Vec3 f    = f_A.tail<3>();
    Vec6 f_B;
    f_B << q_BA * (f_A.head<3>() - X_AB.p.cross(f)), q_BA * f;
    return f_B;
}

/// 6x6 matrix of motion_act(X_AB, .).
inline Mat6 motion_matrix(const Transform3& X_AB)
{
    const Mat3 R = X_AB.q.toRotationMatrix();
    Mat6 X = Mat6::Zero();
    X.topLeftCorner<3, 3>()     = R;
    X.bottomLeftCorner<3, 3>()  = skew(X_AB.p) * R;
    X.bottomRightCorner<3, 3>() = R;
    return X;
}

/// 6x6 matrix of force_act(X_AB, .).
inline Mat6 force_matrix(const Transform3& X_AB)
{
    const Mat3 R = X_AB.q.toRotationMatrix();
    Mat6 X = Mat6::Zero();
    X.topLeftCorner<3, 3>()     = R;
    X.topRightCorner<3, 3>()    = skew(X_AB.p) * R;
    X.bottomRightCorner<3, 3>() = R;
    return X;
}

//------------------------------------------------------------------------------
// Cross products
//------------------------------------------------------------------------------

/// m1 x m2 for two motion vectors (Featherstone's crm). The rate of change of
/// m2 seen from a frame moving with velocity m1.
inline Vec6 motion_cross_motion(const Vec6& m1, const Vec6& m2)
{
    const Vec3 w1 = m1.head<3>(), v1 = m1.tail<3>();
    const Vec3 w2 = m2.head<3>(), v2 = m2.tail<3>();
    Vec6 out;
    out << w1.cross(w2), w1.cross(v2) + v1.cross(w2);
    return out;
}

/// m x* f for a motion and a force vector (Featherstone's crf). Equal to
/// -(m x)^T f, so that (m x m2) . f + m2 . (m x* f) = 0.
inline Vec6 motion_cross_force(const Vec6& m, const Vec6& f)
{
    const Vec3 w = m.head<3>(), v = m.tail<3>();
    const Vec3 n = f.head<3>(), fl = f.tail<3>();
    Vec6 out;
    out << w.cross(n) + v.cross(fl), w.cross(fl);
    return out;
}

//------------------------------------------------------------------------------
// Rigid-body inertia
//------------------------------------------------------------------------------
//
// RigidBodyInertia holds the mass, the centre of mass c in the body frame and
// the rotational inertia about the centre of mass, in body axes. As a spatial
// inertia about the body-frame origin:
//
//     I = [ I_c + m [c]x [c]x^T   m [c]x ]
//         [ m [c]x^T              m 1    ]

/// The 6x6 spatial inertia about the body-frame origin.
inline Mat6 spatial_inertia_matrix(const RigidBodyInertia& I)
{
    const Mat3 cx = skew(I.com_B);
    Mat6 M;
    M.topLeftCorner<3, 3>()     = I.I_com_B + I.mass * cx * cx.transpose();
    M.topRightCorner<3, 3>()    = I.mass * cx;
    M.bottomLeftCorner<3, 3>()  = I.mass * cx.transpose();
    M.bottomRightCorner<3, 3>() = I.mass * Mat3::Identity();
    return M;
}

/// I * m: the momentum of a body moving with motion vector m, without
/// forming the matrix. Linear momentum h = m (v + w x c); angular momentum
/// about the origin I_c w + c x h.
inline Vec6 spatial_inertia_times(const RigidBodyInertia& I, const Vec6& m)
{
    const Vec3 w = m.head<3>(), v = m.tail<3>();
    const Vec3 h = I.mass * (v + w.cross(I.com_B));
    Vec6 out;
    out << I.I_com_B * w + I.com_B.cross(h), h;
    return out;
}

/// A 6x6 spatial inertia given in B, expressed in A:
/// I_A = force_matrix(X_AB) * I_B * force_matrix(X_AB)^T.
inline Mat6 inertia_act(const Transform3& X_AB, const Mat6& I_B)
{
    const Mat6 Xf = force_matrix(X_AB);
    return Xf * I_B * Xf.transpose();
}

//------------------------------------------------------------------------------
// Exponential maps
//------------------------------------------------------------------------------
//
// exp3(w) is the rotation by the angle |w| about w; log3 inverts it, choosing
// the angle in [0, pi]. exp6(xi) is the rigid motion of a frame that moves for
// unit time at the constant body-fixed velocity xi = [w; v] (a screw motion):
// rotation exp3(w), translation V(w) v with
//
//     V(w) = I + (1 - cos t) / t^2 [w]x + (t - sin t) / t^3 [w]x^2,  t = |w|.
//
// log6 inverts it. Both use series near t = 0, where the closed forms lose
// precision.

/// Rotation vector to unit quaternion.
inline Quat exp3(const Vec3& w)
{
    return delta_rotation_from_omega(w, Real(1.0));
}

/// Unit quaternion to rotation vector, angle in [0, pi].
inline Vec3 log3(const Quat& q)
{
    // q and -q are the same rotation: take the one with w >= 0.
    const Real sign = q.w() < 0.0 ? Real(-1.0) : Real(1.0);
    const Vec3 u = sign * q.vec();
    const Real s = u.norm();
    const Real c = sign * q.w();
    // The angle is 2 atan2(s, c), and angle / s tends to 2 / c as s -> 0.
    const Real k = s < Real(1e-8) ? Real(2.0) / c : Real(2.0) * std::atan2(s, c) / s;
    return k * u;
}

/// Body-fixed velocity [w; v] held for unit time, to the motion it produces.
inline Transform3 exp6(const Vec6& xi)
{
    const Vec3 w = xi.head<3>(), v = xi.tail<3>();
    const Real t = w.norm();
    Real a, b;  // (1 - cos t) / t^2 and (t - sin t) / t^3
    if (t < Real(1e-2)) {
        const Real t2 = t * t;
        a = Real(1.0 / 2.0) - t2 / Real(24.0) + t2 * t2 / Real(720.0);
        b = Real(1.0 / 6.0) - t2 / Real(120.0) + t2 * t2 / Real(5040.0);
    } else {
        const Real sh = std::sin(Real(0.5) * t);
        a = Real(2.0) * sh * sh / (t * t);
        b = (t - std::sin(t)) / (t * t * t);
    }
    const Vec3 wv = w.cross(v);
    return Transform3(exp3(w), v + a * wv + b * w.cross(wv));
}

/// Inverse of exp6: the body-fixed velocity that moves the identity to X in
/// unit time, with a rotation angle in [0, pi].
inline Vec6 log6(const Transform3& X)
{
    const Vec3 w = log3(X.q);
    const Real t = w.norm();
    // V(w)^-1 = I - [w]x / 2 + c [w]x^2,  c = (1 - (t/2) cot(t/2)) / t^2.
    Real c;
    if (t < Real(1e-2)) {
        const Real t2 = t * t;
        c = Real(1.0 / 12.0) + t2 / Real(720.0) + t2 * t2 / Real(30240.0);
    } else {
        const Real h = Real(0.5) * t;
        c = (Real(1.0) - h * std::cos(h) / std::sin(h)) / (t * t);
    }
    const Vec3 wp = w.cross(X.p);
    Vec6 xi;
    xi << w, X.p - Real(0.5) * wp + c * w.cross(wp);
    return xi;
}

} // namespace mbd
