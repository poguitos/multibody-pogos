// Loop-closing constraints of the kinematics kernel. Notation in
// include/mbd/kernel/constraints.hpp; derivations in docs/kernel.md.
//
// Each Jacobian row is a sum of terms c_w . w + c_p . v_P over the bodies of
// the two markers (w a body's angular velocity, v_P the velocity of a point P
// on it). Such a term is the power of the world wrench [c_w + P x c_p; c_p]
// on the body, which add_wrench_row turns into generalized forces.

#include "mbd/kernel/constraints.hpp"

#include <cmath>
#include <string>
#include <utility>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/spatial/spatial.hpp"

namespace mbd::kernel {

namespace {

/// The world wrench whose power on a body is c_w . w + c_p . v_P, for the
/// body point P at p.
Vec6 wrench(const Vec3& c_w, const Vec3& c_p, const Vec3& p)
{
    Vec6 f;
    f << c_w + p.cross(c_p), c_p;
    return f;
}

void check_axis(int axis)
{
    MBD_THROW_IF(axis < 0 || axis > 2, "kernel constraint: axis must be 0 (X), 1 (Y) or 2 (Z)");
}

/// a_i . a_j for axis_i of marker i and axis_j of marker j. Adds sign times its
/// Jacobian to row r of J; returns the value and the velocity-product part of
/// its second derivative.
struct DotTerms {
    Real value;
    Real bias;
};

DotTerms dot1_terms(const Model& model, const Data& data,
                    const Marker& mi, const MarkerKinematics& ki, int axis_i,
                    const Marker& mj, const MarkerKinematics& kj, int axis_j,
                    Real sign, MatRef J, Index r)
{
    const Vec3 ai = ki.R.col(axis_i);
    const Vec3 aj = kj.R.col(axis_j);
    const Vec3 n  = ai.cross(aj);
    // d/dt (a_i . a_j) = (w_i x a_i) . a_j + a_i . (w_j x a_j) = n . (w_i - w_j)
    add_wrench_row(model, data, mi.body, wrench(sign * n, Vec3::Zero(), Vec3::Zero()), J, r);
    add_wrench_row(model, data, mj.body, wrench(-sign * n, Vec3::Zero(), Vec3::Zero()), J, r);
    const Vec3 n_dot = ki.w.cross(ai).cross(aj) + ai.cross(kj.w.cross(aj));
    return {ai.dot(aj), n_dot.dot(ki.w - kj.w) + n.dot(ki.alpha - kj.alpha)};
}

} // namespace

// --- Helpers -------------------------------------------------------------------

MarkerKinematics marker_kinematics(const Data& data, const Marker& m)
{
    MarkerKinematics k;
    const Transform3& X_WB = data.oMi[m.body];
    const Transform3  X_WM = X_WB * m.X_BM;
    k.p = X_WM.p;
    k.R = X_WM.rotation_matrix();
    if (m.body == 0) {
        k.w.setZero();
        k.v.setZero();
        k.alpha.setZero();
        k.a.setZero();
        return k;
    }
    const Vec6 V = body_velocity_world(data, m.body);
    const Vec6 A = body_acceleration_world(data, m.body);
    const Vec3 r = k.p - X_WB.p;
    k.w     = V.head<3>();
    k.v     = V.tail<3>() + k.w.cross(r);
    k.alpha = A.head<3>();
    k.a     = A.tail<3>() + k.alpha.cross(r) + k.w.cross(k.w.cross(r));
    return k;
}

void add_wrench_row(const Model& model, const Data& data, int body, const Vec6& f,
                    MatRef J, Index r)
{
    for (int j = body; j > 0; j = model.parent[j]) {
        const int nvj = model.nvs[j];
        if (nvj == 0) continue;
        // The power of f on joint j's motions, in body j's frame.
        const Vec6 f_j = force_act_inv(data.oMi[j], f);
        J.row(r).segment(model.idx_v[j], nvj) += (data.S[j].transpose() * f_j).transpose();
    }
}

// --- TimeFunction ----------------------------------------------------------------

TimeFunction TimeFunction::constant(Real value)
{
    TimeFunction f;
    f.constant_ = value;
    return f;
}

TimeFunction::TimeFunction(std::function<Real(Real)> s,
                           std::function<Real(Real)> ds,
                           std::function<Real(Real)> dds)
    : s_(std::move(s)), ds_(std::move(ds)), dds_(std::move(dds))
{
    MBD_THROW_IF(!s_ || !ds_ || !dds_, "TimeFunction: the function and both derivatives are needed");
}

// --- Primitives ------------------------------------------------------------------

void PointCoincidence::calc(const Model& model, const Data& data, Real /*t*/,
                            VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    J.topRows(3).setZero();
    for (int k = 0; k < 3; ++k) {
        const Vec3 e = Vec3::Unit(k);
        add_wrench_row(model, data, j_.body, wrench(Vec3::Zero(), e, kj.p), J, k);
        add_wrench_row(model, data, i_.body, wrench(Vec3::Zero(), -e, ki.p), J, k);
    }
    phi.head<3>()   = kj.p - ki.p;
    nu.head<3>().setZero();
    gamma.head<3>() = ki.a - kj.a;
}

Dot1::Dot1(Marker i, int axis_i, Marker j, int axis_j, TimeFunction s)
    : i_(i), j_(j), axis_i_(axis_i), axis_j_(axis_j), s_(std::move(s))
{
    check_axis(axis_i);
    check_axis(axis_j);
}

void Dot1::calc(const Model& model, const Data& data, Real t,
                VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    J.topRows(1).setZero();
    const DotTerms d = dot1_terms(model, data, i_, ki, axis_i_, j_, kj, axis_j_, 1.0, J, 0);
    phi(0)   = d.value - s_.value(t);
    nu(0)    = s_.rate(t);
    gamma(0) = -d.bias + s_.acceleration(t);
}

Dot2::Dot2(Marker i, int axis_i, Marker j, TimeFunction s)
    : i_(i), j_(j), axis_i_(axis_i), s_(std::move(s))
{
    check_axis(axis_i);
}

void Dot2::calc(const Model& model, const Data& data, Real t,
                VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    const Vec3 ai = ki.R.col(axis_i_);
    const Vec3 d  = kj.p - ki.p;
    // d/dt (a_i . d) = (a_i x d) . w_i + a_i . (v_j - v_i)
    J.topRows(1).setZero();
    add_wrench_row(model, data, i_.body, wrench(ai.cross(d), -ai, ki.p), J, 0);
    add_wrench_row(model, data, j_.body, wrench(Vec3::Zero(), ai, kj.p), J, 0);
    const Vec3 ai_dot = ki.w.cross(ai);
    const Real bias = (ki.alpha.cross(ai) + ki.w.cross(ai_dot)).dot(d)
                    + 2.0 * ai_dot.dot(kj.v - ki.v)
                    + ai.dot(kj.a - ki.a);
    phi(0)   = ai.dot(d) - s_.value(t);
    nu(0)    = s_.rate(t);
    gamma(0) = -bias + s_.acceleration(t);
}

Distance::Distance(Marker i, Marker j, Real length)
    : Distance(i, j, TimeFunction::constant(length), length)
{}

Distance::Distance(Marker i, Marker j, TimeFunction length, Real nominal_length)
    : i_(i), j_(j), L_(std::move(length)), L0_(nominal_length)
{
    MBD_THROW_IF(!(nominal_length > 0.0), "kernel::Distance: the nominal length must be positive");
}

void Distance::calc(const Model& model, const Data& data, Real t,
                    VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    const Vec3 d  = kj.p - ki.p;
    const Vec3 dv = kj.v - ki.v;
    const Real L = L_.value(t), L_dot = L_.rate(t), L_ddot = L_.acceleration(t);
    // d/dt phi = d . (v_j - v_i) / L0 - L L_dot / L0
    J.topRows(1).setZero();
    add_wrench_row(model, data, j_.body, wrench(Vec3::Zero(), d / L0_, kj.p), J, 0);
    add_wrench_row(model, data, i_.body, wrench(Vec3::Zero(), -d / L0_, ki.p), J, 0);
    phi(0)   = (d.squaredNorm() - L * L) / (2.0 * L0_);
    nu(0)    = L * L_dot / L0_;
    gamma(0) = (L_dot * L_dot + L * L_ddot - dv.squaredNorm() - d.dot(kj.a - ki.a)) / L0_;
}

void NoTwist::calc(const Model& model, const Data& data, Real /*t*/,
                   VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    J.topRows(1).setZero();
    const DotTerms xy = dot1_terms(model, data, i_, ki, 0, j_, kj, 1, 1.0, J, 0);
    const DotTerms yx = dot1_terms(model, data, i_, ki, 1, j_, kj, 0, -1.0, J, 0);
    phi(0)   = xy.value - yx.value;
    nu(0)    = 0.0;
    gamma(0) = yx.bias - xy.bias;
}

JointDriver::JointDriver(const Model& model, int body, TimeFunction s)
    : body_(body), iv_(0), revolute_(false), s_(std::move(s))
{
    MBD_THROW_IF(body < 1 || body >= model.nbodies(), "kernel::JointDriver: no such body");
    const std::string kind = model.joint[body]->name();
    MBD_THROW_IF(kind != "revolute" && kind != "prismatic",
                 "kernel::JointDriver: the joint must be revolute or prismatic");
    revolute_ = kind == "revolute";
    iv_ = model.idx_v[body];
}

void JointDriver::calc(const Model& /*model*/, const Data& data, Real t,
                       VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const Real s = s_.value(t);
    const Transform3& X_J = data.joint[body_].X_J;
    Real q;
    if (revolute_) {
        // Rotation about Z by q: the quaternion is (cos(q/2), 0, 0, sin(q/2)).
        const Real angle = 2.0 * std::atan2(X_J.q.z(), X_J.q.w());
        q = s + std::remainder(angle - s, 2.0 * pi);
    } else {
        q = X_J.p.z();
    }
    J.topRows(1).setZero();
    J(0, iv_) = 1.0;
    phi(0)   = q - s;
    nu(0)    = s_.rate(t);
    gamma(0) = s_.acceleration(t);
}

// --- Composites ------------------------------------------------------------------

void ConstraintSet::add(std::shared_ptr<const ConstraintModel> c)
{
    MBD_THROW_IF(!c, "kernel::ConstraintSet::add: no constraint");
    size_ += c->size();
    parts_.push_back(std::move(c));
}

void ConstraintSet::calc(const Model& model, const Data& data, Real t,
                         VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    Index r = 0;
    for (const auto& c : parts_) {
        const Index m = c->size();
        c->calc(model, data, t, phi.segment(r, m), J.middleRows(r, m),
                nu.segment(r, m), gamma.segment(r, m));
        r += m;
    }
}

std::shared_ptr<ConstraintSet> point_on_line(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("point on line");
    c->add(std::make_shared<Dot2>(i, 0, j));
    c->add(std::make_shared<Dot2>(i, 1, j));
    return c;
}

std::shared_ptr<ConstraintSet> point_on_plane(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("point on plane");
    c->add(std::make_shared<Dot2>(i, 2, j));
    return c;
}

std::shared_ptr<ConstraintSet> spherical_closure(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("spherical closure");
    c->add(std::make_shared<PointCoincidence>(i, j));
    return c;
}

std::shared_ptr<ConstraintSet> revolute_closure(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("revolute closure");
    c->add(std::make_shared<PointCoincidence>(i, j));
    c->add(std::make_shared<Dot1>(i, 2, j, 0));
    c->add(std::make_shared<Dot1>(i, 2, j, 1));
    return c;
}

std::shared_ptr<ConstraintSet> universal_closure(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("universal closure");
    c->add(std::make_shared<PointCoincidence>(i, j));
    c->add(std::make_shared<Dot1>(i, 2, j, 2));
    return c;
}

std::shared_ptr<ConstraintSet> cylindrical_closure(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("cylindrical closure");
    c->add(point_on_line(i, j));
    c->add(std::make_shared<Dot1>(i, 2, j, 0));
    c->add(std::make_shared<Dot1>(i, 2, j, 1));
    return c;
}

std::shared_ptr<ConstraintSet> prismatic_closure(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("prismatic closure");
    c->add(cylindrical_closure(i, j));
    c->add(std::make_shared<Dot1>(i, 0, j, 1));
    return c;
}

std::shared_ptr<ConstraintSet> constant_velocity_closure(Marker i, Marker j)
{
    auto c = std::make_shared<ConstraintSet>("constant-velocity closure");
    c->add(std::make_shared<PointCoincidence>(i, j));
    c->add(std::make_shared<NoTwist>(i, j));
    return c;
}

} // namespace mbd::kernel
