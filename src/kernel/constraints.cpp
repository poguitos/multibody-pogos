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
    MBD_THROW_IF(axis < 0 || axis > 2, "MBD-K020: kernel constraint: axis must be 0 (X), 1 (Y) or 2 (Z)");
}

/// Where the world wrenches of a constraint's rows go: into the Jacobian
/// (calc), or, weighted by the multipliers, onto the bodies (add_wrenches).
/// Each primitive states its rows once, in a function that takes a sink.
class RowSink {
public:
    virtual void add(int body, const Vec6& wrench_W, Index row) = 0;

protected:
    ~RowSink() = default;
};

class JacobianSink final : public RowSink {
public:
    JacobianSink(const Model& model, const Data& data, MatRef J) : model_(model), data_(data), J_(J) {}
    void add(int body, const Vec6& wrench_W, Index row) override
    {
        add_wrench_row(model_, data_, body, wrench_W, J_, row);
    }

private:
    const Model& model_;
    const Data& data_;
    MatRef J_;
};

class WrenchSink final : public RowSink {
public:
    WrenchSink(ConstVecRef lambda, std::vector<Vec6>& wrenches) : lambda_(lambda), wrenches_(wrenches) {}
    void add(int body, const Vec6& wrench_W, Index row) override
    {
        wrenches_[static_cast<std::size_t>(body)] += lambda_(row) * wrench_W;
    }

private:
    ConstVecRef lambda_;
    std::vector<Vec6>& wrenches_;
};

void point_coincidence_rows(const Marker& mi, const MarkerKinematics& ki,
                            const Marker& mj, const MarkerKinematics& kj, RowSink& sink)
{
    for (int k = 0; k < 3; ++k) {
        const Vec3 e = Vec3::Unit(k);
        sink.add(mj.body, wrench(Vec3::Zero(), e, kj.p), k);
        sink.add(mi.body, wrench(Vec3::Zero(), -e, ki.p), k);
    }
}

/// Row r of a_i . a_j, for axis_i of marker i and axis_j of marker j, times
/// sign: d/dt (a_i . a_j) = (w_i x a_i) . a_j + a_i . (w_j x a_j) = n . (w_i - w_j).
void dot1_rows(const Marker& mi, const MarkerKinematics& ki, int axis_i,
               const Marker& mj, const MarkerKinematics& kj, int axis_j,
               Real sign, Index r, RowSink& sink)
{
    const Vec3 n = ki.R.col(axis_i).cross(kj.R.col(axis_j));
    sink.add(mi.body, wrench(sign * n, Vec3::Zero(), Vec3::Zero()), r);
    sink.add(mj.body, wrench(-sign * n, Vec3::Zero(), Vec3::Zero()), r);
}

/// d/dt (a_i . d) = (a_i x d) . w_i + a_i . (v_j - v_i)
void dot2_rows(const Marker& mi, const MarkerKinematics& ki, int axis_i,
               const Marker& mj, const MarkerKinematics& kj, RowSink& sink)
{
    const Vec3 ai = ki.R.col(axis_i);
    const Vec3 d  = kj.p - ki.p;
    sink.add(mi.body, wrench(ai.cross(d), -ai, ki.p), 0);
    sink.add(mj.body, wrench(Vec3::Zero(), ai, kj.p), 0);
}

/// d/dt phi = d . (v_j - v_i) / L0 - L L_dot / L0
void distance_rows(const Marker& mi, const MarkerKinematics& ki,
                   const Marker& mj, const MarkerKinematics& kj, Real L0, RowSink& sink)
{
    const Vec3 d = kj.p - ki.p;
    sink.add(mj.body, wrench(Vec3::Zero(), d / L0, kj.p), 0);
    sink.add(mi.body, wrench(Vec3::Zero(), -d / L0, ki.p), 0);
}

/// x_i . y_j - y_i . x_j
void no_twist_rows(const Marker& mi, const MarkerKinematics& ki,
                   const Marker& mj, const MarkerKinematics& kj, RowSink& sink)
{
    dot1_rows(mi, ki, 0, mj, kj, 1, 1.0, 0, sink);
    dot1_rows(mi, ki, 1, mj, kj, 0, -1.0, 0, sink);
}

/// a_i . a_j: the value and the velocity-product part of its second
/// derivative.
struct DotTerms {
    Real value;
    Real bias;
};

DotTerms dot1_terms(const MarkerKinematics& ki, int axis_i, const MarkerKinematics& kj, int axis_j)
{
    const Vec3 ai = ki.R.col(axis_i);
    const Vec3 aj = kj.R.col(axis_j);
    const Vec3 n  = ai.cross(aj);
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
    MBD_THROW_IF(!s_ || !ds_ || !dds_, "MBD-K021: TimeFunction: the function and both derivatives are needed");
}

// --- Primitives ------------------------------------------------------------------

void PointCoincidence::calc(const Model& model, const Data& data, Real /*t*/,
                            VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    J.topRows(3).setZero();
    JacobianSink sink(model, data, J);
    point_coincidence_rows(i_, ki, j_, kj, sink);
    phi.head<3>()   = kj.p - ki.p;
    nu.head<3>().setZero();
    gamma.head<3>() = ki.a - kj.a;
}

void PointCoincidence::add_wrenches(const Model& /*model*/, const Data& data, Real /*t*/,
                                    ConstVecRef lambda, std::vector<Vec6>& wrenches) const
{
    WrenchSink sink(lambda, wrenches);
    point_coincidence_rows(i_, marker_kinematics(data, i_), j_, marker_kinematics(data, j_), sink);
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
    JacobianSink sink(model, data, J);
    dot1_rows(i_, ki, axis_i_, j_, kj, axis_j_, 1.0, 0, sink);
    const DotTerms d = dot1_terms(ki, axis_i_, kj, axis_j_);
    phi(0)   = d.value - s_.value(t);
    nu(0)    = s_.rate(t);
    gamma(0) = -d.bias + s_.acceleration(t);
}

void Dot1::add_wrenches(const Model& /*model*/, const Data& data, Real /*t*/,
                        ConstVecRef lambda, std::vector<Vec6>& wrenches) const
{
    WrenchSink sink(lambda, wrenches);
    dot1_rows(i_, marker_kinematics(data, i_), axis_i_, j_, marker_kinematics(data, j_), axis_j_,
              1.0, 0, sink);
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
    J.topRows(1).setZero();
    JacobianSink sink(model, data, J);
    dot2_rows(i_, ki, axis_i_, j_, kj, sink);
    const Vec3 ai_dot = ki.w.cross(ai);
    const Real bias = (ki.alpha.cross(ai) + ki.w.cross(ai_dot)).dot(d)
                    + 2.0 * ai_dot.dot(kj.v - ki.v)
                    + ai.dot(kj.a - ki.a);
    phi(0)   = ai.dot(d) - s_.value(t);
    nu(0)    = s_.rate(t);
    gamma(0) = -bias + s_.acceleration(t);
}

void Dot2::add_wrenches(const Model& /*model*/, const Data& data, Real /*t*/,
                        ConstVecRef lambda, std::vector<Vec6>& wrenches) const
{
    WrenchSink sink(lambda, wrenches);
    dot2_rows(i_, marker_kinematics(data, i_), axis_i_, j_, marker_kinematics(data, j_), sink);
}

Distance::Distance(Marker i, Marker j, Real length)
    : Distance(i, j, TimeFunction::constant(length), length)
{}

Distance::Distance(Marker i, Marker j, TimeFunction length, Real nominal_length)
    : i_(i), j_(j), L_(std::move(length)), L0_(nominal_length)
{
    MBD_THROW_IF(!(nominal_length > 0.0), "MBD-K022: kernel::Distance: the nominal length must be positive");
}

void Distance::calc(const Model& model, const Data& data, Real t,
                    VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    const Vec3 d  = kj.p - ki.p;
    const Vec3 dv = kj.v - ki.v;
    const Real L = L_.value(t), L_dot = L_.rate(t), L_ddot = L_.acceleration(t);
    J.topRows(1).setZero();
    JacobianSink sink(model, data, J);
    distance_rows(i_, ki, j_, kj, L0_, sink);
    phi(0)   = (d.squaredNorm() - L * L) / (2.0 * L0_);
    nu(0)    = L * L_dot / L0_;
    gamma(0) = (L_dot * L_dot + L * L_ddot - dv.squaredNorm() - d.dot(kj.a - ki.a)) / L0_;
}

void Distance::add_wrenches(const Model& /*model*/, const Data& data, Real /*t*/,
                            ConstVecRef lambda, std::vector<Vec6>& wrenches) const
{
    WrenchSink sink(lambda, wrenches);
    distance_rows(i_, marker_kinematics(data, i_), j_, marker_kinematics(data, j_), L0_, sink);
}

void NoTwist::calc(const Model& model, const Data& data, Real /*t*/,
                   VecRef phi, MatRef J, VecRef nu, VecRef gamma) const
{
    const MarkerKinematics ki = marker_kinematics(data, i_);
    const MarkerKinematics kj = marker_kinematics(data, j_);
    J.topRows(1).setZero();
    JacobianSink sink(model, data, J);
    no_twist_rows(i_, ki, j_, kj, sink);
    const DotTerms xy = dot1_terms(ki, 0, kj, 1);
    const DotTerms yx = dot1_terms(ki, 1, kj, 0);
    phi(0)   = xy.value - yx.value;
    nu(0)    = 0.0;
    gamma(0) = yx.bias - xy.bias;
}

void NoTwist::add_wrenches(const Model& /*model*/, const Data& data, Real /*t*/,
                           ConstVecRef lambda, std::vector<Vec6>& wrenches) const
{
    WrenchSink sink(lambda, wrenches);
    no_twist_rows(i_, marker_kinematics(data, i_), j_, marker_kinematics(data, j_), sink);
}

void ConstraintModel::add_wrenches(const Model& /*model*/, const Data& /*data*/, Real /*t*/,
                                   ConstVecRef /*lambda*/, std::vector<Vec6>& /*wrenches*/) const
{}

JointDriver::JointDriver(const Model& model, int body, TimeFunction s)
    : body_(body), iv_(0), revolute_(false), s_(std::move(s))
{
    MBD_THROW_IF(body < 1 || body >= model.nbodies(), "MBD-K023: kernel::JointDriver: no such body");
    const std::string kind = model.joint[body]->name();
    MBD_THROW_IF(kind != "revolute" && kind != "prismatic",
                 "MBD-K023: kernel::JointDriver: the joint must be revolute or prismatic");
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
    MBD_THROW_IF(!c, "MBD-K024: kernel::ConstraintSet::add: no constraint");
    size_ += c->size();
    parts_.push_back(std::move(c));
}

std::vector<int> ConstraintSet::bodies() const
{
    std::vector<int> out;
    for (const auto& c : parts_) {
        const std::vector<int> b = c->bodies();
        out.insert(out.end(), b.begin(), b.end());
    }
    return out;
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

void ConstraintSet::add_wrenches(const Model& model, const Data& data, Real t,
                                 ConstVecRef lambda, std::vector<Vec6>& wrenches) const
{
    Index r = 0;
    for (const auto& c : parts_) {
        const Index m = c->size();
        c->add_wrenches(model, data, t, lambda.segment(r, m), wrenches);
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
