#pragma once

// Loop-closing constraints of the kinematics kernel (plan task 2.5,
// docs/kernel.md).
//
// The tree of Model has no loops. A loop is closed by constraint equations
// phi(q, t) = 0 between markers: frames fixed on bodies. Every closure is
// assembled from a few primitive equations on markers, as in E. J. Haug,
// "Computer Aided Kinematics and Dynamics of Mechanical Systems" (1989), and
// A. A. Shabana, "Computational Dynamics" (chapter 3):
//
//   point coincidence    p_j - p_i = 0                          3 equations
//   dot-1                a_i . a_j - s(t) = 0                   1
//   dot-2                a_i . (p_j - p_i) - s(t) = 0           1
//   distance             (|p_j - p_i|^2 - L(t)^2) / (2 L0) = 0  1
//   point on line        dot-2 with two axes across the line    2
//   point on plane       dot-2 with the plane normal            1
//   no twist             x_i . y_j - y_i . x_j = 0              1
//
// where p is a marker's origin and a one of its axes, all in world axes.
// The joint closures (revolute, universal, cylindrical, prismatic, constant
// velocity) are combinations of these.
//
// Each constraint gives, at time t, its value phi, its Jacobian J, and the
// right-hand sides of the velocity and acceleration equations:
//
//   J v = nu,      J v_dot = gamma,
//
// with nu = -d(phi)/dt and gamma = -(dJ/dt) v - d2(phi)/dt2 (the partial time
// derivatives at fixed q).

#include <functional>
#include <memory>
#include <vector>

#include "mbd/kernel/model.hpp"

namespace mbd::kernel {

using MatRef = Eigen::Ref<MatX>;

/// A frame fixed on a body: the body (0 for ground) and the frame's placement
/// in the body frame.
struct Marker {
    int body{0};
    Transform3 X_BM;
};

/// A prescribed function of time, with its first two derivatives, for driven
/// constraints. The default is the constant zero.
class TimeFunction {
public:
    TimeFunction() = default;

    /// The constant `value`.
    static TimeFunction constant(Real value);

    /// Any function: s(t), ds/dt, d2s/dt2.
    TimeFunction(std::function<Real(Real)> s,
                 std::function<Real(Real)> ds,
                 std::function<Real(Real)> dds);

    Real value(Real t) const { return s_ ? s_(t) : constant_; }
    Real rate(Real t) const { return ds_ ? ds_(t) : 0.0; }
    Real acceleration(Real t) const { return dds_ ? dds_(t) : 0.0; }

private:
    Real constant_{0.0};
    std::function<Real(Real)> s_, ds_, dds_;
};

/// Interface of all constraints. Constraint models hold no state.
class ConstraintModel {
public:
    virtual ~ConstraintModel() = default;

    /// Short name, for messages and test output.
    virtual const char* name() const = 0;

    /// Number of equations.
    virtual int size() const = 0;

    /// phi, J, nu and gamma at time t (see the top of this file), written to
    /// the first size() rows of the arguments. `data` must hold
    /// forward_kinematics(q, v, 0): placements, velocities, and the body
    /// accelerations at zero joint accelerations. J is overwritten in the
    /// columns of every joint.
    virtual void calc(const Model& model, const Data& data, Real t,
                      VecRef phi, MatRef J, VecRef nu, VecRef gamma) const = 0;

protected:
    ConstraintModel() = default;
};

// ============================================================================
// Primitives
// ============================================================================

/// p_j - p_i = 0, in world axes: the origins of two markers coincide.
class PointCoincidence final : public ConstraintModel {
public:
    PointCoincidence(Marker i, Marker j) : i_(i), j_(j) {}
    const char* name() const override { return "point coincidence"; }
    int size() const override { return 3; }
    void calc(const Model& model, const Data& data, Real t,
              VecRef phi, MatRef J, VecRef nu, VecRef gamma) const override;

private:
    Marker i_, j_;
};

/// a_i . a_j - s(t) = 0, with a_i axis `axis_i` (0 = X, 1 = Y, 2 = Z) of
/// marker i and a_j axis `axis_j` of marker j. With s = 0 the axes stay
/// perpendicular (Haug's dot-1).
class Dot1 final : public ConstraintModel {
public:
    Dot1(Marker i, int axis_i, Marker j, int axis_j, TimeFunction s = {});
    const char* name() const override { return "dot-1"; }
    int size() const override { return 1; }
    void calc(const Model& model, const Data& data, Real t,
              VecRef phi, MatRef J, VecRef nu, VecRef gamma) const override;

private:
    Marker i_, j_;
    int axis_i_, axis_j_;
    TimeFunction s_;
};

/// a_i . (p_j - p_i) - s(t) = 0, with a_i axis `axis_i` of marker i: the
/// origin of marker j stays at signed distance s from the plane through p_i
/// normal to a_i (Haug's dot-2).
class Dot2 final : public ConstraintModel {
public:
    Dot2(Marker i, int axis_i, Marker j, TimeFunction s = {});
    const char* name() const override { return "dot-2"; }
    int size() const override { return 1; }
    void calc(const Model& model, const Data& data, Real t,
              VecRef phi, MatRef J, VecRef nu, VecRef gamma) const override;

private:
    Marker i_, j_;
    int axis_i_;
    TimeFunction s_;
};

/// (|p_j - p_i|^2 - L(t)^2) / (2 L0) = 0: the origins of two markers stay a
/// distance L apart. Dividing by the nominal length L0 gives phi the units of
/// a length (it equals |p_j - p_i| - L to first order) without the
/// singularity of |p_j - p_i| at zero.
class Distance final : public ConstraintModel {
public:
    /// Constant length L0.
    Distance(Marker i, Marker j, Real length);
    /// Length L(t), scaled by the nominal length L0 > 0.
    Distance(Marker i, Marker j, TimeFunction length, Real nominal_length);
    const char* name() const override { return "distance"; }
    int size() const override { return 1; }
    void calc(const Model& model, const Data& data, Real t,
              VecRef phi, MatRef J, VecRef nu, VecRef gamma) const override;

private:
    Marker i_, j_;
    TimeFunction L_;
    Real L0_;
};

/// x_i . y_j - y_i . x_j = 0: no relative rotation about the Z axes, when
/// those are close to aligned. This is the condition of a constant-velocity
/// joint: the relative rotation of the two markers is a pure bend, about an
/// axis across their Z axes.
class NoTwist final : public ConstraintModel {
public:
    NoTwist(Marker i, Marker j) : i_(i), j_(j) {}
    const char* name() const override { return "no twist"; }
    int size() const override { return 1; }
    void calc(const Model& model, const Data& data, Real t,
              VecRef phi, MatRef J, VecRef nu, VecRef gamma) const override;

private:
    Marker i_, j_;
};

// ============================================================================
// Composites
// ============================================================================

/// Several constraints stacked, in order.
class ConstraintSet final : public ConstraintModel {
public:
    explicit ConstraintSet(const char* set_name) : name_(set_name) {}

    void add(std::shared_ptr<const ConstraintModel> c);

    const char* name() const override { return name_; }
    int size() const override { return size_; }
    void calc(const Model& model, const Data& data, Real t,
              VecRef phi, MatRef J, VecRef nu, VecRef gamma) const override;

    const std::vector<std::shared_ptr<const ConstraintModel>>& parts() const { return parts_; }

private:
    const char* name_;
    int size_{0};
    std::vector<std::shared_ptr<const ConstraintModel>> parts_;
};

/// The origin of marker j stays on the Z axis of marker i (two dot-2).
std::shared_ptr<ConstraintSet> point_on_line(Marker i, Marker j);

/// The origin of marker j stays on the XY plane of marker i (one dot-2).
std::shared_ptr<ConstraintSet> point_on_plane(Marker i, Marker j);

// Joint closures. Marker i is on one body, marker j on the other; they
// coincide when the joint is assembled. The joint axis is Z, as for the
// joint models of the tree.

/// Origins coincide (3 equations).
std::shared_ptr<ConstraintSet> spherical_closure(Marker i, Marker j);

/// Origins coincide and the Z axes stay aligned (5 equations).
std::shared_ptr<ConstraintSet> revolute_closure(Marker i, Marker j);

/// Origins coincide and the Z axes stay perpendicular: the cross of a
/// Cardan joint, arms along Z of i and Z of j (4 equations).
std::shared_ptr<ConstraintSet> universal_closure(Marker i, Marker j);

/// The origin of j stays on the Z axis of i and the Z axes stay aligned
/// (4 equations).
std::shared_ptr<ConstraintSet> cylindrical_closure(Marker i, Marker j);

/// As cylindrical, without rotation about Z (5 equations).
std::shared_ptr<ConstraintSet> prismatic_closure(Marker i, Marker j);

/// Origins coincide and the shafts, along the Z axes, turn at the same rate
/// (4 equations).
std::shared_ptr<ConstraintSet> constant_velocity_closure(Marker i, Marker j);

// ============================================================================
// Helpers
// ============================================================================

/// World kinematics of a marker, from Data after forward_kinematics(q, v, 0).
/// The accelerations are those at zero joint accelerations.
struct MarkerKinematics {
    Vec3 p;      ///< Origin
    Mat3 R;      ///< Axes, as columns
    Vec3 w;      ///< Angular velocity
    Vec3 v;      ///< Velocity of the origin
    Vec3 alpha;  ///< Angular acceleration at zero joint accelerations
    Vec3 a;      ///< Acceleration of the origin at zero joint accelerations
};

MarkerKinematics marker_kinematics(const Data& data, const Marker& m);

/// Row r of J += f^T J_body: the generalized force of the world wrench
/// f = [moment about the world origin; force] applied to `body`. Every
/// Jacobian row of a constraint is a sum of such terms.
void add_wrench_row(const Model& model, const Data& data, int body, const Vec6& f,
                    MatRef J, Index r);

} // namespace mbd::kernel
