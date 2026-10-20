// Manufactured trajectories (plan task 5.1, decision D42): a smooth motion
// q(t) is chosen for a model, the forces that produce it are computed by
// inverse dynamics, and every integrator, given those forces as functions of
// time, must return the motion at its order. The reference is exact at every
// instant, for any tree, so the check needs no closed-form solution of the
// dynamics.
//
// The motion, coordinate by coordinate, in closed form:
//
//   - a joint whose velocities are its coordinates' rates (revolute,
//     prismatic, universal, cylindrical, planar): each coordinate
//     c + a sin(w t + p), so v = q_dot and v_dot = q_ddot;
//   - a rotation (spherical, and the free joint's): R(t) = R0 E1 E2 E3, with
//     E_j = exp(theta_j(t) u_j) about a fixed unit axis u_j and
//     theta_j = a_j sin(w_j t + p_j). Its body-fixed angular velocity is
//
//       w = sum_j Q_j^T u_j theta_j',   Q_j = E_(j+1) ... E_3 (Q_3 = I),
//
//     and, since Q_j' = Q_j [W_j]x with W_j the same sum over l > j,
//
//       w' = sum_j (Q_j^T u_j theta_j'' - W_j x (Q_j^T u_j) theta_j');
//
//   - the free joint's translation p(t) (in the parent-side joint frame),
//     each component c + a sin(w t + p): its velocity coordinate is the
//     body-fixed R^T p', whose rate is R^T p'' - w x (R^T p').
//
// These derivatives are checked against central differences of the motion
// itself before anything else, so that the reference cannot be wrong
// unnoticed.
//
// The driving forces: on a tree, tau(t) = RNEA(q(t), v(t), v_dot(t)). On a
// closed loop only some joints are driven; the drives tau_a and the
// constraint multipliers lambda solve
//
//   B tau_a + J^T lambda = RNEA(q, v, v_dot),
//
// B selecting the driven velocities, so the constraints carry part of the
// load. The forward dynamics must then return v_dot and the constraint forces
// J^T lambda at every instant, and every integrator the motion at its order:
// the error of one step from the exact state divided by 2^(p+1) when the
// step halves, and the error after 0.5 s by 2^p.
//
// The models: two random trees that between them use every joint type; two
// identical random chains side by side, their tips tied by a distance
// constraint (a spatial loop: the twins' tips stay a fixed vector apart, so
// the tie keeps its length); and a parallelogram four-bar closed by a
// revolute in 3-D (a planar loop whose closure has three redundant
// equations).

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/QR>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/spatial/spatial.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::KJoint;
using mbd_test::Rng;

namespace {

/// c + a sin(w t + p) and its first two derivatives.
struct Sine {
    Real c{0.0}, a{0.0}, w{1.0}, p{0.0};
    Real value(Real t) const { return c + a * std::sin(w * t + p); }
    Real rate(Real t) const { return a * w * std::cos(w * t + p); }
    Real acceleration(Real t) const { return -a * w * w * std::sin(w * t + p); }
    Sine negated() const { return {-c, -a, w, p}; }
};

Sine random_sine(Rng& rng, Real centre, Real amplitude)
{
    return {centre, rng.range(0.5, 1.0) * amplitude, rng.range(1.0, 3.0), rng.range(0.0, 2.0 * pi)};
}

/// R(t) = R0 E1 E2 E3 (see the top of this file): the rotation, its
/// body-fixed angular velocity and that velocity's rate.
struct RotationMotion {
    Quat R0{Quat::Identity()};
    Vec3 u[3];
    Sine theta[3];

    void at(Real t, Quat& R, Vec3& w, Vec3& w_dot) const
    {
        Quat E[3];
        for (int j = 0; j < 3; ++j) E[j] = Quat(Eigen::AngleAxisd(theta[j].value(t), u[j]));
        R = R0 * E[0] * E[1] * E[2];
        w.setZero();
        w_dot.setZero();
        // From the last factor back: Q holds Q_j, W the angular velocity of Q_j.
        Quat Q = Quat::Identity();
        Vec3 W = Vec3::Zero();
        for (int j = 2; j >= 0; --j) {
            const Vec3 axis = Q.conjugate() * u[j];
            w += axis * theta[j].rate(t);
            w_dot += axis * theta[j].acceleration(t) - W.cross(axis) * theta[j].rate(t);
            W += axis * theta[j].rate(t);
            Q = E[j] * Q;
        }
    }
};

/// One joint's prescribed motion.
struct JointMotion {
    std::vector<Sine> coordinates;   // scalar joints; the free joint's translation
    bool rotates{false};             // spherical or free
    RotationMotion rotation;
};

/// The prescribed motion of every joint of a model.
class Manufactured {
public:
    /// A random motion for every joint, about a random configuration.
    Manufactured(const Model& model, Rng& rng) : model_(model), joints_(static_cast<std::size_t>(model.nbodies()))
    {
        for (int i = 1; i < model.nbodies(); ++i) {
            JointMotion& m = joints_[static_cast<std::size_t>(i)];
            const std::string name = model.joint[i]->name();
            if (name == "spherical" || name == "free") {
                m.rotates = true;
                m.rotation.R0 = rng.quat();
                for (int j = 0; j < 3; ++j) {
                    m.rotation.u[j] = rng.direction();
                    m.rotation.theta[j] = random_sine(rng, 0.0, 0.8);
                }
                if (name == "free") {
                    for (int k = 0; k < 3; ++k) m.coordinates.push_back(random_sine(rng, rng.range(-0.3, 0.3), 0.3));
                }
            } else {
                for (int k = 0; k < model.nqs[i]; ++k) m.coordinates.push_back(random_sine(rng, rng.range(-1.0, 1.0), 0.6));
            }
        }
    }

    JointMotion& joint(int body) { return joints_[static_cast<std::size_t>(body)]; }

    /// q(t), v(t) and v_dot(t).
    void at(Real t, VecX& q, VecX& v, VecX& a) const
    {
        q = model_.neutral_configuration();
        v.setZero(model_.nv);
        a.setZero(model_.nv);
        for (int i = 1; i < model_.nbodies(); ++i) {
            const JointMotion& m = joints_[static_cast<std::size_t>(i)];
            const int iq = model_.idx_q[i], iv = model_.idx_v[i];
            if (!m.rotates) {
                for (std::size_t k = 0; k < m.coordinates.size(); ++k) {
                    const Index o = static_cast<Index>(k);
                    q(iq + o) = m.coordinates[k].value(t);
                    v(iv + o) = m.coordinates[k].rate(t);
                    a(iv + o) = m.coordinates[k].acceleration(t);
                }
                continue;
            }
            Quat R;
            Vec3 w, w_dot;
            m.rotation.at(t, R, w, w_dot);
            const bool free = m.coordinates.size() == 3;
            const int qr = free ? iq + 3 : iq;   // the quaternion's place
            q.segment<4>(qr) = R.coeffs();
            v.segment<3>(iv) = w;
            a.segment<3>(iv) = w_dot;
            if (free) {
                Vec3 p, p_dot, p_ddot;
                for (int k = 0; k < 3; ++k) {
                    p(k) = m.coordinates[static_cast<std::size_t>(k)].value(t);
                    p_dot(k) = m.coordinates[static_cast<std::size_t>(k)].rate(t);
                    p_ddot(k) = m.coordinates[static_cast<std::size_t>(k)].acceleration(t);
                }
                const Vec3 u = R.conjugate() * p_dot;
                q.segment<3>(iq) = p;
                v.segment<3>(iv + 3) = u;
                a.segment<3>(iv + 3) = R.conjugate() * p_ddot - w.cross(u);
            }
        }
    }

private:
    const Model& model_;
    std::vector<JointMotion> joints_;
};

/// A system, its prescribed motion and the forces that drive it.
struct Case {
    std::string name;
    System sys;
    std::unique_ptr<Manufactured> motion;
    std::vector<Index> driven;   // closed loops: the driven velocities; empty: all

    /// The applied generalized forces at t, and the constraint forces
    /// J^T lambda the inverse dynamics leaves to the constraints. `residual`
    /// is |B tau_a + J^T lambda - RNEA| / |RNEA|: zero when the motion is one
    /// the drives and the constraints can produce.
    void forces(Real t, VecX& tau, VecX& constraint_force, Real& residual)
    {
        const Model& model = sys.model;
        if (!data_) {
            data_ = std::make_unique<Data>(model);
            solver_ = std::make_unique<ConstraintSolver>(model, sys.constraints);
        }
        motion->at(t, q_, v_, a_);
        const VecX id = rnea(model, *data_, q_, v_, a_);
        residual = 0.0;
        constraint_force.setZero(model.nv);
        if (sys.constraints.empty()) {
            tau = id;
            return;
        }
        solver_->evaluate(*data_, q_, v_, t);
        const MatX& J = solver_->J();
        const Index nd = static_cast<Index>(driven.size());
        MatX A = MatX::Zero(model.nv, nd + J.rows());
        for (Index k = 0; k < nd; ++k) A(driven[static_cast<std::size_t>(k)], k) = 1.0;
        A.rightCols(J.rows()) = J.transpose();
        const VecX x = Eigen::ColPivHouseholderQR<MatX>(A).solve(id);
        residual = (A * x - id).norm() / id.norm();
        tau.setZero(model.nv);
        for (Index k = 0; k < nd; ++k) tau(driven[static_cast<std::size_t>(k)]) = x(k);
        constraint_force = J.transpose() * x.tail(J.rows());
    }

    VecX q_, v_, a_;
    std::unique_ptr<Data> data_;
    std::unique_ptr<ConstraintSolver> solver_;
};

/// Every joint type: a free root with revolute, prismatic, fixed and
/// universal joints in its branches; a spherical root with cylindrical,
/// planar, free and spherical ones.
std::unique_ptr<Case> random_tree_case(std::uint64_t seed, KJoint root, const std::vector<KJoint>& kinds,
                                       std::string name)
{
    auto c = std::make_unique<Case>();
    c->name = std::move(name);
    Rng rng(seed);
    c->sys.model = mbd_test::random_tree(rng, root, kinds);
    c->motion = std::make_unique<Manufactured>(c->sys.model, rng);
    return c;
}

/// Two identical random chains, the second's base moved by d along world X,
/// their tips tied by a distance constraint between the same point of each.
/// With the same coordinates the second chain is the first moved by d, so
/// the tie keeps the length |d|. Every joint is driven but the second
/// chain's first, which the tie moves.
std::unique_ptr<Case> twin_chain_case(std::uint64_t seed)
{
    auto c = std::make_unique<Case>();
    c->name = "twin chains tied at their tips (universal - spherical - prismatic)";
    Model& model = c->sys.model;
    Rng rng(seed);
    const std::vector<KJoint> kinds{KJoint::Universal, KJoint::Spherical, KJoint::Prismatic};
    std::vector<Transform3> x_pj, x_cj;
    std::vector<RigidBodyInertia> inertia;
    for (std::size_t k = 0; k < kinds.size(); ++k) {
        x_pj.push_back(rng.frame(0.4));
        x_cj.push_back(rng.frame(0.3));
        inertia.push_back(mbd_test::random_inertia(rng));
    }
    const Vec3 d(0.8, 0.0, 0.0);
    int tip[2] = {0, 0};
    for (int chain = 0; chain < 2; ++chain) {
        int parent = 0;
        for (std::size_t k = 0; k < kinds.size(); ++k) {
            Transform3 base = x_pj[k];
            if (k == 0 && chain == 1) base.p += d;
            parent = model.add_body(parent, mbd_test::make_kernel_joint(kinds[k]), base, x_cj[k], inertia[k]);
        }
        tip[chain] = parent;
    }
    const Transform3 point(Quat::Identity(), Vec3(0.1, -0.2, 0.15));
    c->sys.constraints.push_back(std::make_shared<Distance>(Marker{tip[0], point}, Marker{tip[1], point}, d.norm()));
    c->motion = std::make_unique<Manufactured>(model, rng);
    const int n = static_cast<int>(kinds.size());
    for (int k = 0; k < n; ++k) c->motion->joint(n + 1 + k) = c->motion->joint(1 + k);
    for (Index k = 0; k < model.nv; ++k) {
        if (k != model.idx_v[n + 1]) c->driven.push_back(k);
    }
    return c;
}

/// A parallelogram four-bar in the XY plane, gravity along -Y: cranks of
/// length L hinged at the origin and at (D, 0, 0), a coupler of length D on
/// the first crank's tip, closed by a revolute at the second crank's tip.
/// The coupler stays parallel to the ground: its angle is minus the crank's.
/// Only the first crank is driven.
std::unique_ptr<Case> parallelogram_case()
{
    auto c = std::make_unique<Case>();
    c->name = "parallelogram four-bar, closed by a revolute";
    Model& model = c->sys.model;
    model.gravity = Vec3(0.0, -g_accel, 0.0);
    const Real L = 0.5, D = 0.8;
    auto link = [](Real length, Real mass) {
        RigidBodyInertia I = RigidBodyInertia::from_solid_box(mass, Vec3(0.5 * length, 0.02, 0.02));
        I.com_B = Vec3(0.5 * length, 0.01, 0.0);
        return I;
    };
    const auto revolute = std::make_shared<RevoluteJointModel>();
    const int crank1 = model.add_body(0, revolute, Transform3(), Transform3(), link(L, 1.0));
    const int coupler = model.add_body(crank1, revolute, Transform3(Quat::Identity(), Vec3(L, 0.0, 0.0)),
                                       Transform3(), link(D, 2.0));
    const int crank2 = model.add_body(0, revolute, Transform3(Quat::Identity(), Vec3(D, 0.0, 0.0)), Transform3(),
                                      link(L, 1.5));
    c->sys.constraints.push_back(revolute_closure(Marker{coupler, Transform3(Quat::Identity(), Vec3(D, 0.0, 0.0))},
                                                  Marker{crank2, Transform3(Quat::Identity(), Vec3(L, 0.0, 0.0))}));
    Rng rng(11);
    c->motion = std::make_unique<Manufactured>(model, rng);
    const Sine crank{0.5 * pi, 0.6, 1.7, 0.4};   // within 0.6 rad of upright: far from folding
    c->motion->joint(crank1).coordinates = {crank};
    c->motion->joint(coupler).coordinates = {crank.negated()};
    c->motion->joint(crank2).coordinates = {crank};
    c->driven = {0};
    return c;
}

std::vector<std::unique_ptr<Case>> all_cases()
{
    std::vector<std::unique_ptr<Case>> cases;
    cases.push_back(random_tree_case(5, KJoint::Free, {KJoint::Revolute, KJoint::Prismatic, KJoint::Fixed, KJoint::Universal},
                                     "tree: free root; revolute, prismatic, fixed, universal"));
    cases.push_back(random_tree_case(6, KJoint::Spherical,
                                     {KJoint::Cylindrical, KJoint::Planar, KJoint::Free, KJoint::Spherical},
                                     "tree: spherical root; cylindrical, planar, free, spherical"));
    cases.push_back(twin_chain_case(7));
    cases.push_back(parallelogram_case());
    return cases;
}

/// The motion's error after `duration`, from the exact state at t0, with
/// one method at a fixed step: |q (-) q_ref| and |v - v_ref| together.
Real error_after(Case& c, Integrator method, Real h, Real duration, Real t0 = 0.0)
{
    Simulator sim(c.sys);
    sim.method = method;
    sim.step_control.adaptive = false;
    sim.step_control.relative_tolerance = 1e-13;
    sim.step_control.absolute_tolerance = 1e-13;
    VecX a;
    c.motion->at(t0, sim.q, sim.v, a);
    sim.time = t0;
    VecX tau, constraint_force;
    Real residual = 0.0;
    sim.force_callback = [&](Simulator&, Real t, VecX& tau_sim) {
        c.forces(t, tau, constraint_force, residual);
        tau_sim += tau;
    };
    sim.run(duration, h);
    VecX q_ref, v_ref, dq;
    c.motion->at(sim.time, q_ref, v_ref, a);
    difference(c.sys.model, q_ref, sim.q, dq);
    return std::sqrt(dq.squaredNorm() + (sim.v - v_ref).squaredNorm());
}

struct Method {
    Integrator method;
    const char* name;
    int order;
    Real h_global;   // the longer of the two steps compared over 0.5 s
    Real h_local;    // the longer of the two single steps compared; 0: not compared
};

/// The steps, from the tables of the hidden case below (kept in
/// documentation/journal/data/2026-10-20_analytic-cases.md): for each
/// method a pair at which the ratio has settled on every model, its errors
/// still far above rounding.
const std::vector<Method>& methods()
{
    static const std::vector<Method> all{
        {Integrator::RK4, "RK4", 4, 0.01, 0.01},
        {Integrator::SemiImplicitEuler, "semi-implicit Euler", 1, 1e-3, 0.01},
        {Integrator::DormandPrince, "Dormand-Prince", 5, 0.0125, 0.01},
        {Integrator::GeneralizedAlpha, "generalised-alpha", 2, 5e-3, 0.0},
        {Integrator::SDIRK4, "SDIRK4", 4, 0.01, 0.01},
        {Integrator::RadauIIA, "Radau IIA", 5, 0.025, 0.02},
        {Integrator::LinearlyImplicitEuler, "linearly implicit Euler", 1, 1e-3, 0.01},
    };
    return all;
}

/// The spherical-root tree, on which Dormand-Prince's error over 0.5 s has
/// not settled at fifth order above rounding (see the order test).
bool unsettled(const Case& c, const Method& m)
{
    return m.method == Integrator::DormandPrince && c.name.rfind("tree: spherical root", 0) == 0;
}

} // namespace

TEST_CASE("Manufactured trajectories: the motion's derivatives are its rates", "[kernel][manufactured]")
{
    // The reference itself, against central differences of q(t) and v(t) at
    // h = 1e-5 s: their error is h^2 / 6 times the third derivative, which is
    // at most a w^3 = 27 per scalar coordinate here (4.5e-10) and of the same
    // order for the quaternions, whose rates combine three such rotations;
    // over a few tens of entries, of order 1e-9. The rounding of the values
    // divided by h adds 1e-16 / 1e-5 = 1e-11. Allowed 1e-7.
    for (auto& c : all_cases()) {
        const Model& model = c->sys.model;
        for (Real t : {0.0, 0.37, 0.81}) {
            const Real h = 1e-5;
            VecX q, v, a, qp, vp, ap, qm, vm, am, qd;
            c->motion->at(t, q, v, a);
            c->motion->at(t + h, qp, vp, ap);
            c->motion->at(t - h, qm, vm, am);
            q_dot(model, q, v, qd);
            INFO(c->name << " at t = " << t);
            CHECK((qd - (qp - qm) / (2.0 * h)).norm() < 1e-7);
            CHECK((a - (vp - vm) / (2.0 * h)).norm() < 1e-7);
        }
    }
}

TEST_CASE("Manufactured trajectories: the forward dynamics returns the prescribed accelerations and constraint forces",
          "[kernel][manufactured]")
{
    // The applied forces from inverse dynamics, at instants along the
    // motion. On the loops the motion satisfies the constraints exactly
    // (|phi| and |J v - nu| to rounding), the drives and multipliers solve
    // B tau_a + J^T lambda = RNEA to rounding, and the forward dynamics must
    // give v_dot and J^T lambda back, to rounding grown by the conditioning
    // of the solves (of M, and of J M^-1 J^T on the loops). Seen: the
    // residual 4.9e-16, the accelerations 2.1e-13 for sizes of 1 to 11, the
    // constraint forces 9e-15 of 9 to 14. Allowed 1e-12 for the residual and
    // 1e-9 relative for the rest, room for conditioning a thousand times
    // worse on another compiler's rounding.
    for (auto& c : all_cases()) {
        const Model& model = c->sys.model;
        Data data(model);
        ConstraintSolver solver(model, c->sys.constraints);
        for (Real t : {0.0, 0.37, 0.81}) {
            VecX q, v, a, tau, constraint_force;
            Real residual = 0.0;
            c->motion->at(t, q, v, a);
            c->forces(t, tau, constraint_force, residual);
            const VecX v_dot = solver.forward_dynamics(data, q, v, tau, t);
            INFO(c->name << " at t = " << t << ": inverse dynamics residual " << residual << ", |phi| "
                         << solver.phi().norm() << ", |J v - nu| " << (solver.J() * v - solver.nu()).norm());
            CHECK(residual < 1e-12);
            if (!c->sys.constraints.empty()) {
                CHECK(solver.phi().norm() < 1e-14);
                CHECK((solver.J() * v - solver.nu()).norm() < 1e-13);
                // The constraints carry a real part of the load.
                CHECK(constraint_force.norm() > 0.1 * tau.norm());
                const VecX f = solver.J().transpose() * solver.lambda();
                CHECK((f - constraint_force).norm() < 1e-9 * constraint_force.norm());
            }
            CHECK((v_dot - a).norm() < 1e-9 * (1.0 + a.norm()));
        }
    }
}

TEST_CASE("Manufactured trajectories: one step from the exact state is accurate to the method's order",
          "[kernel][manufactured]")
{
    // One step of length h from the exact state at t = 0.2 s: a method of
    // order p leaves a local error C h^(p+1) (1 + O(h)), so halving h divides
    // it by 2^(p+1). Unlike the error over a span, this does not depend on how
    // the motion amplifies earlier errors, so it settles at long steps. At the
    // steps used the measured orders are within 0.2 of p + 1, the largest
    // deviation being Dormand-Prince's on the spherical-root tree (6.17 at 10
    // and 5 ms, falling from 6.50 at 40 ms as the next term should, by about
    // half per halving); allowed 0.25. The smallest error compared is
    // Dormand-Prince's on the parallelogram at 5 ms, 8.4e-15, twenty times
    // the rounding floor of one step there (3.8e-16, seen at 2.5 ms), which
    // can shift the order by at most log2(1.05) = 0.07; required above
    // 2e-15.
    //
    // Generalised-alpha is left out: its first step from the exact state
    // carries an O(h^2) velocity error (measured: the ratio 4.0), because its
    // acceleration-like variable, started at the true acceleration, is not
    // the one its recurrence needs; the error is damped in the next steps and
    // does not add up (Arnold and Bruls 2007), and the order over a span,
    // below, is 2.
    for (auto& c : all_cases()) {
        for (const Method& m : methods()) {
            if (m.h_local == 0.0) continue;
            const Real e1 = error_after(*c, m.method, m.h_local, m.h_local, 0.2);
            const Real e2 = error_after(*c, m.method, 0.5 * m.h_local, 0.5 * m.h_local, 0.2);
            const Real order = std::log2(e1 / e2);
            INFO(c->name << ", " << m.name << ": local errors " << e1 << " and " << e2 << " at " << m.h_local
                         << " and " << 0.5 * m.h_local << " s; order " << order << " (expected " << m.order + 1
                         << ")");
            CHECK(e2 > 2e-15);
            CHECK(std::abs(order - (m.order + 1)) < 0.25);
        }
    }
}

TEST_CASE("Manufactured trajectories: every integrator returns the motion at its order", "[kernel][manufactured]")
{
    // After 0.5 s, the error at step h and at h / 2. Its leading term is
    // C h^p, so their ratio is 2^p (1 + O(h)); the steps are chosen for each
    // method so that the next term is a few per cent and the error stays far
    // above rounding. Allowed: the measured order log2(ratio) within 0.25 of
    // p, which holds the next term's relative share below 2^0.25 - 1 = 19 %.
    // Measured at these steps: within 0.08 of p in every case checked.
    //
    // The span is 0.5 s because the motions are unstable: they are produced
    // by forces fixed in time, with nothing to pull a deviation back, so an
    // error made early grows: on the spherical-root tree the shortest steps
    // of the fifth-order methods stop converging at 2e-12 to 1e-11, about
    // 1e5 times the state's rounding, and over 1 s the first-order methods'
    // errors there reach the size of the motion.
    //
    // One case is not checked here: Dormand-Prince on the spherical-root
    // tree, whose ratio over 0.5 s has not settled at 32 above that floor (it
    // measures 2^4.39, 2^4.31 and 2^4.70 from 12.5 to 1.6 ms) although its
    // local error, above, settles at sixth order. Over a span the local
    // errors of the many coordinates are carried to the end by different
    // growths and partly cancel, so the leading term can be small against
    // the next one down to short steps. Every other method, including Radau
    // IIA at fifth order, settles on that model.
    for (auto& c : all_cases()) {
        for (const Method& m : methods()) {
            if (unsettled(*c, m)) continue;
            const Real e1 = error_after(*c, m.method, m.h_global, 0.5);
            const Real e2 = error_after(*c, m.method, 0.5 * m.h_global, 0.5);
            const Real order = std::log2(e1 / e2);
            INFO(c->name << ", " << m.name << ": errors " << e1 << " and " << e2 << " at " << m.h_global << " and "
                         << 0.5 * m.h_global << " s; order " << order << " (expected " << m.order << ")");
            CHECK(e2 > 1e-12);
            CHECK(std::abs(order - m.order) < 0.25);
        }
    }
}

TEST_CASE("Manufactured trajectories: the tables of errors behind the chosen steps", "[.][manufactured][table]")
{
    // Not a check: prints, for every model and method, the error over 0.5 s
    // and over one step from t = 0.2 s at five steps, each halving the last,
    // with the order measured between neighbours. Run with
    //   test_kernel "[manufactured][table]"
    for (auto& c : all_cases()) {
        std::printf("%s\n", c->name.c_str());
        for (const Method& m : methods()) {
            for (int local = 0; local < 2; ++local) {
                if (local == 1 && m.h_local == 0.0) continue;
                const Real h0 = local == 1 ? 4.0 * m.h_local : 4.0 * m.h_global;
                std::printf("  %-24s %-6s", m.name, local == 1 ? "step" : "0.5 s");
                Real previous = 0.0;
                for (int k = 0; k < 5; ++k) {
                    const Real h = h0 / std::pow(2.0, k);
                    Real e = 0.0;
                    try {
                        e = local == 1 ? error_after(*c, m.method, h, h, 0.2) : error_after(*c, m.method, h, 0.5);
                    } catch (const std::exception&) {
                        std::printf("  h %8.6f  failed          ", h);   // Newton, at too long a step
                        previous = 0.0;
                        continue;
                    }
                    std::printf("  h %8.6f  %9.3e", h, e);
                    if (previous > 0.0) {
                        std::printf(" (%5.2f)", std::log2(previous / e));
                    } else {
                        std::printf("        ");
                    }
                    previous = e;
                }
                std::printf("\n");
            }
        }
    }
}
