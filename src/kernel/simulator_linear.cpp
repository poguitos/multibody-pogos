// Linearly implicit Euler on the kernel's simulator (plan task 4.5, decision
// D35): the real-time integrator, every step the same work.
//
// One evaluation gives the accelerations a and the states' rates x_dot at
// (q, v, x). Backward Euler, q+ = q (+) h v+, linearized once about the
// step's start in the applied forces and the states' rates, with
//
//     K = d tau / dq, C = d tau / dv, Tx = d tau / dx    (tau: the applied
//                                                         generalized forces)
//     Xv, X = d x_dot / dv, dx,
//
// gives for delta = v+ - v - h a (the change from explicit Euler) and dx = x+
// - x the linear equations
//
//     (M - h (C - D) - h^2 K) delta - h Tx dx = h^2 (K (v + h a) + (C - D) a) + J^T mu
//     -h Xv delta + (I - h X) dx = h x_dot + h^2 Xv a
//     J delta = nu - J (v + h a),
//
// D = d b / dv the exact derivative of the inertia's velocity products (the
// Coriolis, centrifugal and gyroscopic terms b; bias_force_velocity_derivative),
// and the last equation putting the new velocities on the constraints, J v+ =
// nu (the projection after the step takes up the positions' drift). The
// second is solved for dx first, dx = dx0 + G delta with G = (I - h X)^-1 h
// Xv, which leaves (W - h Tx G) delta = r + h Tx dx0 + J^T mu: one
// constrained solve in the velocities. K, C and Tx G enter by their
// symmetric parts, in the matrix and on the right (D35); D enters whole, so
// the matrix is not symmetric: an LU, its symmetric part checked for positive
// definiteness (D38). Both halves of D38 were measured. Taken explicitly, the
// velocity products made a top spinning at 30 rad/s blow up within 100 s at
// 5 ms; through D, backward Euler's damping of its nutation, exactly, for flat
// and long bodies alike. With the old restriction J delta = 0 the new
// velocities kept the explicit part of the constraints' own velocity terms,
// and a closed four-bar gained energy until it blew up; on the constraints,
// their impulse does no work on v+ and it only loses energy. (A Coriolis
// matrix frozen at the step's start, B with B v = b and B + B^T = dM/dt,
// which provably cannot gain kinetic energy, made a long free body tumble.)
// Any approximation of the derivatives keeps the method consistent (it is
// then a W-method); what it must capture is the stiffness. That includes the stiff
// coupling of a wheel's spin and its hub's velocity through the tyre's slip:
// without Tx and Xv a car on tyres at walking pace accelerated a fifth too
// slowly at 1 ms. It excludes the states' dependence on the positions, Xq =
// d x_dot / dq: on the tyres it comes from the rolling radius's dependence
// on the deflection, is not stiff, and in the symmetric matrix made a car
// coasting at 0.5 m/s gain 0.5 % of its speed in 2 s at 5 ms (5 % at 10 ms);
// without it the car keeps its speed within 2e-5. The inertial terms'
// dependence on the positions, and the derivatives of the joints' geometry,
// stay explicit.
//
// K and C are built element by element: each element's loads on its bodies
// ([moment about the body's origin; force], world axes) are differentiated
// with respect to those bodies' poses (a small rotation about the origin and
// a displacement of it, world axes) and velocities ([angular velocity;
// velocity of the origin]), by forward differences of that element's apply()
// alone, without kinematics or dynamics, and mapped through the bodies'
// Jacobians, which have the same rows: K += J_b^T (d load_b / d pose_c) J_c.
// After each apply() with a perturbed velocity the state elements' rates are
// taken again, so that Xv follows through the outputs of the elements they
// read (a tyre's force, for a wheel's spin). Joint forces are differentiated
// with respect to the coordinates of the joints they name; X and Tx with
// respect to each state, with the force elements applied again.

#include "mbd/kernel/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/forces.hpp"

#include "dense_lu.hpp"
#include "linear_work.hpp"

namespace mbd::kernel {

Simulator::~Simulator() = default;

namespace {

/// A body's load [moment about its origin; force].
Vec6 load_of(const RigidBodyForces& f)
{
    Vec6 out;
    out << f.tau_W, f.f_W;
    return out;
}

/// Forward-difference steps: 1e-7 of the entry, at least 1e-7.
Real difference_step(Real value)
{
    return 1e-7 * std::max(Real(1.0), std::abs(value));
}

} // namespace

void Simulator::linear_force_jacobian(bool with_states)
{
    const Model& model = system.model;
    const Index nv = model.nv;
    const int nb = model.nbodies();
    const Index nx = x.size();
    if (!linear_) linear_ = std::make_unique<LinearWork>();
    LinearWork& w = *linear_;
    if (w.force_elements != system.force_elements.size() || w.joint_forces != system.joint_forces.size()
        || w.nv != nv || w.nx != nx) {
        // First use, or elements or states added since: size everything
        // (allocates).
        w.force_elements = system.force_elements.size();
        w.joint_forces = system.joint_forces.size();
        w.nv = nv;
        w.nx = nx;
        w.element_bodies.clear();
        w.max_bodies = 1;
        w.body_needed.assign(static_cast<std::size_t>(nb), 0);
        for (const auto& element : system.force_elements) {
            w.element_bodies.push_back(element->bodies());
            w.max_bodies = std::max(w.max_bodies, static_cast<int>(w.element_bodies.back().size()));
            for (BodyIndex b : w.element_bodies.back()) {
                if (b > 0) w.body_needed[static_cast<std::size_t>(b)] = 1;
            }
        }
        w.joint_bodies.clear();
        for (const auto& jf : system.joint_forces) w.joint_bodies.push_back(jf->bodies());
        w.body_columns.assign(static_cast<std::size_t>(nb), {});
        w.body_jacobian.assign(static_cast<std::size_t>(nb), MatX());
        for (int b = 1; b < nb; ++b) {
            if (!w.body_needed[static_cast<std::size_t>(b)]) continue;
            auto& columns = w.body_columns[static_cast<std::size_t>(b)];
            for (int j = b; j > 0; j = model.parent[j]) {
                for (int k = 0; k < model.nvs[j]; ++k) columns.push_back(model.idx_v[j] + k);
            }
            w.body_jacobian[static_cast<std::size_t>(b)] = MatX::Zero(6, static_cast<Index>(columns.size()));
        }
        w.jacobian_full = MatX::Zero(6, nv);
        w.load_K = MatX::Zero(6 * w.max_bodies, 6 * w.max_bodies);
        w.load_C = MatX::Zero(6 * w.max_bodies, 6 * w.max_bodies);
        w.load_X = MatX::Zero(nx, 6 * w.max_bodies);
        w.states = states_;
        w.forces = forces_;
        w.load0.assign(static_cast<std::size_t>(w.max_bodies), Vec6::Zero());
        w.K = MatX::Zero(nv, nv);
        w.C = MatX::Zero(nv, nv);
        w.W = MatX::Zero(nv, nv);
        w.D = MatX::Zero(nv, nv);
        w.W_sym = MatX::Zero(nv, nv);
        w.llt_sym = Eigen::LLT<MatX>(nv);
        w.a = VecX::Zero(nv);
        w.r = VecX::Zero(nv);
        w.dv = VecX::Zero(nv);
        w.v1 = VecX::Zero(nv);
        w.q1 = q;
        w.tau0 = VecX::Zero(nv);
        w.tau1 = VecX::Zero(nv);
        w.step = VecX::Zero(nv);
        w.X = MatX::Zero(nx, nx);
        w.Xv = MatX::Zero(nx, nv);
        w.Tx = MatX::Zero(nv, nx);
        w.Xm = MatX::Zero(nx, nx);
        w.G = MatX::Zero(nx, nv);
        w.X_pivots.assign(static_cast<std::size_t>(nx), 0);
        w.xd0 = VecX::Zero(nx);
        w.xd1 = VecX::Zero(nx);
        w.x1 = VecX::Zero(nx);
        w.tau_x0 = VecX::Zero(nv);
        w.tau_x1 = VecX::Zero(nv);
        w.rhs_x = VecX::Zero(nx);
        w.dx0 = VecX::Zero(nx);
        w.dx = VecX::Zero(nx);
    }

    ++stats_.jacobians;
    w.K.setZero();
    w.C.setZero();
    const bool rates = with_states && nx > 0;
    if (rates) w.Xv.setZero();

    // The bodies' Jacobians, at the columns that can be nonzero.
    for (int b = 1; b < nb; ++b) {
        if (!w.body_needed[static_cast<std::size_t>(b)]) continue;
        body_jacobian_world(model, data_, b, w.jacobian_full);
        const auto& columns = w.body_columns[static_cast<std::size_t>(b)];
        MatX& J = w.body_jacobian[static_cast<std::size_t>(b)];
        for (std::size_t k = 0; k < columns.size(); ++k) J.col(static_cast<Index>(k)) = w.jacobian_full.col(columns[k]);
    }

    // Force elements, one at a time on a copy of the body states.
    w.states = states_;
    for (std::size_t e = 0; e < system.force_elements.size(); ++e) {
        const ForceElement& element = *system.force_elements[e];
        const auto& bodies = w.element_bodies[e];
        const int n = static_cast<int>(bodies.size());
        auto apply = [&] {
            for (BodyIndex b : bodies) {
                w.forces[static_cast<std::size_t>(b)].f_W.setZero();
                w.forces[static_cast<std::size_t>(b)].tau_W.setZero();
            }
            element.apply(w.states, w.forces);
        };
        apply();
        for (int i = 0; i < n; ++i) w.load0[static_cast<std::size_t>(i)] = load_of(w.forces[static_cast<std::size_t>(bodies[i])]);
        for (int j = 0; j < n; ++j) {
            const BodyIndex c = bodies[static_cast<std::size_t>(j)];
            if (c == 0) continue;   // the ground does not move
            RigidBodyState& s = w.states[static_cast<std::size_t>(c)];
            const RigidBodyState keep = s;
            for (int d = 0; d < 12; ++d) {
                Real eps = 1e-7;
                const int axis = d % 3;
                if (d < 3) {
                    s.q_WB = Quat(Eigen::AngleAxis<Real>(eps, Vec3::Unit(axis))) * keep.q_WB;
                } else if (d < 6) {
                    s.p_WB(axis) = keep.p_WB(axis) + difference_step(keep.p_WB(axis));
                    eps = s.p_WB(axis) - keep.p_WB(axis);
                } else if (d < 9) {
                    s.w_WB(axis) = keep.w_WB(axis) + difference_step(keep.w_WB(axis));
                    eps = s.w_WB(axis) - keep.w_WB(axis);
                } else {
                    s.v_WB(axis) = keep.v_WB(axis) + difference_step(keep.v_WB(axis));
                    eps = s.v_WB(axis) - keep.v_WB(axis);
                }
                apply();
                s = keep;
                MatX& target = d < 6 ? w.load_K : w.load_C;
                const Index column = 6 * j + d % 6;
                for (int i = 0; i < n; ++i) {
                    target.block<6, 1>(6 * i, column) =
                        (load_of(w.forces[static_cast<std::size_t>(bodies[i])]) - w.load0[static_cast<std::size_t>(i)])
                        / eps;
                }
                if (rates && d >= 6) {
                    // The states' rates, through this element's outputs.
                    for (std::size_t m = 0; m < state_elements_.size(); ++m) {
                        const Index ns = state_elements_[m]->size();
                        state_elements_[m]->rates(*this, time, x.segment(state_offsets_[m], ns),
                                                  w.xd1.segment(state_offsets_[m], ns));
                    }
                    w.load_X.col(6 * j + d - 6) = (w.xd1 - x_dot_) / eps;
                }
            }
        }
        apply();   // leave the element as it was at the base state
        if (rates) {
            for (std::size_t m = 0; m < state_elements_.size(); ++m) {
                const Index ns = state_elements_[m]->size();
                state_elements_[m]->rates(*this, time, x.segment(state_offsets_[m], ns),
                                          w.xd1.segment(state_offsets_[m], ns));
            }
        }

        // K += J_b^T (d load_b / d pose_c) J_c, C likewise, column by column;
        // Xv += (d x_dot / d twist_c) J_c.
        for (int j = 0; j < n; ++j) {
            const BodyIndex c = bodies[static_cast<std::size_t>(j)];
            if (c == 0) continue;
            const auto& cols_c = w.body_columns[static_cast<std::size_t>(c)];
            const MatX& Jc = w.body_jacobian[static_cast<std::size_t>(c)];
            for (int i = 0; i < n; ++i) {
                const BodyIndex b = bodies[static_cast<std::size_t>(i)];
                if (b == 0) continue;
                const auto& cols_b = w.body_columns[static_cast<std::size_t>(b)];
                const MatX& Jb = w.body_jacobian[static_cast<std::size_t>(b)];
                const auto Kbc = w.load_K.block<6, 6>(6 * i, 6 * j);
                const auto Cbc = w.load_C.block<6, 6>(6 * i, 6 * j);
                for (std::size_t s = 0; s < cols_c.size(); ++s) {
                    const Vec6 kc = Kbc * Jc.col(static_cast<Index>(s));
                    const Vec6 cc = Cbc * Jc.col(static_cast<Index>(s));
                    for (std::size_t r = 0; r < cols_b.size(); ++r) {
                        w.K(cols_b[r], cols_c[s]) += Jb.col(static_cast<Index>(r)).dot(kc);
                        w.C(cols_b[r], cols_c[s]) += Jb.col(static_cast<Index>(r)).dot(cc);
                    }
                }
            }
            if (rates) {
                for (std::size_t s = 0; s < cols_c.size(); ++s) {
                    const Index col = cols_c[s];
                    w.Xv.col(col).noalias() += w.load_X.middleCols<6>(6 * j) * Jc.col(static_cast<Index>(s));
                }
            }
        }
    }

    // Joint forces, with respect to the coordinates of the joints they name.
    for (std::size_t e = 0; e < system.joint_forces.size(); ++e) {
        const JointForce& jf = *system.joint_forces[e];
        w.tau0.setZero();
        jf.apply(q, v, w.tau0);
        for (int b : w.joint_bodies[e]) {
            for (int k = 0; k < model.nvs[b]; ++k) {
                const Index j = model.idx_v[b] + k;
                const Real eps_q = 1e-7;
                w.step.setZero();
                w.step(j) = eps_q;
                integrate(model, q, w.step, 1.0, w.q1);
                w.tau1.setZero();
                jf.apply(w.q1, v, w.tau1);
                w.K.col(j) += (w.tau1 - w.tau0) / eps_q;
                w.v1 = v;
                w.v1(j) += difference_step(v(j));
                const Real eps_v = w.v1(j) - v(j);
                w.tau1.setZero();
                jf.apply(q, w.v1, w.tau1);
                w.C.col(j) += (w.tau1 - w.tau0) / eps_v;
            }
        }
    }
}

void Simulator::linear_state_jacobian(Real t)
{
    const Model& model = system.model;
    LinearWork& w = *linear_;
    const Index nx = x.size();
    if (nx == 0) return;
    // At states x1 and the evaluation's kinematics: each element is handed x1,
    // the force elements (whose outputs the rates may read) are applied
    // again, their generalized forces formed, then the rates taken.
    auto at = [&](const VecX& x1, VecX& rates, VecX& tau) {
        for (std::size_t i = 0; i < state_elements_.size(); ++i) {
            state_elements_[i]->begin_evaluation(*this, t, x1.segment(state_offsets_[i], state_elements_[i]->size()));
        }
        for (auto& f : w.forces) {
            f.f_W.setZero();
            f.tau_W.setZero();
        }
        for (const auto& element : system.force_elements) element->apply(states_, w.forces);
        generalized_forces(model, data_, w.forces, tau);
        for (std::size_t i = 0; i < state_elements_.size(); ++i) {
            const Index n = state_elements_[i]->size();
            state_elements_[i]->rates(*this, t, x1.segment(state_offsets_[i], n), rates.segment(state_offsets_[i], n));
        }
    };
    at(x, w.xd0, w.tau_x0);
    for (Index j = 0; j < nx; ++j) {
        w.x1 = x;
        w.x1(j) += difference_step(x(j));
        const Real eps = w.x1(j) - x(j);
        at(w.x1, w.xd1, w.tau_x1);
        w.X.col(j) = (w.xd1 - w.xd0) / eps;
        w.Tx.col(j) = (w.tau_x1 - w.tau_x0) / eps;
    }
    at(x, w.xd1, w.tau_x1);   // leave the elements as at the base state
}

void Simulator::force_jacobian(MatX& K, MatX& C)
{
    acceleration(q, v, x, time);
    linear_force_jacobian(false);
    K = linear_->K;
    C = linear_->C;
}

void Simulator::step_linearly_implicit(Real h)
{
    const Model& model = system.model;
    const Index nx = x.size();
    const VecX& a = evaluate(q, v, x, time);
    // The Coriolis matrix, from the evaluation's kinematics
    // before anything else uses them.
    bias_force_velocity_derivative(model, data_, q, v);
    linear_force_jacobian(true);   // sizes the work on the first step
    LinearWork& w = *linear_;
    w.a = a;   // the solver's storage, which the constrained solve reuses
    w.v1 = v + h * w.a;
    w.D = data_.dtau_dv;

    // The states first: dx = dx0 + G delta.
    if (nx > 0) {
        linear_state_jacobian(time);
        w.Xm = (-h) * w.X;
        w.Xm.diagonal().array() += 1.0;
        w.rhs_x = h * w.xd0;
        w.rhs_x.noalias() += (h * h) * (w.Xv * w.a);
        w.G = h * w.Xv;
        if (detail::lu_factor(w.Xm, w.X_pivots)) {
            w.dx0 = w.rhs_x;
            detail::lu_solve(w.Xm, w.X_pivots, w.dx0);
            for (Index j = 0; j < w.G.cols(); ++j) {
                auto column = w.G.col(j);
                w.dx = column;
                detail::lu_solve(w.Xm, w.X_pivots, w.dx);
                column = w.dx;
            }
        } else {
            // I - h X singular: the states explicit, uncoupled.
            w.dx0 = h * w.xd0;
            w.G.setZero();
        }
    }

    // W = M - h C_s - h^2 K_s - h (Tx G)_s + h D, M the evaluation's mass
    // matrix; r = h^2 (K_s (v + h a) + C_s a - D a) + h Tx dx0. Without the
    // stiffness, then without every derivative, where W is not positive
    // definite.
    auto assemble = [&](bool stiffness) {
        w.W = data_.M;
        w.W -= (0.5 * h) * (w.C + w.C.transpose());
        w.W.noalias() += h * w.D;
        w.r.noalias() = w.C * w.a;
        w.r.noalias() += w.C.transpose() * w.a;
        w.r.noalias() -= 2.0 * (w.D * w.a);
        if (stiffness) {
            w.W -= (0.5 * h * h) * (w.K + w.K.transpose());
            w.r.noalias() += w.K * w.v1;
            w.r.noalias() += w.K.transpose() * w.v1;
        }
        w.r *= 0.5 * h * h;
        if (nx > 0) {
            w.W.noalias() -= (0.5 * h) * (w.Tx * w.G);
            w.W.noalias() -= (0.5 * h) * (w.G.transpose() * w.Tx.transpose());
            w.r.noalias() += h * (w.Tx * w.dx0);
        }
    };
    assemble(true);
    bool solved = inertia_solve();
    if (!solved) {
        // Not positive definite: a force that pushes away from where it is
        // (negative stiffness or damping), or a body spinning so fast for
        // the step that its gyroscopic terms do, beyond what the inertia
        // holds at this step. The step drops the stiffness, then all the
        // derivatives.
        report_not_positive_definite("linearly implicit");
        assemble(false);
        solved = inertia_solve();
        if (!solved) w.dv.setZero();   // explicit: M is positive definite
    }
    v = w.v1 + w.dv;
    integrate(model, q, v, h, q);
    if (nx > 0) {
        if (!solved) w.G.setZero();
        w.dx = w.dx0;
        w.dx.noalias() += w.G * w.dv;
        x += w.dx;
    }
}

bool Simulator::inertia_solve()
{
    LinearWork& w = *linear_;
    w.W_sym = 0.5 * (w.W + w.W.transpose());
    w.llt_sym.compute(w.W_sym);
    if (w.llt_sym.info() != Eigen::Success) return false;
    // The velocities of the step's end on the constraints, J v+ = nu (D38):
    // J delta = nu - J (v + h a).
    drift_work_ = solver_.nu();
    drift_work_.noalias() -= solver_.J() * w.v1;
    return solver_.constrained_solve_lu(w.W, w.r, drift_work_, w.dv);
}

void Simulator::report_not_positive_definite(const char* step)
{
    LinearWork& w = *linear_;
    if (w.fallback_reported) return;
    w.fallback_reported = true;
    report_warning("MBD-K045: kernel::Simulator: at t = " + std::to_string(time) + " s the " + std::string(step)
                   + " step's matrix M - h C - h^2 K + h D is not positive definite; steps like it drop the "
                     "forces' stiffness, then all the derivatives, and are less stable. Further cases are not "
                     "reported.");
}

void Simulator::step_semi_implicit_euler(Real h)
{
    // The velocities first, by explicit Euler but with the inertia's velocity
    // products (Coriolis, centrifugal and gyroscopic terms) linearly implicit
    // through their derivative D: (M + h D) delta = -h^2 D a, the new
    // velocities on the constraints (J v+ = nu); then the positions with the
    // new velocities (D38). Taken explicitly, those terms made a spinning body
    // gain energy at every step.
    // The states x by explicit Euler, from the rates at the step's start.
    const Model& model = system.model;
    const Index nv = model.nv;
    const VecX& a = evaluate(q, v, x, time);
    bias_force_velocity_derivative(model, data_, q, v);
    if (!linear_) linear_ = std::make_unique<LinearWork>();
    LinearWork& w = *linear_;
    if (w.D.rows() != nv) {
        // First step (the linearly implicit step sizes all of it): the part
        // this step uses (allocates).
        w.W = MatX::Zero(nv, nv);
        w.D = MatX::Zero(nv, nv);
        w.W_sym = MatX::Zero(nv, nv);
        w.llt_sym = Eigen::LLT<MatX>(nv);
        w.a = VecX::Zero(nv);
        w.r = VecX::Zero(nv);
        w.dv = VecX::Zero(nv);
        w.v1 = VecX::Zero(nv);
    }
    w.a = a;
    w.v1 = v + h * w.a;
    w.D = data_.dtau_dv;
    w.W = data_.M;
    w.W.noalias() += h * w.D;
    w.r.noalias() = (-h * h) * (w.D * w.a);
    if (!inertia_solve()) {
        report_not_positive_definite("semi-implicit Euler");
        w.dv.setZero();
    }
    x += h * x_dot_;
    v = w.v1 + w.dv;
    integrate(model, q, v, h, q);
}

} // namespace mbd::kernel
