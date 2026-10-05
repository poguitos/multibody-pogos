// Recursive algorithms of the kinematics kernel. Notation of Featherstone,
// "Rigid Body Dynamics Algorithms" (2008), with body-frame spatial vectors:
// for body i with parent p = parent[i],
//
//   v_i = iXp v_p + S_i qd_i
//   a_i = iXp a_p + S_i qdd_i + c_i,     c_i = (dS_i/dt) qd_i + v_i x (S_i qd_i)
//
// where iXp = motion_act_inv(liMi[i], .). Gravity enters the dynamics as an
// upward acceleration of the ground, a_0 = -[0; g] (data.a_gf).

#include "mbd/kernel/algorithms.hpp"

#include "mbd/spatial/spatial.hpp"

#include "checks.hpp"

namespace mbd::kernel {

namespace {

using DMat6 = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic, 0, 6, 6>;

/// Joint i's placement, motion subspace and liMi, oMi. `v` is passed to the
/// joint model, which uses it for c.
void placement_step(const Model& model, Data& data, int i, const VecX& q, const VecX& v)
{
    const JointModel& jm = *model.joint[i];
    JointData& jd = data.joint[i];
    jm.calc(q.segment(model.idx_q[i], model.nqs[i]),
            v.segment(model.idx_v[i], model.nvs[i]), jd);
    data.liMi[i] = model.X_PJ[i] * jd.X_J * model.X_JC[i];
    data.oMi[i]  = data.oMi[model.parent[i]] * data.liMi[i];
    data.S[i].noalias() = model.X_CJ_motion[i] * jd.S;
}

/// Body i's velocity and velocity-product acceleration, after placement_step.
void velocity_step(const Model& model, Data& data, int i, const VecX& v)
{
    Vec6 vJ = Vec6::Zero();
    if (model.nvs[i] > 0) {
        vJ.noalias() = data.S[i] * v.segment(model.idx_v[i], model.nvs[i]);
    }
    data.v[i] = motion_act_inv(data.liMi[i], data.v[model.parent[i]]) + vJ;
    data.c[i] = motion_act(model.X_CJ[i], data.joint[i].c)
              + motion_cross_motion(data.v[i], vJ);
}

/// The ground's acceleration minus gravity.
Vec6 ground_acceleration_minus_gravity(const Model& model)
{
    Vec6 a0;
    a0 << Vec3::Zero(), -model.gravity;
    return a0;
}

/// Body i's physical acceleration from its acceleration minus gravity.
Vec6 add_gravity(const Model& model, const Data& data, int i, const Vec6& a_gf)
{
    Vec6 a = a_gf;
    a.tail<3>() += data.oMi[i].q.conjugate() * model.gravity;
    return a;
}

/// The backward pass of RNEA: the joint forces data.f, each a body's own,
/// accumulated from the leaves, and their projections onto the joints in
/// data.tau.
void transmit_forces(const Model& model, Data& data)
{
    for (int i = model.nbodies() - 1; i > 0; --i) {
        if (model.nvs[i] > 0) {
            data.tau.segment(model.idx_v[i], model.nvs[i]).noalias()
                = data.S[i].transpose() * data.f[i];
        }
        const int p = model.parent[i];
        if (p > 0) data.f[p] += force_act(data.liMi[i], data.f[i]);
    }
}

} // namespace

// --- Kinematics ----------------------------------------------------------------

void forward_kinematics(const Model& model, Data& data, const VecX& q)
{
    checks::data("kernel::forward_kinematics", model, data);
    checks::q("kernel::forward_kinematics", model, q);
    data.oMi[0] = Transform3::Identity();
    for (int i = 1; i < model.nbodies(); ++i) {
        placement_step(model, data, i, q, data.zero_v);
    }
}

void forward_kinematics(const Model& model, Data& data, const VecX& q, const VecX& v)
{
    checks::data("kernel::forward_kinematics", model, data);
    checks::q("kernel::forward_kinematics", model, q);
    checks::v("kernel::forward_kinematics", model, v);
    data.oMi[0] = Transform3::Identity();
    data.v[0].setZero();
    for (int i = 1; i < model.nbodies(); ++i) {
        placement_step(model, data, i, q, v);
        velocity_step(model, data, i, v);
    }
}

void forward_kinematics(const Model& model, Data& data,
                        const VecX& q, const VecX& v, const VecX& a)
{
    checks::data("kernel::forward_kinematics", model, data);
    checks::q("kernel::forward_kinematics", model, q);
    checks::v("kernel::forward_kinematics", model, v);
    checks::v("kernel::forward_kinematics", model, a, "a");
    data.oMi[0] = Transform3::Identity();
    data.v[0].setZero();
    data.a[0].setZero();
    for (int i = 1; i < model.nbodies(); ++i) {
        placement_step(model, data, i, q, v);
        velocity_step(model, data, i, v);
        Vec6 ai = motion_act_inv(data.liMi[i], data.a[model.parent[i]]) + data.c[i];
        if (model.nvs[i] > 0) {
            ai.noalias() += data.S[i] * a.segment(model.idx_v[i], model.nvs[i]);
        }
        data.a[i] = ai;
    }
}

Vec6 body_velocity_world(const Data& data, int i)
{
    checks::body("kernel::body_velocity_world", i, data.oMi.size());
    const Quat& R = data.oMi[i].q;
    Vec6 out;
    out << R * data.v[i].head<3>(), R * data.v[i].tail<3>();
    return out;
}

Vec6 body_acceleration_world(const Data& data, int i)
{
    checks::body("kernel::body_acceleration_world", i, data.oMi.size());
    // The spatial acceleration's linear part is the acceleration of the body
    // origin less w x v_origin (Featherstone, section 2.11).
    const Quat& R = data.oMi[i].q;
    const Vec6& a = data.a[i];
    const Vec6& v = data.v[i];
    Vec6 out;
    out << R * a.head<3>(), R * (a.tail<3>() + v.head<3>().cross(v.tail<3>()));
    return out;
}

void body_jacobian_world(const Model& model, const Data& data, int i, MatX& J)
{
    checks::data("kernel::body_jacobian_world", model, data);
    checks::body("kernel::body_jacobian_world", i, data.oMi.size());
    J.setZero(6, model.nv);
    const Vec3& p_i = data.oMi[i].p;
    for (int j = i; j > 0; j = model.parent[j]) {
        for (int k = 0; k < model.nvs[j]; ++k) {
            // Column k of S_j at the world origin, then at body i's origin.
            const Vec6 s = motion_act(data.oMi[j], data.S[j].col(k));
            J.col(model.idx_v[j] + k) << s.head<3>(), s.tail<3>() + s.head<3>().cross(p_i);
        }
    }
}

// --- Dynamics --------------------------------------------------------------------

const VecX& rnea(const Model& model, Data& data,
                 const VecX& q, const VecX& v, const VecX& a)
{
    checks::data("kernel::rnea", model, data);
    checks::q("kernel::rnea", model, q);
    checks::v("kernel::rnea", model, v);
    checks::v("kernel::rnea", model, a, "a");
    const int nb = model.nbodies();
    data.oMi[0] = Transform3::Identity();
    data.v[0].setZero();
    data.a_gf[0] = ground_acceleration_minus_gravity(model);

    for (int i = 1; i < nb; ++i) {
        placement_step(model, data, i, q, v);
        velocity_step(model, data, i, v);
        Vec6 a_gf = motion_act_inv(data.liMi[i], data.a_gf[model.parent[i]]) + data.c[i];
        if (model.nvs[i] > 0) {
            a_gf.noalias() += data.S[i] * a.segment(model.idx_v[i], model.nvs[i]);
        }
        data.a_gf[i] = a_gf;
        data.a[i]    = add_gravity(model, data, i, a_gf);
        const Vec6 h = model.I[i] * data.v[i];
        data.f[i] = model.I[i] * a_gf + motion_cross_force(data.v[i], h);
    }
    transmit_forces(model, data);
    return data.tau;
}

const VecX& bias_forces(const Model& model, Data& data)
{
    checks::data("kernel::bias_forces", model, data);
    const int nb = model.nbodies();
    data.a_gf[0] = ground_acceleration_minus_gravity(model);
    for (int i = 1; i < nb; ++i) {
        // Gravity is a uniform field, so rnea's acceleration minus gravity is
        // the physical acceleration less gravity in the body's axes.
        Vec6 a_gf = data.a[i];
        a_gf.tail<3>() -= data.oMi[i].q.conjugate() * model.gravity;
        data.a_gf[i] = a_gf;
        const Vec6 h = model.I[i] * data.v[i];
        data.f[i] = model.I[i] * a_gf + motion_cross_force(data.v[i], h);
    }
    transmit_forces(model, data);
    return data.tau;
}

const MatX& crba(const Model& model, Data& data, const VecX& q)
{
    checks::data("kernel::crba", model, data);
    checks::q("kernel::crba", model, q);
    forward_kinematics(model, data, q);
    return mass_matrix(model, data);
}

const MatX& mass_matrix(const Model& model, Data& data)
{
    checks::data("kernel::mass_matrix", model, data);
    const int nb = model.nbodies();
    for (int i = 1; i < nb; ++i) data.Ic[i] = model.I[i];
    data.M.setZero();

    for (int i = nb - 1; i > 0; --i) {
        const int nvi = model.nvs[i];
        if (nvi > 0) {
            // F: the force that gives the composite body i acceleration S_i,
            // carried up the tree to each ancestor j to fill M(j, i).
            const int iv = model.idx_v[i];
            Mat6X F = data.Ic[i] * data.S[i];
            data.M.block(iv, iv, nvi, nvi).noalias() = data.S[i].transpose() * F;
            int j = i;
            while (model.parent[j] > 0) {
                for (int k = 0; k < nvi; ++k) F.col(k) = force_act(data.liMi[j], F.col(k));
                j = model.parent[j];
                const int nvj = model.nvs[j];
                if (nvj == 0) continue;
                const int jv = model.idx_v[j];
                data.M.block(jv, iv, nvj, nvi).noalias() = data.S[j].transpose() * F;
                data.M.block(iv, jv, nvi, nvj) = data.M.block(jv, iv, nvj, nvi).transpose();
            }
        }
        const int p = model.parent[i];
        if (p > 0) data.Ic[p] += inertia_act(data.liMi[i], data.Ic[i]);
    }
    return data.M;
}

const VecX& aba(const Model& model, Data& data,
                const VecX& q, const VecX& v, const VecX& tau)
{
    checks::data("kernel::aba", model, data);
    checks::q("kernel::aba", model, q);
    checks::v("kernel::aba", model, v);
    checks::v("kernel::aba", model, tau, "tau");
    const int nb = model.nbodies();
    data.oMi[0] = Transform3::Identity();
    data.v[0].setZero();

    // Pass 1: kinematics; articulated inertias and bias forces of the bodies
    // on their own.
    for (int i = 1; i < nb; ++i) {
        placement_step(model, data, i, q, v);
        velocity_step(model, data, i, v);
        data.Ic[i] = model.I[i];
        data.pA[i] = motion_cross_force(data.v[i], model.I[i] * data.v[i]);
    }

    // Pass 2: articulated inertias and bias forces, from the leaves.
    for (int i = nb - 1; i > 0; --i) {
        const int nvi = model.nvs[i];
        const int p   = model.parent[i];
        if (nvi > 0) {
            data.U[i].noalias() = data.Ic[i] * data.S[i];
            const DMat6 D = data.S[i].transpose() * data.U[i];
            if (nvi == 1) {
                data.Dinv[i](0, 0) = 1.0 / D(0, 0);
            } else {
                data.Dinv[i] = D.inverse();
            }
            // In two steps: Eigen evaluates "y - A b" through a temporary sized
            // like y, here a block of a dynamic vector, so on the heap.
            data.u[i] = tau.segment(model.idx_v[i], nvi);
            data.u[i].noalias() -= data.S[i].transpose() * data.pA[i];
        }
        if (p == 0) continue;

        Mat6 Ia = data.Ic[i];
        Vec6 pa = data.pA[i];
        if (nvi > 0) {
            Ia.noalias() -= data.U[i] * data.Dinv[i] * data.U[i].transpose();
            pa += Ia * data.c[i] + data.U[i] * (data.Dinv[i] * data.u[i]);
        } else {
            pa += Ia * data.c[i];
        }
        data.Ic[p] += inertia_act(data.liMi[i], Ia);
        data.pA[p] += force_act(data.liMi[i], pa);
    }

    // Pass 3: accelerations, from the root.
    data.a_gf[0] = ground_acceleration_minus_gravity(model);
    for (int i = 1; i < nb; ++i) {
        const int nvi = model.nvs[i];
        Vec6 a_gf = motion_act_inv(data.liMi[i], data.a_gf[model.parent[i]]) + data.c[i];
        if (nvi > 0) {
            auto ddq_i = data.ddq.segment(model.idx_v[i], nvi);
            ddq_i.noalias() = data.Dinv[i] * (data.u[i] - data.U[i].transpose() * a_gf);
            a_gf.noalias() += data.S[i] * ddq_i;
        }
        data.a_gf[i] = a_gf;
        data.a[i]    = add_gravity(model, data, i, a_gf);
    }
    return data.ddq;
}

// --- Whole-system quantities ---------------------------------------------------

Vec6 momentum_world(const Model& model, const Data& data)
{
    checks::data("kernel::momentum_world", model, data);
    Vec6 h = Vec6::Zero();
    for (int i = 1; i < model.nbodies(); ++i) {
        h += force_act(data.oMi[i], model.I[i] * data.v[i]);
    }
    return h;
}

Real kinetic_energy(const Model& model, const Data& data)
{
    checks::data("kernel::kinetic_energy", model, data);
    Real T = 0.0;
    for (int i = 1; i < model.nbodies(); ++i) {
        T += data.v[i].dot(model.I[i] * data.v[i]);
    }
    return 0.5 * T;
}

Real potential_energy(const Model& model, const Data& data)
{
    checks::data("kernel::potential_energy", model, data);
    Real V = 0.0;
    for (int i = 1; i < model.nbodies(); ++i) {
        V -= model.inertia[i].mass * model.gravity.dot(data.oMi[i].apply(model.inertia[i].com_B));
    }
    return V;
}

Vec3 center_of_mass(const Model& model, const Data& data)
{
    checks::data("kernel::center_of_mass", model, data);
    Real m = 0.0;
    Vec3 mc = Vec3::Zero();
    for (int i = 1; i < model.nbodies(); ++i) {
        m  += model.inertia[i].mass;
        mc += model.inertia[i].mass * data.oMi[i].apply(model.inertia[i].com_B);
    }
    return m > 0.0 ? Vec3(mc / m) : Vec3::Zero();
}

// --- Configuration space -------------------------------------------------------

void q_dot(const Model& model, const VecX& q, const VecX& v, VecX& qd)
{
    checks::q("kernel::q_dot", model, q);
    checks::v("kernel::q_dot", model, v);
    MBD_THROW_IF(&qd == &q, "MBD-K004: kernel::q_dot: qd must not be q itself");
    qd.resize(model.nq);
    for (int i = 1; i < model.nbodies(); ++i) {
        model.joint[i]->q_dot(q.segment(model.idx_q[i], model.nqs[i]),
                              v.segment(model.idx_v[i], model.nvs[i]),
                              qd.segment(model.idx_q[i], model.nqs[i]));
    }
}

void normalize(const Model& model, VecX& q)
{
    checks::q("kernel::normalize", model, q);
    for (int i = 1; i < model.nbodies(); ++i) {
        model.joint[i]->normalize(q.segment(model.idx_q[i], model.nqs[i]));
    }
}

void integrate(const Model& model, const VecX& q, const VecX& v, Real dt, VecX& q_out)
{
    checks::q("kernel::integrate", model, q);
    checks::v("kernel::integrate", model, v);
    if (&q_out != &q) q_out.resize(model.nq);
    Eigen::Matrix<Real, Eigen::Dynamic, 1, 0, 6, 1> dv;
    for (int i = 1; i < model.nbodies(); ++i) {
        const int iq = model.idx_q[i], nqi = model.nqs[i];
        dv = dt * v.segment(model.idx_v[i], model.nvs[i]);
        model.joint[i]->integrate(q.segment(iq, nqi), dv, q_out.segment(iq, nqi));
    }
}

void difference(const Model& model, const VecX& q0, const VecX& q1, VecX& dv)
{
    checks::q("kernel::difference", model, q0, "q0");
    checks::q("kernel::difference", model, q1, "q1");
    dv.resize(model.nv);
    for (int i = 1; i < model.nbodies(); ++i) {
        const int iq = model.idx_q[i], nqi = model.nqs[i];
        model.joint[i]->difference(q0.segment(iq, nqi), q1.segment(iq, nqi),
                                   dv.segment(model.idx_v[i], model.nvs[i]));
    }
}

} // namespace mbd::kernel
