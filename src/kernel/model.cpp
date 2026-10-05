#include "mbd/kernel/model.hpp"

#include <utility>

#include "mbd/spatial/spatial.hpp"

namespace mbd::kernel {

Model::Model()
{
    // Body 0 is ground: it has no joint and no inertia.
    parent.push_back(-1);
    joint.push_back(nullptr);
    X_PJ.emplace_back();
    X_CJ.emplace_back();
    inertia.emplace_back();
    idx_q.push_back(0);
    idx_v.push_back(0);
    nqs.push_back(0);
    nvs.push_back(0);
    name.emplace_back("ground");
    X_JC.emplace_back();
    X_CJ_motion.push_back(Mat6::Identity());
    I.push_back(Mat6::Zero());
}

int Model::add_body(int parent_body,
                    std::shared_ptr<const JointModel> joint_model,
                    const Transform3& x_pj,
                    const Transform3& x_cj,
                    const RigidBodyInertia& body_inertia,
                    std::string body_name)
{
    MBD_THROW_IF(parent_body < 0 || parent_body >= nbodies(),
                 "MBD-K010: kernel::Model::add_body: parent body does not exist");
    MBD_THROW_IF(!joint_model, "MBD-K010: kernel::Model::add_body: no joint model");
    MBD_THROW_IF(body_inertia.mass < 0.0, "MBD-K011: kernel::Model::add_body: negative mass");

    const int nq_joint = joint_model->nq();
    const int nv_joint = joint_model->nv();
    MBD_THROW_IF(nv_joint < 0 || nv_joint > 6 || nq_joint < nv_joint,
                 "MBD-K010: kernel::Model::add_body: joint has an invalid number of coordinates");

    parent.push_back(parent_body);
    joint.push_back(std::move(joint_model));
    X_PJ.push_back(x_pj);
    X_CJ.push_back(x_cj);
    inertia.push_back(body_inertia);
    idx_q.push_back(nq);
    idx_v.push_back(nv);
    nqs.push_back(nq_joint);
    nvs.push_back(nv_joint);
    name.push_back(std::move(body_name));
    X_JC.push_back(x_cj.inverse());
    X_CJ_motion.push_back(motion_matrix(x_cj));
    I.push_back(spatial_inertia_matrix(body_inertia));

    nq += nq_joint;
    nv += nv_joint;
    return nbodies() - 1;
}

VecX Model::neutral_configuration() const
{
    VecX q = VecX::Zero(nq);
    for (int i = 1; i < nbodies(); ++i) {
        joint[i]->neutral(q.segment(idx_q[i], nqs[i]));
    }
    return q;
}

Data::Data(const Model& model)
{
    const auto n = static_cast<std::size_t>(model.nbodies());
    joint.resize(n);
    S.assign(n, Mat6X::Zero(6, 0));
    liMi.resize(n);
    oMi.resize(n);
    v.assign(n, Vec6::Zero());
    a.assign(n, Vec6::Zero());
    a_gf.assign(n, Vec6::Zero());
    c.assign(n, Vec6::Zero());
    f.assign(n, Vec6::Zero());
    Ic.assign(n, Mat6::Zero());
    pA.assign(n, Vec6::Zero());
    U.assign(n, Mat6X::Zero(6, 0));
    Dinv.resize(n);
    u.resize(n);

    for (int i = 1; i < model.nbodies(); ++i) {
        const auto k   = static_cast<std::size_t>(i);
        const int  nvi = model.nvs[i];
        joint[k].S.setZero(6, nvi);
        S[k].setZero(6, nvi);
        U[k].setZero(6, nvi);
        Dinv[k].setZero(nvi, nvi);
        u[k].setZero(nvi);
    }

    M      = MatX::Zero(model.nv, model.nv);
    tau    = VecX::Zero(model.nv);
    ddq    = VecX::Zero(model.nv);
    zero_v = VecX::Zero(model.nv);
}

} // namespace mbd::kernel
