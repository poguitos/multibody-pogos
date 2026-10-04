#include "mbd/kernel/forces.hpp"

#include "mbd/kernel/algorithms.hpp"
#include "mbd/spatial/spatial.hpp"

namespace mbd::kernel {

RigidBodyState body_state(const Data& data, int i)
{
    RigidBodyState s;
    const Transform3& X = data.oMi[i];
    s.p_WB = X.p;
    s.q_WB = X.q;
    s.w_WB = X.q * data.v[i].head<3>();
    s.v_WB = X.q * data.v[i].tail<3>();
    return s;
}

void body_states(const Model& model, const Data& data, std::vector<RigidBodyState>& states)
{
    states.resize(static_cast<std::size_t>(model.nbodies()));
    for (int i = 0; i < model.nbodies(); ++i) states[static_cast<std::size_t>(i)] = body_state(data, i);
}

void generalized_forces(const Model& model, Data& data,
                        const std::vector<RigidBodyForces>& forces, VecX& tau)
{
    MBD_ASSERT(static_cast<int>(forces.size()) == model.nbodies());
    if (tau.size() != model.nv) tau.resize(model.nv);
    tau.setZero();
    const int nb = model.nbodies();

    // Each body's force as a spatial force in its own frame: the moment is
    // already about the origin, so only the axes change.
    for (int i = 1; i < nb; ++i) {
        const auto& f = forces[static_cast<std::size_t>(i)];
        const Quat R_BW = data.oMi[i].q.conjugate();
        data.f[i] << R_BW * f.tau_W, R_BW * f.f_W;
    }

    // From the leaves: each joint carries the forces on everything below it.
    for (int i = nb - 1; i > 0; --i) {
        if (model.nvs[i] > 0) {
            tau.segment(model.idx_v[i], model.nvs[i]).noalias() = data.S[i].transpose() * data.f[i];
        }
        const int p = model.parent[i];
        if (p > 0) data.f[p] += force_act(data.liMi[i], data.f[i]);
    }
}

} // namespace mbd::kernel
