#include "mbd/kernel/outputs.hpp"

#include <string>

#include "mbd/kernel/algorithms.hpp"
#include "mbd/spatial/spatial.hpp"

#include "checks.hpp"

namespace mbd::kernel {

void joint_reactions(const Model& model, Data& data, const std::vector<Vec6>& external_W,
                     std::vector<Vec6>& reactions)
{
    checks::data("kernel::joint_reactions", model, data);
    const int nb = model.nbodies();
    MBD_THROW_IF(external_W.size() != static_cast<std::size_t>(nb),
                 "MBD-K006: kernel::joint_reactions: " + std::to_string(external_W.size())
                     + " external wrenches given, but the model has " + std::to_string(nb)
                     + " bodies.");

    // Each body's net force less the external ones is what its joint and its
    // children's joints transmit; gravity enters as the ground's upward
    // acceleration, as in rnea.
    for (int i = 1; i < nb; ++i) {
        Vec6 a_gf = data.a[i];
        a_gf.tail<3>() -= data.oMi[i].q.conjugate() * model.gravity;
        const Vec6 h = model.I[i] * data.v[i];
        data.f[i] = model.I[i] * a_gf + motion_cross_force(data.v[i], h)
                  - force_act_inv(data.oMi[i], external_W[static_cast<std::size_t>(i)]);
    }
    for (int i = nb - 1; i > 0; --i) {
        const int p = model.parent[i];
        if (p > 0) data.f[p] += force_act(data.liMi[i], data.f[i]);
    }

    reactions.assign(static_cast<std::size_t>(nb), Vec6::Zero());
    for (int i = 1; i < nb; ++i) {
        reactions[static_cast<std::size_t>(i)] = force_act_inv(model.X_CJ[i], data.f[i]);
    }
}

Loads compute_loads(Simulator& sim)
{
    const Model& model = sim.system.model;
    const int nb = model.nbodies();
    Loads out;
    // The evaluation leaves the kinematics, the body states and the forces of
    // the elements at this state.
    out.v_dot = sim.acceleration(sim.q, sim.v, sim.time);
    out.lambda = sim.solver().lambda();

    out.constraint_W.assign(static_cast<std::size_t>(nb), Vec6::Zero());
    Index row = 0;
    for (const auto& c : sim.system.constraints) {
        const Index m = c->size();
        c->add_wrenches(model, sim.data(), sim.time, out.lambda.segment(row, m), out.constraint_W);
        row += m;
    }

    std::vector<Vec6> external = out.constraint_W;
    for (int i = 1; i < nb; ++i) {
        const auto k = static_cast<std::size_t>(i);
        const RigidBodyForces& f = sim.forces()[k];
        external[k].head<3>() += f.tau_W + sim.states()[k].p_WB.cross(f.f_W);
        external[k].tail<3>() += f.f_W;
    }

    Data data(model);
    forward_kinematics(model, data, sim.q, sim.v, out.v_dot);
    joint_reactions(model, data, external, out.joint_reaction);
    return out;
}

} // namespace mbd::kernel
