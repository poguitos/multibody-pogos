#pragma once

#include <Eigen/Dense>
#include <vector>
#include "mbd/system.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

struct SolverConfig {
    Real alpha = 10.0;
    Real beta  = 10.0;
};

/// Compute the 6x6 inverse mass matrix for a body.
/// Ground (index 0) returns zero (infinite mass = zero inverse mass).
inline Mat6 compute_body_M_inv(const MultibodySystem& system, BodyIndex body_idx)
{
    Mat6 M_inv = Mat6::Zero();

    if (system.is_ground(body_idx)) {
        return M_inv;
    }

    const auto& inertia = system.inertias[body_idx];
    const auto& state   = system.states[body_idx];

    Real inv_mass = Real(1.0) / inertia.mass;
    M_inv.block<3,3>(0,0) = Mat3::Identity() * inv_mass;

    Mat3 R = state.q_WB.toRotationMatrix();
    Mat3 I_W = R * inertia.I_com_B * R.transpose();
    M_inv.block<3,3>(3,3) = I_W.inverse();

    return M_inv;
}

/// Solves for Lagrange multipliers and adds constraint forces to system.forces.
inline void solve_constraints(MultibodySystem& system,
                              const Vec3& gravity_W,
                              const SolverConfig& config = SolverConfig())
{
    if (system.constraints.empty()) return;

    int total_eqs = 0;
    for (const auto& c : system.constraints) {
        total_eqs += c->equation_count();
    }
    if (total_eqs == 0) return;

    Eigen::MatrixXd A(total_eqs, total_eqs);
    Eigen::VectorXd b(total_eqs);
    A.setZero();
    b.setZero();

    // Unconstrained accelerations
    std::vector<Vec6> a_unc(static_cast<size_t>(system.body_count()));
    for (BodyIndex i = 0; i < system.body_count(); ++i) {
        if (system.is_ground(i)) {
            a_unc[static_cast<size_t>(i)].setZero();
            continue;
        }
        Vec3 a_lin, a_ang;
        compute_rigid_body_acceleration(
            system.inertias[i], system.states[i], system.forces[i], gravity_W,
            a_lin, a_ang);
        a_unc[static_cast<size_t>(i)] << a_lin, a_ang;
    }

    // Inverse mass matrices
    std::vector<Mat6> M_inv(static_cast<size_t>(system.body_count()));
    for (BodyIndex i = 0; i < system.body_count(); ++i) {
        M_inv[static_cast<size_t>(i)] = compute_body_M_inv(system, i);
    }

    // Assemble system
    struct ConInfo { int row_start; int count; };
    std::vector<ConInfo> con_infos;

    int row_offset = 0;
    for (const auto& constr : system.constraints) {
        int n_eq = constr->equation_count();
        con_infos.push_back({row_offset, n_eq});

        BodyIndex b1 = constr->body1_idx;
        BodyIndex b2 = constr->body2_idx;

        Eigen::MatrixXd J1, J2;
        constr->jacobian(system, J1, J2);

        Eigen::VectorXd gamma_vec;
        constr->velocity_bias(system, gamma_vec);

        Eigen::VectorXd phi;
        constr->evaluate(system, phi);

        Vec6 v1, v2;
        v1 << system.states[b1].v_WB, system.states[b1].w_WB;
        v2 << system.states[b2].v_WB, system.states[b2].w_WB;

        Eigen::VectorXd phi_dot = J1 * v1 + J2 * v2;

        Eigen::VectorXd stab_term = (2.0 * config.alpha) * phi_dot +
                                    (config.beta * config.beta) * phi;

        b.segment(row_offset, n_eq) -= (gamma_vec + stab_term);
        b.segment(row_offset, n_eq) -= (J1 * a_unc[static_cast<size_t>(b1)] +
                                        J2 * a_unc[static_cast<size_t>(b2)]);

        // Build LHS: J * M_inv * J^T
        int col_offset = 0;
        for (const auto& other_constr : system.constraints) {
            int other_n_eq = other_constr->equation_count();
            Eigen::MatrixXd J1_o, J2_o;
            other_constr->jacobian(system, J1_o, J2_o);
            BodyIndex ob1 = other_constr->body1_idx;
            BodyIndex ob2 = other_constr->body2_idx;

            Eigen::MatrixXd block(n_eq, other_n_eq);
            block.setZero();

            if (b1 == ob1) block += J1 * M_inv[static_cast<size_t>(b1)] * J1_o.transpose();
            if (b1 == ob2) block += J1 * M_inv[static_cast<size_t>(b1)] * J2_o.transpose();
            if (b2 == ob1) block += J2 * M_inv[static_cast<size_t>(b2)] * J1_o.transpose();
            if (b2 == ob2) block += J2 * M_inv[static_cast<size_t>(b2)] * J2_o.transpose();

            A.block(row_offset, col_offset, n_eq, other_n_eq) = block;
            col_offset += other_n_eq;
        }

        row_offset += n_eq;
    }

    // Solve
    Eigen::VectorXd lambda = A.ldlt().solve(b);

    // Apply constraint forces
    row_offset = 0;
    for (size_t i = 0; i < system.constraints.size(); ++i) {
        const auto& constr = system.constraints[i];
        int n_eq = con_infos[i].count;

        Eigen::VectorXd lambda_local = lambda.segment(row_offset, n_eq);

        Eigen::MatrixXd J1, J2;
        constr->jacobian(system, J1, J2);

        Vec6 Fc1 = J1.transpose() * lambda_local;
        Vec6 Fc2 = J2.transpose() * lambda_local;

        system.forces[constr->body1_idx].f_W   += Fc1.head<3>();
        system.forces[constr->body1_idx].tau_W += Fc1.tail<3>();

        system.forces[constr->body2_idx].f_W   += Fc2.head<3>();
        system.forces[constr->body2_idx].tau_W += Fc2.tail<3>();

        row_offset += n_eq;
    }
}

} // namespace mbd