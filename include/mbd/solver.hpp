#pragma once

#include <Eigen/Dense>
#include <vector>
#include "mbd/system.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

struct SolverConfig {
    // Baumgarte stabilization parameters
    // 2 * alpha is the damping coefficient for velocity error.
    // beta^2 is the spring coefficient for position error.
    // Typical values rely on time step, e.g., alpha = 1/dt, beta = 1/dt.
    Real alpha = 10.0; 
    Real beta  = 10.0;
};

// Solves for Lagrange multipliers (lambda) and adds constraint forces to system.forces.
//
// Equation:
// (J * M^-1 * J^T) * lambda = -gamma - J * a_unc - 2*alpha*Phi_dot - beta^2*Phi
//
inline void solve_constraints(MultibodySystem& system, 
                              const Vec3& gravity_W,
                              const SolverConfig& config = SolverConfig()) 
{
    if (system.constraints.empty()) return;

    // 1. Count equations
    int total_eqs = 0;
    for (const auto& c : system.constraints) {
        total_eqs += c->equation_count();
    }
    if (total_eqs == 0) return;

    // 2. Prepare Global Matrices
    Eigen::MatrixXd A(total_eqs, total_eqs);
    Eigen::VectorXd b(total_eqs);
    A.setZero();
    b.setZero();

    // 3. Compute Unconstrained Accelerations (a_unc)
    std::vector<Vec6> a_unc(system.body_count());
    for (size_t i = 0; i < system.body_count(); ++i) {
        Vec3 a_lin, a_ang;
        compute_rigid_body_acceleration(
            system.inertias[i], system.states[i], system.forces[i], gravity_W, 
            a_lin, a_ang);
        a_unc[i] << a_lin, a_ang;
    }

    // 4. Precompute Inverse Mass Matrices
    std::vector<Mat6> M_inv(system.body_count());
    for (size_t i = 0; i < system.body_count(); ++i) {
        M_inv[i].setZero();
        Real inv_mass = Real(1.0) / system.inertias[i].mass;
        M_inv[i].block<3,3>(0,0) = Mat3::Identity() * inv_mass;
        
        Mat3 R = system.states[i].pose_WB().rotation();
        Mat3 I_W = R * system.inertias[i].I_com_B * R.transpose();
        M_inv[i].block<3,3>(3,3) = I_W.inverse();
    }

    // 5. Assemble System (A and b)
    int row_offset = 0;
    struct ConInfo { int row_start; int count; };
    std::vector<ConInfo> con_infos;

    for (const auto& constr : system.constraints) {
        int n_eq = constr->equation_count();
        con_infos.push_back({row_offset, n_eq});

        BodyIndex b1 = constr->body1_idx;
        BodyIndex b2 = constr->body2_idx;

        // Jacobian and Bias (gamma = dot(J)*v)
        Eigen::MatrixXd J1, J2;
        constr->jacobian(system, J1, J2);

        Eigen::VectorXd gamma;
        constr->velocity_bias(system, gamma);

        // --- Baumgarte Terms ---
        // Position Error (Phi)
        Eigen::VectorXd phi;
        constr->evaluate(system, phi);

        // Velocity Error (Phi_dot = J * v)
        // Extract velocities
        Vec6 v1, v2;
        v1 << system.states[b1].v_WB, system.states[b1].w_WB;
        v2 << system.states[b2].v_WB, system.states[b2].w_WB;
        
        Eigen::VectorXd phi_dot = J1 * v1 + J2 * v2;

        // --- Build RHS (b) ---
        // b = -gamma - J*a_unc - 2*alpha*phi_dot - beta^2*phi
        
        Eigen::VectorXd stab_term = (2.0 * config.alpha) * phi_dot + 
                                    (config.beta * config.beta) * phi;
                                    
        b.segment(row_offset, n_eq) -= (gamma + stab_term);

        // Subtract J * a_unc
        Vec6 a1_u = a_unc[b1];
        Vec6 a2_u = a_unc[b2];
        Eigen::VectorXd Ja = J1 * a1_u + J2 * a2_u;
        b.segment(row_offset, n_eq) -= Ja;

        // --- Build LHS (A) ---
        int col_offset = 0;
        for (const auto& other_constr : system.constraints) {
            int other_n_eq = other_constr->equation_count();
            Eigen::MatrixXd J1_o, J2_o;
            other_constr->jacobian(system, J1_o, J2_o);
            BodyIndex ob1 = other_constr->body1_idx;
            BodyIndex ob2 = other_constr->body2_idx;

            Eigen::MatrixXd block(n_eq, other_n_eq);
            block.setZero();

            if (b1 == ob1) block += J1 * M_inv[b1] * J1_o.transpose();
            if (b1 == ob2) block += J1 * M_inv[b1] * J2_o.transpose();
            if (b2 == ob1) block += J2 * M_inv[b2] * J1_o.transpose();
            if (b2 == ob2) block += J2 * M_inv[b2] * J2_o.transpose();

            A.block(row_offset, col_offset, n_eq, other_n_eq) = block;
            col_offset += other_n_eq;
        }

        row_offset += n_eq;
    }

    // 6. Solve
    Eigen::VectorXd lambda = A.ldlt().solve(b);

    // 7. Apply Forces
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