#pragma once

#include <Eigen/Dense>
#include <vector>
#include "mbd/system.hpp"
#include "mbd/dynamics.hpp"

namespace mbd {

// Solves for Lagrange multipliers (lambda) and adds constraint forces to system.forces.
//
// Solves the linear system: (J * M^-1 * J^T) * lambda = -gamma - J * a_unc
// Then applies:             F_c = J^T * lambda
//
// Arguments:
//   system:    The multibody system containing bodies, constraints, and current states.
//   gravity_W: Global gravity vector (needed to compute unconstrained acceleration a_unc).
inline void solve_constraints(MultibodySystem& system, const Vec3& gravity_W) {
    if (system.constraints.empty()) return;

    // 1. Count total constraint equations to size the global matrix
    int total_eqs = 0;
    for (const auto& c : system.constraints) {
        total_eqs += c->equation_count();
    }
    if (total_eqs == 0) return;

    // 2. Prepare Global Matrices (Dense solver approach)
    // A * lambda = b
    // A is the "effective inverse mass" seen by the constraints.
    Eigen::MatrixXd A(total_eqs, total_eqs);
    Eigen::VectorXd b(total_eqs);
    A.setZero();
    b.setZero();

    // 3. Compute Unconstrained Accelerations (a_unc) for all bodies
    // We assume the system.forces already contain external forces (springs, etc.) 
    // but NOT yet the constraint forces.
    std::vector<Vec6> a_unc(system.body_count());
    
    for (size_t i = 0; i < system.body_count(); ++i) {
        Vec3 a_lin, a_ang;
        // Compute F=ma and Tau=Ia for the body as if it were free
        compute_rigid_body_acceleration(
            system.inertias[i], system.states[i], system.forces[i], gravity_W, 
            a_lin, a_ang);
        a_unc[i] << a_lin, a_ang;
    }

    // 4. Precompute Inverse Mass Matrices (Block Diagonal)
    // M_inv_i is 6x6.
    std::vector<Mat6> M_inv(system.body_count());
    for (size_t i = 0; i < system.body_count(); ++i) {
        M_inv[i].setZero();
        
        // Linear part: 1/m * I (Identity)
        Real inv_mass = Real(1.0) / system.inertias[i].mass;
        M_inv[i].block<3,3>(0,0) = Mat3::Identity() * inv_mass;
        
        // Angular part: I_W^-1
        // We must rotate the body inertia to world frame first: I_W = R * I_B * R^T
        Mat3 R = system.states[i].pose_WB().rotation();
        Mat3 I_W = R * system.inertias[i].I_com_B * R.transpose();
        
        // Invert the inertia matrix (Cholesky or standard inverse)
        M_inv[i].block<3,3>(3,3) = I_W.inverse();
    }

    // 5. Assemble System (A and b)
    // We iterate over all constraints to fill the rows of the linear system.
    int row_offset = 0;
    
    // Helper struct to map global rows back to specific constraints later
    struct ConInfo { int row_start; int count; };
    std::vector<ConInfo> con_infos;

    for (const auto& constr : system.constraints) {
        int n_eq = constr->equation_count();
        con_infos.push_back({row_offset, n_eq});

        BodyIndex b1 = constr->body1_idx;
        BodyIndex b2 = constr->body2_idx;

        // Get Jacobians and Bias for this constraint
        Eigen::MatrixXd J1, J2;
        constr->jacobian(system, J1, J2);

        Eigen::VectorXd gamma;
        constr->velocity_bias(system, gamma);

        // --- Build RHS Vector b ---
        // b = -gamma - J * a_unc
        b.segment(row_offset, n_eq) -= gamma;

        // Calculate term: J * a_unc = J1 * a1 + J2 * a2
        Vec6 a1 = a_unc[b1];
        Vec6 a2 = a_unc[b2];
        Eigen::VectorXd Ja = J1 * a1 + J2 * a2;
        b.segment(row_offset, n_eq) -= Ja;

        // --- Build LHS Matrix A ---
        // We need to compute the block: J * M^-1 * J^T
        // For a diagonal M^-1 (uncoupled bodies), A_ij involves summing over shared bodies.
        
        // We perform a double loop over constraints to fill the matrix A.
        // This handles cases where constraints are coupled (e.g. two links sharing a body).
        int col_offset = 0;
        for (const auto& other_constr : system.constraints) {
            int other_n_eq = other_constr->equation_count();
            
            // Re-fetch Jacobians of the 'other' constraint
            // (Optimization: cache these in a real engine)
            Eigen::MatrixXd J1_other, J2_other;
            other_constr->jacobian(system, J1_other, J2_other);
            BodyIndex ob1 = other_constr->body1_idx;
            BodyIndex ob2 = other_constr->body2_idx;

            Eigen::MatrixXd block(n_eq, other_n_eq);
            block.setZero();

            // Accumulate contributions from shared bodies.
            // Formula: Sum( J_current_k * M_k^-1 * J_other_k^T ) for body k
            
            // Check Body 1 of C1 vs Body 1 of C2
            if (b1 == ob1) block += J1 * M_inv[b1] * J1_other.transpose();
            // Check Body 1 of C1 vs Body 2 of C2
            if (b1 == ob2) block += J1 * M_inv[b1] * J2_other.transpose();
            // Check Body 2 of C1 vs Body 1 of C2
            if (b2 == ob1) block += J2 * M_inv[b2] * J1_other.transpose();
            // Check Body 2 of C1 vs Body 2 of C2
            if (b2 == ob2) block += J2 * M_inv[b2] * J2_other.transpose();

            // Fill the sub-block in A
            A.block(row_offset, col_offset, n_eq, other_n_eq) = block;
            col_offset += other_n_eq;
        }

        row_offset += n_eq;
    }

    // 6. Solve for Lambda (Constraint Forces/Impulses)
    // A is usually Symmetric Positive Definite (unless redundant constraints exist).
    // LDLT is a robust choice for symmetric systems.
    Eigen::VectorXd lambda = A.ldlt().solve(b);

    // 7. Apply Constraint Forces back to the bodies
    // F_c = J^T * lambda
    row_offset = 0;
    for (size_t i = 0; i < system.constraints.size(); ++i) {
        const auto& constr = system.constraints[i];
        int n_eq = con_infos[i].count;
        
        // Extract the lambda subset for this constraint
        Eigen::VectorXd lambda_local = lambda.segment(row_offset, n_eq);
        
        Eigen::MatrixXd J1, J2;
        constr->jacobian(system, J1, J2);
        
        // Calculate force vectors in 6D space (force + torque)
        Vec6 Fc1 = J1.transpose() * lambda_local;
        Vec6 Fc2 = J2.transpose() * lambda_local;
        
        // Accumulate into the system's force buffers
        system.forces[constr->body1_idx].f_W   += Fc1.head<3>();
        system.forces[constr->body1_idx].tau_W += Fc1.tail<3>();
        
        system.forces[constr->body2_idx].f_W   += Fc2.head<3>();
        system.forces[constr->body2_idx].tau_W += Fc2.tail<3>();

        row_offset += n_eq;
    }
}

} // namespace mbd