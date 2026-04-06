#pragma once

// Simulation orchestrator: time-stepping loop for tree-based multibody systems.
//
// Usage:
//   MultibodySystem sys;
//   // ... add bodies, joints ...
//   Simulator sim(sys);
//   sim.set_gravity({0, -9.81, 0});
//   sim.initialize();
//   for (int i = 0; i < 1000; ++i) {
//       sim.step(0.001);
//   }

#include "mbd/system.hpp"
#include "mbd/algorithms.hpp"

#include <vector>
#include <functional>

namespace mbd {

/// A snapshot of simulation state at a given time.
struct StateRecord {
    Real time;
    VecX q;
    VecX q_dot;
};

/// Integration method selection.
enum class IntegrationMethod {
    SemiImplicitEuler,
    RK4
};

/// Simulation orchestrator for tree-based multibody systems.
class Simulator {
public:
    MultibodySystem& system;

    Vec3 gravity{Vec3(0.0, -g_accel, 0.0)};

    /// Applied generalized forces (size = total_dof).
    /// Accumulated by the user or force callbacks, cleared after each step.
    VecX tau_applied;

    /// Current simulation time.
    Real time{0.0};

    /// Integration method (default: RK4).
    IntegrationMethod method{IntegrationMethod::RK4};

    /// Recorded state history (populated if recording is enabled).
    std::vector<StateRecord> history;
    bool recording{false};

    /// Baumgarte stabilization parameters for loop-closing constraints.
    Real constraint_alpha{5.0};
    Real constraint_beta{5.0};

    /// Optional callback invoked before each dynamics evaluation.
    /// Use this to apply time-dependent or state-dependent forces.
    /// Signature: void(MultibodySystem& sys, Real time, VecX& tau)
    std::function<void(MultibodySystem&, Real, VecX&)> force_callback;

    /// Called after kinematics but before force elements. Use for drivetrain
    /// to set wheel omega before tire slip computation.
    std::function<void(MultibodySystem&, Real)> pre_force_callback;

    /// Called once after each complete time step (not during RK4 sub-stages).
    /// Use for drivetrain wheel spin integration. Receives (sys, dt).
    std::function<void(MultibodySystem&, Real)> post_step_callback;

    explicit Simulator(MultibodySystem& sys)
        : system(sys)
    {}

    /// Initialize the simulator: compute initial FK and allocate tau.
    /// Must be called after all bodies and joints are added.
    void initialize()
    {
        tau_applied = VecX::Zero(system.total_dof);
        system.compute_kinematics();

        if (recording) {
            history.clear();
            record_state();
        }
    }

    /// Set gravity vector.
    void set_gravity(const Vec3& g) { gravity = g; }

    /// Enable/disable state recording.
    void set_recording(bool enabled) { recording = enabled; }

    /// Advance the simulation by dt seconds.
    void step(Real dt)
    {
        switch (method) {
            case IntegrationMethod::SemiImplicitEuler:
                step_semi_implicit_euler(dt);
                break;
            case IntegrationMethod::RK4:
                step_rk4(dt);
                break;
        }

        time += dt;

        if (recording) {
            record_state();
        }

        // Clear applied forces for next step
        tau_applied.setZero();

        if (post_step_callback) {
            post_step_callback(system, dt);
        }
    }

    /// Run the simulation for a total duration with fixed time step.
    /// Returns the number of steps taken.
    int run(Real duration, Real dt)
    {
        int steps = static_cast<int>(std::round(duration / dt));
        for (int i = 0; i < steps; ++i) {
            step(dt);
        }
        return steps;
    }

private:
    /// Record current state to history.
    void record_state()
    {
        history.push_back({time, system.q, system.q_dot});
    }

    /// Evaluate the derivative: returns [q_dot, q_ddot].
    /// Applies registered force elements, projects them to joint space,
    /// then calls forward dynamics.
    VecX evaluate_derivative(const VecX& q_in, const VecX& qd_in, Real t)
    {
        const int n = system.total_dof;

        system.q     = q_in;
        system.q_dot = qd_in;
        system.compute_kinematics();

        if (pre_force_callback) {
            pre_force_callback(system, t);
        }

        // Apply registered force elements and project to joint space
        system.clear_forces();
        system.apply_force_elements();
        VecX tau = tau_applied + project_body_forces_to_joint_space(system);

        if (force_callback) {
            force_callback(system, t, tau);
        }

        VecX q_ddot = constrained_forward_dynamics(
            system, tau, gravity, constraint_alpha, constraint_beta);

        VecX dstate(2 * n);
        dstate.head(n) = qd_in;
        dstate.tail(n) = q_ddot;
        return dstate;
    }

    /// Semi-implicit (symplectic) Euler: update velocities first, then positions.
    void step_semi_implicit_euler(Real dt)
    {
        system.compute_kinematics();

        if (pre_force_callback) {
            pre_force_callback(system, time);
        }

        // Apply registered force elements and project to joint space
        system.clear_forces();
        system.apply_force_elements();
        VecX tau = tau_applied + project_body_forces_to_joint_space(system);

        if (force_callback) {
            force_callback(system, time, tau);
        }

        VecX q_ddot = constrained_forward_dynamics(
            system, tau, gravity, constraint_alpha, constraint_beta);

        // Update velocities first (semi-implicit)
        system.q_dot += q_ddot * dt;

        // Update positions with new velocities
        system.q += system.q_dot * dt;

        // Recompute FK for consistency
        system.compute_kinematics();
    }

    /// Classical 4th-order Runge-Kutta.
    void step_rk4(Real dt)
    {
        const int n = system.total_dof;
        const VecX q0  = system.q;
        const VecX qd0 = system.q_dot;

        // k1
        VecX k1 = evaluate_derivative(q0, qd0, time);

        // k2
        VecX k2 = evaluate_derivative(
            q0  + 0.5 * dt * k1.head(n),
            qd0 + 0.5 * dt * k1.tail(n),
            time + 0.5 * dt);

        // k3
        VecX k3 = evaluate_derivative(
            q0  + 0.5 * dt * k2.head(n),
            qd0 + 0.5 * dt * k2.tail(n),
            time + 0.5 * dt);

        // k4
        VecX k4 = evaluate_derivative(
            q0  + dt * k3.head(n),
            qd0 + dt * k3.tail(n),
            time + dt);

        // Combine
        VecX dstate = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;

        system.q     = q0  + dt * dstate.head(n);
        system.q_dot = qd0 + dt * dstate.tail(n);

        // Final FK
        system.compute_kinematics();
    }
};

} // namespace mbd