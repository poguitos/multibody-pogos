#pragma once

// Elements with first-order states of their own (plan task 4.1): a
// drivetrain's wheel spins, a tyre's relaxation lag, an actuator, a
// controller. Their states x are integrated together with the multibody state
// (q, v), by the same method and in the same stages, so a force that depends
// on them is as accurate as the integrator:
//
//     x_dot = f(t, q, v, x).
//
// At every evaluation the simulator first computes the kinematics, then calls
// each element's begin_evaluation() with the states of that evaluation (the
// element hands them to whatever needs them: the drivetrain gives the wheel
// spins to the tyres), then the forces and the constrained dynamics, then each
// element's rates(). After every completed step it calls end_step(), where an
// element makes discrete changes (a gear shift, a brake locking a wheel).

#include "mbd/core/core.hpp"
#include "mbd/kernel/joint_model.hpp"   // VecRef, ConstVecRef

namespace mbd::kernel {

class Simulator;

class StateElement {
public:
    virtual ~StateElement() = default;

    /// Short name, for messages.
    virtual const char* name() const = 0;

    /// Number of states.
    virtual int size() const = 0;

    /// The states' values when the element is attached to a simulator.
    virtual void initial_states(VecRef x) const = 0;

    /// At every evaluation, after the kinematics and before the forces: the
    /// states of this evaluation. The simulator's states() and data() hold its
    /// kinematics; acceleration() and refresh() must not be called from here.
    virtual void begin_evaluation(const Simulator& sim, Real t, ConstVecRef x) = 0;

    /// At every evaluation, after the forces and the constrained dynamics:
    /// the states' rates.
    virtual void rates(const Simulator& sim, Real t, ConstVecRef x, VecRef x_dot) = 0;

    /// After every completed step, at the step's end state: discrete changes,
    /// which may change the states.
    virtual void end_step(const Simulator& sim, Real t, VecRef x)
    {
        (void)sim;
        (void)t;
        (void)x;
    }

protected:
    StateElement() = default;
    StateElement(const StateElement&) = default;
    StateElement& operator=(const StateElement&) = default;
};

} // namespace mbd::kernel
