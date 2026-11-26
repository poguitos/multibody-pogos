#pragma once

// A virtual damper dyno (plan task 7.6, decisions D42 and D66): one end of an
// element held, the other driven through a sinusoidal stroke along an axis,
// as a test rig drives a damper; the force against the stroke and against
// the velocity recorded.
//
// The held end is the ground. The driven end is a body on a prismatic joint
// along `axis`, its axes the world's, its origin at length * axis at
// mid-stroke; a joint driver moves it by
//
//     stroke(t) = amplitude sin(2 pi frequency t),
//
// so it starts at mid-stroke moving outward at its fastest. The element is
// made between the two (any force element; a state element among them is
// integrated with the rest), and the system run by the kernel's simulator
// without gravity. After each step the forces are evaluated at the step's
// end state and the element's force on the driven body read: the force
// recorded is its component against the stroke's direction, positive when
// the element resists extension (a damper's force at a positive velocity),
// the force a rig's load cell reads.

#include <functional>
#include <memory>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/core/math.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/simulator.hpp"

namespace mbd {

struct DynoSettings {
    Real amplitude{0.025};      ///< Of the stroke [m], > 0
    Real frequency{1.0};        ///< [Hz], > 0
    int cycles{3};              ///< Full cycles run, >= 1
    int steps_per_cycle{1000};  ///< Steps of the integrator in each cycle, >= 4
    /// The distance between the two ends at mid-stroke [m], >= 0: an
    /// element's length between its anchors, or zero for a bushing.
    Real length{0.0};
    /// The stroke's direction, from the held end to the driven one (world
    /// axes; normalised).
    Vec3 axis{Vec3::UnitZ()};
    kernel::Integrator method{kernel::Integrator::RK4};
    /// For the methods that choose their own steps within each step of the
    /// dyno (DormandPrince, SDIRK4, RadauIIA): the simulator's step control.
    kernel::StepControl step_control;
};

/// What the dyno recorded: one sample at the start and one after each step.
struct DynoRecord {
    std::vector<Real> time;       ///< [s]
    std::vector<Real> stroke;     ///< The driven end's displacement from mid-stroke [m]
    std::vector<Real> velocity;   ///< Its rate [m/s]
    std::vector<Real> force;      ///< The element's force against the stroke [N]
    int steps_per_cycle{0};

    /// The energy the element took in during cycle k (from 0) [J]: the area
    /// of its loop, the integral of force times velocity over the cycle's
    /// time, by the trapezoidal rule over its samples. Over a period of a
    /// smooth periodic motion that rule is exact for every harmonic below
    /// the number of samples; the sum of force times the stroke's changes,
    /// by contrast, is short by a factor sin(w h) / (w h) (task 7.6).
    Real energy(int cycle) const;
};

/// Makes the element under test between the held body (the ground) and the
/// driven one, each end at its body's origin.
using DynoElement = std::function<std::shared_ptr<ForceElement>(BodyIndex held, BodyIndex driven)>;

/// Run the dyno (see the top of this file). Throws MBD-A030 for settings it
/// cannot use and for no element.
DynoRecord run_damper_dyno(const DynoElement& make, const DynoSettings& settings = {});

} // namespace mbd
