// Checks that stay on (plan task 2.9): validate() gives each malformed system
// its own message and counts the degrees of freedom, and the algorithms,
// the constraint solver and the simulator reject inputs of the wrong size or
// references to bodies that do not exist, in every build.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "mbd/core/core.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/kernel/algorithms.hpp"
#include "mbd/kernel/constrained_dynamics.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/simulator.hpp"
#include "mbd/kernel/validate.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;

namespace {

RigidBodyInertia box(Real mass, const Vec3& com = Vec3::Zero())
{
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(mass, Vec3(0.05, 0.05, 0.05));
    I.com_B = com;
    return I;
}

/// True if one message contains every fragment.
bool has_message(const std::vector<std::string>& messages,
                 std::initializer_list<const char*> fragments)
{
    for (const auto& m : messages) {
        bool all = true;
        for (const char* f : fragments) all = all && m.find(f) != std::string::npos;
        if (all) return true;
    }
    return false;
}

/// A planar four-bar in the XY plane: crank (up, length 1), coupler (right,
/// length 2) and rocker (down, length 1) on revolutes about Z, the rocker's
/// end held at the ground pivot (2, 0, 0) by a revolute closure. At the
/// neutral configuration it is a 1 x 2 rectangle, assembled.
System four_bar()
{
    System sys;
    const auto revolute = std::make_shared<RevoluteJointModel>();
    const int crank = sys.model.add_body(0, revolute, Transform3(), Transform3(),
                                         box(1.0, Vec3(0.0, 0.5, 0.0)), "crank");
    const int coupler = sys.model.add_body(crank, revolute, Transform3::FromTranslation(Vec3(0.0, 1.0, 0.0)),
                                           Transform3(), box(2.0, Vec3(1.0, 0.0, 0.0)), "coupler");
    const int rocker = sys.model.add_body(coupler, revolute, Transform3::FromTranslation(Vec3(2.0, 0.0, 0.0)),
                                          Transform3(), box(1.0, Vec3(0.0, -0.5, 0.0)), "rocker");
    sys.constraints.push_back(revolute_closure(
        Marker{rocker, Transform3::FromTranslation(Vec3(0.0, -1.0, 0.0))},
        Marker{0, Transform3::FromTranslation(Vec3(2.0, 0.0, 0.0))}));
    return sys;
}

/// One body on a revolute joint about Z.
System pendulum()
{
    System sys;
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(),
                       box(1.0, Vec3(0.5, 0.0, 0.0)), "arm");
    return sys;
}

} // namespace

// --- Sound systems: the counts -----------------------------------------------------

TEST_CASE("Validate: a planar four-bar closed in 3D has one degree of freedom", "[kernel][validate]")
{
    // Three revolutes, five closure equations of which only two act in the
    // plane: rank 2, three redundant, 3 - 2 = 1 degree of freedom.
    const System sys = four_bar();
    const ValidationReport r = validate(sys);
    INFO(r.summary());
    CHECK(r.ok());
    CHECK(r.warnings.empty());
    CHECK(r.bodies == 3);
    CHECK(r.coordinates == 3);
    CHECK(r.velocities == 3);
    CHECK(r.constraint_equations == 5);
    CHECK(r.independent_equations == 2);
    CHECK(r.redundant_equations == 3);
    CHECK(r.degrees_of_freedom == 1);
    CHECK(has_message(r.notes, {"3 of the 5", "redundant"}));
}

TEST_CASE("Validate: a free body with nothing on it is noted as floating", "[kernel][validate]")
{
    System sys;
    sys.model.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(), box(1.0),
                       "probe");
    const ValidationReport r = validate(sys);
    CHECK(r.ok());
    CHECK(r.degrees_of_freedom == 6);
    CHECK(has_message(r.notes, {"body 1 (probe)", "floats freely"}));

    // A spring to the ground holds it: no note.
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, 1, Vec3::Zero(), Vec3::Zero(), 100.0, 0.0, 0.0));
    CHECK(validate(sys).notes.empty());
}

TEST_CASE("Validate: the summary gives the counts and every message", "[kernel][validate]")
{
    System sys = pendulum();
    sys.model.inertia[1].mass = 3.0;   // changed after add_body: an error
    const std::string s = validate(sys).summary();
    CHECK_THAT(s, ContainsSubstring("bodies 1, coordinates 1, velocities 1"));
    CHECK_THAT(s, ContainsSubstring("Error: body 1 (arm)"));
}

// --- Malformed models ------------------------------------------------------------------

TEST_CASE("Validate: bodies out of topological order", "[kernel][validate]")
{
    System sys = four_bar();
    sys.model.parent[2] = 3;   // the coupler hung from the rocker, which comes after it
    const ValidationReport r = validate(sys);
    CHECK_FALSE(r.ok());
    CHECK(has_message(r.errors, {"body 2 (coupler)", "parent is body 3", "topological order"}));
}

TEST_CASE("Validate: inertias no real body has", "[kernel][validate]")
{
    auto with_inertia = [](const Mat3& I_com) {
        System sys;
        RigidBodyInertia I = box(1.0);
        I.I_com_B = I_com;
        sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(), Transform3(), I,
                           "odd");
        return validate(sys);
    };

    // A negative principal moment.
    const ValidationReport negative = with_inertia(Mat3(Vec3(-0.1, 1.0, 1.0).asDiagonal()));
    CHECK(has_message(negative.errors, {"body 1 (odd)", "negative principal moment"}));

    // Moments 1, 1, 3: the two smaller add up to less than the largest.
    const ValidationReport triangle = with_inertia(Mat3(Vec3(1.0, 1.0, 3.0).asDiagonal()));
    CHECK(has_message(triangle.errors, {"body 1 (odd)", "triangle inequality"}));

    // Not symmetric.
    Mat3 skewed = Mat3::Identity();
    skewed(0, 1) = 0.2;
    const ValidationReport asymmetric = with_inertia(skewed);
    CHECK(has_message(asymmetric.errors, {"body 1 (odd)", "not symmetric"}));

    // Not a number.
    Mat3 nan = Mat3::Identity();
    nan(2, 2) = std::numeric_limits<Real>::quiet_NaN();
    const ValidationReport not_finite = with_inertia(nan);
    CHECK(has_message(not_finite.errors, {"body 1 (odd)", "not a finite number"}));
}

TEST_CASE("Validate: an inertia or joint frame changed after the body was added",
          "[kernel][validate]")
{
    System sys = four_bar();
    sys.model.inertia[1].mass = 5.0;
    sys.model.X_CJ[3] = Transform3::FromTranslation(Vec3(0.1, 0.0, 0.0));
    const ValidationReport r = validate(sys);
    CHECK(has_message(r.errors, {"body 1 (crank)", "inertia was changed after"}));
    CHECK(has_message(r.errors, {"body 3 (rocker)", "X_CJ was changed after"}));
}

TEST_CASE("Validate: a joint that moves nothing with mass makes the mass matrix singular",
          "[kernel][validate]")
{
    // A massless body on a slider at the end of a pendulum: nothing resists
    // the slide, so the mass matrix has a zero row.
    System sys = pendulum();
    sys.model.add_body(1, std::make_shared<PrismaticJointModel>(), Transform3(), Transform3(),
                       RigidBodyInertia{}, "ghost");
    const ValidationReport r = validate(sys);
    CHECK_FALSE(r.ok());
    CHECK(has_message(r.errors, {"body 2 (ghost)", "mass matrix is singular"}));
    CHECK_FALSE(has_message(r.errors, {"body 1 (arm)"}));
}

TEST_CASE("Validate: constraints and force elements on bodies that do not exist",
          "[kernel][validate]")
{
    System sys = pendulum();
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{9, Transform3()}, Marker{0, Transform3()}));
    sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(
        0, 7, Vec3::Zero(), Vec3::Zero(), 100.0, 0.0, 0.1));
    const ValidationReport r = validate(sys);
    CHECK(has_message(r.errors, {"constraint 0 (point coincidence)", "body 9", "0 to 1"}));
    CHECK(has_message(r.errors, {"force element 0 (spring-damper)", "body 7", "0 to 1"}));

    // These checks stay on outside validate() as well: the solver and the
    // simulator refuse such systems when they are built.
    CHECK_THROWS_WITH(ConstraintSolver(sys.model, sys.constraints), ContainsSubstring("body 9"));
    sys.constraints.clear();
    CHECK_THROWS_WITH(Simulator(sys), ContainsSubstring("body 7"));
}

TEST_CASE("Validate: a constraint with both markers on one body", "[kernel][validate]")
{
    System sys = pendulum();
    sys.constraints.push_back(std::make_shared<PointCoincidence>(
        Marker{1, Transform3()}, Marker{1, Transform3::FromTranslation(Vec3(0.1, 0.0, 0.0))}));
    const ValidationReport r = validate(sys);
    CHECK(has_message(r.errors, {"constraint 0 (point coincidence)", "body 1 (arm)",
                                 "cannot restrain anything"}));
}

TEST_CASE("Validate: constraints not satisfied at the configuration", "[kernel][validate]")
{
    // The arm's tip is 1 m from the ground point; the distance asks for 2 m.
    System sys = pendulum();
    sys.constraints.push_back(std::make_shared<Distance>(
        Marker{0, Transform3::FromTranslation(Vec3(0.0, 0.0, 0.0))},
        Marker{1, Transform3::FromTranslation(Vec3(1.0, 0.0, 0.0))}, 2.0));
    const ValidationReport r = validate(sys);
    CHECK(r.ok());
    CHECK(has_message(r.warnings, {"not satisfied", "constraint 0 (distance)"}));

    // At a configuration given with the wrong size: an error, nothing evaluated.
    const ValidationReport wrong = validate(sys, VecX::Zero(4));
    CHECK(has_message(wrong.errors, {"4 entries", "nq = 1"}));
}

// --- Size checks in the algorithms -------------------------------------------------------

TEST_CASE("Kernel algorithms check the sizes of their arguments in every build",
          "[kernel][validate]")
{
    const System sys = four_bar();
    const Model& model = sys.model;
    Data data(model);
    const VecX q = model.neutral_configuration();
    const VecX v = VecX::Zero(model.nv);
    const VecX wrong = VecX::Zero(model.nv + 1);

    CHECK_THROWS_WITH(forward_kinematics(model, data, wrong), ContainsSubstring("nq = 3"));
    CHECK_THROWS_WITH(forward_kinematics(model, data, q, wrong), ContainsSubstring("nv = 3"));
    CHECK_THROWS_WITH(rnea(model, data, q, v, wrong), ContainsSubstring("nv = 3"));
    CHECK_THROWS_WITH(crba(model, data, wrong), ContainsSubstring("nq = 3"));
    CHECK_THROWS_WITH(aba(model, data, q, wrong, v), ContainsSubstring("nv = 3"));

    // Data sized for another model.
    const System other = pendulum();
    Data small(other.model);
    CHECK_THROWS_WITH(forward_kinematics(model, small, q), ContainsSubstring("Data"));

    ConstraintSolver solver(model, sys.constraints);
    CHECK_THROWS_WITH(solver.forward_dynamics(data, q, v, wrong, 0.0), ContainsSubstring("nv = 3"));

    System copy = four_bar();
    Simulator sim(copy);
    sim.v = wrong;
    CHECK_THROWS_WITH(sim.step(1e-3), ContainsSubstring("nv = 3"));
}
