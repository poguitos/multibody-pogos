#include <catch2/catch_test_macros.hpp>

#include "invariants/invariant_helpers.hpp"

// Every constraint type is tested between two bodies in two arrangements:
//
//   pair  : two free bodies;
//   chain : a body hanging from a free body by a revolute joint, against a
//           second free body. This exercises the composition of the
//           constraint's own Jacobian with the body Jacobians of the tree.
//
// Attachment points and axes are generic, away from the body origins.

using namespace mbd_test;

namespace
{
    enum class ConstraintKind {
        DistanceAtOrigins,
        DistanceOffset,
        CoincidentPoint,
        RevoluteLoop,
        PointCoordinate,
        StrutLineGround,
        StrutLineTwoBody
    };

    const ConstraintKind kKinds[] = {
        ConstraintKind::DistanceAtOrigins, ConstraintKind::DistanceOffset,
        ConstraintKind::CoincidentPoint,   ConstraintKind::RevoluteLoop,
        ConstraintKind::PointCoordinate,   ConstraintKind::StrutLineGround,
        ConstraintKind::StrutLineTwoBody};

    const char* constraint_name(ConstraintKind k)
    {
        switch (k) {
            case ConstraintKind::DistanceAtOrigins: return "distance, anchors at the body origins";
            case ConstraintKind::DistanceOffset:    return "distance, anchors away from the origins";
            case ConstraintKind::CoincidentPoint:   return "coincident point";
            case ConstraintKind::RevoluteLoop:      return "revolute (5 equations)";
            case ConstraintKind::PointCoordinate:   return "point coordinate";
            case ConstraintKind::StrutLineGround:   return "strut line, top mount on ground";
            case ConstraintKind::StrutLineTwoBody:  return "strut line, top mount on a body";
        }
        return "?";
    }

    const char* kArrangements[] = {"pair", "chain"};
    constexpr int kSeeds = 3;

    struct BodyPair {
        MultibodySystem sys;
        BodyIndex first{0};
        BodyIndex second{0};
    };

    /// Build the two bodies and give them a generic state.
    void make_bodies(BodyPair& bp, Rng& rng, int arrangement)
    {
        if (arrangement == 0) {
            bp.first  = add_random_body(bp.sys, rng, JointKind::Free, mbd::kGroundIndex, "first");
            bp.second = add_random_body(bp.sys, rng, JointKind::Free, mbd::kGroundIndex, "second");
        } else {
            const BodyIndex base =
                add_random_body(bp.sys, rng, JointKind::Free, mbd::kGroundIndex, "base");
            bp.first  = add_random_body(bp.sys, rng, JointKind::Revolute, base, "first");
            bp.second = add_random_body(bp.sys, rng, JointKind::Free, mbd::kGroundIndex, "second");
        }
        randomize_state(bp.sys, rng);
    }

    /// Add one constraint between the two bodies. With `satisfied` the
    /// parameters are chosen so that the constraint holds exactly in the
    /// current configuration; otherwise they are generic.
    void add_constraint(BodyPair& bp, Rng& rng, ConstraintKind kind, bool satisfied)
    {
        auto& sys = bp.sys;
        const auto& s1 = sys.states[bp.first];
        const auto& s2 = sys.states[bp.second];

        const Vec3 a1 = rng.vec(0.3);                  // point on the first body
        const Vec3 a1_W = s1.p_WB + s1.q_WB * a1;
        // The same world point, seen from the second body.
        const Vec3 a1_in_second = s2.q_WB.conjugate() * (a1_W - s2.p_WB);
        const Vec3 a2 = rng.vec(0.3);                  // point on the second body
        const Vec3 a2_W = s2.p_WB + s2.q_WB * a2;

        switch (kind) {
            case ConstraintKind::DistanceAtOrigins: {
                const Real d = satisfied ? (s2.p_WB - s1.p_WB).norm() : Real(1.0);
                sys.constraints.push_back(std::make_shared<mbd::DistanceConstraint>(
                    bp.first, bp.second, Vec3::Zero(), Vec3::Zero(), d));
                break;
            }
            case ConstraintKind::DistanceOffset: {
                const Real d = satisfied ? (a2_W - a1_W).norm() : Real(1.0);
                sys.constraints.push_back(std::make_shared<mbd::DistanceConstraint>(
                    bp.first, bp.second, a1, a2, d));
                break;
            }
            case ConstraintKind::CoincidentPoint: {
                sys.constraints.push_back(std::make_shared<mbd::CoincidentPointConstraint>(
                    bp.first, bp.second, a1, satisfied ? a1_in_second : a2));
                break;
            }
            case ConstraintKind::RevoluteLoop: {
                const Vec3 axis1 = rng.direction();
                const Vec3 axis1_in_second = s2.q_WB.conjugate() * (s1.q_WB * axis1);
                const Vec3 axis2 = rng.direction();
                sys.constraints.push_back(std::make_shared<mbd::RevoluteJoint>(
                    bp.first, bp.second,
                    a1, axis1,
                    satisfied ? a1_in_second : a2,
                    satisfied ? axis1_in_second : axis2));
                break;
            }
            case ConstraintKind::PointCoordinate: {
                const Real target = satisfied ? a2_W.y() : Real(0.5);
                sys.constraints.push_back(std::make_shared<mbd::PointCoordinateConstraint>(
                    bp.second, a2, 1, target));
                break;
            }
            case ConstraintKind::StrutLineGround: {
                const Vec3 top_W = rng.vec(0.8);
                const Vec3 axis_W = satisfied ? (top_W - a2_W).normalized() : rng.direction();
                sys.constraints.push_back(std::make_shared<mbd::StrutLineConstraint>(
                    bp.second, top_W, a2, s2.q_WB.conjugate() * axis_W));
                break;
            }
            case ConstraintKind::StrutLineTwoBody: {
                const Vec3 axis_W = satisfied ? (a1_W - a2_W).normalized() : rng.direction();
                sys.constraints.push_back(std::make_shared<mbd::StrutLineTwoBodyConstraint>(
                    bp.first, bp.second, a1, a2, s2.q_WB.conjugate() * axis_W));
                break;
            }
        }
    }

    std::uint64_t seed_for(int seed, ConstraintKind kind, int arrangement)
    {
        return 5000u * static_cast<unsigned>(seed)
             + 10u * static_cast<unsigned>(kind)
             + static_cast<unsigned>(arrangement);
    }
}

TEST_CASE("Invariant: a constraint's Jacobian is the time derivative of its equation",
          "[invariants][constraints]")
{
    for (ConstraintKind kind : kKinds) {
        for (int arrangement = 0; arrangement < 2; ++arrangement) {
            DYNAMIC_SECTION(constraint_name(kind) << " [" << kArrangements[arrangement] << "]") {
                for (int seed = 1; seed <= kSeeds; ++seed) {
                    Rng rng(seed_for(seed, kind, arrangement));
                    BodyPair bp;
                    make_bodies(bp, rng, arrangement);
                    add_constraint(bp, rng, kind, false);

                    const ConstraintErrors e = constraint_errors(bp.sys).front();
                    CAPTURE(seed, e.jacobian);
                    CHECK(e.jacobian < 1e-6 * e.scale_vel);
                }
            }
        }
    }
}

TEST_CASE("Invariant: a constraint's acceleration term is the second time derivative of its equation",
          "[invariants][constraints]")
{
    for (ConstraintKind kind : kKinds) {
        for (int arrangement = 0; arrangement < 2; ++arrangement) {
            DYNAMIC_SECTION(constraint_name(kind) << " [" << kArrangements[arrangement] << "]") {
                for (int seed = 1; seed <= kSeeds; ++seed) {
                    Rng rng(seed_for(seed, kind, arrangement));
                    BodyPair bp;
                    make_bodies(bp, rng, arrangement);
                    add_constraint(bp, rng, kind, false);

                    const ConstraintErrors e = constraint_errors(bp.sys).front();
                    CAPTURE(seed, e.bias);
                    // Second differences with a step of 1e-4 are accurate to
                    // about 1e-7; a missing term shows up at 1e-2 or more.
                    CHECK(e.bias < 1e-5 * e.scale_acc);
                }
            }
        }
    }
}

TEST_CASE("Invariant: the equations of one constraint are independent",
          "[invariants][constraints]")
{
    for (ConstraintKind kind : kKinds) {
        for (int arrangement = 0; arrangement < 2; ++arrangement) {
            DYNAMIC_SECTION(constraint_name(kind) << " [" << kArrangements[arrangement] << "]") {
                for (int seed = 1; seed <= kSeeds; ++seed) {
                    Rng rng(seed_for(seed, kind, arrangement));
                    BodyPair bp;
                    make_bodies(bp, rng, arrangement);
                    add_constraint(bp, rng, kind, true);

                    const ConstraintErrors e = constraint_errors(bp.sys).front();
                    CAPTURE(seed, e.rank_ratio);
                    // Smallest over largest singular value of the constraint's
                    // rows of the joint-space Jacobian. A redundant equation
                    // drives this to rounding level.
                    CHECK(e.rank_ratio > 1e-6);
                }
            }
        }
    }
}

TEST_CASE("Invariant: a constraint holds without drift correction when its accelerations are exact",
          "[invariants][constraints][simulation]")
{
    const Vec3 gravity(0.0, -mbd::g_accel, 0.0);

    for (ConstraintKind kind : kKinds) {
        for (int arrangement = 0; arrangement < 2; ++arrangement) {
            DYNAMIC_SECTION(constraint_name(kind) << " [" << kArrangements[arrangement] << "]") {
                for (int seed = 1; seed <= kSeeds; ++seed) {
                    Rng rng(seed_for(seed, kind, arrangement));
                    BodyPair bp;
                    make_bodies(bp, rng, arrangement);
                    add_constraint(bp, rng, kind, true);

                    // Projection and Baumgarte terms are switched off, so the
                    // only source of violation is the RK4 truncation error. At
                    // this step it stays below 1e-9 even for the fastest
                    // tumbling bodies generated here; a wrong acceleration term
                    // produces a violation growing as t^2, from millimetres to
                    // a metre over the same run.
                    const Real drift = drift_without_stabilisation(bp.sys, gravity, 0.5, 2.5e-4);
                    CAPTURE(seed, drift);
                    CHECK(drift < 1e-8);
                }
            }
        }
    }
}
