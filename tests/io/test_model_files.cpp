// Model files (plan task 6.4, D56): a kernel::System as JSON and back.
//
//   - a system with every kind of part the format holds survives a round
//     trip unchanged: the text it writes again is the same, character for
//     character, and so are its accelerations at a generic state, bit for
//     bit, and its description;
//   - a four-bar on a spring saved to a file and loaded runs exactly as the
//     original;
//   - a small model's text is the documented format (docs/model-files.md);
//   - what a file cannot hold is refused by name (MBD-I008);
//   - every error in a file names its place, and a constructor's own check
//     keeps its code with the place added.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include "mbd/core/core.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/contact.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/user_force.hpp"
#include "mbd/io/model_file.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/describe.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/simulator.hpp"

#include "kernel/four_bar.hpp"
#include "kernel/hmmwv.hpp"
#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::StartsWith;

namespace {

Transform3 frame(Real angle, const Vec3& axis, const Vec3& p)
{
    return Transform3(Quat(Eigen::AngleAxisd(angle, axis.normalized())), p);
}

RigidBodyInertia tilted_inertia(Real mass, const Vec3& half, const Vec3& com, Real angle)
{
    RigidBodyInertia I = RigidBodyInertia::from_solid_box(mass, half);
    const Mat3 R = Quat(Eigen::AngleAxisd(angle, Vec3(1.0, 2.0, 3.0).normalized())).toRotationMatrix();
    I.I_com_B = R * I.I_com_B * R.transpose();
    I.com_B = com;
    return I;
}

/// Every kind of part the format holds, with generic values: eight bodies,
/// one on each joint, in a chain with a branch; every primitive constraint
/// and every closure, constants for the time functions; two sets made by
/// hand, one of them named like a closure but not one; every force element
/// the format holds, with lines, tables, zero curves and infinite
/// clearances; a joint force with a table, friction and one infinite limit.
System every_kind()
{
    System sys;
    Model& m = sys.model;
    m.gravity = Vec3(0.1, -9.81, 0.2);
    const int b1 = m.add_body(0, std::make_shared<RevoluteJointModel>(), frame(0.3, Vec3(1, 2, 3), Vec3(0.1, 0.2, 0.3)),
                              frame(-0.2, Vec3(0, 1, 1), Vec3(0.0, 0.0, -0.4)),
                              tilted_inertia(2.0, Vec3(0.2, 0.1, 0.05), Vec3(0.01, -0.02, 0.03), 0.7), "arm \"left\"");
    const int b2 = m.add_body(b1, std::make_shared<PrismaticJointModel>(), frame(1.1, Vec3(0, 0, 1), Vec3(0.5, 0, 0)),
                              Transform3(), tilted_inertia(1.5, Vec3(0.1, 0.1, 0.1), Vec3::Zero(), 0.0), "slider");
    const int b3 = m.add_body(b2, std::make_shared<UniversalJointModel>(), Transform3::FromTranslation(Vec3(0, 0.3, 0)),
                              frame(0.4, Vec3(1, 0, 0), Vec3::Zero()),
                              tilted_inertia(0.7, Vec3(0.05, 0.2, 0.05), Vec3(0, 0.1, 0), 1.3), "");
    const int b4 = m.add_body(b3, std::make_shared<CylindricalJointModel>(), Transform3(), Transform3(),
                              tilted_inertia(0.9, Vec3(0.1, 0.05, 0.3), Vec3(0, 0, 0.2), 0.1), "cylinder");
    const int b5 = m.add_body(b1, std::make_shared<SphericalJointModel>(), frame(2.0, Vec3(3, -1, 2), Vec3(0, 0, 0.6)),
                              Transform3(), tilted_inertia(1.2, Vec3(0.1, 0.2, 0.3), Vec3(0.05, 0, 0), 2.2), "ball");
    const int b6 = m.add_body(b5, std::make_shared<PlanarJointModel>(), Transform3(), frame(0.5, Vec3(0, 1, 0), Vec3(0.1, 0, 0)),
                              tilted_inertia(0.4, Vec3(0.05, 0.05, 0.05), Vec3::Zero(), 0.0), "plate");
    const int b7 = m.add_body(0, std::make_shared<FreeJointModel>(), Transform3(), Transform3(),
                              tilted_inertia(3.0, Vec3(0.3, 0.2, 0.1), Vec3(0.02, 0.01, -0.03), 0.9), "free");
    const int b8 = m.add_body(b7, std::make_shared<FixedJointModel>(), frame(0.9, Vec3(1, 1, 0), Vec3(0.2, 0.2, 0)),
                              Transform3(), tilted_inertia(0.5, Vec3(0.1, 0.1, 0.1), Vec3::Zero(), 0.0), "welded");

    const Marker on1{b1, frame(0.2, Vec3(0, 0, 1), Vec3(0.3, 0, 0))}, on4{b4, frame(-0.6, Vec3(1, 0, 0), Vec3(0, 0.1, 0))};
    const Marker on6{b6, Transform3::FromTranslation(Vec3(0.05, 0, 0))}, on8{b8, frame(1.0, Vec3(0, 1, 0), Vec3::Zero())};
    const Marker ground{0, frame(0.1, Vec3(1, 1, 1), Vec3(1.0, 0.5, 0.0))};
    auto& c = sys.constraints;
    c.push_back(std::make_shared<PointCoincidence>(on1, on8));
    c.push_back(std::make_shared<Dot1>(on1, 2, on4, 0, TimeFunction::constant(0.25)));
    c.push_back(std::make_shared<Dot2>(ground, 1, on6, TimeFunction::constant(-0.1)));
    c.push_back(std::make_shared<Distance>(on4, ground, 1.3));
    c.push_back(std::make_shared<Distance>(on6, on8, TimeFunction::constant(0.8), 1.1));
    c.push_back(std::make_shared<NoTwist>(on1, on6));
    c.push_back(std::make_shared<JointDriver>(m, b1, TimeFunction::constant(0.3)));
    for (auto make : {&point_on_line, &point_on_plane, &spherical_closure, &revolute_closure, &universal_closure,
                      &cylindrical_closure, &prismatic_closure, &constant_velocity_closure}) {
        c.push_back(make(ground, on4));
    }
    auto custom = std::make_shared<ConstraintSet>("custom pair");
    custom->add(std::make_shared<Dot1>(on4, 0, on8, 1));
    custom->add(revolute_closure(on1, on6));
    c.push_back(custom);
    auto impostor = std::make_shared<ConstraintSet>("revolute closure");   // not what revolute_closure makes
    impostor->add(std::make_shared<PointCoincidence>(on6, on8));
    c.push_back(impostor);

    auto& f = sys.force_elements;
    f.push_back(std::make_shared<LinearSpringDamper>(0, b7, Vec3(0, 0, 2), Vec3(0.1, 0, 0), 500.0, 12.5, 1.25));
    SpringDamperParams sp;
    sp.free_length = 0.35;
    sp.spring = Curve::table({-0.1, 0.0, 0.05, 0.1}, {-2500.0, 0.0, 1400.0, 3600.0});
    sp.preload = 500.0;
    sp.damper = Curve::linear(1200.0, -3.0);
    sp.bump_clearance = 0.06;
    sp.bump_stop = Curve::table({0.0, 0.01, 0.02}, {0.0, 2000.0, 8000.0});
    f.push_back(std::make_shared<SpringDamper>(b2, b5, Vec3(0.1, 0.2, 0.3), Vec3(-0.1, 0, 0), sp));
    RotationalSpringDamperParams rp;
    rp.reference = 0.1;
    rp.spring = Curve::table({-1.0, 0.0, 0.5, 1.0}, {-300.0, 0.0, 120.0, 400.0});
    rp.preload = 4.0;
    rp.damper = Curve::linear(2.0);
    f.push_back(std::make_shared<RotationalSpringDamper>(b3, frame(0.3, Vec3(0, 1, 0), Vec3::Zero()), b4,
                                                         frame(-0.1, Vec3(1, 0, 0), Vec3(0, 0, 0.1)), rp));
    BushingParams bp;
    bp.stiffness = {Curve::linear(1e5), Curve::linear(2e5), Curve::table({-0.01, 0.0, 0.01}, {-900.0, 0.0, 1100.0})};
    bp.damping = {Curve::linear(100.0), Curve(), Curve::linear(300.0)};
    bp.rotational_stiffness = {Curve::linear(500.0), Curve(), Curve::linear(700.0)};
    f.push_back(std::make_shared<Bushing>(b7, frame(0.2, Vec3(0, 0, 1), Vec3(0.3, 0, 0)), b8, Transform3(), bp));
    ContactParams cp;
    cp.stiffness = 1e5;
    cp.exponent = 1.5;
    cp.damping = 200.0;
    cp.damping_depth = 1e-5;
    cp.mu_static = 0.5;
    cp.mu_dynamic = 0.4;
    cp.v_static = 1e-3;
    cp.v_dynamic = 1e-2;
    f.push_back(std::make_shared<PlaneContact>(b6, std::vector<Vec3>{Vec3(0.1, 0, -0.05), Vec3(-0.1, 0, -0.05)}, 0.02, 0,
                                               frame(0.05, Vec3(1, 0, 0), Vec3::Zero()), cp));

    JointCoordinateForceParams jp;
    jp.reference = 0.02;
    jp.spring = Curve::table({-0.2, 0.0, 0.1, 0.2}, {-900.0, 0.0, 300.0, 1100.0});
    jp.preload = 15.0;
    jp.damper = Curve::linear(3.0);
    jp.friction = 2.0;
    jp.friction_velocity = 0.01;
    jp.lower_limit = -0.15;   // the upper limit left infinite
    jp.limit_stiffness = 1e5;
    jp.limit_damping = 50.0;
    sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(m, b2, jp));
    return sys;
}

/// Bit for bit, signs of zero included.
bool identical(const VecX& a, const VecX& b)
{
    if (a.size() != b.size()) return false;
    for (Index k = 0; k < a.size(); ++k) {
        if (std::signbit(a(k)) != std::signbit(b(k)) || !(a(k) == b(k) || (std::isnan(a(k)) && std::isnan(b(k))))) {
            return false;
        }
    }
    return true;
}

std::string error_of(const std::string& text)
{
    try {
        (void)io::model_from_json(text);
    } catch (const MbdError& e) {
        return e.what();
    }
    return "no error";
}

/// A file of one body with `body` as its entry, the rest as written.
std::string one_body(const std::string& body)
{
    return R"({"format": "multibody model", "version": 1, "bodies": [)" + body + "]}";
}

} // namespace

TEST_CASE("Model files: a system with every kind of part survives a round trip unchanged", "[io]")
{
    System original = every_kind();
    const std::string text = io::model_to_json(original);
    System read = io::model_from_json(text);
    INFO(text);

    // The same text again: nothing written was lost or changed on reading.
    CHECK(io::model_to_json(read) == text);
    CHECK(describe(read) == describe(original));

    // Closures as the calls that make them; the hand-made sets part by part.
    CHECK_THAT(text, ContainsSubstring(R"("type": "constant_velocity_closure")"));
    CHECK_THAT(text, ContainsSubstring(R"("name": "custom pair")"));
    CHECK_THAT(text, ContainsSubstring(R"("name": "revolute closure")"));
    CHECK_THAT(text, ContainsSubstring(R"("upper_limit": "inf")"));
    CHECK_THAT(text, ContainsSubstring(R"("name": "arm \"left\"")"));

    // The same dynamics, bit for bit, at a generic state.
    mbd_test::Rng rng(29);
    const VecX q = mbd_test::random_configuration(original.model, rng);
    const VecX v = mbd_test::random_vector(original.model.nv, rng, 0.8);
    Simulator a(original), b(read);
    const VecX acc_a = a.acceleration(q, v, 0.3), acc_b = b.acceleration(q, v, 0.3);
    CHECK(acc_a.allFinite());
    CHECK(identical(acc_a, acc_b));
}

TEST_CASE("Model files: a four-bar on a spring, saved and loaded, runs exactly as the original", "[io]")
{
    mbd_test::FourBar fb;
    fb.sys.force_elements.push_back(std::make_shared<LinearSpringDamper>(0, fb.coupler, Vec3(0.35, 0.9, 0.0),
                                                                         Vec3(0.4, 0.0, 0.0), 300.0, 2.0, 0.2));
    JointCoordinateForceParams damper;
    damper.damper = Curve::linear(0.5);
    fb.sys.joint_forces.push_back(std::make_shared<JointCoordinateForce>(fb.sys.model, fb.crank, damper));

    const std::filesystem::path path = std::filesystem::temp_directory_path() / "mbd_test_four_bar.json";
    io::save_model(fb.sys, path.string());
    System loaded = io::load_model(path.string());
    std::ifstream in(path, std::ios::binary);
    const std::string on_disk((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();
    std::filesystem::remove(path);
    CHECK(on_disk == io::model_to_json(fb.sys));

    Simulator original(fb.sys), copy(loaded);
    for (Simulator* sim : {&original, &copy}) {
        sim->q = mbd_test::FourBar::configuration(1.2);
        sim->v = 3.0 * mbd_test::FourBar::motion(sim->q);
        sim->initialize();
        sim->run(1.0, 1e-3);
    }
    CHECK(identical(original.q, copy.q));
    CHECK(identical(original.v, copy.v));
    CHECK(original.q(0) != mbd_test::FourBar::configuration(1.2)(0));   // it moved
}

TEST_CASE("Model files: the HMMWV, its tyres left out, survives a round trip", "[io]")
{
    // The car of the benchmark (task 5.4), with joints and with bushings,
    // which task 6.5 plays back from a file. Its tyres are forces written as
    // code: refused by name. Without them every part is in the format, and
    // the car read back writes the same text and has the same accelerations,
    // bit for bit, at its design position moving forward.
    for (const bool bushed : {false, true}) {
        INFO((bushed ? "with bushings" : "with joints"));
        mbd_test::Hmmwv car(bushed);
        try {
            (void)io::model_to_json(car.sys);
            FAIL("the tyres were written");
        } catch (const MbdError& e) {
            CHECK_THAT(std::string(e.what()), StartsWith("MBD-I008: force_elements[") && ContainsSubstring(" tyre')"));
        }
        System drawn = car.sys;
        std::erase_if(drawn.force_elements, [](const std::shared_ptr<ForceElement>& e) {
            return dynamic_cast<const UserForce*>(e.get()) != nullptr;
        });
        CHECK(drawn.force_elements.size() + 4 == car.sys.force_elements.size());
        const std::string text = io::model_to_json(drawn);
        System read = io::model_from_json(text);
        CHECK(io::model_to_json(read) == text);
        CHECK(read.model.nbodies() == 15);

        Simulator a(drawn), b(read);
        car.design(a);
        VecX v = VecX::Zero(a.v.size());
        v(drawn.model.idx_v[static_cast<std::size_t>(car.chassis)] + 3) = mbd_test::Hmmwv::kSpeed;
        CHECK(identical(a.acceleration(a.q, v, 0.0), b.acceleration(a.q, v, 0.0)));
    }
}

TEST_CASE("Model files: a small model's text is the documented format", "[io]")
{
    // The pendulum of docs/model-files.md: one body hinged about Z, its joint
    // frame 1 m above it, gravity along -Y. Short objects and lists on one
    // line, within 100 columns.
    System sys;
    sys.model.gravity = Vec3(0.0, -9.81, 0.0);
    sys.model.add_body(0, std::make_shared<RevoluteJointModel>(), Transform3(),
                       Transform3::FromTranslation(Vec3(0.0, 1.0, 0.0)),
                       RigidBodyInertia(1.0, Vec3::Zero(), Mat3::Zero()), "bob");
    const std::string expected = R"({
  "format": "multibody model",
  "version": 1,
  "units": "SI",
  "gravity": [0.0, -9.81, 0.0],
  "bodies": [
    {
      "name": "bob",
      "parent": 0,
      "joint": "revolute",
      "x_pj": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
      "x_cj": {"quaternion": [1.0, 0.0, 0.0, 0.0], "translation": [0.0, 1.0, 0.0]},
      "inertia": {
        "mass": 1.0,
        "com_B": [0.0, 0.0, 0.0],
        "I_com_B": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
      }
    }
  ],
  "constraints": [],
  "force_elements": [],
  "joint_forces": []
}
)";
    CHECK(io::model_to_json(sys) == expected);

    // Written by hand with the optional keys left out, it is the same model.
    const System brief = io::model_from_json(R"({"format": "multibody model", "version": 1,
        "gravity": [0, -9.81, 0],
        "bodies": [{"name": "bob", "parent": 0, "joint": "revolute",
                    "x_cj": {"translation": [0, 1, 0]}, "inertia": {"mass": 1}}]})");
    CHECK(io::model_to_json(brief) == expected);
}

TEST_CASE("Model files: what a file cannot hold is refused by name", "[io]")
{
    auto refused = [](const System& sys) {
        try {
            (void)io::model_to_json(sys);
        } catch (const MbdError& e) {
            return std::string(e.what());
        }
        return std::string("no error");
    };
    mbd_test::FourBar fb;
    {
        System sys = fb.sys;
        sys.force_elements.push_back(std::make_shared<UserForce>("push", std::vector<BodyIndex>{1},
                                                                 [](const auto&, auto&) {}));
        CHECK_THAT(refused(sys), StartsWith("MBD-I008: force_elements[0]: a force written as code (UserForce 'push')"));
    }
    {
        System sys = fb.sys;
        sys.constraints.push_back(std::make_shared<JointDriver>(
            sys.model, fb.crank,
            TimeFunction([](Real t) { return t; }, [](Real) { return 1.0; }, [](Real) { return 0.0; })));
        CHECK_THAT(refused(sys), StartsWith("MBD-I008: constraints[1]: a time function given as code"));
    }
    {
        // An element the format does not know, written in C++.
        struct Gust final : ForceElement {
            const char* name() const override { return "gust"; }
            std::vector<BodyIndex> bodies() const override { return {1}; }
            void apply(const std::vector<RigidBodyState>&, std::vector<RigidBodyForces>&) const override {}
        };
        System sys = fb.sys;
        sys.force_elements.push_back(std::make_shared<Gust>());
        CHECK_THAT(refused(sys), StartsWith("MBD-I008: force_elements[0]: a force element of the kind 'gust'"));
    }
}

TEST_CASE("Model files: every error in a file names its place", "[io]")
{
    const std::string body = R"({"parent": 0, "joint": "revolute", "inertia": {"mass": 1.0}})";
    REQUIRE(error_of(one_body(body)) == "no error");

    CHECK_THAT(error_of(one_body(R"({"parent": 0, "joint": "revolute", "inertia": {"mass": 1.0}, "nmae": "x"})")),
               StartsWith("MBD-I004: bodies[0]: unknown key 'nmae'; the keys here are name, parent, joint"));
    CHECK_THAT(error_of(one_body(R"({"parent": 0, "inertia": {"mass": 1.0}})")),
               StartsWith("MBD-I005: bodies[0]: the key 'joint' is missing"));
    CHECK_THAT(error_of(one_body(R"({"parent": 0, "joint": "revolute", "inertia": {"mass": "one"}})")),
               StartsWith("MBD-I006: bodies[0].inertia.mass: expected a number, found the text \"one\""));
    CHECK_THAT(error_of(one_body(R"({"parent": 0, "joint": "hinge", "inertia": {"mass": 1.0}})")),
               StartsWith("MBD-I007: bodies[0].joint: unknown joint 'hinge'; the joints are revolute"));
    CHECK_THAT(error_of(one_body(R"({"parent": 0, "joint": "revolute", "inertia": {"mass": 1.0},
                                     "x_pj": {"quaternion": [0.7071, 0.7071, 0, 0]}})")),
               StartsWith("MBD-I006: bodies[0].x_pj.quaternion: a rotation's quaternion must have unit length"));
    CHECK_THAT(error_of(one_body(R"({"parent": 2, "joint": "revolute", "inertia": {"mass": 1.0}})")),
               StartsWith("MBD-K010") && ContainsSubstring("(at bodies[0] in the model file)"));
    CHECK_THAT(error_of(one_body(R"({"parent": 0, "joint": "revolute", "joint": "free", "inertia": {"mass": 1.0}})")),
               StartsWith("MBD-I004: the key 'joint' is given twice in one object"));

    const std::string head = R"({"format": "multibody model", "version": 1, "bodies": [)" + body + "], ";
    CHECK_THAT(error_of(head + R"("constraints": [{"type": "Hinge"}]})"),
               StartsWith("MBD-I007: constraints[0].type: unknown constraint type 'Hinge'"));
    CHECK_THAT(error_of(head + R"("constraints": [{"type": "NoTwist", "i": {"body": 0}, "j": {"body": 5}}]})"),
               StartsWith("MBD-I006: constraints[0].j.body: there is no body 5 (the model has bodies 0, the ground, to 1)"));
    CHECK_THAT(error_of(head + R"("force_elements": [{"type": "SpringDamper", "body1": 0, "body2": 1,
                                  "anchor1": [0, 0, 0], "anchor2": [0, 0, 0], "params": {"free_length": -1}}]})"),
               StartsWith("MBD-F003") && ContainsSubstring("(at force_elements[0] in the model file)"));
    CHECK_THAT(error_of(head + R"("joint_forces": [{"type": "JointCoordinateForce", "body": 1,
                                  "params": {"upper_limit": "infinite"}}]})"),
               StartsWith("MBD-I006: joint_forces[0].params.upper_limit: expected a number (or \"inf\" or \"-inf\")"));

    CHECK_THAT(error_of(R"({"format": "multibody model", "version": 1,)"), StartsWith("MBD-I002: not valid JSON"));
    CHECK_THAT(error_of(R"({"bodies": []})"), StartsWith("MBD-I003: the file: not a model file"));
    CHECK_THAT(error_of(R"({"format": "multibody model", "version": 2})"),
               StartsWith("MBD-I003: version: this program reads version 1; the file has the number 2"));
    CHECK_THAT(error_of(R"({"format": "multibody model", "version": 1, "units": "mm"})"),
               StartsWith("MBD-I003: units: this program reads SI units"));

    // A quaternion typed a little off unit length is normalised.
    const System typed = io::model_from_json(one_body(R"({"parent": 0, "joint": "revolute", "inertia": {"mass": 1.0},
                                                         "x_pj": {"quaternion": [1.000000001, 0, 0, 0]}})"));
    CHECK(typed.model.X_PJ[1].q.w() == 1.0);

    CHECK_THROWS_WITH(io::load_model((std::filesystem::temp_directory_path() / "no_such_dir" / "m.json").string()),
                      StartsWith("MBD-I001: cannot read the model file"));
}
