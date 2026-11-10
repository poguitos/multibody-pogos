// Model files (plan task 6.4, decision D56): a kernel::System as JSON and
// back. See include/mbd/io/model_file.hpp and docs/model-files.md.
//
// Writing goes through nlohmann::ordered_json, so that keys keep the order
// they are written in, and a printer of its own that keeps short objects
// and arrays on one line. Numbers are printed by nlohmann's shortest
// round-trip form, so each reads back as the same double; infinite values,
// which JSON cannot hold, are the strings "inf" and "-inf".
//
// Reading checks every object's keys against those it may have, names the
// place of every error in the file, and builds the system through the same
// constructors as code does. Transforms are set member by member: the
// Transform3 constructor normalises its quaternion, which could move a
// written quaternion's last bit.

#include "mbd/io/model_file.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// No implicit conversions from a JSON value: each is read through get<>(),
// and the helpers below that share a name for writing and reading cannot be
// confused.
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include <nlohmann/json.hpp>

#include "mbd/core/core.hpp"
#include "mbd/forces/bushing.hpp"
#include "mbd/forces/contact.hpp"
#include "mbd/forces/curve.hpp"
#include "mbd/forces/force_element.hpp"
#include "mbd/forces/rotational_spring_damper.hpp"
#include "mbd/forces/spring_damper.hpp"
#include "mbd/forces/user_force.hpp"
#include "mbd/kernel/constraints.hpp"
#include "mbd/kernel/joint_forces.hpp"
#include "mbd/kernel/joint_model.hpp"

namespace mbd::io {
namespace {

using Json = nlohmann::ordered_json;
using namespace mbd::kernel;

const char* const kFormat = "multibody model";
constexpr int kVersion = 1;
constexpr std::size_t kWidth = 100;   // the printer's line length

// --- What the format knows --------------------------------------------------

/// A joint model of the tree, by the name it gives itself; null if unknown.
std::shared_ptr<const JointModel> make_joint(const std::string& name)
{
    if (name == "revolute") return std::make_shared<RevoluteJointModel>();
    if (name == "prismatic") return std::make_shared<PrismaticJointModel>();
    if (name == "fixed") return std::make_shared<FixedJointModel>();
    if (name == "universal") return std::make_shared<UniversalJointModel>();
    if (name == "cylindrical") return std::make_shared<CylindricalJointModel>();
    if (name == "planar") return std::make_shared<PlanarJointModel>();
    if (name == "spherical") return std::make_shared<SphericalJointModel>();
    if (name == "free") return std::make_shared<FreeJointModel>();
    return nullptr;
}

const char* const kJoints = "revolute, prismatic, fixed, universal, cylindrical, planar, spherical, free";

/// The closures, written by the name of the function that makes them.
struct ClosureKind {
    const char* type;       // in the file
    const char* set_name;   // ConstraintSet::name() of what the function makes
    std::shared_ptr<ConstraintSet> (*make)(Marker, Marker);
};

const ClosureKind kClosures[] = {
    {"point_on_line", "point on line", &point_on_line},
    {"point_on_plane", "point on plane", &point_on_plane},
    {"spherical_closure", "spherical closure", &spherical_closure},
    {"revolute_closure", "revolute closure", &revolute_closure},
    {"universal_closure", "universal closure", &universal_closure},
    {"cylindrical_closure", "cylindrical closure", &cylindrical_closure},
    {"prismatic_closure", "prismatic closure", &prismatic_closure},
    {"constant_velocity_closure", "constant-velocity closure", &constant_velocity_closure},
};

const char* const kConstraints =
    "PointCoincidence, Dot1, Dot2, Distance, NoTwist, JointDriver, point_on_line, point_on_plane, "
    "spherical_closure, revolute_closure, universal_closure, cylindrical_closure, prismatic_closure, "
    "constant_velocity_closure, ConstraintSet";
const char* const kForceElements = "LinearSpringDamper, SpringDamper, RotationalSpringDamper, Bushing, PlaneContact";
const char* const kJointForces = "JointCoordinateForce";

// --- Writing ----------------------------------------------------------------

[[noreturn]] void cannot_write(const std::string& where, const std::string& what)
{
    throw MbdError("MBD-I008: " + where + ": " + what + " cannot be written to a model file");
}

Json real(Real x, const std::string& where)
{
    if (std::isnan(x)) cannot_write(where, "a value that is not a number");
    if (std::isinf(x)) return Json(x > 0.0 ? "inf" : "-inf");
    return Json(x);
}

Json reals(const std::vector<Real>& x, const std::string& where)
{
    Json a = Json::array();
    for (const Real v : x) a.push_back(real(v, where));
    return a;
}

Json vec(const Vec3& v, const std::string& where)
{
    return Json::array({real(v.x(), where), real(v.y(), where), real(v.z(), where)});
}

Json mat(const Mat3& M, const std::string& where)
{
    Json rows = Json::array();
    for (Index r = 0; r < 3; ++r) rows.push_back(vec(Vec3(M.row(r).transpose()), where));
    return rows;
}

Json transform(const Transform3& X, const std::string& where)
{
    Json j;
    j["quaternion"] = Json::array({real(X.q.w(), where), real(X.q.x(), where), real(X.q.y(), where),
                                   real(X.q.z(), where)});
    j["translation"] = vec(X.p, where);
    return j;
}

Json marker(const Marker& m, const std::string& where)
{
    Json j;
    j["body"] = m.body;
    j["X_BM"] = transform(m.X_BM, where);
    return j;
}

Json curve(const Curve& c, const std::string& where)
{
    Json j;
    if (c.is_table()) {
        j["x"] = reals(c.table_x(), where);
        j["y"] = reals(c.table_y(), where);
    } else {
        j["slope"] = real(c.line_slope(), where);
        j["offset"] = real(c.line_offset(), where);
    }
    return j;
}

Json curves(const std::array<Curve, 3>& c, const std::string& where)
{
    return Json::array({curve(c[0], where), curve(c[1], where), curve(c[2], where)});
}

Json constant(const TimeFunction& s, const std::string& where)
{
    if (!s.is_constant()) cannot_write(where, "a time function given as code (only a constant can be)");
    return real(s.constant_value(), where);
}

Json params(const SpringDamperParams& p, const std::string& where)
{
    Json j;
    j["free_length"] = real(p.free_length, where);
    j["spring"] = curve(p.spring, where);
    j["preload"] = real(p.preload, where);
    j["damper"] = curve(p.damper, where);
    j["bump_clearance"] = real(p.bump_clearance, where);
    j["bump_stop"] = curve(p.bump_stop, where);
    j["rebound_clearance"] = real(p.rebound_clearance, where);
    j["rebound_stop"] = curve(p.rebound_stop, where);
    return j;
}

Json params(const RotationalSpringDamperParams& p, const std::string& where)
{
    Json j;
    j["reference"] = real(p.reference, where);
    j["spring"] = curve(p.spring, where);
    j["preload"] = real(p.preload, where);
    j["damper"] = curve(p.damper, where);
    return j;
}

Json params(const BushingParams& p, const std::string& where)
{
    Json j;
    j["rotational_stiffness"] = curves(p.rotational_stiffness, where);
    j["stiffness"] = curves(p.stiffness, where);
    j["rotational_damping"] = curves(p.rotational_damping, where);
    j["damping"] = curves(p.damping, where);
    return j;
}

Json params(const ContactParams& p, const std::string& where)
{
    Json j;
    j["stiffness"] = real(p.stiffness, where);
    j["exponent"] = real(p.exponent, where);
    j["damping"] = real(p.damping, where);
    j["damping_depth"] = real(p.damping_depth, where);
    j["mu_static"] = real(p.mu_static, where);
    j["mu_dynamic"] = real(p.mu_dynamic, where);
    j["v_static"] = real(p.v_static, where);
    j["v_dynamic"] = real(p.v_dynamic, where);
    return j;
}

Json params(const JointCoordinateForceParams& p, const std::string& where)
{
    Json j;
    j["reference"] = real(p.reference, where);
    j["spring"] = curve(p.spring, where);
    j["preload"] = real(p.preload, where);
    j["damper"] = curve(p.damper, where);
    j["friction"] = real(p.friction, where);
    j["friction_velocity"] = real(p.friction_velocity, where);
    j["lower_limit"] = real(p.lower_limit, where);
    j["upper_limit"] = real(p.upper_limit, where);
    j["limit_stiffness"] = real(p.limit_stiffness, where);
    j["limit_damping"] = real(p.limit_damping, where);
    return j;
}

Json constraint(const ConstraintModel& c, const std::string& where);

Json parts(const ConstraintSet& set, const std::string& where)
{
    Json a = Json::array();
    for (std::size_t k = 0; k < set.parts().size(); ++k) {
        a.push_back(constraint(*set.parts()[k], where + ".parts[" + std::to_string(k) + "]"));
    }
    return a;
}

/// The markers of the first primitive of a closure: those it was made from.
bool first_markers(const ConstraintModel& c, Marker& i, Marker& j)
{
    if (const auto* set = dynamic_cast<const ConstraintSet*>(&c)) {
        return !set->parts().empty() && first_markers(*set->parts().front(), i, j);
    }
    if (const auto* p = dynamic_cast<const PointCoincidence*>(&c)) {
        i = p->i();
        j = p->j();
        return true;
    }
    if (const auto* p = dynamic_cast<const Dot2*>(&c)) {
        i = p->i();
        j = p->j();
        return true;
    }
    return false;
}

Json constraint(const ConstraintModel& c, const std::string& where)
{
    Json j;
    if (const auto* p = dynamic_cast<const PointCoincidence*>(&c)) {
        j["type"] = "PointCoincidence";
        j["i"] = marker(p->i(), where);
        j["j"] = marker(p->j(), where);
    } else if (const auto* d1 = dynamic_cast<const Dot1*>(&c)) {
        j["type"] = "Dot1";
        j["i"] = marker(d1->i(), where);
        j["axis_i"] = d1->axis_i();
        j["j"] = marker(d1->j(), where);
        j["axis_j"] = d1->axis_j();
        j["s"] = constant(d1->s(), where);
    } else if (const auto* d2 = dynamic_cast<const Dot2*>(&c)) {
        j["type"] = "Dot2";
        j["i"] = marker(d2->i(), where);
        j["axis_i"] = d2->axis_i();
        j["j"] = marker(d2->j(), where);
        j["s"] = constant(d2->s(), where);
    } else if (const auto* d = dynamic_cast<const Distance*>(&c)) {
        j["type"] = "Distance";
        j["i"] = marker(d->i(), where);
        j["j"] = marker(d->j(), where);
        j["length"] = constant(d->length(), where);
        j["nominal_length"] = real(d->nominal_length(), where);
    } else if (const auto* n = dynamic_cast<const NoTwist*>(&c)) {
        j["type"] = "NoTwist";
        j["i"] = marker(n->i(), where);
        j["j"] = marker(n->j(), where);
    } else if (const auto* drive = dynamic_cast<const JointDriver*>(&c)) {
        j["type"] = "JointDriver";
        j["body"] = drive->body();
        j["s"] = constant(drive->s(), where);
    } else if (const auto* set = dynamic_cast<const ConstraintSet*>(&c)) {
        // A closure is written as the call that makes it, provided that call
        // makes exactly this set (compared as written, signs of zero
        // included); any other set part by part.
        const std::string written = parts(*set, where).dump();
        for (const ClosureKind& kind : kClosures) {
            Marker mi, mj;
            if (std::string(set->name()) == kind.set_name && first_markers(*set, mi, mj)
                && parts(*kind.make(mi, mj), where).dump() == written) {
                j["type"] = kind.type;
                j["i"] = marker(mi, where);
                j["j"] = marker(mj, where);
                return j;
            }
        }
        j["type"] = "ConstraintSet";
        j["name"] = set->name();
        j["parts"] = parts(*set, where);
    } else {
        cannot_write(where, std::string("a constraint of the kind '") + c.name() + "'");
    }
    return j;
}

Json force_element(const ForceElement& e, const std::string& where)
{
    Json j;
    const std::vector<BodyIndex> b = e.bodies();
    if (const auto* s = dynamic_cast<const LinearSpringDamper*>(&e)) {
        j["type"] = "LinearSpringDamper";
        j["body1"] = s->body1_idx;
        j["body2"] = s->body2_idx;
        j["anchor1"] = vec(s->anchor1_B, where);
        j["anchor2"] = vec(s->anchor2_B, where);
        j["stiffness"] = real(s->k, where);
        j["damping"] = real(s->c, where);
        j["rest_length"] = real(s->rest_length, where);
    } else if (const auto* sd = dynamic_cast<const SpringDamper*>(&e)) {
        j["type"] = "SpringDamper";
        j["body1"] = b[0];
        j["body2"] = b[1];
        j["anchor1"] = vec(sd->anchor1(), where);
        j["anchor2"] = vec(sd->anchor2(), where);
        j["params"] = params(sd->params(), where);
    } else if (const auto* r = dynamic_cast<const RotationalSpringDamper*>(&e)) {
        j["type"] = "RotationalSpringDamper";
        j["body1"] = b[0];
        j["X_1M"] = transform(r->X_1M(), where);
        j["body2"] = b[1];
        j["X_2M"] = transform(r->X_2M(), where);
        j["params"] = params(r->params(), where);
    } else if (const auto* bu = dynamic_cast<const Bushing*>(&e)) {
        j["type"] = "Bushing";
        j["body1"] = b[0];
        j["X_1M"] = transform(bu->X_1M(), where);
        j["body2"] = b[1];
        j["X_2M"] = transform(bu->X_2M(), where);
        j["params"] = params(bu->params(), where);
    } else if (const auto* pc = dynamic_cast<const PlaneContact*>(&e)) {
        j["type"] = "PlaneContact";
        j["body"] = b[0];
        Json centres = Json::array();
        for (const Vec3& c : pc->centres()) centres.push_back(vec(c, where));
        j["centres"] = centres;
        j["radius"] = real(pc->radius(), where);
        j["plane_body"] = b[1];
        j["plane_X"] = transform(pc->plane_frame(), where);
        j["params"] = params(pc->params(), where);
    } else if (dynamic_cast<const UserForce*>(&e)) {
        cannot_write(where, std::string("a force written as code (UserForce '") + e.name() + "')");
    } else {
        cannot_write(where, std::string("a force element of the kind '") + e.name() + "'");
    }
    return j;
}

Json joint_force(const JointForce& f, const std::string& where)
{
    Json j;
    if (const auto* c = dynamic_cast<const JointCoordinateForce*>(&f)) {
        j["type"] = "JointCoordinateForce";
        j["body"] = c->body();
        j["params"] = params(c->params(), where);
    } else {
        cannot_write(where, std::string("a joint force of the kind '") + f.name() + "'");
    }
    return j;
}

Json body(const Model& model, int b)
{
    const auto i = static_cast<std::size_t>(b);
    const std::string where = "bodies[" + std::to_string(b - 1) + "]";
    const std::string joint = model.joint[i]->name();
    if (!make_joint(joint)) cannot_write(where, "a joint of the kind '" + joint + "'");
    const RigidBodyInertia& I = model.inertia[i];
    Json inertia;
    inertia["mass"] = real(I.mass, where);
    inertia["com_B"] = vec(I.com_B, where);
    inertia["I_com_B"] = mat(I.I_com_B, where);
    Json j;
    j["name"] = model.name[i];
    j["parent"] = model.parent[i];
    j["joint"] = joint;
    j["x_pj"] = transform(model.X_PJ[i], where);
    j["x_cj"] = transform(model.X_CJ[i], where);
    j["inertia"] = inertia;
    return j;
}

/// j on one line, with a space after each comma and colon.
std::string one_line(const Json& j)
{
    if (!j.is_structured()) return j.dump();
    std::string s = j.is_object() ? "{" : "[";
    bool first = true;
    for (auto it = j.begin(); it != j.end(); ++it) {
        if (!first) s += ", ";
        first = false;
        if (j.is_object()) s += Json(it.key()).dump() + ": ";
        s += one_line(it.value());
    }
    return s + (j.is_object() ? "}" : "]");
}

/// j as text: on one line where it fits within kWidth columns (starting at
/// `column`), otherwise one entry per line, indented.
void print(const Json& j, std::size_t indent, std::size_t column, std::string& out)
{
    const std::string line = one_line(j);
    if (!j.is_structured() || j.empty() || column + line.size() <= kWidth) {
        out += line;
        return;
    }
    const std::string pad(indent + 2, ' ');
    out += j.is_object() ? "{\n" : "[\n";
    std::size_t k = 0;
    for (auto it = j.begin(); it != j.end(); ++it, ++k) {
        out += pad;
        std::size_t at = pad.size();
        if (j.is_object()) {
            const std::string key = Json(it.key()).dump() + ": ";
            out += key;
            at += key.size();
        }
        print(it.value(), indent + 2, at, out);
        out += k + 1 < j.size() ? ",\n" : "\n";
    }
    out += std::string(indent, ' ') + (j.is_object() ? "}" : "]");
}

// --- Reading ----------------------------------------------------------------

[[noreturn]] void fail(const char* code, const std::string& where, const std::string& what)
{
    throw MbdError(std::string(code) + ": " + (where.empty() ? "the file" : where) + ": " + what);
}

std::string kind_of(const Json& j)
{
    if (j.is_object()) return "an object";
    if (j.is_array()) return "a list";
    if (j.is_string()) return "the text " + j.dump();
    if (j.is_boolean()) return j.get<bool>() ? "true" : "false";
    if (j.is_null()) return "null";
    return "the number " + j.dump();
}

std::string place(const std::string& where, const std::string& key)
{
    return where.empty() ? key : where + "." + key;
}

std::string item(const std::string& where, std::size_t k)
{
    return where + "[" + std::to_string(k) + "]";
}

/// An object of the file and the keys it may have: any other key is an
/// error (MBD-I004), and a key it must have and has not is one too (I005).
class Fields {
public:
    Fields(const Json& j, std::string where, std::initializer_list<const char*> keys) : j_(j), where_(std::move(where))
    {
        if (!j.is_object()) fail("MBD-I006", where_, "expected an object, found " + kind_of(j));
        for (auto it = j.begin(); it != j.end(); ++it) {
            bool known = false;
            for (const char* key : keys) known = known || it.key() == key;
            if (!known) {
                std::string list;
                for (const char* key : keys) list += (list.empty() ? "" : ", ") + std::string(key);
                fail("MBD-I004", where_, "unknown key '" + it.key() + "'; the keys here are " + list);
            }
        }
    }

    bool has(const char* key) const { return j_.contains(key); }

    const Json& operator[](const char* key) const
    {
        if (!has(key)) fail("MBD-I005", where_, "the key '" + std::string(key) + "' is missing");
        return j_.at(key);
    }

    std::string at(const char* key) const { return place(where_, key); }
    const std::string& where() const { return where_; }

private:
    const Json& j_;
    std::string where_;
};

Real real(const Json& j, const std::string& where, bool infinite_allowed = false)
{
    if (j.is_number()) {
        const Real x = j.get<Real>();
        if (!std::isfinite(x)) fail("MBD-I006", where, "the number " + j.dump() + " is out of range");
        return x;
    }
    if (infinite_allowed && j.is_string()) {
        if (j.get<std::string>() == "inf") return std::numeric_limits<Real>::infinity();
        if (j.get<std::string>() == "-inf") return -std::numeric_limits<Real>::infinity();
    }
    fail("MBD-I006", where,
         std::string("expected a number") + (infinite_allowed ? " (or \"inf\" or \"-inf\")" : "") + ", found "
             + kind_of(j));
}

int integer(const Json& j, const std::string& where)
{
    if (j.is_number_integer()) {
        const long long n = j.get<long long>();
        if (n >= std::numeric_limits<int>::min() && n <= std::numeric_limits<int>::max()) return static_cast<int>(n);
    }
    fail("MBD-I006", where, "expected a whole number, found " + kind_of(j));
}

std::string text(const Json& j, const std::string& where)
{
    if (!j.is_string()) fail("MBD-I006", where, "expected a text in quotes, found " + kind_of(j));
    return j.get<std::string>();
}

const Json& list(const Json& j, const std::string& where)
{
    if (!j.is_array()) fail("MBD-I006", where, "expected a list, found " + kind_of(j));
    return j;
}

std::vector<Real> reals(const Json& j, const std::string& where)
{
    std::vector<Real> x;
    for (std::size_t k = 0; k < list(j, where).size(); ++k) x.push_back(real(j[k], item(where, k)));
    return x;
}

Vec3 vec(const Json& j, const std::string& where)
{
    if (!j.is_array() || j.size() != 3) fail("MBD-I006", where, "expected 3 numbers, found " + kind_of(j));
    return Vec3(real(j[0], item(where, 0)), real(j[1], item(where, 1)), real(j[2], item(where, 2)));
}

Mat3 mat(const Json& j, const std::string& where)
{
    if (!j.is_array() || j.size() != 3) fail("MBD-I006", where, "expected 3 rows of 3 numbers, found " + kind_of(j));
    Mat3 M;
    for (std::size_t r = 0; r < 3; ++r) M.row(static_cast<Index>(r)) = vec(j[r], item(where, r)).transpose();
    return M;
}

/// [w, x, y, z]: of unit length to 1e-6; normalised if typed further than
/// 1e-12 from it, kept as written otherwise, so that a written one reads
/// back unchanged.
Quat quaternion(const Json& j, const std::string& where)
{
    if (!j.is_array() || j.size() != 4) fail("MBD-I006", where, "expected 4 numbers [w, x, y, z], found " + kind_of(j));
    Quat q(real(j[0], item(where, 0)), real(j[1], item(where, 1)), real(j[2], item(where, 2)),
           real(j[3], item(where, 3)));
    const Real n = q.norm();
    if (!(std::abs(n - 1.0) <= 1e-6)) {
        std::ostringstream s;
        s << "a rotation's quaternion must have unit length; its length is " << n;
        fail("MBD-I006", where, s.str());
    }
    if (std::abs(n - 1.0) > 1e-12) q.normalize();
    return q;
}

Transform3 transform(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"quaternion", "translation"});
    Transform3 X;
    if (f.has("quaternion")) X.q = quaternion(f["quaternion"], f.at("quaternion"));
    if (f.has("translation")) X.p = vec(f["translation"], f.at("translation"));
    return X;
}

int body_index(const Json& j, const std::string& where, const Model& model)
{
    const int b = integer(j, where);
    if (b < 0 || b >= model.nbodies()) {
        fail("MBD-I006", where,
             "there is no body " + std::to_string(b) + " (the model has bodies 0, the ground, to "
                 + std::to_string(model.nbodies() - 1) + ")");
    }
    return b;
}

Marker marker(const Json& j, const std::string& where, const Model& model)
{
    const Fields f(j, where, {"body", "X_BM"});
    Marker m;
    m.body = body_index(f["body"], f.at("body"), model);
    if (f.has("X_BM")) m.X_BM = transform(f["X_BM"], f.at("X_BM"));
    return m;
}

/// The constructors' own checks, with the place in the file added.
template <class Make>
auto made(const std::string& where, Make&& make) -> decltype(make())
{
    try {
        return make();
    } catch (const MbdError& e) {
        throw MbdError(std::string(e.what()) + " (at " + where + " in the model file)");
    }
}

Curve curve(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"slope", "offset", "x", "y"});
    if (f.has("x") || f.has("y")) {
        if (f.has("slope") || f.has("offset")) fail("MBD-I006", where, "a curve is either a table (x, y) or a line (slope, offset)");
        const std::vector<Real> x = reals(f["x"], f.at("x")), y = reals(f["y"], f.at("y"));
        return made(where, [&] { return Curve::table(x, y); });
    }
    return Curve::linear(f.has("slope") ? real(f["slope"], f.at("slope")) : 0.0,
                         f.has("offset") ? real(f["offset"], f.at("offset")) : 0.0);
}

std::array<Curve, 3> curves(const Json& j, const std::string& where)
{
    if (!j.is_array() || j.size() != 3) fail("MBD-I006", where, "expected 3 curves, found " + kind_of(j));
    return {curve(j[0], item(where, 0)), curve(j[1], item(where, 1)), curve(j[2], item(where, 2))};
}

TimeFunction constant(const Fields& f, const char* key)
{
    return f.has(key) ? TimeFunction::constant(real(f[key], f.at(key))) : TimeFunction();
}

// Each setting optional, its default the C++ one.
#define MBD_REAL(p, f, key) if (f.has(#key)) p.key = real(f[#key], f.at(#key))
#define MBD_REAL_INF(p, f, key) if (f.has(#key)) p.key = real(f[#key], f.at(#key), true)
#define MBD_CURVE(p, f, key) if (f.has(#key)) p.key = curve(f[#key], f.at(#key))
#define MBD_CURVES(p, f, key) if (f.has(#key)) p.key = curves(f[#key], f.at(#key))

SpringDamperParams spring_damper_params(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"free_length", "spring", "preload", "damper", "bump_clearance", "bump_stop",
                              "rebound_clearance", "rebound_stop"});
    SpringDamperParams p;
    MBD_REAL(p, f, free_length);
    MBD_CURVE(p, f, spring);
    MBD_REAL(p, f, preload);
    MBD_CURVE(p, f, damper);
    MBD_REAL_INF(p, f, bump_clearance);
    MBD_CURVE(p, f, bump_stop);
    MBD_REAL_INF(p, f, rebound_clearance);
    MBD_CURVE(p, f, rebound_stop);
    return p;
}

RotationalSpringDamperParams rotational_params(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"reference", "spring", "preload", "damper"});
    RotationalSpringDamperParams p;
    MBD_REAL(p, f, reference);
    MBD_CURVE(p, f, spring);
    MBD_REAL(p, f, preload);
    MBD_CURVE(p, f, damper);
    return p;
}

BushingParams bushing_params(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"rotational_stiffness", "stiffness", "rotational_damping", "damping"});
    BushingParams p;
    MBD_CURVES(p, f, rotational_stiffness);
    MBD_CURVES(p, f, stiffness);
    MBD_CURVES(p, f, rotational_damping);
    MBD_CURVES(p, f, damping);
    return p;
}

ContactParams contact_params(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"stiffness", "exponent", "damping", "damping_depth", "mu_static", "mu_dynamic",
                              "v_static", "v_dynamic"});
    ContactParams p;
    MBD_REAL(p, f, stiffness);
    MBD_REAL(p, f, exponent);
    MBD_REAL(p, f, damping);
    MBD_REAL(p, f, damping_depth);
    MBD_REAL(p, f, mu_static);
    MBD_REAL(p, f, mu_dynamic);
    MBD_REAL(p, f, v_static);
    MBD_REAL(p, f, v_dynamic);
    return p;
}

JointCoordinateForceParams joint_force_params(const Json& j, const std::string& where)
{
    const Fields f(j, where, {"reference", "spring", "preload", "damper", "friction", "friction_velocity",
                              "lower_limit", "upper_limit", "limit_stiffness", "limit_damping"});
    JointCoordinateForceParams p;
    MBD_REAL(p, f, reference);
    MBD_CURVE(p, f, spring);
    MBD_REAL(p, f, preload);
    MBD_CURVE(p, f, damper);
    MBD_REAL(p, f, friction);
    MBD_REAL(p, f, friction_velocity);
    MBD_REAL_INF(p, f, lower_limit);
    MBD_REAL_INF(p, f, upper_limit);
    MBD_REAL(p, f, limit_stiffness);
    MBD_REAL(p, f, limit_damping);
    return p;
}

#undef MBD_REAL
#undef MBD_REAL_INF
#undef MBD_CURVE
#undef MBD_CURVES

/// The "type" of an entry of a list, before its other keys are read.
std::string type_of(const Json& j, const std::string& where)
{
    if (!j.is_object()) fail("MBD-I006", where, "expected an object, found " + kind_of(j));
    if (!j.contains("type")) fail("MBD-I005", where, "the key 'type' is missing");
    return text(j.at("type"), place(where, "type"));
}

std::shared_ptr<const ConstraintModel> read_constraint(const Json& j, const std::string& where, const Model& model)
{
    const std::string type = type_of(j, where);
    if (type == "PointCoincidence") {
        const Fields f(j, where, {"type", "i", "j"});
        return std::make_shared<PointCoincidence>(marker(f["i"], f.at("i"), model), marker(f["j"], f.at("j"), model));
    }
    if (type == "Dot1") {
        const Fields f(j, where, {"type", "i", "axis_i", "j", "axis_j", "s"});
        const Marker i = marker(f["i"], f.at("i"), model), mj = marker(f["j"], f.at("j"), model);
        const int ai = integer(f["axis_i"], f.at("axis_i")), aj = integer(f["axis_j"], f.at("axis_j"));
        const TimeFunction s = constant(f, "s");
        return made(where, [&] { return std::make_shared<Dot1>(i, ai, mj, aj, s); });
    }
    if (type == "Dot2") {
        const Fields f(j, where, {"type", "i", "axis_i", "j", "s"});
        const Marker i = marker(f["i"], f.at("i"), model), mj = marker(f["j"], f.at("j"), model);
        const int ai = integer(f["axis_i"], f.at("axis_i"));
        const TimeFunction s = constant(f, "s");
        return made(where, [&] { return std::make_shared<Dot2>(i, ai, mj, s); });
    }
    if (type == "Distance") {
        // A constant length L is Distance(i, j, L): the time function L and
        // the nominal length L, unless one is given.
        const Fields f(j, where, {"type", "i", "j", "length", "nominal_length"});
        const Marker i = marker(f["i"], f.at("i"), model), mj = marker(f["j"], f.at("j"), model);
        const Real L = real(f["length"], f.at("length"));
        const Real L0 = f.has("nominal_length") ? real(f["nominal_length"], f.at("nominal_length")) : L;
        return made(where, [&] { return std::make_shared<Distance>(i, mj, TimeFunction::constant(L), L0); });
    }
    if (type == "NoTwist") {
        const Fields f(j, where, {"type", "i", "j"});
        return std::make_shared<NoTwist>(marker(f["i"], f.at("i"), model), marker(f["j"], f.at("j"), model));
    }
    if (type == "JointDriver") {
        const Fields f(j, where, {"type", "body", "s"});
        const int b = body_index(f["body"], f.at("body"), model);
        const TimeFunction s = constant(f, "s");
        return made(where, [&] { return std::make_shared<JointDriver>(model, b, s); });
    }
    for (const ClosureKind& kind : kClosures) {
        if (type == kind.type) {
            const Fields f(j, where, {"type", "i", "j"});
            return kind.make(marker(f["i"], f.at("i"), model), marker(f["j"], f.at("j"), model));
        }
    }
    if (type == "ConstraintSet") {
        const Fields f(j, where, {"type", "name", "parts"});
        auto set = std::make_shared<ConstraintSet>(text(f["name"], f.at("name")));
        const Json& parts_list = list(f["parts"], f.at("parts"));
        for (std::size_t k = 0; k < parts_list.size(); ++k) {
            set->add(read_constraint(parts_list[k], item(f.at("parts"), k), model));
        }
        return set;
    }
    fail("MBD-I007", place(where, "type"), "unknown constraint type '" + type + "'; the types are " + kConstraints);
}

std::shared_ptr<ForceElement> read_force_element(const Json& j, const std::string& where, const Model& model)
{
    const std::string type = type_of(j, where);
    if (type == "LinearSpringDamper") {
        const Fields f(j, where, {"type", "body1", "body2", "anchor1", "anchor2", "stiffness", "damping", "rest_length"});
        const int b1 = body_index(f["body1"], f.at("body1"), model), b2 = body_index(f["body2"], f.at("body2"), model);
        const Vec3 a1 = vec(f["anchor1"], f.at("anchor1")), a2 = vec(f["anchor2"], f.at("anchor2"));
        const Real k = real(f["stiffness"], f.at("stiffness"));
        const Real c = f.has("damping") ? real(f["damping"], f.at("damping")) : 0.0;
        const Real L = real(f["rest_length"], f.at("rest_length"));
        return made(where, [&] { return std::make_shared<LinearSpringDamper>(b1, b2, a1, a2, k, c, L); });
    }
    if (type == "SpringDamper") {
        const Fields f(j, where, {"type", "body1", "body2", "anchor1", "anchor2", "params"});
        const int b1 = body_index(f["body1"], f.at("body1"), model), b2 = body_index(f["body2"], f.at("body2"), model);
        const Vec3 a1 = vec(f["anchor1"], f.at("anchor1")), a2 = vec(f["anchor2"], f.at("anchor2"));
        const SpringDamperParams p = spring_damper_params(f["params"], f.at("params"));
        return made(where, [&] { return std::make_shared<SpringDamper>(b1, b2, a1, a2, p); });
    }
    if (type == "RotationalSpringDamper" || type == "Bushing") {
        const Fields f(j, where, {"type", "body1", "X_1M", "body2", "X_2M", "params"});
        const int b1 = body_index(f["body1"], f.at("body1"), model), b2 = body_index(f["body2"], f.at("body2"), model);
        const Transform3 X1 = f.has("X_1M") ? transform(f["X_1M"], f.at("X_1M")) : Transform3();
        const Transform3 X2 = f.has("X_2M") ? transform(f["X_2M"], f.at("X_2M")) : Transform3();
        if (type == "Bushing") {
            const BushingParams p = bushing_params(f["params"], f.at("params"));
            return made(where, [&] { return std::make_shared<Bushing>(b1, X1, b2, X2, p); });
        }
        const RotationalSpringDamperParams p = rotational_params(f["params"], f.at("params"));
        return made(where, [&] { return std::make_shared<RotationalSpringDamper>(b1, X1, b2, X2, p); });
    }
    if (type == "PlaneContact") {
        const Fields f(j, where, {"type", "body", "centres", "radius", "plane_body", "plane_X", "params"});
        const int b = body_index(f["body"], f.at("body"), model);
        std::vector<Vec3> centres;
        const Json& c = list(f["centres"], f.at("centres"));
        for (std::size_t k = 0; k < c.size(); ++k) centres.push_back(vec(c[k], item(f.at("centres"), k)));
        const Real r = f.has("radius") ? real(f["radius"], f.at("radius")) : 0.0;
        const int pb = body_index(f["plane_body"], f.at("plane_body"), model);
        const Transform3 X = f.has("plane_X") ? transform(f["plane_X"], f.at("plane_X")) : Transform3();
        const ContactParams p = contact_params(f["params"], f.at("params"));
        return made(where, [&] { return std::make_shared<PlaneContact>(b, centres, r, pb, X, p); });
    }
    fail("MBD-I007", place(where, "type"), "unknown force element type '" + type + "'; the types are " + kForceElements);
}

std::shared_ptr<const JointForce> read_joint_force(const Json& j, const std::string& where, const Model& model)
{
    const std::string type = type_of(j, where);
    if (type == "JointCoordinateForce") {
        const Fields f(j, where, {"type", "body", "params"});
        const int b = body_index(f["body"], f.at("body"), model);
        const JointCoordinateForceParams p = joint_force_params(f["params"], f.at("params"));
        return made(where, [&] { return std::make_shared<JointCoordinateForce>(model, b, p); });
    }
    fail("MBD-I007", place(where, "type"), "unknown joint force type '" + type + "'; the types are " + kJointForces);
}

void read_body(const Json& j, const std::string& where, Model& model)
{
    const Fields f(j, where, {"name", "parent", "joint", "x_pj", "x_cj", "inertia"});
    const std::string name = f.has("name") ? text(f["name"], f.at("name")) : std::string();
    const int parent = integer(f["parent"], f.at("parent"));
    const std::string kind = text(f["joint"], f.at("joint"));
    const auto joint = make_joint(kind);
    if (!joint) fail("MBD-I007", f.at("joint"), "unknown joint '" + kind + "'; the joints are " + kJoints);
    const Transform3 x_pj = f.has("x_pj") ? transform(f["x_pj"], f.at("x_pj")) : Transform3();
    const Transform3 x_cj = f.has("x_cj") ? transform(f["x_cj"], f.at("x_cj")) : Transform3();
    const Fields fi(f["inertia"], f.at("inertia"), {"mass", "com_B", "I_com_B"});
    RigidBodyInertia inertia;
    inertia.mass = real(fi["mass"], fi.at("mass"));
    if (fi.has("com_B")) inertia.com_B = vec(fi["com_B"], fi.at("com_B"));
    if (fi.has("I_com_B")) inertia.I_com_B = mat(fi["I_com_B"], fi.at("I_com_B"));
    made(where, [&] { return model.add_body(parent, joint, x_pj, x_cj, inertia, name); });
}

/// Parsed text, each object's keys checked for one given twice (which the
/// parser would otherwise keep one of without a word).
Json parse(const std::string& text)
{
    std::vector<std::set<std::string>> keys;
    const Json::parser_callback_t check = [&keys](int, Json::parse_event_t event, Json& parsed) {
        if (event == Json::parse_event_t::object_start) keys.emplace_back();
        if (event == Json::parse_event_t::object_end) keys.pop_back();
        if (event == Json::parse_event_t::key && !keys.back().insert(parsed.get<std::string>()).second) {
            throw MbdError("MBD-I004: the key '" + parsed.get<std::string>() + "' is given twice in one object");
        }
        return true;
    };
    try {
        return Json::parse(text, check);
    } catch (const Json::parse_error& e) {
        throw MbdError(std::string("MBD-I002: not valid JSON: ") + e.what());
    }
}

} // namespace

std::string model_to_json(const System& system)
{
    const Model& model = system.model;
    Json j;
    j["format"] = kFormat;
    j["version"] = kVersion;
    j["units"] = "SI";
    j["gravity"] = vec(model.gravity, "gravity");
    Json bodies = Json::array();
    for (int b = 1; b < model.nbodies(); ++b) bodies.push_back(body(model, b));
    j["bodies"] = bodies;
    Json constraints = Json::array();
    for (std::size_t k = 0; k < system.constraints.size(); ++k) {
        constraints.push_back(constraint(*system.constraints[k], item("constraints", k)));
    }
    j["constraints"] = constraints;
    Json elements = Json::array();
    for (std::size_t k = 0; k < system.force_elements.size(); ++k) {
        elements.push_back(force_element(*system.force_elements[k], item("force_elements", k)));
    }
    j["force_elements"] = elements;
    Json joint_forces = Json::array();
    for (std::size_t k = 0; k < system.joint_forces.size(); ++k) {
        joint_forces.push_back(joint_force(*system.joint_forces[k], item("joint_forces", k)));
    }
    j["joint_forces"] = joint_forces;
    std::string out;
    print(j, 0, 0, out);
    return out + "\n";
}

System model_from_json(const std::string& json_text)
{
    const Json j = parse(json_text);
    const Fields f(j, "", {"format", "version", "units", "gravity", "bodies", "constraints", "force_elements",
                           "joint_forces"});
    if (!f.has("format") || !f["format"].is_string() || f["format"].get<std::string>() != kFormat) {
        fail("MBD-I003", "", std::string("not a model file: it must begin with \"format\": \"") + kFormat + "\"");
    }
    if (!f.has("version") || !f["version"].is_number_integer() || f["version"].get<long long>() != kVersion) {
        fail("MBD-I003", "version",
             std::string("this program reads version ") + std::to_string(kVersion) + "; the file has "
                 + (f.has("version") ? kind_of(f["version"]) : std::string("none")));
    }
    if (f.has("units") && text(f["units"], "units") != "SI") {
        fail("MBD-I003", "units", "this program reads SI units (m, kg, s, rad, N), \"units\": \"SI\"");
    }

    System system;
    Model& model = system.model;
    if (f.has("gravity")) model.gravity = vec(f["gravity"], "gravity");
    if (f.has("bodies")) {
        const Json& bodies = list(f["bodies"], "bodies");
        for (std::size_t k = 0; k < bodies.size(); ++k) read_body(bodies[k], item("bodies", k), model);
    }
    if (f.has("constraints")) {
        const Json& constraints = list(f["constraints"], "constraints");
        for (std::size_t k = 0; k < constraints.size(); ++k) {
            system.constraints.push_back(read_constraint(constraints[k], item("constraints", k), model));
        }
    }
    if (f.has("force_elements")) {
        const Json& elements = list(f["force_elements"], "force_elements");
        for (std::size_t k = 0; k < elements.size(); ++k) {
            system.force_elements.push_back(read_force_element(elements[k], item("force_elements", k), model));
        }
    }
    if (f.has("joint_forces")) {
        const Json& forces = list(f["joint_forces"], "joint_forces");
        for (std::size_t k = 0; k < forces.size(); ++k) {
            system.joint_forces.push_back(read_joint_force(forces[k], item("joint_forces", k), model));
        }
    }
    return system;
}

void save_model(const System& system, const std::string& path)
{
    const std::string text = model_to_json(system);
    std::ofstream out(path, std::ios::binary);
    MBD_THROW_IF(!out, "MBD-I001: cannot write the model file " + path);
    out << text;
    out.close();
    MBD_THROW_IF(!out, "MBD-I001: cannot write the model file " + path);
}

System load_model(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    MBD_THROW_IF(!in, "MBD-I001: cannot read the model file " + path);
    std::ostringstream text;
    text << in.rdbuf();
    try {
        return model_from_json(text.str());
    } catch (const MbdError& e) {
        throw MbdError(std::string(e.what()) + " (in " + path + ")");
    }
}

} // namespace mbd::io
