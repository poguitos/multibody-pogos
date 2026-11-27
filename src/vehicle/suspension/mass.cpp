#include "mbd/vehicle/suspension/mass.hpp"

namespace mbd {

Mat3 point_inertia(Real m, const Vec3& r)
{
    return m * (r.squaredNorm() * Mat3::Identity() - r * r.transpose());
}

RigidBodyInertia rod_inertia(Real mass, const Vec3& a, const Vec3& b)
{
    MBD_THROW_IF(!(mass > 0.0), "MBD-V001: rod_inertia: the mass must be > 0");
    const Vec3 d = b - a;
    const Real L = d.norm();
    const Vec3 u = L > 0.0 ? Vec3(d / L) : Vec3(Vec3::UnitX());
    const Real across = mass * (3.0 * rod_radius * rod_radius + L * L) / 12.0;
    const Real along = 0.5 * mass * rod_radius * rod_radius;
    const Mat3 I = across * (Mat3::Identity() - u * u.transpose()) + along * u * u.transpose();
    return RigidBodyInertia(mass, 0.5 * (a + b), I);
}

RigidBodyInertia lumped_inertia(Real mass, const std::vector<Vec3>& points)
{
    MBD_THROW_IF(!(mass > 0.0) || points.empty(), "MBD-V001: lumped_inertia: the mass must be > 0, with points");
    const Real share = mass / static_cast<Real>(points.size());
    Vec3 com = Vec3::Zero();
    for (const Vec3& p : points) com += p;
    com /= static_cast<Real>(points.size());
    Mat3 I = Mat3::Zero();
    for (const Vec3& p : points) {
        I += point_inertia(share, p - com) + 0.4 * share * lump_radius * lump_radius * Mat3::Identity();
    }
    return RigidBodyInertia(mass, com, I);
}

RigidBodyInertia combine(const RigidBodyInertia& a, const RigidBodyInertia& b)
{
    const Real m = a.mass + b.mass;
    const Vec3 com = (a.mass * a.com_B + b.mass * b.com_B) / m;
    const Mat3 I = a.I_com_B + point_inertia(a.mass, a.com_B - com) + b.I_com_B + point_inertia(b.mass, b.com_B - com);
    return RigidBodyInertia(m, com, I);
}

} // namespace mbd
