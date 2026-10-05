#include "mbd/analysis/position_kinematics.hpp"

namespace mbd {

Real extract_camber(const RigidBodyState& state)
{
    const Vec3 spin_axis_W = state.q_WB * Vec3::UnitY();
    return std::atan2(spin_axis_W.z(), spin_axis_W.y());
}

Real extract_toe(const RigidBodyState& state)
{
    const Vec3 fwd_W = state.q_WB * Vec3::UnitX();
    return std::atan2(fwd_W.z(), fwd_W.x());
}

void KinematicSweepResult::export_csv(const std::string& filename) const
{
    std::ofstream file(filename);
    file << std::fixed << std::setprecision(6);
    file << "bump_mm,camber_deg,toe_deg,wheel_center_y_mm,converged\n";

    for (const auto& p : points) {
        file << p.bump * 1000.0 << ","
             << p.camber * 180.0 / 3.14159265358979323846 << ","
             << p.toe * 180.0 / 3.14159265358979323846 << ","
             << p.wheel_y * 1000.0 << ","
             << (p.converged ? 1 : 0) << "\n";
    }
}

Real KinematicSweepResult::camber_gain() const
{
    if (points.size() < 2) return 0.0;
    const auto& first = points.front();
    const auto& last  = points.back();
    Real dbump = last.bump - first.bump;
    if (std::abs(dbump) < 1e-12) return 0.0;
    return (last.camber - first.camber) / dbump;
}

Kinematics::Kinematics(const kernel::System& sys)
    : system_(sys)
    , data_(sys.model)
    , solver_(sys.model, sys.constraints)
    , v_zero_(VecX::Zero(sys.model.nv))
    , v_(VecX::Zero(sys.model.nv))
    , a_(VecX::Zero(sys.model.nv))
    , q(sys.model.neutral_configuration())
{
    update();
}

bool Kinematics::solve(int max_iterations, Real tolerance)
{
    VecX v = v_zero_;
    const kernel::ProjectionInfo info =
        solver_.project(data_, q, v, t, tolerance, max_iterations);
    update();
    return info.converged;
}

const VecX& Kinematics::phi()
{
    solver_.evaluate(data_, q, v_zero_, t);
    return solver_.phi();
}

const VecX& Kinematics::velocities()
{
    // J v = nu, nu = -d(phi)/dt. The complete orthogonal decomposition gives
    // the exact solution when J has full column rank, even with redundant
    // rows, and the least-norm one otherwise.
    solver_.evaluate(data_, q, v_zero_, t);
    v_ = solver_.J().completeOrthogonalDecomposition().solve(solver_.nu());
    update();
    return v_;
}

const VecX& Kinematics::accelerations()
{
    // J a = gamma, whose velocity-product terms need v.
    solver_.evaluate(data_, q, v_, t);
    a_ = solver_.J().completeOrthogonalDecomposition().solve(solver_.gamma());
    update();
    return a_;
}

KinematicSweepResult sweep_bump_travel(Kinematics& k, int upright_body,
                                       Real bump_min, Real bump_max, int n_steps)
{
    KinematicSweepResult result;
    result.points.reserve(static_cast<std::size_t>(n_steps));
    VecX q_start = k.q;
    for (int i = 0; i < n_steps; ++i) {
        const Real bump = bump_min + (bump_max - bump_min) * i / std::max(n_steps - 1, 1);
        k.t = bump;
        k.q = q_start;
        const bool ok = k.solve();

        KinematicSweepPoint pt;
        pt.bump = bump;
        pt.converged = ok;
        const RigidBodyState s = k.state(upright_body);
        pt.wheel_y = s.p_WB.y();
        pt.camber = extract_camber(s);
        pt.toe = extract_toe(s);
        result.points.push_back(pt);

        if (ok) q_start = k.q;
    }
    return result;
}

std::shared_ptr<const kernel::ConstraintModel> point_height_driver(
    int body, const Vec3& point_B, int axis, Real nominal)
{
    return std::make_shared<kernel::Dot2>(
        kernel::Marker{0, Transform3::Identity()}, axis,
        kernel::Marker{body, Transform3::FromTranslation(point_B)},
        kernel::TimeFunction([nominal](Real t) { return nominal + t; },
                             [](Real) { return 1.0; },
                             [](Real) { return 0.0; }));
}

} // namespace mbd
