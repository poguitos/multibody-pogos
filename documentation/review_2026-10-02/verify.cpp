// Standalone verification of suspected defects found while reading the engine.
// Not part of the project; lives in the session scratchpad.

#include "mbd/model/system.hpp"
#include "mbd/model/constraint.hpp"
#include "mbd/algorithms/dynamics.hpp"
#include "mbd/integrators/simulator.hpp"
#include "mbd/vehicle/vehicle.hpp"
#include "mbd/vehicle/vehicle_template.hpp"
#include "mbd/vehicle/drivetrain.hpp"

#include <Eigen/SVD>
#include <chrono>
#include <cstdio>
#include <functional>

using namespace mbd;

static double now_us()
{
    using namespace std::chrono;
    return duration<double, std::micro>(steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------------------
// A. Kinematic consistency: finite-difference FK vs velocity FK vs Jacobian
// ---------------------------------------------------------------------------
static void check_kinematics(MultibodySystem& sys, const char* label)
{
    const VecX q0 = sys.q, qd = sys.q_dot;
    const double h = 1e-6;
    const int nb = sys.body_count();
    std::vector<Vec3> pp(nb), pm(nb);
    std::vector<Quat> rp(nb), rm(nb);

    sys.q = q0 + h * qd; sys.compute_forward_kinematics();
    for (int i = 0; i < nb; ++i) { pp[i] = sys.states[i].p_WB; rp[i] = sys.states[i].q_WB; }
    sys.q = q0 - h * qd; sys.compute_forward_kinematics();
    for (int i = 0; i < nb; ++i) { pm[i] = sys.states[i].p_WB; rm[i] = sys.states[i].q_WB; }

    sys.q = q0; sys.q_dot = qd; sys.compute_kinematics();

    std::printf("[A] %s\n", label);
    for (int i = 1; i < nb; ++i) {
        const Vec3 v_num = (pp[i] - pm[i]) / (2 * h);
        Quat dq = rp[i] * rm[i].conjugate();
        if (dq.w() < 0) dq.coeffs() *= -1.0;
        const Vec3 w_num = 2.0 * dq.vec() / (2 * h);

        const BodyJacobian bj = compute_body_jacobian_origin(sys, i);
        const Vec3 v_jac = bj.J_v * qd;
        const Vec3 w_jac = bj.J_omega * qd;

        std::printf("    body %d (%s): |v_fk-v_num|=%.3e |w_fk-w_num|=%.3e |v_jac-v_num|=%.3e |w_jac-w_num|=%.3e  (|v_num|=%.3f |w_num|=%.3f)\n",
            i, sys.body_infos[i].name.c_str(),
            (sys.states[i].v_WB - v_num).norm(), (sys.states[i].w_WB - w_num).norm(),
            (v_jac - v_num).norm(), (w_jac - w_num).norm(), v_num.norm(), w_num.norm());
    }
}

// ---------------------------------------------------------------------------
// B. Dynamics consistency: RNEA vs mass matrix, and RNEA bias vs Lagrange
// ---------------------------------------------------------------------------
static VecX lagrangian_bias(MultibodySystem& sys)   // gravity = 0
{
    const VecX q0 = sys.q, qd = sys.q_dot;
    const int n = sys.total_dof;
    const double h = 1e-6;
    auto Mat = [&](const VecX& q) {
        sys.q = q; sys.compute_forward_kinematics();
        return compute_mass_matrix(sys);
    };
    const MatX Mdot = (Mat(q0 + h * qd) - Mat(q0 - h * qd)) / (2 * h);
    VecX hb = Mdot * qd;
    for (int k = 0; k < n; ++k) {
        VecX e = VecX::Zero(n); e(k) = h;
        const double Tp = 0.5 * qd.dot(Mat(q0 + e) * qd);
        const double Tm = 0.5 * qd.dot(Mat(q0 - e) * qd);
        hb(k) -= (Tp - Tm) / (2 * h);
    }
    sys.q = q0; sys.q_dot = qd; sys.compute_kinematics();
    return hb;
}

static void check_dynamics(MultibodySystem& sys, const char* label)
{
    const int n = sys.total_dof;
    sys.compute_kinematics();
    const MatX M = compute_mass_matrix(sys);
    VecX qdd(n);
    for (int i = 0; i < n; ++i) qdd(i) = 0.3 + 0.37 * i * ((i % 2) ? -1.0 : 1.0);
    const Vec3 g0 = Vec3::Zero();
    const VecX tau1 = inverse_dynamics(sys, qdd, g0);
    const VecX tau0 = inverse_dynamics(sys, VecX::Zero(n), g0);
    const VecX hl = lagrangian_bias(sys);
    std::printf("[B] %s\n", label);
    std::printf("    |(RNEA(qdd)-RNEA(0)) - M*qdd| = %.3e   (|M*qdd| = %.3e)\n",
        ((tau1 - tau0) - M * qdd).norm(), (M * qdd).norm());
    std::printf("    |h_RNEA - h_Lagrange|         = %.3e   (|h_Lagrange| = %.3e)\n",
        (tau0 - hl).norm(), hl.norm());
    std::printf("    h_RNEA     = ["); for (int i = 0; i < n; ++i) std::printf(" %.5f", tau0(i)); std::printf(" ]\n");
    std::printf("    h_Lagrange = ["); for (int i = 0; i < n; ++i) std::printf(" %.5f", hl(i));   std::printf(" ]\n");
}

// ---------------------------------------------------------------------------
// C. Constraint Jacobian and acceleration bias vs finite differences
// ---------------------------------------------------------------------------
static void check_constraints(MultibodySystem& sys, const std::vector<const char*>& names)
{
    const VecX q0 = sys.q, qd = sys.q_dot;
    const int n = sys.total_dof;
    auto phi_at = [&](const VecX& q) {
        sys.q = q; sys.compute_forward_kinematics();
        return evaluate_all_constraints(sys);
    };
    const double h = 1e-4;
    const VecX php = phi_at(q0 + h * qd), phm = phi_at(q0 - h * qd), ph0 = phi_at(q0);
    const VecX phid_num  = (php - phm) / (2 * h);
    const VecX phidd_num = (php - 2 * ph0 + phm) / (h * h);

    sys.q = q0; sys.q_dot = qd; sys.compute_kinematics();
    const MatX Jq = build_constraint_jacobian(sys);
    const VecX phid_code = Jq * qd;
    const auto acc = compute_body_accelerations(sys, VecX::Zero(n));

    std::printf("[C] constraint velocity/acceleration consistency (two free bodies, q_ddot = 0)\n");
    int row = 0, k = 0;
    for (const auto& c : sys.constraints) {
        const int ne = c->equation_count();
        Eigen::MatrixXd J1, J2; c->jacobian(sys, J1, J2);
        Eigen::VectorXd gamma;  c->velocity_bias(sys, gamma);
        Vec6 a1; a1 << acc[c->body1_idx].a, acc[c->body1_idx].alpha;
        Vec6 a2; a2 << acc[c->body2_idx].a, acc[c->body2_idx].alpha;
        const VecX cb = J1 * a1 + J2 * a2 + gamma;
        std::printf("    %-34s |J*qd - dPhi/dt|=%.3e   |c_bias - d2Phi/dt2|=%.3e   (|d2Phi/dt2|=%.3e)\n",
            names[k],
            (phid_code.segment(row, ne) - phid_num.segment(row, ne)).norm(),
            (cb - phidd_num.segment(row, ne)).norm(),
            phidd_num.segment(row, ne).norm());
        row += ne; ++k;
    }
}

static void time_steps(const char* label, Simulator& sim, int n, double dt)
{
    double tmax = 0, ttot = 0;
    for (int i = 0; i < n; ++i) {
        const double t0 = now_us();
        sim.step(dt);
        const double d = now_us() - t0;
        ttot += d; if (d > tmax) tmax = d;
    }
    std::printf("[J] %-34s avg %8.1f us/step, worst %8.1f us, real-time factor at dt=%.0e s: %.2fx\n",
        label, ttot / n, tmax, dt, dt * 1e6 / (ttot / n));
}

template <class F>
static double time_call(F&& f, int reps)
{
    const double t0 = now_us();
    for (int i = 0; i < reps; ++i) f();
    return (now_us() - t0) / reps;
}

static void section(const char* name, const std::function<void()>& f)
{
    try { f(); }
    catch (const std::exception& e) { std::printf("!! %s threw: %s\n", name, e.what()); }
    std::fflush(stdout);
}

int main()
{
    const Mat3 Ry90 = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitY()).toRotationMatrix();
    const Mat3 Rx90 = Eigen::AngleAxisd(pi / 2.0, Vec3::UnitX()).toRotationMatrix();

    // ---- 1. bead on a rotating rod: revolute(Z) -> prismatic along the rod ----
    auto make_bead = [&](MultibodySystem& s) {
        auto I_rod  = RigidBodyInertia::from_solid_box(2.0, Vec3(0.5, 0.02, 0.02));
        auto I_bead = RigidBodyInertia::from_solid_box(1.0, Vec3(0.05, 0.05, 0.05));
        BodyIndex rod = s.add_body(I_rod, RigidBodyState{}, "rod", kGroundIndex);
        s.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3::Identity(), Transform3::Identity(), kGroundIndex, rod));
        BodyIndex bead = s.add_body(I_bead, RigidBodyState{}, "bead", rod);
        s.add_joint(std::make_unique<PrismaticCoordJoint>(
            Transform3(Ry90, Vec3::Zero()), Transform3::FromRotation(Ry90), rod, bead));
    };

    section("bead kinematics", [&] {
        MultibodySystem s; make_bead(s);
        s.q << 0.3, 0.5; s.q_dot << 2.0, 0.3;
        check_kinematics(s, "bead on rotating rod (revolute -> prismatic), theta_dot=2, d=0.5, d_dot=0.3");
        check_dynamics(s, "bead on rotating rod; analytic h = [2*m*d*d_dot*w, -m*d*w^2] = [0.6, -2.0]");
    });

    section("bead simulation", [&] {
        MultibodySystem s; make_bead(s);
        Simulator sim(s); sim.set_gravity(Vec3::Zero()); sim.method = IntegrationMethod::RK4; sim.initialize();
        s.q << 0.0, 0.5; s.q_dot << 2.0, 0.0; s.compute_kinematics();
        const double E0 = 0.5 * s.q_dot.dot(compute_mass_matrix(s) * s.q_dot);
        sim.run(1.0, 1e-4);
        s.compute_kinematics();
        const double E1 = 0.5 * s.q_dot.dot(compute_mass_matrix(s) * s.q_dot);

        // reference: (I0 + m d^2) th'' = -2 m d d' th' ;  d'' = d th'^2
        const double m = 1.0, I0 = (2.0 / 3.0) * (0.25 + 0.0004) + (1.0 / 3.0) * (0.0025 + 0.0025);
        double y[4] = {0.0, 0.5, 2.0, 0.0};
        auto f = [&](const double* a, double* o) {
            o[0] = a[2]; o[1] = a[3];
            o[2] = -2.0 * m * a[1] * a[3] * a[2] / (I0 + m * a[1] * a[1]);
            o[3] = a[1] * a[2] * a[2];
        };
        const double dt = 1e-4;
        for (int i = 0; i < 10000; ++i) {
            double k1[4], k2[4], k3[4], k4[4], t[4];
            f(y, k1);
            for (int j = 0; j < 4; ++j) t[j] = y[j] + 0.5 * dt * k1[j]; f(t, k2);
            for (int j = 0; j < 4; ++j) t[j] = y[j] + 0.5 * dt * k2[j]; f(t, k3);
            for (int j = 0; j < 4; ++j) t[j] = y[j] + dt * k3[j];       f(t, k4);
            for (int j = 0; j < 4; ++j) y[j] += dt / 6.0 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
        }
        std::printf("[E] bead on rod after 1 s (no gravity, theta_dot0 = 2 rad/s, d0 = 0.5 m):\n");
        std::printf("    engine:    d = %.5f m, theta_dot = %.5f rad/s, kinetic energy %.5f -> %.5f J\n",
            s.q(1), s.q_dot(0), E0, E1);
        std::printf("    reference: d = %.5f m, theta_dot = %.5f rad/s (energy conserved)\n", y[1], y[2]);
    });

    // ---- 2. universal / spherical pendulum ----
    section("universal", [&] {
        MultibodySystem s;
        auto I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.1, 0.1));
        BodyIndex b = s.add_body(I, RigidBodyState{}, "universal_bob", kGroundIndex);
        s.add_joint(std::make_unique<UniversalCoordJoint>(
            Transform3::Identity(), Transform3::FromTranslation(Vec3(0, 1.0, 0)), kGroundIndex, b));
        s.q << 0.7, 0.5; s.q_dot << 1.0, -0.8;
        check_kinematics(s, "universal joint at q = (0.7, 0.5)");
        s.q << 0.2, 0.3; s.q_dot << 1.0, -0.8;
        check_kinematics(s, "universal joint at q = (0.2, 0.3)  [angles used by the existing energy test]");
    });
    section("spherical", [&] {
        MultibodySystem s;
        auto I = RigidBodyInertia::from_solid_box(1.0, Vec3(0.1, 0.2, 0.3));
        BodyIndex b = s.add_body(I, RigidBodyState{}, "spherical_bob", kGroundIndex);
        s.add_joint(std::make_unique<SphericalCoordJoint>(
            Transform3::Identity(), Transform3::FromTranslation(Vec3(0, 1.0, 0)), kGroundIndex, b));
        s.q << 0.4, -0.3, 0.6; s.q_dot << 0.5, 1.0, -0.7;
        check_kinematics(s, "spherical joint");
        check_dynamics(s, "spherical joint");
    });

    // ---- 3. free chassis -> prismatic wheel (the simple-vehicle topology) ----
    section("free->prismatic", [&] {
        MultibodySystem s;
        auto Ic = RigidBodyInertia::from_solid_box(1400.0, Vec3(1.5, 0.3, 0.8));
        auto Iw = RigidBodyInertia::from_solid_box(40.0, Vec3(0.15, 0.15, 0.15));
        BodyIndex c = s.add_body(Ic, RigidBodyState{}, "chassis", kGroundIndex);
        s.add_joint(std::make_unique<FreeCoordJoint>(
            Transform3::Identity(), Transform3::Identity(), kGroundIndex, c));
        BodyIndex w = s.add_body(Iw, RigidBodyState{}, "wheel", c);
        s.add_joint(std::make_unique<PrismaticCoordJoint>(
            Transform3(Rx90, Vec3(1.35, 0.0, 0.8)), Transform3::FromRotation(Rx90), c, w));
        s.q << 1.0, 0.5, -2.0, 0.05, -0.3, 0.04, 0.25;
        s.q_dot << 20.0, 0.1, 0.5, 0.3, 0.6, -0.4, 0.2;
        check_kinematics(s, "free chassis -> prismatic wheel, susp travel 0.25 m, chassis rates (0.3,0.6,-0.4) rad/s");
        check_dynamics(s, "free chassis -> prismatic wheel");
    });

    // ---- 4. free joint with rotated/offset joint frames, then a revolute ----
    section("free rotated", [&] {
        MultibodySystem s;
        auto I = RigidBodyInertia::from_solid_box(3.0, Vec3(0.2, 0.1, 0.3));
        BodyIndex a = s.add_body(I, RigidBodyState{}, "free_rotated_frames", kGroundIndex);
        const Quat qpj = Quat(Eigen::AngleAxisd(0.5, Vec3::UnitZ())) * Quat(Eigen::AngleAxisd(0.3, Vec3::UnitX()));
        const Quat qcj(Eigen::AngleAxisd(0.4, Vec3::UnitY()));
        s.add_joint(std::make_unique<FreeCoordJoint>(
            Transform3(qpj, Vec3(0.2, -0.1, 0.4)), Transform3(qcj, Vec3(0.1, 0.2, -0.3)), kGroundIndex, a));
        BodyIndex b = s.add_body(I, RigidBodyState{}, "revolute_child", a);
        s.add_joint(std::make_unique<RevoluteCoordJoint>(
            Transform3(Ry90, Vec3(0.3, 0.1, 0.0)), Transform3(Rx90, Vec3(0.0, 0.2, 0.1)), a, b));
        s.q << 0.5, -0.4, 0.3, 0.6, -0.2, 0.7, 0.9;
        s.q_dot << 0.3, 0.2, -0.5, 0.4, 0.8, -0.6, 1.1;
        check_kinematics(s, "free joint with rotated X_PJ / X_CJ -> revolute child");
        check_dynamics(s, "free joint with rotated X_PJ / X_CJ -> revolute child");
    });

    // ---- C. constraints between two free bodies ----
    section("constraints", [&] {
        MultibodySystem s;
        auto I = RigidBodyInertia::from_solid_box(2.0, Vec3(0.2, 0.1, 0.3));
        BodyIndex A = s.add_body(I, RigidBodyState{}, "A", kGroundIndex);
        s.add_joint(std::make_unique<FreeCoordJoint>(Transform3::Identity(), Transform3::Identity(), kGroundIndex, A));
        BodyIndex B = s.add_body(I, RigidBodyState{}, "B", kGroundIndex);
        s.add_joint(std::make_unique<FreeCoordJoint>(Transform3::Identity(), Transform3::Identity(), kGroundIndex, B));
        s.q << 0.2, 0.5, -0.3, 0.3, -0.2, 0.5,   1.0, 0.7, 0.4, -0.4, 0.6, 0.1;
        s.q_dot << 0.5, -0.2, 0.3, 0.8, -0.5, 0.6,   -0.3, 0.4, 0.2, -0.7, 0.9, 0.4;

        const Vec3 a1(0.3, 0.1, -0.2), a2(-0.1, 0.25, 0.15);
        std::vector<const char*> names;
        s.constraints.push_back(std::make_shared<DistanceConstraint>(A, B, a1, a2, 1.0));
        names.push_back("Distance (off-origin anchors)");
        s.constraints.push_back(std::make_shared<DistanceConstraint>(A, B, Vec3::Zero(), Vec3::Zero(), 1.0));
        names.push_back("Distance (anchors at origins)");
        s.constraints.push_back(std::make_shared<CoincidentPointConstraint>(A, B, a1, a2));
        names.push_back("CoincidentPoint");
        s.constraints.push_back(std::make_shared<RevoluteJoint>(A, B, a1, Vec3::UnitZ(), a2, Vec3::UnitY()));
        names.push_back("RevoluteJoint (5 eq)");
        s.constraints.push_back(std::make_shared<PointCoordinateConstraint>(B, Vec3(0.1, 0.2, 0.3), 1, 0.5));
        names.push_back("PointCoordinate");
        s.constraints.push_back(std::make_shared<StrutLineTwoBodyConstraint>(
            A, B, Vec3(0.1, 0.4, 0.2), Vec3(0.0, 0.1, 0.05), Vec3(0.2, 1.0, 0.1)));
        names.push_back("StrutLineTwoBody (McPherson)");
        s.constraints.push_back(std::make_shared<StrutLineConstraint>(
            B, Vec3(0.5, 1.0, 0.2), Vec3(0.0, 0.1, 0.05), Vec3(0.2, 1.0, 0.1)));
        names.push_back("StrutLine (ground)");
        check_constraints(s, names);
    });

    // ---- D. McPherson corner: rank of J M^-1 J^T ----
    section("mcpherson rank", [&] {
        MultibodySystem s;
        auto Ic = RigidBodyInertia::from_solid_box(1400.0, Vec3(1.5, 0.3, 0.8));
        BodyIndex c = s.add_body(Ic, RigidBodyState{}, "chassis", kGroundIndex);
        s.add_joint(std::make_unique<FreeCoordJoint>(Transform3::Identity(), Transform3::Identity(), kGroundIndex, c));
        McPhersonParams p; p.arm_mass = 5.0; p.upright_mass = 15.0;
        build_mcpherson_corner_dynamic(s, c, p);
        s.q.setZero(); s.q_dot.setZero(); s.compute_kinematics();
        const MatX M = compute_mass_matrix(s);
        const MatX J = build_constraint_jacobian(s);
        Eigen::LLT<MatX> llt(M);
        const MatX A = J * llt.solve(J.transpose());
        Eigen::JacobiSVD<MatX> svd(A);
        std::printf("[D] McPherson corner on a free chassis: %d constraint equations, singular values of J*M^-1*J^T:\n    ",
            static_cast<int>(A.rows()));
        for (int i = 0; i < svd.singularValues().size(); ++i) std::printf("%.3e ", svd.singularValues()(i));
        VecX x = VecX::LinSpaced(A.rows(), 1.0, 2.0);
        const VecX rhs = A * x;
        const VecX lam = A.ldlt().solve(rhs);
        std::printf("\n    LDLT solve used by constrained_forward_dynamics: |A*lambda - rhs| = %.3e, |lambda| = %.3e (true |x| = %.3e)\n",
            (A * lam - rhs).norm(), lam.norm(), x.norm());
    });

    // ---- I. braking with the RWD simple vehicle (the failing test) ----
    section("braking", [&] {
        MultibodySystem sys; VehicleParams vp; auto vm = build_simple_vehicle(sys, vp);
        Simulator sim(sys); sim.set_gravity(Vec3(0.0, -g_accel, 0.0)); sim.method = IntegrationMethod::RK4; sim.initialize();
        set_vehicle_equilibrium(sys, vm);
        sys.q_dot(0) = 20.0; sys.compute_kinematics();
        Drivetrain dt; dt.params.layout = DriveLayout::RWD; dt.initialize(sys, vm);
        dt.throttle = 0.0; dt.brake = 0.0; dt.connect(sim, vm);
        sim.run(0.3, 0.001);
        const double Vb = sys.q_dot(0);
        dt.brake = 1.0;
        sim.run(1.0, 0.001);
        std::printf("[I] RWD simple vehicle, full brake, 1 s after application (V = %.2f m/s):\n", sys.q_dot(0));
        const char* nm[4] = {"FL", "FR", "RL", "RR"};
        for (int c = 0; c < 4; ++c) {
            std::printf("    %s: Fx = %9.1f N  Fz = %8.1f N  kappa = %7.3f  wheel_omega = %7.2f rad/s  brake torque = %7.1f Nm  free-roll forced: %s\n",
                nm[c], vm.tires[c]->get_Fx(), vm.tires[c]->get_vertical_force(), vm.tires[c]->get_slip_ratio(),
                dt.wheel_omega[c], dt.brake_torque_out[c], vm.tires[c]->auto_free_roll ? "yes" : "no");
        }
        sim.run(1.0, 0.001);
        std::printf("    V before = %.2f m/s, after 2 s of full braking = %.2f m/s (mean decel %.2f m/s^2; test expects < %.2f m/s)\n",
            Vb, sys.q_dot(0), (Vb - sys.q_dot(0)) / 2.0, 0.3 * Vb);
    });

    // ---- tyre load sensitivity of the default parameter set ----
    section("tyre", [&] {
        PacejkaTire t;
        std::printf("[M] default tyre:");
        for (double Fz : {2000.0, 4000.0, 8000.0})
            std::printf("  Fz=%.0f N: mu_y=%.4f, C_alpha=%.0f N/rad;", Fz, t.peak_mu_lateral(Fz), t.cornering_stiffness(Fz));
        std::printf("\n");
    });

    // ---- J/K/L/O. vehicles: mass accounting, start residual, rack, timing ----
    section("simple sedan timing", [&] {
        auto tmpl = VehicleTemplate::DefaultSedan();
        MultibodySystem sys; auto vh = build_vehicle(sys, tmpl);
        Simulator sim(sys); sim.set_gravity(Vec3(0.0, -g_accel, 0.0)); sim.method = IntegrationMethod::RK4; sim.initialize();
        set_vehicle_equilibrium(sys, vh);
        Drivetrain dt(tmpl.drivetrain); dt.initialize(sys, vh); dt.connect(sim, vh); dt.throttle = 0.3;
        std::printf("[J] timings (this machine, /O2, single thread)\n");
        time_steps("simple sedan, 10 DOF, RK4", sim, 2000, 0.001);
        sim.method = IntegrationMethod::SemiImplicitEuler;
        time_steps("simple sedan, 10 DOF, semi-impl. Euler", sim, 2000, 0.001);
    });

    section("dwb sedan", [&] {
        auto tmpl = VehicleTemplate::DefaultSedan();
        tmpl.front_axle.suspension_type = SuspensionType::DoubleWishbone;
        tmpl.rear_axle.suspension_type  = SuspensionType::DoubleWishbone;
        MultibodySystem sys; auto vh = build_vehicle(sys, tmpl);
        Simulator sim(sys); sim.set_gravity(Vec3(0.0, -g_accel, 0.0)); sim.method = IntegrationMethod::RK4; sim.initialize();
        set_vehicle_equilibrium(sys, vh);

        double m_sum = 0; for (int i = 1; i < sys.body_count(); ++i) m_sum += sys.inertias[i].mass;
        std::printf("[L] all-DWB sedan: %d bodies, %d DOF, %d constraint equations; mass in model %.1f kg vs template total_mass() %.1f kg\n",
            sys.body_count() - 1, sys.total_dof, total_constraint_equations(sys), m_sum, tmpl.total_mass());

        sys.compute_kinematics(); sys.clear_forces(); sys.apply_force_elements();
        const VecX tau = project_body_forces_to_joint_space(sys);
        const VecX qdd = constrained_forward_dynamics(sys, tau, sim.gravity, 5.0, 5.0);
        std::printf("[K] all-DWB sedan at set_vehicle_equilibrium(): chassis vertical accel %.2f m/s^2, max |q_ddot| = %.2f\n",
            qdd(1), qdd.cwiseAbs().maxCoeff());

        vh.set_steering(0.10);
        std::printf("[O] set_steering(0.10 rad): tie-rod inner displacement FL = %.5f m, FR = %.5f m (one rigid rack would give equal values)\n",
            vh.corners[0].tierod_constraint->anchor1_B.z() - vh.corners[0].tierod_inner_ref.z(),
            vh.corners[1].tierod_constraint->anchor1_B.z() - vh.corners[1].tierod_inner_ref.z());
        vh.set_steering(0.0);

        Drivetrain dt(tmpl.drivetrain); dt.initialize(sys, vh); dt.connect(sim, vh); dt.throttle = 0.3;
        time_steps("all-DWB sedan, 26 DOF, RK4", sim, 1000, 0.001);

        sys.compute_kinematics();
        const int n = sys.total_dof;
        std::printf("    breakdown per call [us]: kinematics %.1f | mass matrix %.1f | RNEA %.1f | constraint Jacobian %.1f | force elements+projection %.1f | constraint projection %.1f\n",
            time_call([&] { sys.compute_kinematics(); }, 300),
            time_call([&] { volatile double x = compute_mass_matrix(sys)(0, 0); (void)x; }, 300),
            time_call([&] { volatile double x = inverse_dynamics(sys, VecX::Zero(n), sim.gravity)(0); (void)x; }, 300),
            time_call([&] { volatile double x = build_constraint_jacobian(sys)(0, 0); (void)x; }, 300),
            time_call([&] { sys.clear_forces(); sys.apply_force_elements(); volatile double x = project_body_forces_to_joint_space(sys)(0); (void)x; }, 300),
            time_call([&] { project_onto_constraints(sys, 1e-10); }, 300));
    });

    section("sports car mass", [&] {
        auto tmpl = VehicleTemplate::SportsCar();
        MultibodySystem sys; auto vh = build_vehicle(sys, tmpl);
        double m_sum = 0; for (int i = 1; i < sys.body_count(); ++i) m_sum += sys.inertias[i].mass;
        std::printf("[L] SportsCar preset: mass in model %.1f kg vs template total_mass() %.1f kg\n", m_sum, tmpl.total_mass());
        (void)vh;
    });

    return 0;
}
