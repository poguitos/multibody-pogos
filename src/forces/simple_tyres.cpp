#include "mbd/forces/simple_tyres.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

// Equation numbers are Pacejka's, Tire and Vehicle Dynamics, 3rd ed. (2012),
// chapter 3; the Fiala tyre's longitudinal force and friction are MSC Adams's,
// as Project Chrono's ChFialaTire has them (mbd/forces/simple_tyres.hpp).

namespace mbd {

namespace {

Real sgn(Real x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }

void positive(Real v, const char* model, const char* name)
{
    MBD_THROW_IF(!(v > 0.0), std::string("MBD-F055: ") + model + " tyre: " + name + " must be > 0; it is "
                                 + std::to_string(v));
}

void not_negative(Real v, const char* model, const char* name)
{
    MBD_THROW_IF(!(v >= 0.0), std::string("MBD-F055: ") + model + " tyre: " + name + " must be >= 0; it is "
                                  + std::to_string(v));
}

void check_common(const SimpleTyreCommon& p, const char* model)
{
    positive(p.free_radius, model, "free_radius");
    not_negative(p.vertical_stiffness, model, "vertical_stiffness");
    not_negative(p.vertical_damping, model, "vertical_damping");
    positive(p.nominal_load, model, "nominal_load");
    positive(p.low_speed, model, "low_speed");
    not_negative(p.rolling_resistance, model, "rolling_resistance");
    positive(p.carcass_stiffness_x, model, "carcass_stiffness_x");
    positive(p.carcass_stiffness_y, model, "carcass_stiffness_y");
}

TyreBody common_body(const SimpleTyreCommon& p)
{
    return {p.free_radius, p.vertical_stiffness, p.vertical_damping, p.nominal_load, p.low_speed, TyreSide::Left};
}

/// The rolling resistance moment, against the rolling (the tyre on a wheel
/// takes its sign from the spin, below its V_roll).
Real rolling_resistance(const SimpleTyreCommon& p, const TyreInput& in)
{
    return -(in.Vcx < 0.0 ? -1.0 : 1.0) * p.rolling_resistance * p.free_radius * in.Fz;
}

} // namespace

// --- Linear with saturation ---------------------------------------------------

LinearTyre::LinearTyre(const LinearTyreParams& parameters)
    : p_(parameters)
{
    check_common(p_, "linear");
    positive(p_.slip_stiffness, "linear", "slip_stiffness");
    positive(p_.cornering_stiffness, "linear", "cornering_stiffness");
    positive(p_.friction, "linear", "friction");
    not_negative(p_.trail, "linear", "trail");
}

TyreBody LinearTyre::body() const { return common_body(p_); }

TyreOutput LinearTyre::evaluate(const TyreInput& in) const
{
    TyreOutput out;
    out.Fz = std::max(in.Fz, Real(0.0));
    if (!(in.Fz > 0.0)) return out;
    const Real load = in.Fz / p_.nominal_load;
    const Real K = p_.slip_stiffness * load;
    const Real C = p_.cornering_stiffness * load;
    const Real mu = p_.friction * in.friction;
    const Real limit = mu * in.Fz;

    // The linear forces, scaled back onto the friction circle when beyond it.
    Real Fx = K * in.kappa;
    Real Fy = -C * in.alpha;
    const Real F = std::hypot(Fx, Fy);
    if (F > limit) {
        Fx *= limit / F;
        Fy *= limit / F;
    }
    out.Fx = Fx;
    out.Fy = Fy;
    out.Mz = -p_.trail * Fy;
    out.Fx0 = std::clamp(K * in.kappa, -limit, limit);
    out.Fy0 = std::clamp(-C * in.alpha, -limit, limit);
    out.Mz0 = -p_.trail * out.Fy0;
    out.trail = p_.trail;
    out.My = rolling_resistance(p_, in);
    out.Kxk = K;
    out.Kya = -C;
    out.mux = out.muy = mu;
    out.sigma_kappa = K / p_.carcass_stiffness_x;
    out.sigma_alpha = C / p_.carcass_stiffness_y;
    return out;
}

// --- Fiala ---------------------------------------------------------------------

FialaTyre::FialaTyre(const FialaTyreParams& parameters)
    : p_(parameters)
{
    check_common(p_, "Fiala");
    positive(p_.slip_stiffness, "Fiala", "slip_stiffness");
    positive(p_.cornering_stiffness, "Fiala", "cornering_stiffness");
    positive(p_.friction_max, "Fiala", "friction_max");
    positive(p_.friction_min, "Fiala", "friction_min");
    not_negative(p_.width, "Fiala", "width");
}

TyreBody FialaTyre::body() const { return common_body(p_); }

TyreOutput FialaTyre::evaluate(const TyreInput& in) const
{
    TyreOutput out;
    out.Fz = std::max(in.Fz, Real(0.0));
    if (!(in.Fz > 0.0)) return out;
    const Real Fz = in.Fz;
    const Real C = p_.slip_stiffness;
    const Real Ca = p_.cornering_stiffness;

    // The forces and the moment at a longitudinal slip and the tangent of a
    // slip angle, the friction set by their combined slip.
    struct Forces {
        Real Fx, Fy, Mz;
    };
    auto forces = [&](Real kappa, Real tan_alpha) {
        const Real combined = std::min(Real(1.0), std::hypot(kappa, tan_alpha));
        const Real U = (p_.friction_max - (p_.friction_max - p_.friction_min) * combined) * in.friction;
        const Real limit = U * Fz;
        Forces f{};
        // Longitudinal: linear up to kappa_c = U Fz / (2 C), then saturating
        // with a continuous slope.
        if (std::abs(kappa) < limit / (2.0 * C)) {
            f.Fx = C * kappa;
        } else {
            f.Fx = sgn(kappa) * (limit - limit * limit / (4.0 * std::abs(kappa) * C));
        }
        // Lateral and the aligning torque: the brush model at pure side slip,
        // (3.11) and (3.12), with H = 1 - theta |tan alpha|, theta = C_alpha /
        // (3 U Fz), and the tyre's width for the half contact length.
        const Real H = 1.0 - Ca * std::abs(tan_alpha) / (3.0 * limit);
        if (H > 0.0) {
            f.Fy = -limit * (1.0 - H * H * H) * sgn(tan_alpha);
            f.Mz = limit * p_.width * (1.0 - H) * H * H * H * sgn(tan_alpha);
        } else {
            f.Fy = -limit * sgn(tan_alpha);
            f.Mz = 0.0;
        }
        return f;
    };
    const Real tan_alpha = std::tan(in.alpha);
    const Forces f = forces(in.kappa, tan_alpha);
    out.Fx = f.Fx;
    out.Fy = f.Fy;
    out.Mz = f.Mz;
    out.Fx0 = forces(in.kappa, 0.0).Fx;
    const Forces side = forces(0.0, tan_alpha);
    out.Fy0 = side.Fy;
    out.Mz0 = side.Mz;
    out.trail = f.Fy != 0.0 ? -f.Mz / f.Fy : 0.0;
    out.My = rolling_resistance(p_, in);
    out.Kxk = C;
    out.Kya = -Ca;
    out.mux = out.muy = p_.friction_max * in.friction;
    out.sigma_kappa = C / p_.carcass_stiffness_x;
    out.sigma_alpha = Ca / p_.carcass_stiffness_y;
    return out;
}

// --- Brush -----------------------------------------------------------------------

BrushTyre::BrushTyre(const BrushTyreParams& parameters)
    : p_(parameters)
{
    check_common(p_, "brush");
    positive(p_.slip_stiffness, "brush", "slip_stiffness");
    positive(p_.half_contact_length, "brush", "half_contact_length");
    positive(p_.friction, "brush", "friction");
}

TyreBody BrushTyre::body() const { return common_body(p_); }

TyreOutput BrushTyre::evaluate(const TyreInput& in) const
{
    TyreOutput out;
    out.Fz = std::max(in.Fz, Real(0.0));
    if (!(in.Fz > 0.0)) return out;
    const Real Fz = in.Fz;
    const Real load = Fz / p_.nominal_load;
    const Real a = p_.half_contact_length * std::sqrt(load);   // a grows as sqrt(Fz)
    const Real C = p_.slip_stiffness * load;                   // 2 c_p a^2 (3.3), (3.27)
    const Real mu = p_.friction * in.friction;
    const Real theta = C / (3.0 * mu * Fz);                    // (3.46)
    const Real inf = std::numeric_limits<Real>::infinity();

    // The force's magnitude and the trail at a theoretical slip s: with u =
    // theta s, mu Fz (3 u - 3 u^2 + u^3) = mu Fz (1 - (1 - u)^3) up to full
    // sliding at u = 1 (3.48), and the trail a / 3 (1 - u)^3 / (1 - u + u^2
    // / 3) (3.13), zero beyond.
    auto force = [&](Real s) {
        const Real u = std::min(theta * s, Real(1.0));
        return mu * Fz * (1.0 - (1.0 - u) * (1.0 - u) * (1.0 - u));
    };
    auto trail = [&](Real s) {
        const Real u = theta * s;
        return u < 1.0 ? a / 3.0 * (1.0 - u) * (1.0 - u) * (1.0 - u) / (1.0 - u + u * u / 3.0) : 0.0;
    };

    // The theoretical slip (3.34), in ISO's axes: s = (kappa, -tan alpha) /
    // (1 + kappa), the force along it (3.49), against the slip speed. At or
    // past a locked wheel (1 + kappa <= 0) it is full sliding.
    const Real kappa = in.kappa;
    const Real tan_alpha = std::tan(in.alpha);
    const Real rolling = std::abs(1.0 + kappa);
    const Real n = std::hypot(kappa, tan_alpha);
    if (n > 0.0) {
        const Real s = rolling > 0.0 ? n / rolling : inf;
        const Real F = force(s);
        out.Fx = F * kappa / n;
        out.Fy = -F * tan_alpha / n;
        out.trail = trail(s);
        out.Mz = -out.trail * out.Fy;                          // (3.50)
    } else {
        out.trail = a / 3.0;                                   // (3.14)
    }
    out.Fx0 = sgn(kappa) * force(rolling > 0.0 ? std::abs(kappa) / rolling : inf);
    out.Fy0 = -sgn(tan_alpha) * force(std::abs(tan_alpha));
    out.Mz0 = -trail(std::abs(tan_alpha)) * out.Fy0;
    out.My = rolling_resistance(p_, in);
    out.Kxk = C;
    out.Kya = -C;
    out.mux = out.muy = mu;
    out.sigma_kappa = C / p_.carcass_stiffness_x;
    out.sigma_alpha = C / p_.carcass_stiffness_y;
    return out;
}

} // namespace mbd
