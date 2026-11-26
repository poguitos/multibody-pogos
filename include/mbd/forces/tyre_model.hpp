#pragma once

// A tyre's law (plan task 7.5, D65): its forces and moments at the contact
// from its load, slips and inclination, whatever the model. The Magic Formula
// (mbd/forces/magic_formula.hpp) is one; the simple tyres
// (mbd/forces/simple_tyres.hpp: linear with saturation, Fiala, brush) are
// the others. The tyre on a wheel (FullTireForce, mbd/forces/tire.hpp) holds
// one, and lags its slips over the relaxation lengths the model gives.
//
// Axes and signs are ISO, the tyre's W axes at the contact: x along the
// wheel's heading in the road plane, y to its left, z along the road's normal,
// up. The longitudinal slip kappa = -Vsx / |Vcx| is positive in traction;
// the slip angle alpha, with tan(alpha) = Vcy / |Vcx|, is positive when the
// contact moves to the left, so that a car tyre's cornering stiffness is
// negative; the inclination gamma is positive when the wheel's top leans to
// the right. The outputs are the forces and moments of the road on the tyre.

#include "mbd/core/core.hpp"

namespace mbd {

/// The side of a vehicle a tyre's parameters describe (a .tir file's
/// TYRESIDE): a tyre mounted on the other side is evaluated mirrored.
enum class TyreSide { Left, Right };

/// The tyre's state for one evaluation.
struct TyreInput {
    Real Fz{0.0};         ///< Normal load [N]; the outputs are zero at Fz <= 0
    Real kappa{0.0};      ///< Longitudinal slip, -Vsx / |Vcx| [-]
    Real alpha{0.0};      ///< Slip angle [rad], tan(alpha) = Vcy / |Vcx|
    Real gamma{0.0};      ///< Inclination angle [rad], ISO
    /// Forward speed of the contact centre [m/s]: its sign, and with the slips
    /// the slip speed (Vsx = -kappa |Vcx|, Vcy = tan(alpha) |Vcx|), and the
    /// rolling resistance.
    Real Vcx{10.0};
    Real pressure{-1.0};  ///< Inflation pressure [Pa] (the Magic Formula 6.1); below 0, the model's own
    /// The road's friction, a scale on the tyre's own (1: the surface its
    /// parameters were measured on).
    Real friction{1.0};
};

/// The forces and moments, and the quantities behind them (W axes, at the
/// contact point, on the tyre). A model fills what it has; the rest is zero.
struct TyreOutput {
    Real Fx{0.0}, Fy{0.0}, Fz{0.0}, Mx{0.0}, My{0.0}, Mz{0.0};
    Real Fx0{0.0}, Fy0{0.0}, Mz0{0.0};   ///< At pure slip
    Real Kxk{0.0};          ///< Longitudinal slip stiffness dFx0/dkappa at kappa_x = 0 [N]
    Real Kya{0.0};          ///< Cornering stiffness dFy0/dalpha at alpha_y = 0 [N/rad]
    Real mux{0.0}, muy{0.0};
    Real trail{0.0};        ///< Pneumatic trail t [m]
    Real Mzr{0.0};          ///< Residual aligning torque [N m]
    Real s{0.0};            ///< Moment arm of Fx in Mz [m]
    Real sigma_kappa{0.0};  ///< Longitudinal relaxation length [m]
    Real sigma_alpha{0.0};  ///< Lateral relaxation length [m]
};

/// What a tyre gives the wheel it is on besides its forces.
struct TyreBody {
    Real free_radius{0.0};         ///< Unloaded radius [m]
    Real vertical_stiffness{0.0};  ///< [N/m]
    Real vertical_damping{0.0};    ///< [N s/m]
    Real nominal_load{0.0};        ///< [N]
    Real low_speed{1.0};           ///< [m/s] Below it the tyre on a wheel damps its slips and fades its shifts
    TyreSide side{TyreSide::Left}; ///< The side of a vehicle the parameters describe
};

/// A tyre's law.
class TyreModel {
public:
    virtual ~TyreModel() = default;

    /// Short name, for messages and tables.
    virtual const char* name() const = 0;

    /// The steady-state forces and moments.
    virtual TyreOutput evaluate(const TyreInput& in) const = 0;

    /// The quantities that do not depend on the slips, at a load, an
    /// inclination, a pressure (below 0, the model's own) and a road's
    /// friction: the outputs Kxk, Kya, mux, muy, sigma_kappa and sigma_alpha.
    /// The tyre on a wheel needs them before it has its slips. By default,
    /// evaluate() at zero slip and zero speed.
    virtual TyreOutput characteristics(Real Fz, Real gamma, Real pressure = -1.0, Real friction = 1.0) const;

    /// The tyre's radius, vertical stiffness and damping, nominal load, low
    /// speed and side.
    virtual TyreBody body() const = 0;

protected:
    TyreModel() = default;
    TyreModel(const TyreModel&) = default;
    TyreModel& operator=(const TyreModel&) = default;
};

} // namespace mbd
