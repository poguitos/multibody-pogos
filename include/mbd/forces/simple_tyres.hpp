#pragma once

// Simple tyres (plan task 7.5, D65), kept beside the Magic Formula
// (mbd/forces/magic_formula.hpp) for real-time use and for teaching: a few
// parameters each, closed forms, a fraction of the Magic Formula's cost.
//
//   - LinearTyre: slip stiffnesses in proportion to the load, the force
//     limited to the friction circle, the aligning torque from a constant
//     pneumatic trail.
//   - FialaTyre: the Fiala tyre as MSC Adams and Project Chrono have it
//     (Chrono's ChFialaTire, BSD-3, read for this): the side force and
//     aligning torque of the brush model at pure side slip, Pacejka's (3.11)
//     and (3.12), with the tyre's width as the torque's arm; the longitudinal
//     force linear up to kappa_c = mu Fz / (2 C_kappa), then mu Fz - (mu
//     Fz)^2 / (4 |kappa| C_kappa); the friction falling from mu_max at no slip
//     to mu_min at a combined slip sqrt(kappa^2 + tan^2 alpha) of 1. The two
//     forces do not reduce each other: at combined slip their sum can pass mu
//     Fz.
//   - BrushTyre: the brush model of Pacejka (3rd ed., 2012), section 3.2, with
//     a parabolic pressure distribution and the same tread stiffness and
//     friction in every direction: the theoretical slip s = (kappa, -tan
//     alpha) / (1 + kappa) (3.34), the force mu Fz (3 theta s - 3 (theta s)^2 +
//     (theta s)^3) against the slip up to full sliding at theta s = 1 (3.48,
//     3.49), the aligning torque -t Fy with the trail of (3.13) at theta s
//     (3.50); theta = C / (3 mu Fz), C = 2 c_p a^2 the slip stiffness, a the
//     half contact length, growing as the square root of the load.
//
// Common to all three: the rolling resistance My = -f_r R0 Fz against the
// rolling, no overturning moment, no camber, no pressure; the relaxation
// lengths the slip stiffnesses over the carcass's stiffnesses, as MF 6.1's.
// The defaults describe roughly the Magic Formula's Appendix 3 tyre at its
// nominal load. Signs are ISO (mbd/forces/tyre_model.hpp).

#include "mbd/core/core.hpp"
#include "mbd/forces/tyre_model.hpp"

namespace mbd {

/// What the simple tyres share: the wheel's tyre, the rolling resistance,
/// the carcass. Defaults: the Appendix 3 tyre's.
struct SimpleTyreCommon {
    Real free_radius{0.3135};             ///< R0 [m]
    Real vertical_stiffness{209651.0};    ///< [N/m]
    Real vertical_damping{50.0};          ///< [N s/m]
    Real nominal_load{4000.0};            ///< Fz0 [N]
    Real low_speed{1.0};                  ///< [m/s] (the tyre on a wheel's V_low)
    Real rolling_resistance{0.007};       ///< f_r [-]: My = -f_r R0 Fz, against the rolling
    Real carcass_stiffness_x{358066.0};   ///< c_x [N/m]: sigma_kappa = Kxk / c_x
    Real carcass_stiffness_y{102673.0};   ///< c_y [N/m]: sigma_alpha = |Kya| / c_y
};

/// Linear with saturation.
struct LinearTyreParams : SimpleTyreCommon {
    Real slip_stiffness{86748.0};        ///< Kxk at the nominal load [N], in proportion to the load
    Real cornering_stiffness{53356.0};   ///< |Kya| at the nominal load [N/rad], in proportion to the load
    Real friction{1.0};                  ///< mu: the force's limit mu Fz, in any direction
    Real trail{0.028};                   ///< The pneumatic trail [m]: Mz = -trail Fy
};

/// Fiala, as MSC Adams and Project Chrono have it.
struct FialaTyreParams : SimpleTyreCommon {
    Real slip_stiffness{86748.0};        ///< C_kappa [N], whatever the load
    Real cornering_stiffness{53356.0};   ///< C_alpha [N/rad], whatever the load
    Real friction_max{1.0};              ///< mu at no slip
    Real friction_min{0.9};              ///< mu at a combined slip of 1 and beyond
    Real width{0.205};                   ///< The aligning torque's arm [m]: the tyre's width, as Adams has it
};

/// The isotropic brush model with a parabolic pressure distribution.
struct BrushTyreParams : SimpleTyreCommon {
    Real slip_stiffness{53356.0};        ///< C = 2 c_p a^2 at the nominal load [N], longitudinal and lateral
    Real half_contact_length{0.075};     ///< a at the nominal load [m]; a = a0 sqrt(Fz / Fz0)
    Real friction{1.0};                  ///< mu, the same in every direction
};

/// Linear slip stiffnesses, the force limited to the friction circle.
class LinearTyre final : public TyreModel {
public:
    /// Throws MBD-F055 for a parameter that must be positive and is not.
    explicit LinearTyre(const LinearTyreParams& parameters);
    const LinearTyreParams& parameters() const { return p_; }
    const char* name() const override { return "linear"; }
    TyreOutput evaluate(const TyreInput& in) const override;
    TyreBody body() const override;

private:
    LinearTyreParams p_;
};

/// The Fiala tyre.
class FialaTyre final : public TyreModel {
public:
    /// Throws MBD-F055 for a parameter that must be positive and is not.
    explicit FialaTyre(const FialaTyreParams& parameters);
    const FialaTyreParams& parameters() const { return p_; }
    const char* name() const override { return "Fiala"; }
    TyreOutput evaluate(const TyreInput& in) const override;
    TyreBody body() const override;

private:
    FialaTyreParams p_;
};

/// The brush tyre.
class BrushTyre final : public TyreModel {
public:
    /// Throws MBD-F055 for a parameter that must be positive and is not.
    explicit BrushTyre(const BrushTyreParams& parameters);
    const BrushTyreParams& parameters() const { return p_; }
    const char* name() const override { return "brush"; }
    TyreOutput evaluate(const TyreInput& in) const override;
    TyreBody body() const override;

private:
    BrushTyreParams p_;
};

} // namespace mbd
