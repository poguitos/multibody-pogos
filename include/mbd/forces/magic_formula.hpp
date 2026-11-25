#pragma once

// The Magic Formula tyre: steady-state forces and moments (plan task 7.4).
//
// The equations are TNO's MF-Tyre 6.1 as the appendix of Besselink, Schmeitz
// and Pacejka (2010) lists them, (33) to (90); Pacejka's Tire and Vehicle
// Dynamics, 3rd ed. (2012), section 4.3.2, (4.E1) to (4.E78), gives the same
// in most places. Where the two differ, the paper is followed, because .tir
// files are fitted with TNO's tools and MFeval's comparisons with TNO's
// solver side with the paper (decision D63): the slip angle and the
// inclination enter as they are (not the book's alpha* = tan(alpha) sgn(Vcx)
// and gamma* = sin(gamma)); the trail's camber terms (84), (86); cos(alpha)
// once in the residual torque (88), (90); the equivalent slips (80), (81);
// the rolling resistance (50); the overturning couple with qsx12 to qsx14
// (49); no degressive friction factor (the book's A_mu taken as 1). Turn slip
// is not modelled (every zeta factor 1). docs/tyres.md lists every choice
// with its source.
//
// Two versions, chosen by a parameter set's FITTYP as a .tir file gives it
// (mbd/io/tir_file.hpp):
//   - MF 6.1 (FITTYP 61 or 62), as above;
//   - MF 5.2 (FITTYP 6 or 21): the same equations with MF-Tyre 5.2's forms
//     where the paper says 6.1 changed them: camber shifts the side force
//     curve in proportion (pHy3) rather than through a camber stiffness; the
//     aligning torque's arm takes the side force less its slip-induced part,
//     at the inclination; the rolling resistance is R0 Fz (...); the
//     relaxation lengths come from pTx and pTy; there is no inflation
//     pressure; the inclination is scaled by LGAX, LGAY and LGAZ.
//
// Axes and signs are those of a .tir file: ISO, the tyre's W axes (x along the
// wheel's heading, y to its left, z along the road's normal, up). Inputs:
// the longitudinal slip kappa = -Vsx / |Vcx| (positive in traction), the slip
// angle alpha with tan(alpha) = Vcy / |Vcx| (positive when the contact moves
// to the left; a car tyre's cornering stiffness is then negative), the
// inclination gamma (positive when the wheel's top leans to the right), the
// normal load Fz > 0. Outputs: the forces and moments of the road on the tyre
// at the contact point, in the same axes. Rolling backwards (Vcx < 0), the
// trail and the residual torque change sign with Vcx, as the book's sgn(Vcx)
// in (4.E42) and (4.E47) has them, and the rolling resistance opposes the
// motion; the comparison with MFeval is at Vcx > 0 only.

#include "mbd/core/core.hpp"

#include <string>

namespace mbd {

/// Which Magic Formula a parameter set is evaluated with.
enum class MagicFormulaVersion {
    MF52,   ///< MF-Tyre 5.2 (PAC2002): FITTYP 6 or 21
    MF61,   ///< MF-Tire 6.1 (and 6.2's steady state): FITTYP 61 or 62
};

// Every parameter once: its .tir section, its name, its default when a file
// leaves it out. Coefficients default to 0, scaling factors to 1, except as
// noted (LMUV 0: no friction decay with slip speed; PKY4 2: the 5.2 load
// curve of the cornering stiffness). The book's qBz6 (4.E40) is not here: it
// is in no .tir file, and the paper's trail (84) has no such term.
#define MBD_MAGIC_FORMULA_PARAMETERS(X)                                                                          \
    X(MODEL, LONGVL, 16.7) X(MODEL, VXLOW, 1.0)                                                                  \
    X(DIMENSION, UNLOADED_RADIUS, 0.3) X(DIMENSION, WIDTH, 0.2) X(DIMENSION, RIM_RADIUS, 0.2)                   \
    X(DIMENSION, RIM_WIDTH, 0.15) X(DIMENSION, ASPECT_RATIO, 0.6)                                               \
    X(OPERATING_CONDITIONS, INFLPRES, 220000.0) X(OPERATING_CONDITIONS, NOMPRES, 220000.0)                      \
    X(VERTICAL, FNOMIN, 4000.0) X(VERTICAL, VERTICAL_STIFFNESS, 200000.0) X(VERTICAL, VERTICAL_DAMPING, 50.0)   \
    X(VERTICAL, BREFF, 0.0) X(VERTICAL, DREFF, 0.0) X(VERTICAL, FREFF, 0.0) X(VERTICAL, Q_RE0, 1.0)            \
    X(VERTICAL, Q_V1, 0.0) X(VERTICAL, Q_V2, 0.0) X(VERTICAL, Q_FZ2, 0.0) X(VERTICAL, Q_FCX, 0.0)               \
    X(VERTICAL, Q_FCY, 0.0) X(VERTICAL, PFZ1, 0.0)                                                              \
    X(STRUCTURAL, LONGITUDINAL_STIFFNESS, 0.0) X(STRUCTURAL, LATERAL_STIFFNESS, 0.0)                            \
    X(STRUCTURAL, PCFX1, 0.0) X(STRUCTURAL, PCFX2, 0.0) X(STRUCTURAL, PCFX3, 0.0)                               \
    X(STRUCTURAL, PCFY1, 0.0) X(STRUCTURAL, PCFY2, 0.0) X(STRUCTURAL, PCFY3, 0.0)                               \
    X(SCALING_COEFFICIENTS, LFZO, 1.0) X(SCALING_COEFFICIENTS, LCX, 1.0) X(SCALING_COEFFICIENTS, LMUX, 1.0)    \
    X(SCALING_COEFFICIENTS, LEX, 1.0) X(SCALING_COEFFICIENTS, LKX, 1.0) X(SCALING_COEFFICIENTS, LHX, 1.0)      \
    X(SCALING_COEFFICIENTS, LVX, 1.0) X(SCALING_COEFFICIENTS, LCY, 1.0) X(SCALING_COEFFICIENTS, LMUY, 1.0)     \
    X(SCALING_COEFFICIENTS, LEY, 1.0) X(SCALING_COEFFICIENTS, LKY, 1.0) X(SCALING_COEFFICIENTS, LKYC, 1.0)     \
    X(SCALING_COEFFICIENTS, LKZC, 1.0) X(SCALING_COEFFICIENTS, LHY, 1.0) X(SCALING_COEFFICIENTS, LVY, 1.0)     \
    X(SCALING_COEFFICIENTS, LTR, 1.0) X(SCALING_COEFFICIENTS, LRES, 1.0) X(SCALING_COEFFICIENTS, LXAL, 1.0)    \
    X(SCALING_COEFFICIENTS, LYKA, 1.0) X(SCALING_COEFFICIENTS, LVYKA, 1.0) X(SCALING_COEFFICIENTS, LS, 1.0)    \
    X(SCALING_COEFFICIENTS, LMX, 1.0) X(SCALING_COEFFICIENTS, LVMX, 1.0) X(SCALING_COEFFICIENTS, LMY, 1.0)     \
    X(SCALING_COEFFICIENTS, LMUV, 0.0) X(SCALING_COEFFICIENTS, LSGKP, 1.0) X(SCALING_COEFFICIENTS, LSGAL, 1.0) \
    X(SCALING_COEFFICIENTS, LGAX, 1.0) X(SCALING_COEFFICIENTS, LGAY, 1.0) X(SCALING_COEFFICIENTS, LGAZ, 1.0)   \
    X(LONGITUDINAL_COEFFICIENTS, PCX1, 0.0) X(LONGITUDINAL_COEFFICIENTS, PDX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PDX2, 0.0) X(LONGITUDINAL_COEFFICIENTS, PDX3, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PEX1, 0.0) X(LONGITUDINAL_COEFFICIENTS, PEX2, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PEX3, 0.0) X(LONGITUDINAL_COEFFICIENTS, PEX4, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PKX1, 0.0) X(LONGITUDINAL_COEFFICIENTS, PKX2, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PKX3, 0.0) X(LONGITUDINAL_COEFFICIENTS, PHX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PHX2, 0.0) X(LONGITUDINAL_COEFFICIENTS, PVX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PVX2, 0.0) X(LONGITUDINAL_COEFFICIENTS, PPX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PPX2, 0.0) X(LONGITUDINAL_COEFFICIENTS, PPX3, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PPX4, 0.0) X(LONGITUDINAL_COEFFICIENTS, RBX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, RBX2, 0.0) X(LONGITUDINAL_COEFFICIENTS, RBX3, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, RCX1, 0.0) X(LONGITUDINAL_COEFFICIENTS, REX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, REX2, 0.0) X(LONGITUDINAL_COEFFICIENTS, RHX1, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PTX1, 0.0) X(LONGITUDINAL_COEFFICIENTS, PTX2, 0.0)                             \
    X(LONGITUDINAL_COEFFICIENTS, PTX3, 0.0)                                                                     \
    X(OVERTURNING_COEFFICIENTS, QSX1, 0.0) X(OVERTURNING_COEFFICIENTS, QSX2, 0.0)                               \
    X(OVERTURNING_COEFFICIENTS, QSX3, 0.0) X(OVERTURNING_COEFFICIENTS, QSX4, 0.0)                               \
    X(OVERTURNING_COEFFICIENTS, QSX5, 0.0) X(OVERTURNING_COEFFICIENTS, QSX6, 0.0)                               \
    X(OVERTURNING_COEFFICIENTS, QSX7, 0.0) X(OVERTURNING_COEFFICIENTS, QSX8, 0.0)                               \
    X(OVERTURNING_COEFFICIENTS, QSX9, 0.0) X(OVERTURNING_COEFFICIENTS, QSX10, 0.0)                              \
    X(OVERTURNING_COEFFICIENTS, QSX11, 0.0) X(OVERTURNING_COEFFICIENTS, QSX12, 0.0)                             \
    X(OVERTURNING_COEFFICIENTS, QSX13, 0.0) X(OVERTURNING_COEFFICIENTS, QSX14, 0.0)                             \
    X(OVERTURNING_COEFFICIENTS, PPMX1, 0.0)                                                                     \
    X(LATERAL_COEFFICIENTS, PCY1, 0.0) X(LATERAL_COEFFICIENTS, PDY1, 0.0) X(LATERAL_COEFFICIENTS, PDY2, 0.0)   \
    X(LATERAL_COEFFICIENTS, PDY3, 0.0) X(LATERAL_COEFFICIENTS, PEY1, 0.0) X(LATERAL_COEFFICIENTS, PEY2, 0.0)   \
    X(LATERAL_COEFFICIENTS, PEY3, 0.0) X(LATERAL_COEFFICIENTS, PEY4, 0.0) X(LATERAL_COEFFICIENTS, PEY5, 0.0)   \
    X(LATERAL_COEFFICIENTS, PKY1, 0.0) X(LATERAL_COEFFICIENTS, PKY2, 0.0) X(LATERAL_COEFFICIENTS, PKY3, 0.0)   \
    X(LATERAL_COEFFICIENTS, PKY4, 2.0) X(LATERAL_COEFFICIENTS, PKY5, 0.0) X(LATERAL_COEFFICIENTS, PKY6, 0.0)   \
    X(LATERAL_COEFFICIENTS, PKY7, 0.0) X(LATERAL_COEFFICIENTS, PHY1, 0.0) X(LATERAL_COEFFICIENTS, PHY2, 0.0)   \
    X(LATERAL_COEFFICIENTS, PHY3, 0.0) X(LATERAL_COEFFICIENTS, PVY1, 0.0) X(LATERAL_COEFFICIENTS, PVY2, 0.0)   \
    X(LATERAL_COEFFICIENTS, PVY3, 0.0) X(LATERAL_COEFFICIENTS, PVY4, 0.0) X(LATERAL_COEFFICIENTS, PPY1, 0.0)   \
    X(LATERAL_COEFFICIENTS, PPY2, 0.0) X(LATERAL_COEFFICIENTS, PPY3, 0.0) X(LATERAL_COEFFICIENTS, PPY4, 0.0)   \
    X(LATERAL_COEFFICIENTS, PPY5, 0.0) X(LATERAL_COEFFICIENTS, RBY1, 0.0) X(LATERAL_COEFFICIENTS, RBY2, 0.0)   \
    X(LATERAL_COEFFICIENTS, RBY3, 0.0) X(LATERAL_COEFFICIENTS, RBY4, 0.0) X(LATERAL_COEFFICIENTS, RCY1, 0.0)   \
    X(LATERAL_COEFFICIENTS, REY1, 0.0) X(LATERAL_COEFFICIENTS, REY2, 0.0) X(LATERAL_COEFFICIENTS, RHY1, 0.0)   \
    X(LATERAL_COEFFICIENTS, RHY2, 0.0) X(LATERAL_COEFFICIENTS, RVY1, 0.0) X(LATERAL_COEFFICIENTS, RVY2, 0.0)   \
    X(LATERAL_COEFFICIENTS, RVY3, 0.0) X(LATERAL_COEFFICIENTS, RVY4, 0.0) X(LATERAL_COEFFICIENTS, RVY5, 0.0)   \
    X(LATERAL_COEFFICIENTS, RVY6, 0.0) X(LATERAL_COEFFICIENTS, PTY1, 0.0) X(LATERAL_COEFFICIENTS, PTY2, 0.0)   \
    X(ROLLING_COEFFICIENTS, QSY1, 0.0) X(ROLLING_COEFFICIENTS, QSY2, 0.0) X(ROLLING_COEFFICIENTS, QSY3, 0.0)   \
    X(ROLLING_COEFFICIENTS, QSY4, 0.0) X(ROLLING_COEFFICIENTS, QSY5, 0.0) X(ROLLING_COEFFICIENTS, QSY6, 0.0)   \
    X(ROLLING_COEFFICIENTS, QSY7, 0.0) X(ROLLING_COEFFICIENTS, QSY8, 0.0)                                       \
    X(ALIGNING_COEFFICIENTS, QBZ1, 0.0) X(ALIGNING_COEFFICIENTS, QBZ2, 0.0) X(ALIGNING_COEFFICIENTS, QBZ3, 0.0) \
    X(ALIGNING_COEFFICIENTS, QBZ4, 0.0) X(ALIGNING_COEFFICIENTS, QBZ5, 0.0) X(ALIGNING_COEFFICIENTS, QBZ9, 0.0) \
    X(ALIGNING_COEFFICIENTS, QBZ10, 0.0) X(ALIGNING_COEFFICIENTS, QCZ1, 0.0)                                    \
    X(ALIGNING_COEFFICIENTS, QDZ1, 0.0) X(ALIGNING_COEFFICIENTS, QDZ2, 0.0) X(ALIGNING_COEFFICIENTS, QDZ3, 0.0) \
    X(ALIGNING_COEFFICIENTS, QDZ4, 0.0) X(ALIGNING_COEFFICIENTS, QDZ6, 0.0) X(ALIGNING_COEFFICIENTS, QDZ7, 0.0) \
    X(ALIGNING_COEFFICIENTS, QDZ8, 0.0) X(ALIGNING_COEFFICIENTS, QDZ9, 0.0) X(ALIGNING_COEFFICIENTS, QDZ10, 0.0) \
    X(ALIGNING_COEFFICIENTS, QDZ11, 0.0) X(ALIGNING_COEFFICIENTS, QEZ1, 0.0) X(ALIGNING_COEFFICIENTS, QEZ2, 0.0) \
    X(ALIGNING_COEFFICIENTS, QEZ3, 0.0) X(ALIGNING_COEFFICIENTS, QEZ4, 0.0) X(ALIGNING_COEFFICIENTS, QEZ5, 0.0) \
    X(ALIGNING_COEFFICIENTS, QHZ1, 0.0) X(ALIGNING_COEFFICIENTS, QHZ2, 0.0) X(ALIGNING_COEFFICIENTS, QHZ3, 0.0) \
    X(ALIGNING_COEFFICIENTS, QHZ4, 0.0) X(ALIGNING_COEFFICIENTS, PPZ1, 0.0) X(ALIGNING_COEFFICIENTS, PPZ2, 0.0) \
    X(ALIGNING_COEFFICIENTS, SSZ1, 0.0) X(ALIGNING_COEFFICIENTS, SSZ2, 0.0) X(ALIGNING_COEFFICIENTS, SSZ3, 0.0) \
    X(ALIGNING_COEFFICIENTS, SSZ4, 0.0)

/// A Magic Formula tyre's parameters, named as in a .tir file (SI units,
/// ISO axes). Read one with read_tir (mbd/io/tir_file.hpp), or fill it.
struct MagicFormulaParameters {
    MagicFormulaVersion version{MagicFormulaVersion::MF61};
    int FITTYP{61};   ///< As the file gives it: 6 or 21 for MF 5.2, 61 or 62 for MF 6.1
#define MBD_MF_MEMBER(section, name, value) Real name{value};
    MBD_MAGIC_FORMULA_PARAMETERS(MBD_MF_MEMBER)
#undef MBD_MF_MEMBER
};

/// The tyre's state for one evaluation.
struct MagicFormulaInput {
    Real Fz{0.0};         ///< Normal load [N]; the outputs are zero at Fz <= 0
    Real kappa{0.0};      ///< Longitudinal slip, -Vsx / |Vcx| [-]
    Real alpha{0.0};      ///< Slip angle [rad], tan(alpha) = Vcy / |Vcx|; it enters the formulas as it is
    Real gamma{0.0};      ///< Inclination angle [rad], ISO
    /// Forward speed of the contact centre [m/s]: its sign, and with the slips
    /// the slip speed (Vsx = -kappa |Vcx|, Vcy = tan(alpha) |Vcx|) of LMUV,
    /// and the rolling resistance.
    Real Vcx{10.0};
    Real pressure{-1.0};  ///< Inflation pressure [Pa]; below 0, the file's INFLPRES
};

/// The forces and moments, and the quantities behind them (W axes, at the
/// contact point, on the tyre).
struct MagicFormulaOutput {
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

/// A Magic Formula tyre: evaluate its forces and moments.
class MagicFormulaTyre {
public:
    /// Checks the parameters: FNOMIN, UNLOADED_RADIUS, LONGVL, LFZO and, for
    /// MF 6.1, NOMPRES positive (MBD-F053).
    explicit MagicFormulaTyre(const MagicFormulaParameters& parameters);

    const MagicFormulaParameters& parameters() const { return p_; }

    /// The steady-state forces and moments (combined slip).
    MagicFormulaOutput evaluate(const MagicFormulaInput& in) const;

private:
    MagicFormulaParameters p_;
};

/// The .tir name of a version, for messages: "MF 5.2" or "MF 6.1".
std::string to_string(MagicFormulaVersion version);

} // namespace mbd
