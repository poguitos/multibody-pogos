#include "mbd/forces/tyre_model.hpp"

namespace mbd {

TyreOutput TyreModel::characteristics(Real Fz, Real gamma, Real pressure, Real friction) const
{
    TyreInput in;
    in.Fz = Fz;
    in.gamma = gamma;
    in.pressure = pressure;
    in.friction = friction;
    in.kappa = 0.0;
    in.alpha = 0.0;
    in.Vcx = 0.0;
    return evaluate(in);
}

} // namespace mbd
