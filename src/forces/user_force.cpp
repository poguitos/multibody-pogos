#include "mbd/forces/user_force.hpp"

#include <utility>

namespace mbd {

UserForce::UserForce(std::string name, std::vector<BodyIndex> bodies, Function function)
    : name_(std::move(name)), bodies_(std::move(bodies)), function_(std::move(function))
{
    MBD_THROW_IF(!function_, "MBD-F004: UserForce: no function given");
}

} // namespace mbd
