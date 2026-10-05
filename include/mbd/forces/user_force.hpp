#pragma once

// A force element given by a function (plan task 3.1): anything that reads
// the bodies' world states and adds world forces at their origins. Time and
// generalized forces go through Simulator::force_callback instead. A user
// force states no derivatives; statics and linearisation take them by finite
// differences (decision D24).

#include <functional>
#include <string>
#include <vector>

#include "mbd/forces/force_element.hpp"

namespace mbd {

class UserForce : public ForceElement {
public:
    using Function = std::function<void(const std::vector<RigidBodyState>& states,
                                        std::vector<RigidBodyForces>& forces)>;

    /// `bodies` lists the bodies the function acts on, for the checks.
    UserForce(std::string name, std::vector<BodyIndex> bodies, Function function);

    const char* name() const override { return name_.c_str(); }
    std::vector<BodyIndex> bodies() const override { return bodies_; }

    void apply(const std::vector<RigidBodyState>& states,
               std::vector<RigidBodyForces>& forces) const override
    {
        function_(states, forces);
    }

private:
    std::string name_;
    std::vector<BodyIndex> bodies_;
    Function function_;
};

} // namespace mbd
