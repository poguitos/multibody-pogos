#include "checks.hpp"

#include <string>

namespace mbd::kernel::checks {

void size_error(const char* function, const char* argument, Index size, const char* dimension,
                int expected)
{
    throw MbdError("MBD-K001: " + std::string(function) + ": " + argument + " has " + std::to_string(size)
                   + " entries, but the model has " + dimension + " = " + std::to_string(expected)
                   + ".");
}

void data_error(const char* function, const Model& model, const Data& data)
{
    throw MbdError("MBD-K002: " + std::string(function) + ": the Data was made for another model ("
                   + std::to_string(data.oMi.size()) + " bodies, nv = "
                   + std::to_string(data.zero_v.size()) + "; this model has "
                   + std::to_string(model.nbodies()) + " bodies, nv = " + std::to_string(model.nv)
                   + ").");
}

void body_error(const char* function, int body, int bodies)
{
    throw MbdError("MBD-K003: " + std::string(function) + ": there is no body " + std::to_string(body)
                   + "; the bodies are 0 to " + std::to_string(bodies - 1) + ".");
}

} // namespace mbd::kernel::checks
