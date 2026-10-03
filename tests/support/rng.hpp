#pragma once

// Deterministic random numbers for tests: the same sequence on every platform
// and compiler, unlike the distributions of <random>.

#include <cstdint>

#include "mbd/core/math.hpp"

namespace mbd_test {

class Rng {
public:
    explicit Rng(std::uint64_t seed) : s_(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}

    /// Uniform in [0, 1).
    mbd::Real unit()
    {
        s_ ^= s_ >> 12;
        s_ ^= s_ << 25;
        s_ ^= s_ >> 27;
        return mbd::Real((s_ * 0x2545F4914F6CDD1Dull) >> 11) / mbd::Real(9007199254740992.0);
    }

    mbd::Real range(mbd::Real lo, mbd::Real hi) { return lo + (hi - lo) * unit(); }

    mbd::Vec3 vec(mbd::Real magnitude)
    {
        const mbd::Real x = range(-magnitude, magnitude);
        const mbd::Real y = range(-magnitude, magnitude);
        const mbd::Real z = range(-magnitude, magnitude);
        return mbd::Vec3(x, y, z);
    }

    mbd::Vec3 direction()
    {
        mbd::Vec3 d = vec(1.0);
        if (d.norm() < mbd::Real(1e-3)) d = mbd::Vec3::UnitX();
        return d.normalized();
    }

    /// A frame with a generic orientation and an offset of the given size.
    mbd::Transform3 frame(mbd::Real offset)
    {
        const mbd::Vec3 axis = direction();
        const mbd::Real angle = range(-1.2, 1.2);
        return mbd::Transform3(mbd::Quat(Eigen::AngleAxisd(angle, axis)), vec(offset));
    }

    /// A generic unit quaternion, any rotation angle.
    mbd::Quat quat()
    {
        const mbd::Vec3 axis = direction();
        return mbd::Quat(Eigen::AngleAxisd(range(-3.0, 3.0), axis));
    }

private:
    std::uint64_t s_;
};

} // namespace mbd_test
