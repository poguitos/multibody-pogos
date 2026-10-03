// Model / Data split (plan task 2.3): one Model shared read-only by several
// threads, each with its own Data, gives the same results as one thread.

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <thread>

#include "mbd/kernel/algorithms.hpp"

#include "kernel/kernel_helpers.hpp"

using namespace mbd;
using namespace mbd::kernel;
using mbd_test::KJoint;
using mbd_test::Rng;

TEST_CASE("Kernel: threads sharing one Model get identical results", "[kernel][threads]")
{
    Rng rng(31);
    const Model model = mbd_test::random_tree(rng, KJoint::Free, mbd_test::all_kernel_joints());
    const VecX q0 = mbd_test::random_configuration(model, rng);
    const VecX v0 = mbd_test::random_vector(model.nv, rng, 1.0);

    auto simulate = [&model, &q0, &v0](VecX& q, VecX& v) {
        Data data(model);
        q = q0;
        v = v0;
        for (int step = 0; step < 500; ++step) mbd_test::rk4_step(model, data, q, v, 1e-3);
    };

    VecX qa, va, qb, vb, q_ref, v_ref;
    std::thread a(simulate, std::ref(qa), std::ref(va));
    std::thread b(simulate, std::ref(qb), std::ref(vb));
    a.join();
    b.join();
    simulate(q_ref, v_ref);

    // Bit for bit: the threads share nothing they write.
    CHECK(qa == q_ref);
    CHECK(va == v_ref);
    CHECK(qb == q_ref);
    CHECK(vb == v_ref);
}
