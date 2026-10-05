#pragma once

// Counts heap allocations made through the global operator new, which the
// test_alloc executable replaces (alloc_counter.cpp). Eigen allocates with
// malloc instead; a build with EIGEN_RUNTIME_NO_MALLOC catches those with
// Eigen's own check.

namespace mbd_test {

/// Start counting, from zero.
void start_counting_allocations();

/// Stop counting; returns the allocations made since the start.
long stop_counting_allocations();

} // namespace mbd_test
