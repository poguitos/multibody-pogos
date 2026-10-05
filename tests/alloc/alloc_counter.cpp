// Replacements of the global operator new and delete for the test_alloc
// executable: plain malloc and free, with a count of the allocations made
// while counting is on.

#include "alloc/alloc_counter.hpp"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(_MSC_VER)
#include <malloc.h>
#endif

namespace {

std::atomic<bool> g_counting{false};
std::atomic<long> g_count{0};

void count()
{
    if (g_counting.load(std::memory_order_relaxed)) g_count.fetch_add(1, std::memory_order_relaxed);
}

void* allocate(std::size_t n)
{
    count();
    if (void* p = std::malloc(n == 0 ? 1 : n)) return p;
    throw std::bad_alloc();
}

void* allocate_aligned(std::size_t n, std::align_val_t alignment)
{
    count();
    const std::size_t a = static_cast<std::size_t>(alignment);
    const std::size_t size = ((n == 0 ? 1 : n) + a - 1) / a * a;
#if defined(_MSC_VER)
    if (void* p = _aligned_malloc(size, a)) return p;
#else
    if (void* p = std::aligned_alloc(a, size)) return p;
#endif
    throw std::bad_alloc();
}

void release_aligned(void* p) noexcept
{
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

} // namespace

namespace mbd_test {

void start_counting_allocations()
{
    g_count.store(0);
    g_counting.store(true);
}

long stop_counting_allocations()
{
    g_counting.store(false);
    return g_count.load();
}

} // namespace mbd_test

void* operator new(std::size_t n) { return allocate(n); }
void* operator new[](std::size_t n) { return allocate(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }

void* operator new(std::size_t n, std::align_val_t a) { return allocate_aligned(n, a); }
void* operator new[](std::size_t n, std::align_val_t a) { return allocate_aligned(n, a); }
void operator delete(void* p, std::align_val_t) noexcept { release_aligned(p); }
void operator delete[](void* p, std::align_val_t) noexcept { release_aligned(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { release_aligned(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { release_aligned(p); }
