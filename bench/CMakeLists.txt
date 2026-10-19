# Timing programs (plan task 2.10). Not built by default:
#   scripts\build.ps1 -Target mbd_bench
add_executable(mbd_bench EXCLUDE_FROM_ALL bench_dynamics.cpp)
target_link_libraries(mbd_bench PRIVATE multibody_core mbd_warnings)

# The integrators' verification report (plan task 4.9): figures, data and
# tables for docs/integrators.md. Not built by default:
#   scripts\build.ps1 -Target mbd_verify
# It reuses the tests' header-only models (tests/kernel/*.hpp).
add_executable(mbd_verify EXCLUDE_FROM_ALL verify_integrators.cpp)
target_link_libraries(mbd_verify PRIVATE multibody_core mbd_warnings)
target_include_directories(mbd_verify PRIVATE ${PROJECT_SOURCE_DIR}/tests)
