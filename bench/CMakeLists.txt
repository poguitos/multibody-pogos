# Timing programs (plan task 2.10). Not built by default:
#   scripts\build.ps1 -Target mbd_bench
add_executable(mbd_bench EXCLUDE_FROM_ALL bench_dynamics.cpp)
target_link_libraries(mbd_bench PRIVATE multibody_core mbd_warnings)
