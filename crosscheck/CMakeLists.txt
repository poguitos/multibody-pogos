# Comparison with independent codes (plan task 5.3). Not built by default:
#   scripts\build.ps1 -Target mbd_crosscheck
# The program writes random trees and closed mechanisms with this library's
# answers; the Python scripts beside it compare them with Pinocchio and
# Project Chrono (see README.md). It reuses the tests' header-only
# generators (tests/kernel, tests/support).
add_executable(mbd_crosscheck EXCLUDE_FROM_ALL main.cpp export_trees.cpp export_mechanisms.cpp)
target_link_libraries(mbd_crosscheck PRIVATE multibody_core mbd_warnings)
target_include_directories(mbd_crosscheck PRIVATE ${PROJECT_SOURCE_DIR}/tests)
