# CMakeLists.txt in tests/
#
# Tests are grouped into one executable per domain, mirroring the
# include/mbd/ folder structure. ctest discovers all Catch2 cases.

include(Catch)

option(MBD_TEST_PCH
    "Share one precompiled header between the test executables" ON)

# Settings common to every test executable.
add_library(mbd_test_config INTERFACE)
target_link_libraries(mbd_test_config
    INTERFACE
        multibody_core
        Catch2::Catch2WithMain
        mbd_warnings
)
target_include_directories(mbd_test_config
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# The engine is header-only, so without a precompiled header every test file
# parses and analyses all of it again: about 60 s of the 90 s that one file
# takes with MSVC. The header is compiled once here and reused by every test
# executable. Each test file still includes what it uses, so the tests also
# build with -DMBD_TEST_PCH=OFF (the Linux CI job does that to keep the
# includes honest).
if (MBD_TEST_PCH)
    add_library(mbd_test_pch OBJECT support/pch.cpp)
    target_link_libraries(mbd_test_pch PRIVATE mbd_test_config)
    target_precompile_headers(mbd_test_pch PRIVATE support/pch.hpp)
endif()

function(mbd_add_test_exe name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE mbd_test_config)
    if (MBD_TEST_PCH)
        target_precompile_headers(${name} REUSE_FROM mbd_test_pch)
    endif()
    catch_discover_tests(${name})
endfunction()

mbd_add_test_exe(test_core
    test_core.cpp
    test_core_types.cpp
    core/test_math.cpp
    core/test_transform3.cpp
)

mbd_add_test_exe(test_model
    model/test_rigid_body.cpp
    model/test_joints.cpp
    model/test_freejoint_conventions.cpp
    test_rigid_body_dynamics.cpp
    test_forward_kinematics.cpp
    test_revolute_joint.cpp
    test_prismatic_joint.cpp
    test_other_joints.cpp
    test_constraints.cpp
)

mbd_add_test_exe(test_algorithms
    algorithms/test_multibody_basics.cpp
    algorithms/test_multibody_chains.cpp
    test_algorithms.cpp
    test_simulator.cpp
    test_constrained_dynamics.cpp
    test_loop_constraints.cpp
    test_orientation_canonicalization.cpp
)

# Checks every joint and constraint type against finite differences and
# conservation laws. New joints, constraints and forces must pass these before
# anything is built on them (see documentation/Master_plan.md, working rule 2).
mbd_add_test_exe(test_invariants
    invariants/test_joint_invariants.cpp
    invariants/test_constraint_invariants.cpp
    invariants/test_conservation.cpp
)

# The kinematics kernel (plan Phase 2): spatial algebra, the joint models and
# the recursive algorithms, checked against invariants and the legacy path.
mbd_add_test_exe(test_kernel
    kernel/test_spatial.cpp
    kernel/test_kernel_invariants.cpp
    kernel/test_kernel_threads.cpp
    kernel/test_kernel_vs_legacy.cpp
)
find_package(Threads REQUIRED)
target_link_libraries(test_kernel PRIVATE Threads::Threads)

mbd_add_test_exe(test_forces
    forces/test_aerodynamics.cpp
    forces/test_anti_roll_bar.cpp
    forces/test_full_tire.cpp
    forces/test_pacejka.cpp
    test_force_elements.cpp
)

mbd_add_test_exe(test_vehicle
    vehicle/test_dwb_dynamic.cpp
    vehicle/test_dwb_steering.cpp
    vehicle/test_mcpherson.cpp
    vehicle/test_vehicle_template_dynamic.cpp
    test_vehicle.cpp
    test_vehicle_template.cpp
    test_steering.cpp
    test_drivetrain.cpp
    test_suspension_kinematics.cpp
    test_mcpherson_dynamic.cpp
    test_multilink.cpp
    test_quarter_car.cpp
)

mbd_add_test_exe(test_analysis
    analysis/test_track.cpp
    analysis/test_lap_vehicle.cpp
    analysis/test_lap_speed_profile.cpp
    analysis/test_lap_simulation.cpp
    analysis/test_lap_validation.cpp
    test_bicycle_model.cpp
    test_optimization.cpp
)
