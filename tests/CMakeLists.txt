# CMakeLists.txt in tests/
#
# Tests are grouped into one executable per domain, mirroring the
# include/mbd/ folder structure. ctest discovers all Catch2 cases.

include(Catch)

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

# Without a precompiled header every test file parses Catch2, Eigen and the
# engine's headers again. The header is compiled once here and reused by
# every test executable. Each test file still includes what it uses, so the
# tests also build with -DMBD_PCH=OFF (the Linux CI job does that to keep the
# includes honest).
if (MBD_PCH)
    add_library(mbd_test_pch OBJECT support/pch.cpp)
    target_link_libraries(mbd_test_pch PRIVATE mbd_test_config)
    target_precompile_headers(mbd_test_pch PRIVATE support/pch.hpp)
endif()

function(mbd_add_test_exe name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE mbd_test_config)
    if (MBD_PCH)
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
)

# The multibody kernel (plan Phase 2): spatial algebra, joint models, recursive
# algorithms, constraints, the constrained solve and the simulator. Every joint
# and constraint type is checked against finite differences, conservation laws
# and closed-form solutions; new joints, constraints and forces must pass these
# before anything is built on them (see documentation/Master_plan.md, working
# rule 2).
mbd_add_test_exe(test_kernel
    kernel/test_spatial.cpp
    kernel/test_kernel_analytic.cpp
    kernel/test_kernel_invariants.cpp
    kernel/test_kernel_threads.cpp
    kernel/test_kernel_constraints.cpp
    kernel/test_kernel_constrained.cpp
    kernel/test_kernel_simulator.cpp
    kernel/test_kernel_joints_physics.cpp
    kernel/test_kernel_validate.cpp
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
