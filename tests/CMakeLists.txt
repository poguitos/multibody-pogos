# CMakeLists.txt in tests/
#
# Tests are grouped into one executable per domain, mirroring the
# include/mbd/ folder structure. ctest discovers all Catch2 cases.

include(Catch)

function(mbd_add_test_exe name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name}
        PRIVATE
            multibody_core
            Catch2::Catch2WithMain
    )
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
