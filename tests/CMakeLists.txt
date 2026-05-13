# CMakeLists.txt in tests/

add_executable(test_core
    # test_core.cpp
    # test_math.cpp
    # test_core_types.cpp
    # test_rigid_body.cpp
    # test_rigid_body_dynamics.cpp
    # test_transform3.cpp
    # test_force_elements.cpp
    # test_constraints.cpp
    # test_joints.cpp
    # test_forward_kinematics.cpp
    # test_algorithms.cpp
    # test_simulator.cpp
    # test_constrained_dynamics.cpp
    # test_quarter_car.cpp
    # test_pacejka.cpp
    # test_full_tire.cpp
    # test_vehicle.cpp
    # test_suspension_kinematics.cpp
    # test_mcpherson.cpp
    # test_multilink.cpp
    # test_steering.cpp
    # test_drivetrain.cpp
    # test_optimization.cpp
    # test_bicycle_model.cpp
    # test_vehicle_template.cpp
    # test_dwb_dynamic.cpp
    # test_mcpherson_dynamic.cpp
    # test_vehicle_template_dynamic.cpp
    # test_anti_roll_bar.cpp
    # test_dwb_steering.cpp
    # test_aerodynamics.cpp
    # test_track.cpp
    # test_lap_vehicle.cpp
    # test_lap_speed_profile.cpp
    # test_lap_simulation.cpp
    # test_lap_validation.cpp
    # test_freejoint_conventions.cpp
    test_multibody_basics.cpp
    test_revolute_joint.cpp
    test_prismatic_joint.cpp
    test_other_joints.cpp
    test_multibody_chains.cpp
    test_loop_constraints.cpp
)

target_link_libraries(test_core
    PRIVATE
        multibody_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(test_core)


