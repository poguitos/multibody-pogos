# CMakeLists.txt in tests/

add_executable(test_core
    test_core.cpp
    test_math.cpp
    test_core_types.cpp
    test_rigid_body.cpp
    test_rigid_body_dynamics.cpp
    test_transform3.cpp
    test_force_elements.cpp
    test_constraints.cpp
    test_joints.cpp
    test_forward_kinematics.cpp
    test_algorithms.cpp
    test_simulator.cpp
    test_constrained_dynamics.cpp
    test_quarter_car.cpp
    test_pacejka.cpp
    test_full_tire.cpp
    test_vehicle.cpp
    test_suspension_kinematics.cpp
    test_mcpherson.cpp
    test_multilink.cpp
    test_steering.cpp
    test_drivetrain.cpp
    test_optimization.cpp
    test_bicycle_model.cpp
)

target_link_libraries(test_core
    PRIVATE
        multibody_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(test_core)


