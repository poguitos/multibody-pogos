# CMakeLists.txt in tests/

add_executable(test_core
    test_core.cpp
    test_math.cpp
    test_core_types.cpp
)

target_link_libraries(test_core
    PRIVATE
        multibody_core
        Catch2::Catch2WithMain
)

include(Catch)
catch_discover_tests(test_core)

