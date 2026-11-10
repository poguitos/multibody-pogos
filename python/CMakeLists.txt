# The Python interface (plan Phase 6, D52): the package `multibody`, its
# Python files from python/multibody/ and the extension module `_mbd` built
# from python/src/, laid out together in <build>/python/multibody/ so that a
# test or a user imports it with PYTHONPATH=<build>/python.

set(MBD_PYTHON_DIR "${CMAKE_BINARY_DIR}/python")
set(MBD_PYTHON_PACKAGE_DIR "${MBD_PYTHON_DIR}/multibody")

pybind11_add_module(_mbd MODULE
    src/module.cpp
    src/bind_core.cpp
    src/bind_model.cpp
    src/bind_constraints.cpp
    src/bind_forces.cpp
    src/bind_simulator.cpp
    src/bind_analysis.cpp
    src/bind_state.cpp
)
target_link_libraries(_mbd PRIVATE multibody_core mbd_warnings)
# The release this build follows: the last tag, raised at each phase gate.
target_compile_definitions(_mbd PRIVATE MBD_VERSION="${PROJECT_VERSION}")
set_target_properties(_mbd PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${MBD_PYTHON_PACKAGE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${MBD_PYTHON_PACKAGE_DIR}"
)

# The package's Python files, copied at every build (a changed file needs no
# reconfiguration).
file(GLOB MBD_PYTHON_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/multibody/*.py")
add_custom_target(mbd_python_files
    COMMAND ${CMAKE_COMMAND} -E make_directory "${MBD_PYTHON_PACKAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${MBD_PYTHON_SOURCES} "${MBD_PYTHON_PACKAGE_DIR}"
    COMMENT "Copying the multibody package's Python files"
)
add_dependencies(_mbd mbd_python_files)

# What pip installs (task 6.2, D53): the module, into the package. The
# package's Python files come from python/multibody/ through pyproject.toml,
# and only this component is installed, so no dependency's install rules run.
install(TARGETS _mbd LIBRARY DESTINATION multibody COMPONENT python)

# Tests: each file in python/tests/ is one CTest test, run by pytest with the
# interpreter the module was built for. They need NumPy and pytest in it
# (python/environment.yml); without them the tests are left out, with a note.
# MBD_TEST_PACKAGE_DIR names the package they must import: a package
# installed with pip install -e into the same interpreter would be imported
# first (its finder precedes PYTHONPATH), and python/tests/conftest.py stops.
if (BUILD_TESTING)
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c "import numpy, pytest"
        RESULT_VARIABLE MBD_PYTHON_TEST_DEPS
        OUTPUT_QUIET ERROR_QUIET
    )
    if (MBD_PYTHON_TEST_DEPS EQUAL 0)
        file(GLOB MBD_PYTHON_TESTS CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.py")
        foreach (test_file IN LISTS MBD_PYTHON_TESTS)
            get_filename_component(test_name "${test_file}" NAME)
            add_test(NAME "Python: ${test_name}"
                COMMAND "${Python_EXECUTABLE}" -B -m pytest -q -p no:cacheprovider "${test_file}"
                WORKING_DIRECTORY "${MBD_PYTHON_DIR}")
            set_tests_properties("Python: ${test_name}" PROPERTIES
                ENVIRONMENT "PYTHONPATH=${MBD_PYTHON_DIR};PYTHONNOUSERSITE=1;MBD_TEST_PACKAGE_DIR=${MBD_PYTHON_PACKAGE_DIR}")
        endforeach()
    else()
        message(STATUS "Python tests left out: ${Python_EXECUTABLE} lacks NumPy or pytest "
                       "(python/environment.yml has both)")
    endif()
endif()
