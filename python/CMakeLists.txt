# CMakeLists.txt in python/

pybind11_add_module(pycore
    pycore.cpp
)

target_link_libraries(pycore
    PRIVATE
        multibody_core
)
