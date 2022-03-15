project(UserTesting LANGUAGES C CXX)

find_package(Python3 REQUIRED)
add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/include/reprod.h
        ${CMAKE_CURRENT_BINARY_DIR}/reprod.cc
        ${CMAKE_CURRENT_BINARY_DIR}/reprod.ll
        ${CMAKE_CURRENT_BINARY_DIR}/reprod.o
        ${CMAKE_CURRENT_BINARY_DIR}/halide_runtime.o
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/reprod.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${CMAKE_CURRENT_LIST_DIR}/reprod.py Halide_Python
)

add_library(ReprodRuntime OBJECT IMPORTED)
set_target_properties(ReprodRuntime PROPERTIES IMPORTED_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/halide_runtime.o)
target_include_directories(ReprodRuntime INTERFACE ${CMAKE_BINARY_DIR}/include)

add_library(ReprodLLVM OBJECT IMPORTED)
set_target_properties(ReprodLLVM PROPERTIES IMPORTED_OBJECTS ${CMAKE_CURRENT_BINARY_DIR}/reprod.o)
target_include_directories(ReprodLLVM INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/include)

add_library(ReprodCC OBJECT ${CMAKE_CURRENT_BINARY_DIR}/reprod.cc)
target_include_directories(ReprodCC PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/include)
set_target_properties(ReprodCC PROPERTIES C_STANDARD 11 CXX_STANDARD 17)

add_executable(ReprodMain main.cpp)
set_property(TARGET ReprodMain PROPERTY CXX_STANDARD 17)
target_link_libraries(ReprodMain PUBLIC ReprodRuntime ReprodLLVM)
