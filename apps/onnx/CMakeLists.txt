cmake_minimum_required(VERSION 3.28)
project(onnx)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)
find_package(Protobuf REQUIRED)

# Download onnx.proto
set(ONNX_PROTO_FILE ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto)
if (NOT EXISTS ${ONNX_PROTO_FILE})
    file(DOWNLOAD
        "https://raw.githubusercontent.com/onnx/onnx/v1.4.1/onnx/onnx.proto"
        ${ONNX_PROTO_FILE}
        SHOW_PROGRESS
        STATUS download_status)
    if (NOT download_status EQUAL 0)
        message(FATAL_ERROR "Failed to download onnx.proto: ${download_status}")
    endif ()

    file(READ ${ONNX_PROTO_FILE} onnx_proto_content)
    string(REPLACE "package onnx;" "package onnx;option optimize_for = LITE_RUNTIME;" onnx_proto_content "${onnx_proto_content}")
    file(WRITE ${ONNX_PROTO_FILE} "${onnx_proto_content}")
endif ()

# Generate sources from onnx.proto
add_custom_target(onnx_proto)
protobuf_generate(TARGET onnx_proto 
    LANGUAGE cpp 
    PROTOC_OUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/onnx 
    PROTOC_OUT_VAR onnx_proto_generated_sources 
    PROTOS ${ONNX_PROTO_FILE}
    IMPORT_DIRS ${CMAKE_CURRENT_BINARY_DIR})

# Add library that converts ONNX models to Halide operators
add_library(oclib STATIC onnx_converter.cc ${CMAKE_CURRENT_BINARY_DIR}/onnx/onnx.pb.cc)
target_include_directories(oclib
    PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Protobuf_INCLUDE_DIRS})
target_link_libraries(oclib PUBLIC Halide::Halide ${Protobuf_LITE_LIBRARIES})
target_compile_options(oclib PUBLIC -g -fPIC)
target_compile_definitions(oclib PUBLIC GOOGLE_PROTOBUF_NO_RTTI)

add_executable(onnx_converter_test onnx_converter_test.cc)
target_link_libraries(onnx_converter_test oclib)
add_test(NAME onnx_converter_test COMMAND onnx_converter_test)
set_tests_properties(onnx_converter_test PROPERTIES
                     LABELS onnx
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")

# Generator
add_halide_generator(onnx_converter.generator SOURCES onnx_converter_generator.cc LINK_LIBRARIES oclib)
target_compile_options(onnx_converter.generator PRIVATE -g -fno-rtti)

# Generate test onnx model
add_custom_command(
    OUTPUT test_model.onnx
    COMMAND 
        cat ${CMAKE_CURRENT_SOURCE_DIR}/test_model_proto.txt | protoc --encode=onnx.ModelProto --proto_path=${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto > test_model.onnx
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/onnx.proto
    COMMENT "Generating test ONNX model from proto content"
    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/test_model_proto.txt
    VERBATIM
)
add_custom_target(test_model_onnx DEPENDS test_model.onnx)
add_dependencies(test_model_onnx onnx_proto)

# Generate static library using halide generator for test onnx model
add_halide_library(test_model FROM onnx_converter.generator
                   GENERATOR onnx_model_generator
                   PARAMS model_file_path=${CMAKE_CURRENT_BINARY_DIR}/test_model.onnx
                   AUTOSCHEDULER Halide::Adams2019)
add_dependencies(test_model test_model_onnx)

# Test the generated static library
add_executable(onnx_converter_generator_test onnx_converter_generator_test.cc)
target_link_libraries(onnx_converter_generator_test PRIVATE Halide::Runtime test_model)
add_test(NAME onnx_converter_generator_test COMMAND onnx_converter_generator_test)
set_tests_properties(onnx_converter_generator_test PROPERTIES
                     LABELS onnx
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")

# Python bindings to convert onnx models to Halide model
find_package(Python3 COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
pybind11_add_module(model_cpp model.cpp)
target_link_libraries(model_cpp PRIVATE Halide::Halide oclib)
target_compile_options(model_cpp PRIVATE -g -fPIC -frtti)
target_link_libraries(model_cpp PRIVATE -Wl,--no-as-needed Halide::Adams2019 -Wl,--as-needed)

# Test python bindings
add_test(NAME model_test
    COMMAND ${Python3_EXECUTABLE} -m unittest ${CMAKE_CURRENT_SOURCE_DIR}/model_test.py -v 
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(model_test PROPERTIES
                     LABELS onnx
                     PASS_REGULAR_EXPRESSION "OK"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]"
                     ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:model_cpp>")

# Test python generator
# add_test(NAME halide_as_onnx_backend_test
#          COMMAND ${Python3_EXECUTABLE} -m unittest ${CMAKE_CURRENT_SOURCE_DIR}/halide_as_onnx_backend_test.py -v
#          WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
# set_tests_properties(halide_as_onnx_backend_test PROPERTIES
#                      LABELS onnx
#                      PASS_REGULAR_EXPRESSION "OK"
#                      SKIP_REGULAR_EXPRESSION "\\[SKIP\\]"
#                      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:model_cpp>")


# Compile zivid robocolor denoise onnx model to halide model
set(zivid_robocolor_denoise_model_path /home/vaibhav/zivid/zivid-sdk/vision-machine-learning/workspace/onnx/robocolor_denoise.onnx)
add_halide_library(zivid_robocolor_denoise_model STMT_HTML 
                   FROM onnx_converter.generator
                   GENERATOR onnx_model_generator
                   FEATURES debug
                   # AUTOSCHEDULER Halide::Adams2019
                   PARAMS model_file_path=${zivid_robocolor_denoise_model_path})

add_executable(zivid_robocolor_denoise_test zivid_robocolor_denoise_test.cc)
target_link_libraries(zivid_robocolor_denoise_test PRIVATE Halide::Runtime zivid_robocolor_denoise_model)
add_test(NAME zivid_robocolor_denoise_test COMMAND zivid_robocolor_denoise_test)
set_tests_properties(zivid_robocolor_denoise_test PROPERTIES
                     LABELS onnx
                     PASS_REGULAR_EXPRESSION "Success!"
                     SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
