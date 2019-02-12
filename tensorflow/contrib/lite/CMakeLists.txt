cmake_minimum_required(VERSION 3.6)

# Project
project(tensorflow-lite C CXX)

# Set compilers options
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )

# Define CMAKE options
# option(BUILD_TFL_TESTS "Build unit tests " OFF) # These don't compile in r1.8
option(BUILD_TFL_EXAMPLES "Build Examples" OFF)

# Define used pathes
get_filename_component(TENSORFLOW_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../" ABSOLUTE)
get_filename_component(TENSORFLOW_LITE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/" ABSOLUTE)
set(TENSORFLOW_LITE_KERNEL_DIR "${TENSORFLOW_LITE_ROOT_DIR}/kernels")
set(TENSORFLOW_LITE_DOWNLOADED_DEPENDENCIES_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads")
set(TENSORFLOW_LITE_GEMMLOWP_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads/gemmlowp")
set(TENSORFLOW_LITE_EIGEN_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads/eigen")
set(TENSORFLOW_LITE_FLATBUFFERS_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads/flatbuffers/include")
set(TENSORFLOW_LITE_FARMHASH_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads/farmhash/src")
set(TENSORFLOW_LITE_FFT2D_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads/fft2d")
set(TENSORFLOW_LITE_ABSL_DIR "${TENSORFLOW_LITE_ROOT_DIR}/downloads/absl")

# External dependencies
set(CMAKE_MODULE_PATH ${TENSORFLOW_LITE_ROOT_DIR}/cmake)

set(PLATFORM_ENV_TIME_SRC ${TENSORFLOW_ROOT_DIR}/tensorflow/core/platform/posix/env_time.cc)

add_definitions(
    -DGEMMLOWP_ALLOW_SLOW_SCALAR_FALLBACK
    -DNOMINMAX
    -DFARMHASH_NO_BUILTIN_EXPECT
)

include_directories(
    ${TENSORFLOW_ROOT_DIR}
    ${TENSORFLOW_LITE_GEMMLOWP_DIR}
    ${TENSORFLOW_LITE_FLATBUFFERS_DIR}
    ${TENSORFLOW_LITE_EIGEN_DIR}
    ${TENSORFLOW_LITE_FARMHASH_DIR}
    ${TENSORFLOW_LITE_ABSL_DIR}
)

file(GLOB_RECURSE TENSORFLOW_LITE_KERNEL_SOURCES
    ${TENSORFLOW_LITE_KERNEL_DIR}/*.cc
    ${TENSORFLOW_LITE_KERNEL_DIR}/*.c
)
list(FILTER TENSORFLOW_LITE_KERNEL_SOURCES EXCLUDE REGEX test)

file(GLOB TENSORFLOW_LITE_SOURCES
    ${TENSORFLOW_LITE_ROOT_DIR}/*.cc
    ${TENSORFLOW_LITE_ROOT_DIR}/*.c
)
list(FILTER TENSORFLOW_LITE_SOURCES EXCLUDE REGEX test)

add_library(tensorflow-lite STATIC
    ${TENSORFLOW_LITE_SOURCES}
    ${TENSORFLOW_LITE_KERNEL_SOURCES}
    ${TENSORFLOW_LITE_FARMHASH_DIR}/farmhash.cc
    ${TENSORFLOW_LITE_FFT2D_DIR}/fftsg.c
)

find_package(Threads REQUIRED)
target_link_libraries(tensorflow-lite Threads::Threads)
target_link_libraries(tensorflow-lite ${CMAKE_DL_LIBS})

target_include_directories(tensorflow-lite PUBLIC $<INSTALL_INTERFACE:include/>)

install(
    TARGETS tensorflow-lite
    EXPORT tensorflow_lite_export
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# install necessary headers
# tensorflow lite headers
file(GLOB tf_lite_headers "${TENSORFLOW_LITE_ROOT_DIR}/*.h")
install(
    FILES ${tf_lite_headers}
    DESTINATION include/tensorflow/contrib/lite/
)
install(
    DIRECTORY ${TENSORFLOW_LITE_ROOT_DIR}/profiling
    DESTINATION include/tensorflow/contrib/lite/
    FILES_MATCHING PATTERN "*.h"
)
install(
    DIRECTORY ${TENSORFLOW_LITE_ROOT_DIR}/schema
    DESTINATION include/tensorflow/contrib/lite/
    FILES_MATCHING PATTERN "*.h"
)
install(
    DIRECTORY ${TENSORFLOW_LITE_ROOT_DIR}/kernels
    DESTINATION include/tensorflow/contrib/lite/
    FILES_MATCHING PATTERN "*.h"
)
# flatbuffer directory
install(
    DIRECTORY ${TENSORFLOW_LITE_FLATBUFFERS_DIR}/flatbuffers
    DESTINATION include/
    FILES_MATCHING PATTERN "*.h"
)

# if (BUILD_TFL_TESTS)
#     enable_testing()

#     include(googletest)

#     include_directories(
#         ${googletest_INCLUDE_DIRS}
#         ${googlemock_INCLUDE_DIRS}
#     )

#     add_library(tensorflow_lite_test_util STATIC
#         ${TENSORFLOW_LITE_KERNEL_DIR}/test_util.cc
#         ${TENSORFLOW_ROOT_DIR}/tensorflow/core/platform/default/logging.cc
#         ${TENSORFLOW_ROOT_DIR}/tensorflow/core/platform/env_time.cc
#         ${TENSORFLOW_ROOT_DIR}/tensorflow/core/platform/posix/env_time.cc
#     )

#     add_dependencies(tensorflow_lite_test_util googletest)

#     function(add_tflite_tests test_directory)
#         file(GLOB TEST_SRC_FILES ${test_directory}/*_test.cc )
#         foreach (TEST_SRC_FILE ${TEST_SRC_FILES})
#             string(REPLACE ".cc" "" TEST_NAME ${TEST_SRC_FILE})
#             string(REPLACE "${TENSORFLOW_LITE_ROOT_DIR}/" "" TEST_NAME ${TEST_NAME})
#             string(REPLACE "/" "_" TEST_NAME ${TEST_NAME})
#             add_executable(${TEST_NAME} ${TEST_SRC_FILE})
#             target_link_libraries(${TEST_NAME} debug tensorflow-lite ${googletest_STATIC_LIBRARIES} tensorflow_lite_test_util)
#             target_link_libraries(${TEST_NAME} optimized tensorflow-lite ${googletest_STATIC_LIBRARIES} tensorflow_lite_test_util)
#             if (${TEST_SRC_FILE} STREQUAL "${TENSORFLOW_LITE_KERNEL_DIR}/internal/batch_to_space_nd_test.cc" OR
#                 ${TEST_SRC_FILE} STREQUAL "${TENSORFLOW_LITE_KERNEL_DIR}/internal/tensor_utils_test.cc")
#                 target_link_libraries(${TEST_NAME} debug ${googletest_MAIN_STATIC_LIBRARIES} )
#                 target_link_libraries(${TEST_NAME} optimized ${googletest_MAIN_STATIC_LIBRARIES})
#             endif()
#             add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} WORKING_DIRECTORY ${TENSORFLOW_ROOT_DIR})
#         endforeach()
#     endfunction()

#     add_tflite_tests(${TENSORFLOW_LITE_ROOT_DIR})
#     add_tflite_tests(${TENSORFLOW_LITE_KERNEL_DIR})
#     add_tflite_tests(${TENSORFLOW_LITE_KERNEL_DIR}/internal)
# endif ()

if (BUILD_TFL_EXAMPLES)
    file (GLOB LABEL_IMAGE_SOURCES "${TENSORFLOW_LITE_ROOT_DIR}/examples/label_image/*.cc")
    list(REMOVE_ITEM LABEL_IMAGE_SOURCES "${TENSORFLOW_LITE_ROOT_DIR}/examples/label_image/label_image_test.cc")
    add_executable(label_image ${LABEL_IMAGE_SOURCES})
    target_link_libraries(label_image tensorflow-lite)

    install(TARGETS label_image DESTINATION bin)
endif ()

# Installation and packaging
# ----------------------------------------------------------------------------
# CPack stuff
# ----------------------------------------------------------------------------
include(CPackLists)
