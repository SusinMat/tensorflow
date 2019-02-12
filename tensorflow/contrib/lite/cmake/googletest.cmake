# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
include (ExternalProject)

set(googletest_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(googlemock_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/include)
set(googletest_URL https://github.com/google/googletest.git)
set(googletest_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest/)
set(googletest_TAG ec44c6c1675c25b9827aacd08c02433cccde7780)

if(WIN32)
  if(${CMAKE_GENERATOR} MATCHES "Visual Studio.*")
    set(googletest_MAIN_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/$(Configuration)/gtest_main.lib)
    set(googletest_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/$(Configuration)/gtest.lib)
  else()
    set(googletest_MAIN_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/gtest_main.lib)
    set(googletest_STATIC_LIBRARIES
        ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/gtest.lib)
  endif()
else()
  set(googletest_MAIN_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/gtest/libgtest_main.a)
  set(googletest_STATIC_LIBRARIES
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/gtest/libgtest.a
      ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googlemock/libgmock.a)
endif()

ExternalProject_Add(googletest
    PREFIX googletest
    GIT_REPOSITORY ${googletest_URL}
    GIT_TAG ${googletest_TAG}
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    BUILD_BYPRODUCTS ${googletest_STATIC_LIBRARIES} ${googletest_MAIN_STATIC_LIBRARIES}
    #PATCH_COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_SOURCE_DIR}/patches/grpc/CMakeLists.txt ${GRPC_BUILD}
    INSTALL_COMMAND ""
    CMAKE_ARGS
       -DCMAKE_C_COMPILER:STRING=${CMAKE_C_COMPILER}
       -DCMAKE_CXX_COMPILER:STRING=${CMAKE_CXX_COMPILER}
       -DCMAKE_C_FLAGS:STRING=${CMAKE_C_FLAGS}
       -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS}
    CMAKE_CACHE_ARGS
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DBUILD_GMOCK:BOOL=ON
        -DBUILD_GTEST:BOOL=ON
        -Dgtest_force_shared_crt:BOOL=ON
)
