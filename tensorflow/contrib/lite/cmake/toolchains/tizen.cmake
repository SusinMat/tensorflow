#  Tizen CMake toolchain file
#  Usage Linux:
#   $ mkdir build && cd build
#   $ cmake -DCMAKE_TOOLCHAIN_FILE=path/to/the/tizen.cmake ..
#   $ make -j8

set (TIZEN_SDK "$ENV{HOME}/tizen-studio" CACHE PATH "Location of tizen studio.")
if ( TIZEN_SDK )
  message(STATUS "Tizen SDK path: ${TIZEN_SDK}")
else()
  message( FATAL_ERROR "Could not find Tizen SDK path."
    "Probably you forgot to specify it! Define the variable: TIZEN_SDK")
endif ()

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_VERSION 1)

set(TIZEN_TARGET "mobile-4.0" CACHE STRING "Tizen device")
set_property(CACHE TIZEN_TARGET PROPERTY STRINGS "iot-headless-4.0;mobile-4.0;wearable-3.0")

option(TIZEN_USE_LLVM "Enable build with LLVM + GCC" OFF)
option(TIZEN_DEVICE "Build for Tizen device" ON)

message(STATUS "Tizen with LLVM + GCC: ${TIZEN_USE_LLVM}")
message(STATUS "Tizen Target: ${TIZEN_TARGET}")
message(STATUS "Tizen SDK: ${TIZEN_SDK}")
message(STATUS "Tizen Device: ${TIZEN_DEVICE}")

STRING(REGEX REPLACE "([^ ]*)-([^ ]*)" "\\1" __tizen_platform ${TIZEN_TARGET})
STRING(REGEX REPLACE "([^ ]*)-([^ ]*)" "\\2" __tizen_platform_version ${TIZEN_TARGET})

if (TIZEN_TARGET STREQUAL "wearable-3.0")
  if (TIZEN_DEVICE)
    set(__tizen_toolchain_prefix "arm-linux-gnueabi-")
    set(__tizen_toolchain "${__tizen_toolchain_prefix}gcc-4.9")
  else () # Emulator
    set(__tizen_toolchain_prefix "i386-linux-gnueabi-")
    set(__tizen_toolchain "${__tizen_toolchain_prefix}gcc-4.9")
  endif()
else () # iot-headless-4.0;mobile-4.0
  if (TIZEN_DEVICE)
    set(__tizen_toolchain_prefix "arm-linux-gnueabi-")
    set(__tizen_toolchain "${__tizen_toolchain_prefix}gcc-6.2")
  else () # Emulator
    set(__tizen_toolchain_prefix "i586-linux-gnueabi-")
    set(__tizen_toolchain "${__tizen_toolchain_prefix}gcc-6.2")
  endif()
endif()

set(__tizen_toolchain_root "${TIZEN_SDK}/tools/${__tizen_toolchain}")

if (TIZEN_USE_LLVM)
  set(CMAKE_C_COMPILER   "${TIZEN_SDK}/tools/llvm-4.0.0/bin/clang")
  set(CMAKE_CXX_COMPILER "${TIZEN_SDK}/tools/llvm-4.0.0/bin/clang++")
else ()
  set( CMAKE_C_COMPILER   "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}gcc"     CACHE PATH "c compiler")
  set( CMAKE_CXX_COMPILER "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}g++"     CACHE PATH "c++ compiler")
endif ()

set( CMAKE_STRIP        "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}strip"   CACHE PATH "strip" )
set( CMAKE_AR           "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}ar"      CACHE PATH "archive" )
set( CMAKE_LINKER       "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}ld"      CACHE PATH "linker" )
set( CMAKE_NM           "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}nm"      CACHE PATH "nm" )
set( CMAKE_OBJCOPY      "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}objcopy" CACHE PATH "objcopy" )
set( CMAKE_OBJDUMP      "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}objdump" CACHE PATH "objdump" )
set( CMAKE_RANLIB       "${__tizen_toolchain_root}/bin/${__tizen_toolchain_prefix}ranlib"  CACHE PATH "ranlib" )


set(__tizen_rootstraps "${TIZEN_SDK}/platforms/tizen-${__tizen_platform_version}/${__tizen_platform}/rootstraps")

if (TIZEN_DEVICE)
  set(TIZEN_SYSROOT "${__tizen_rootstraps}/${TIZEN_TARGET}-device.core"  CACHE PATH "" FORCE)
else () # Emulator
  set(TIZEN_SYSROOT "${__tizen_rootstraps}/${TIZEN_TARGET}-emulator.core"  CACHE PATH "" FORCE)
endif()

#set(CMAKE_FIND_ROOT_PATH ${TIZEN_SYSROOT})
#set(CMAKE_SYSROOT ${TIZEN_SYSROOT})

#message(STATUS "SYSROOT: ${CMAKE_SYSROOT}")
message(STATUS "tizen SYSROOT: ${TIZEN_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH "${TIZEN_SYSROOT}" "${__tizen_toolchain_root}/bin")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

if (TIZEN_USE_LLVM)
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -gcc-toolchain \"${__tizen_toolchain_root}\"")
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -ccc-gcc-name ${__tizen_toolchain_prefix}g++")
endif ()

add_definitions(-DTIZEN_DEPRECATION)
add_definitions(-DDEPRECATION_WARNING)

# common flags
set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} --sysroot=${TIZEN_SYSROOT}")
set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -fmessage-length=0")
set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -Wno-gnu")
set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -fPIE")

# specific flags
if (TIZEN_DEVICE)
  add_definitions(-D_DEBUG)
  add_definitions(-D__ARM_NEON)
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -march=armv7-a")
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -mfloat-abi=softfp")
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -mfpu=neon-vfpv4")
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -mtune=cortex-a8")
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -mthumb")
  if (TIZEN_USE_LLVM)
    set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -target arm-tizen-linux-gnueabi")
  endif ()
else ()
  set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -march=i586")

  if (TIZEN_USE_LLVM)
    set(TIZEN_EXTRA_FLAGS "${TIZEN_EXTRA_FLAGS} -target i586-tizen-linux-gnueabi")
  endif()
endif ()

set( TIZEN_C_FLAGS "${TIZEN_C_FLAGS} ${TIZEN_EXTRA_FLAGS}")
set( TIZEN_CXX_FLAGS "${TIZEN_CXX_FLAGS} ${TIZEN_EXTRA_FLAGS}")
set( TIZEN_FLAGS_RELEASE "-O3" )
set( TIZEN_FLAGS_DEBUG "-O0 -g3 -Wall" )

set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS}           ${TIZEN_CXX_FLAGS}"     CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         ${TIZEN_CXX_FLAGS}"     CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${TIZEN_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   ${TIZEN_FLAGS_RELEASE}" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   ${TIZEN_FLAGS_DEBUG}"   CACHE STRING "" FORCE)
set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     ${TIZEN_FLAGS_DEBUG}"   CACHE STRING "" FORCE)
