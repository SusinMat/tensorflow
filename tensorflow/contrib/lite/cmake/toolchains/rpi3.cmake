SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_VERSION 1)

set(__compiler_prefix "arm-linux-gnu-")

# specify the cross compiler
set(CMAKE_C_COMPILER   "${__compiler_prefix}gcc")
set(CMAKE_CXX_COMPILER "${__compiler_prefix}cpp")

set( CMAKE_STRIP        "${__compiler_prefix}strip"   CACHE PATH "strip" )
set( CMAKE_AR           "${__compiler_prefix}ar"      CACHE PATH "archive" )
set( CMAKE_LINKER       "${__compiler_prefix}ld"      CACHE PATH "linker" )
set( CMAKE_NM           "${__compiler_prefix}nm"      CACHE PATH "nm" )
set( CMAKE_OBJCOPY      "${__compiler_prefix}objcopy" CACHE PATH "objcopy" )
set( CMAKE_OBJDUMP      "${__compiler_prefix}objdump" CACHE PATH "objdump" )
set( CMAKE_RANLIB       "${__compiler_prefix}ranlib"  CACHE PATH "ranlib" )

add_definitions(-D_DEBUG)
add_definitions(-D__ARM_NEON)
set(RPI3_EXTRA_FLAGS "${RPI3_EXTRA_FLAGS} -march=armv7-a")
set(RPI3_EXTRA_FLAGS "${RPI3_EXTRA_FLAGS} -mfloat-abi=softfp")
set(RPI3_EXTRA_FLAGS "${RPI3_EXTRA_FLAGS} -mfpu=neon-vfpv4")
set(RPI3_EXTRA_FLAGS "${RPI3_EXTRA_FLAGS} -funsafe-math-optimizations")
set(RPI3_EXTRA_FLAGS "${RPI3_EXTRA_FLAGS} -ftree-vectorize")

set( RPI3_C_FLAGS "${RPI3_C_FLAGS} ${RPI3_EXTRA_FLAGS}")
set( RPI3_CXX_FLAGS "${RPI3_CXX_FLAGS} ${RPI3_EXTRA_FLAGS}")
set( RPI3_FLAGS_RELEASE "-O3" )
set( RPI3_FLAGS_DEBUG "-O0 -g3 -Wall" )

set(CMAKE_C_FLAGS           "${CMAKE_C_FLAGS}           ${RPI3_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS}         ${RPI3_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${RPI3_FLAGS_RELEASE}")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   ${RPI3_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG}   ${RPI3_FLAGS_DEBUG}")
set(CMAKE_C_FLAGS_DEBUG     "${CMAKE_C_FLAGS_DEBUG}     ${RPI3_FLAGS_DEBUG}")
