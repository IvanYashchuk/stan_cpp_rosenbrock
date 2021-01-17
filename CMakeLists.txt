cmake_minimum_required(VERSION 3.17)

project (rosenbrock)

set (CMAKE_CXX_STANDARD 14)
# set (CMAKE_FIND_DEBUG_MODE 1) # available from 3.17

if (NOT DEFINED ENV{STAN_DIR})
message(FATAL_ERROR "Please set STAN_DIR environment variable.")
endif (NOT DEFINED ENV{STAN_DIR})

message(STATUS "STAN_DIR is set to " $ENV{STAN_DIR})
set (STAN_DIR $ENV{STAN_DIR})

# find_package(StanMath REQUIRED)
add_library(StanMath INTERFACE)
target_include_directories(StanMath INTERFACE
${STAN_DIR}/lib/stan_math/)

# error: "stan-math requires _REENTRANT being defined during
# compilation" "to make lgamma_r available."
target_compile_definitions(StanMath INTERFACE _REENTRANT)

# Find Eigen and link to StanMath
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(StanMath INTERFACE Eigen3::Eigen)

# Find Boost and link to StanMath
find_package(Boost 1.72 REQUIRED NO_MODULE)
target_link_libraries(StanMath INTERFACE Boost::boost)

# Find TBB and link to StanMath
include(${STAN_DIR}/lib/stan_math/lib/tbb_2019_U8/cmake/TBBBuild.cmake)
set (tbb_root ${STAN_DIR}/lib/stan_math/lib/tbb_2019_U8/)
tbb_build(TBB_ROOT ${tbb_root} CONFIG_DIR TBB_DIR)
find_package(TBB REQUIRED NO_MODULE)
target_link_libraries(StanMath INTERFACE TBB::tbb)

# Link Sundials to StanMath
# TODO: here it's assumed that sundials is already built
# In stan-math folder make -f make/standalone math-libs
set (SUNDIALS_DIR ${STAN_DIR}/lib/stan_math/lib/sundials_5.5.0)
add_library(SundialsCvodes STATIC IMPORTED)
SET_TARGET_PROPERTIES(SundialsCvodes PROPERTIES IMPORTED_LOCATION ${SUNDIALS_DIR}/lib/libsundials_cvodes.a)
target_include_directories(SundialsCvodes INTERFACE ${SUNDIALS_DIR}/include)

add_library(SundialsIdas STATIC IMPORTED)
SET_TARGET_PROPERTIES(SundialsIdas PROPERTIES IMPORTED_LOCATION ${SUNDIALS_DIR}/lib/libsundials_idas.a)
target_include_directories(SundialsIdas INTERFACE ${SUNDIALS_DIR}/include)

add_library(SundialsKinsol STATIC IMPORTED)
SET_TARGET_PROPERTIES(SundialsKinsol PROPERTIES IMPORTED_LOCATION ${SUNDIALS_DIR}/lib/libsundials_kinsol.a)
target_include_directories(SundialsKinsol INTERFACE ${SUNDIALS_DIR}/include)

add_library(SundialsNvecserial STATIC IMPORTED)
SET_TARGET_PROPERTIES(SundialsNvecserial PROPERTIES IMPORTED_LOCATION ${SUNDIALS_DIR}/lib/libsundials_nvecserial.a)
target_include_directories(SundialsNvecserial INTERFACE ${SUNDIALS_DIR}/include)

target_link_libraries(StanMath INTERFACE SundialsCvodes SundialsIdas SundialsKinsol SundialsNvecserial)

# find_package(Stan REQUIRED)
add_library(Stan INTERFACE)
target_include_directories(Stan INTERFACE
${STAN_DIR}/src/)

# Link StanMath to Stan
target_link_libraries(Stan INTERFACE StanMath)

add_executable(rosenbrock main.cpp)
target_link_libraries(rosenbrock Stan)
