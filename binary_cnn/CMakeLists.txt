cmake_minimum_required(VERSION 3.9)
project(Fierra CXX)
# set(CMAKE_CXX_FLAGS "-O3") Comment during debugging
set(CMAKE_CXX_STANDARD 17)
message(STATUS "CMake Version: ${CMAKE_VERSION}")
add_compile_options("-Dusing_cmake")
include_directories(CNN)

add_executable(model Example\ MNIST/unit_tests/fc_forward_xtensor.cpp)

find_package(xtl REQUIRED)
find_package(xtensor REQUIRED)
find_package(xsimd REQUIRED)
find_package(xtensor-blas REQUIRED)

add_definitions(-DHAVE_CBLAS=1)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
# find_package(nlohmann_json 3.2.0 REQUIRED)


message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

target_link_libraries(model ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
# add_library( model LIBRARIES/json.hpp )
if(MSVC)
    target_compile_options(model PRIVATE /EHsc /MP /bigobj)
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
endif()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
    CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
    (CMAKE_CXX_COMPILER_ID MATCHES "Intel" AND NOT WIN32))
    target_compile_options(model PRIVATE -march=native -g -fno-stack-protector -std=c++14)
endif()

target_link_libraries(model xtensor)
