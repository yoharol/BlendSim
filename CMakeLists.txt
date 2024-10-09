cmake_minimum_required(VERSION 3.20)
project(simTF
        VERSION 0.1
        DESCRIPTION "Portable and Editable Simulation Framework"
        LANGUAGES CXX
)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_definitions(-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)

find_package(TBB REQUIRED)
message("TBB found at: ${TBB_DIR}")

find_package(OpenMP REQUIRED)

if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "OpenMP not found.")
endif()

get_filename_component(ABS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
add_definitions(-DSIMTF_RESOURCES_PATH="${ABS_SOURCE_DIR}/assets")

message("Linking to OpenUSD")
get_property(importTargets DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
find_package(pxr REQUIRED COMPONENTS
    usd
    usdGeom
    usdSkel
    tf
    vt
    gf
    sdf
    arch
    work)
# print out where is the OpenUSD
message("OpenUSD found at: ${pxr_DIR}")
get_property(importTargetsAfter DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY IMPORTED_TARGETS)
list(REMOVE_ITEM importTargetsAfter ${importTargets})
message("Targets from OpenUSD: ${importTargetsAfter}")

add_subdirectory(external)

file(GLOB_RECURSE CORE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/core/*.cpp")

add_library(SIMTF_CORE STATIC)
target_include_directories(SIMTF_CORE PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/core")
target_sources(SIMTF_CORE PUBLIC ${CORE_SOURCES})
target_link_libraries(SIMTF_CORE PUBLIC ArmorerPhys)

find_package(Python3 REQUIRED COMPONENTS Development)
include_directories(${Python3_INCLUDE_DIRS})

set(LIBS 
    SIMTF_CORE 
    ArmorerPhys 
    usd
    usdGeom
    usdSkel
    tf
    vt
    gf
    sdf
    arch
    work)

file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
file(GLOB EXAMPLE_SOURCES "examples/*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PUBLIC ${LIBS})
    target_include_directories(${TEST_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
endforeach()

foreach(EXAMPLE_SOURCE ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SOURCE} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SOURCE})
    target_link_libraries(${EXAMPLE_NAME} PUBLIC ${LIBS})
    target_include_directories(${EXAMPLE_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/core)
endforeach()
