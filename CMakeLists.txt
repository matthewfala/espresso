project(task3)

cmake_minimum_required (VERSION 2.6)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -g -Wall --std=c++0x")

file(GLOB SOURCES "./Espresso/*.cpp" "databatcher_test.cpp" "./Espresso/DataBatcher.hpp")

add_executable(task3 ${SOURCES})
