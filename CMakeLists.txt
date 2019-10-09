project(task3)

# Require CMAKE 3.1 or higher
#cmake_minimum_required(VERSION 3.1)


cmake_minimum_required (VERSION 2.6)

#set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++0x") # -Wall -g

# Build for C++14
#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB SOURCES "./Espresso/*.cpp" "databatcher_test.cpp")

add_executable(task3 ${SOURCES})
