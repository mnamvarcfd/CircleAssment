cmake_minimum_required(VERSION 3.14)

project(CodeCpp VERSION 1.0.0 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -------------------------------
# Dependencies
# -------------------------------
include(FetchContent)

# spdlog (header-only)
FetchContent_Declare(
    spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG v1.13.0
)
FetchContent_MakeAvailable(spdlog)

# Google Test (only for unit tests)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# System dependencies
find_package(DCMTK REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# ALGLIB - header-only library
set(ALGLIB_INCLUDE_DIR "/usr/local/include/alglib")
if(NOT EXISTS ${ALGLIB_INCLUDE_DIR})
    message(FATAL_ERROR "ALGLIB headers not found at ${ALGLIB_INCLUDE_DIR}")
endif()

# Create ALGLIB library from source files
file(GLOB ALGLIB_SOURCES "${ALGLIB_INCLUDE_DIR}/*.cpp")
add_library(alglib STATIC ${ALGLIB_SOURCES})

# -------------------------------
# Project source
# -------------------------------
add_executable(${PROJECT_NAME}
    main.cpp
    src/logger.cpp
    src/DicomLoader.cpp
    src/TiffMaskLoader.cpp
    src/ComputeQuantity.cpp
    src/SaveDataManager.cpp
    src/MyocardialBloodFlow.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inculde
    ${ALGLIB_INCLUDE_DIR}
)

# Link dependencies
target_link_libraries(${PROJECT_NAME} PRIVATE
    spdlog::spdlog
    ${DCMTK_LIBRARIES}
    ${OpenCV_LIBS}
    alglib
)

# Filesystem library for old GCC (<9)
if(NOT MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# -------------------------------
# Unit Tests
# -------------------------------
enable_testing()

add_executable(SaveDataManagerTest
    tests/SaveDataManagerTest.cpp
    src/SaveDataManager.cpp
    src/logger.cpp
)

# Include directories for test
target_include_directories(SaveDataManagerTest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/inculde
)

target_link_libraries(SaveDataManagerTest PRIVATE
    GTest::gtest
    GTest::gtest_main
    spdlog::spdlog
    ${OpenCV_LIBS}
)

if(NOT MSVC AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9)
    target_link_libraries(SaveDataManagerTest PRIVATE stdc++fs)
endif()

add_test(NAME SaveDataManagerTest COMMAND SaveDataManagerTest)
