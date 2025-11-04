cmake_minimum_required(VERSION 3.14)

project(CodeCpp VERSION 1.0.0 LANGUAGES CXX)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile_commands.json for IDEs
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

# DCMTK - Use vcpkg or system installation
# To install DCMTK via vcpkg: vcpkg install dcmtk
# Or download pre-built binaries from: https://dcmtk.org/download/
find_package(DCMTK QUIET)

if(NOT DCMTK_FOUND)
    message(FATAL_ERROR 
        "DCMTK not found!\n"
        "Please install DCMTK using one of these methods:\n"
        "  1. vcpkg: vcpkg install dcmtk\n"
        "  2. Download from: https://dcmtk.org/download/\n"
        "  3. Set DCMTK_DIR to point to your DCMTK installation"
    )
endif()

message(STATUS "DCMTK found: ${DCMTK_DIR}")
message(STATUS "DCMTK include dirs: ${DCMTK_INCLUDE_DIRS}")
message(STATUS "DCMTK libraries: ${DCMTK_LIBRARIES}")

# OpenCV
find_package(OpenCV REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR 
        "OpenCV not found!\n"
        "Please install OpenCV using one of these methods:\n"
        "  1. Ubuntu/Debian: sudo apt-get install libopencv-dev\n"
        "  2. vcpkg: vcpkg install opencv\n"
        "  3. Build from source: https://opencv.org/releases/"
    )
endif()

message(STATUS "OpenCV found: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# -------------------------------
# Project source
# -------------------------------
add_executable(${PROJECT_NAME}
    main.cpp
    src/logger.cpp
    src/DicomLoader.cpp
    src/TiffMaskLoader.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/inculde)

# Link spdlog
target_link_libraries(${PROJECT_NAME} PRIVATE spdlog::spdlog)

# Link DCMTK
target_link_libraries(${PROJECT_NAME} PRIVATE ${DCMTK_LIBRARIES})
target_include_directories(${PROJECT_NAME} PRIVATE ${DCMTK_INCLUDE_DIRS})

# Link OpenCV (use modern CMake targets if available, otherwise fallback)
if(TARGET opencv_core)
    target_link_libraries(${PROJECT_NAME} PRIVATE opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui)
else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
    target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
endif()

# Filesystem (modern compilers)
if(NOT MSVC)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()

# Optional: Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -pedantic)
endif()

# Custom targets
add_custom_target(run
    COMMAND ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${PROJECT_NAME}
    DEPENDS ${PROJECT_NAME}
    WORKING_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    COMMENT "Running ${PROJECT_NAME}"
)
