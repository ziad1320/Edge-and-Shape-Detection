cmake_minimum_required(VERSION 3.10)
project(image_lab_backend)

# Use C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 1. ADD PYBIND11 (from submodule folder)
add_subdirectory(pybind11)

# 2. FIND OPENCV (This works with the MSYS2 MinGW install)
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV found at ${OpenCV_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})

# 3. DEFINE THE SOURCE FILES
set(SOURCES
    src/main.cpp
    src/lab_functions.cpp
    src/snake.cpp
    src/shapes.cpp
)

# 4. CREATE THE PYBIND11 MODULE
# This will produce 'image_lab_backend.cp[python_version]-win_amd64.pyd'
pybind11_add_module(image_lab_backend ${SOURCES})

# 5. LINK LIBRARIES
target_link_libraries(image_lab_backend PRIVATE ${OpenCV_LIBS})

# Set output name explicitly to avoid python version suffixes if preferred
set_target_properties(image_lab_backend PROPERTIES PREFIX "")
set_target_properties(image_lab_backend PROPERTIES SUFFIX ".pyd")