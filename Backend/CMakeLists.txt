cmake_minimum_required(VERSION 3.15)
project(Backend)

# Generate compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Automatically download and include Pybind11
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v3.0.2
)
FetchContent_MakeAvailable(pybind11)

# Build C++ code into a Python module
pybind11_add_module(task2_backend canny_and_shapes_detection.cpp)
pybind11_add_module(image_lab_backend contour_fitting_objects.cpp)

# Link the required libraries
target_include_directories(task2_backend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(task2_backend PRIVATE ${OpenCV_LIBS})

target_include_directories(image_lab_backend PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(image_lab_backend PRIVATE ${OpenCV_LIBS})