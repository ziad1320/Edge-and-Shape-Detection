#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> // Critical for converting Python lists to C++ vectors
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>

namespace py = pybind11;

// Forward declarations Task 2
py::dict apply_greedy_snake(py::array_t<unsigned char> input_image, std::vector<std::pair<int, int>> initial_points, int iterations, float alpha, float beta, float gamma);

// Forward declarations Task 1 (Now with min/max parameters!)
py::array_t<unsigned char> apply_canny_wrapper(py::array_t<unsigned char> img, int threshold1, int threshold2);
py::array_t<unsigned char> detect_lines_wrapper(py::array_t<unsigned char> img_edges, py::array_t<unsigned char> img_orig, int min_len, int max_len);
py::array_t<unsigned char> detect_circles_wrapper(py::array_t<unsigned char> img_edges, py::array_t<unsigned char> img_orig, int min_r, int max_r);
py::array_t<unsigned char> detect_ellipses_wrapper(py::array_t<unsigned char> img_edges, py::array_t<unsigned char> img_orig, int min_a, int max_a);

// Forward declarations Lab Functions
cv::Mat changeSpatialResolution(const cv::Mat& inputImage, int newWidth, int newHeight);
cv::Mat quantizeImage(const cv::Mat& inputImage, int numBits);
cv::Mat convertToGrayscale(const cv::Mat& inputImage);
cv::Mat getBitPlane(const cv::Mat& inputImage, int planeIndex);

PYBIND11_MODULE(image_lab_backend, m) {
    m.doc() = "Computer Vision Studio Backend";

    m.def("apply_greedy_snake", &apply_greedy_snake, "Greedy Snake algorithm",
          py::arg("input_image"), py::arg("initial_points"), py::arg("iterations") = 100, 
          py::arg("alpha") = 1.0, py::arg("beta") = 1.0, py::arg("gamma") = 1.2);

    m.def("apply_custom_canny", &apply_canny_wrapper, "Apply Canny edge detector from scratch", 
          py::arg("image"), py::arg("threshold1"), py::arg("threshold2"));
          
    // Updated bindings with new arguments
    m.def("detect_lines", &detect_lines_wrapper, "Detect & superimpose lines", 
          py::arg("edges"), py::arg("original"), py::arg("min_len"), py::arg("max_len"));
          
    m.def("detect_circles", &detect_circles_wrapper, "Detect & superimpose circles", 
          py::arg("edges"), py::arg("original"), py::arg("min_r"), py::arg("max_r"));
          
    m.def("detect_ellipses", &detect_ellipses_wrapper, "Detect & superimpose ellipses", 
          py::arg("edges"), py::arg("original"), py::arg("min_a"), py::arg("max_a"));

    m.def("change_spatial_resolution", &changeSpatialResolution, "Changes image resolution",
          py::arg("input_image"), py::arg("new_width"), py::arg("new_height"));
    m.def("quantize_image", &quantizeImage, "Changes intensity resolution",
          py::arg("input_image"), py::arg("num_bits"));
    m.def("convert_to_grayscale", &convertToGrayscale, "Converts to grayscale",
          py::arg("input_image"));
    m.def("get_bit_plane", &getBitPlane, "Extracts bit plane",
          py::arg("input_image"), py::arg("plane_index"));
}