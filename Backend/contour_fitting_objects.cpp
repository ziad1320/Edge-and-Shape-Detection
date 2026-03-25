// image_lab_backend.cpp
// Combined module: Active Contours (snake) + Lab Functions (resolution, quantization, etc.)

#include "binding_utils.h"
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <algorithm>

namespace py = pybind11;

// ==========================================
// --- ACTIVE CONTOURS (GREEDY SNAKE)     ---
// ==========================================

static inline double dist2(const cv::Point& a, const cv::Point& b) {
    double dx = a.x - b.x, dy = a.y - b.y;
    return dx*dx + dy*dy;
}

static double e_cont(const cv::Point& prev, const cv::Point& curr, double d_avg) {
    double d = std::sqrt(dist2(prev, curr));
    double diff = d - d_avg;
    return diff * diff;
}

static double e_curv(const cv::Point& prev, const cv::Point& curr, const cv::Point& next) {
    double ux = curr.x - prev.x, uy = curr.y - prev.y;
    double vx = next.x - curr.x, vy = next.y - curr.y;
    double cx = vx - ux, cy = vy - uy;
    return cx*cx + cy*cy;
}

static double e_image(const cv::Mat& grad_mag, const cv::Point& p) {
    if (p.x < 0 || p.y < 0 || p.x >= grad_mag.cols || p.y >= grad_mag.rows)
        return 1e9;
    return -static_cast<double>(grad_mag.at<float>(p.y, p.x));
}

cv::Mat gradientMag(const cv::Mat& gray) {
    cv::Mat blur, gx, gy, mag;
    cv::GaussianBlur(gray, blur, {5,5}, 1.5);
    cv::Sobel(blur, gx, CV_32F, 1, 0, 3);
    cv::Sobel(blur, gy, CV_32F, 0, 1, 3);
    cv::magnitude(gx, gy, mag);
    cv::normalize(mag, mag, 0, 1, cv::NORM_MINMAX);
    return mag;
}

// --- Chain Code Utilities ---

static const int DX[8] = { 1,  1,  0, -1, -1, -1,  0,  1 };
static const int DY[8] = { 0, -1, -1, -1,  0,  1,  1,  1 };

std::vector<int> chainCode(const std::vector<cv::Point>& pts) {
    const int N = static_cast<int>(pts.size());
    std::vector<int> code(N);
    for (int i = 0; i < N; i++) {
        int nx = pts[(i+1)%N].x - pts[i].x;
        int ny = pts[(i+1)%N].y - pts[i].y;
        if (nx >  1) nx =  1; if (nx < -1) nx = -1;
        if (ny >  1) ny =  1; if (ny < -1) ny = -1;
        int dir = 0;
        for (int d = 0; d < 8; d++)
            if (DX[d] == nx && DY[d] == ny) { dir = d; break; }
        code[i] = dir;
    }
    return code;
}

double perimeterFromCode(const std::vector<int>& code) {
    double p = 0;
    for (int c : code) p += (c % 2 == 0) ? 1.0 : std::sqrt(2.0);
    return p;
}

double areaFromChainCode(const std::vector<cv::Point>& pts) {
    const int N = static_cast<int>(pts.size());
    double A = 0;
    for (int i = 0; i < N; i++) {
        const auto& p0 = pts[i];
        const auto& p1 = pts[(i+1)%N];
        A += static_cast<double>(p0.x) * p1.y - static_cast<double>(p1.x) * p0.y;
    }
    return std::abs(A) * 0.5;
}

// --- Main Snake Function ---

py::dict apply_greedy_snake(py::array_t<unsigned char> input_image, 
                            std::vector<std::pair<int, int>> initial_points, 
                            float min_displacement, float alpha, float beta, float gamma) {
    
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC1, buf.ptr);

    cv::Mat grad = gradientMag(img);

    std::vector<cv::Point> initial_poly;
    for (const auto& p : initial_points) {
        initial_poly.push_back(cv::Point(p.first, p.second));
    }
    int n_init = initial_poly.size();
    if (n_init == 0) return py::dict();

    double total_peri = 0.0;
    std::vector<double> accum_dist(n_init + 1, 0.0);
    for (int i = 0; i < n_init; i++) {
        cv::Point p1 = initial_poly[i];
        cv::Point p2 = initial_poly[(i + 1) % n_init];
        total_peri += std::sqrt(dist2(p1, p2));
        accum_dist[i + 1] = total_peri;
    }

    int num_points = std::max(500, (int)(total_peri / 3.0));
    double spacing = total_peri / num_points;

    std::vector<cv::Point> pts;
    int current_segment = 0;
    for (int i = 0; i < num_points; i++) {
        double target_dist = i * spacing;
        while (current_segment < n_init && target_dist > accum_dist[current_segment + 1]) {
            current_segment++;
        }
        if (current_segment >= n_init) {
            pts.push_back(initial_poly[0]);
            continue;
        }
        double segment_start_dist = accum_dist[current_segment];
        double segment_len = accum_dist[current_segment + 1] - accum_dist[current_segment];
        double t = (segment_len > 0) ? (target_dist - segment_start_dist) / segment_len : 0;
        cv::Point p1 = initial_poly[current_segment];
        cv::Point p2 = initial_poly[(current_segment + 1) % n_init];
        pts.push_back(cv::Point(p1.x + t * (p2.x - p1.x), p1.y + t * (p2.y - p1.y)));
    }

    std::vector<std::vector<std::pair<int, int>>> history;
    auto record_history = [&]() {
        std::vector<std::pair<int, int>> current_state;
        for (const auto& p : pts) current_state.push_back({p.x, p.y});
        history.push_back(current_state);
    };
    record_history();

    int half = 1;
    int max_iter = 400;
    int actual_iters = 0;
    const int N = pts.size();

    for (int it = 0; it < max_iter; it++) {
        actual_iters++;
        double d_sum = 0;
        for (int i = 0; i < N; i++) d_sum += std::sqrt(dist2(pts[i], pts[(i+1)%N]));
        double d_avg = d_sum / N;

        int moved_points = 0;

        for (int i = 0; i < N; i++) {
            const cv::Point& prev = pts[(i - 1 + N) % N];
            const cv::Point& next = pts[(i + 1) % N];

            double best_e = 1e18;
            cv::Point best_p = pts[i];

            for (int dy = -half; dy <= half; dy++) {
                for (int dx = -half; dx <= half; dx++) {
                    cv::Point candidate(pts[i].x + dx, pts[i].y + dy);

                    double ec = alpha * e_cont(prev, candidate, d_avg);
                    double eb = beta  * e_curv(prev, candidate, next);
                    double ei = gamma * e_image(grad, candidate);

                    double e_total = ec + eb + ei;
                    if (e_total < best_e) {
                        best_e  = e_total;
                        best_p  = candidate;
                    }
                }
            }

            if (best_p != pts[i]) {
                pts[i] = best_p;
                moved_points++;
            }
        }
        
        record_history();
        float tolerance_ratio = (min_displacement < 1.0f && min_displacement > 0.0f) ? min_displacement : 0.0f;
        int min_pts_continue = std::max(0, (int)(N * tolerance_ratio));

        if (moved_points <= min_pts_continue) break;
    }

    auto code = chainCode(pts);
    double perim = perimeterFromCode(code);
    double area = areaFromChainCode(pts);

    std::string chain_str = "";
    for (int c : code) chain_str += std::to_string(c);

    py::dict results;
    results["points"] = history.back();
    results["history"] = history;
    results["area"] = area;
    results["perimeter"] = perim;
    results["chain_code"] = chain_str;
    results["iterations_run"] = actual_iters;

    return results;
}

// ==========================================
// --- LAB FUNCTIONS                      ---
// ==========================================

// Task 1a: Change spatial resolution (Sampling/Interpolation)
cv::Mat changeSpatialResolution(const cv::Mat& inputImage, int newWidth, int newHeight) {
    cv::Mat output;
    if (inputImage.empty()) return inputImage;
    inputImage.copyTo(output); 
    return output;
}

// Task 1b: Change intensity resolution (Quantization)
cv::Mat quantizeImage(const cv::Mat& inputImage, int numBits) {
    cv::Mat output;
    if (inputImage.empty()) return inputImage;
    inputImage.copyTo(output); 
    return output;
}

// Task 2: Convert to Grayscale
cv::Mat convertToGrayscale(const cv::Mat& inputImage) {
    cv::Mat output;
    if (inputImage.empty()) return inputImage;
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, output, cv::COLOR_BGR2GRAY);
    } else {
        inputImage.copyTo(output);
    }
    return output;
}

// Task 3: Bit-plane slicing
cv::Mat getBitPlane(const cv::Mat& inputImage, int planeIndex) {
    cv::Mat output;
    if (inputImage.empty() || inputImage.channels() > 1) return inputImage;
    inputImage.copyTo(output); 
    return output;
}

// ==========================================
// --- LAB FUNCTION WRAPPERS              ---
// ==========================================

py::array_t<unsigned char> change_spatial_resolution_wrapper(py::array_t<unsigned char> img, int newWidth, int newHeight) {
    auto src = numpy_to_mat(img);
    cv::Mat result = changeSpatialResolution(src, newWidth, newHeight);
    return mat_to_numpy(result);
}

py::array_t<unsigned char> quantize_image_wrapper(py::array_t<unsigned char> img, int numBits) {
    auto src = numpy_to_mat(img);
    cv::Mat result = quantizeImage(src, numBits);
    return mat_to_numpy(result);
}

py::array_t<unsigned char> convert_to_grayscale_wrapper(py::array_t<unsigned char> img) {
    auto src = numpy_to_mat(img);
    cv::Mat result = convertToGrayscale(src);
    return mat_to_numpy(result);
}

py::array_t<unsigned char> get_bit_plane_wrapper(py::array_t<unsigned char> img, int planeIndex) {
    auto src = numpy_to_mat(img);
    cv::Mat result = getBitPlane(src, planeIndex);
    return mat_to_numpy(result);
}

// ==========================================
// --- PYBIND11 MODULE EXPORT             ---
// ==========================================

PYBIND11_MODULE(image_lab_backend, m) {
    m.doc() = "Image Lab Backend: Active Contours (Snake) + Lab Functions";

    // Active Contours
    m.def("apply_greedy_snake", &apply_greedy_snake,
          "Run greedy snake active contour on a grayscale image",
          py::arg("image"), py::arg("initial_points"),
          py::arg("min_displacement"), py::arg("alpha"),
          py::arg("beta"), py::arg("gamma"));

    // Lab Functions
    m.def("change_spatial_resolution", &change_spatial_resolution_wrapper,
          "Change spatial resolution of an image",
          py::arg("image"), py::arg("new_width"), py::arg("new_height"));

    m.def("quantize_image", &quantize_image_wrapper,
          "Quantize image intensity levels",
          py::arg("image"), py::arg("num_bits"));

    m.def("convert_to_grayscale", &convert_to_grayscale_wrapper,
          "Convert image to grayscale",
          py::arg("image"));

    m.def("get_bit_plane", &get_bit_plane_wrapper,
          "Extract a specific bit plane from a grayscale image",
          py::arg("image"), py::arg("plane_index"));
}
