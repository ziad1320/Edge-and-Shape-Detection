#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>
#include <limits>
#include <algorithm>

namespace py = pybind11;

py::dict apply_greedy_snake(py::array_t<unsigned char> input_image, std::vector<std::pair<int, int>> initial_points, int iterations, float alpha, float beta, float gamma) {
    
    // 1. Convert Python Numpy array to OpenCV Mat
    py::buffer_info buf = input_image.request();
    cv::Mat img(buf.shape[0], buf.shape[1], CV_8UC1, buf.ptr);

    // 2. Precompute the Edge Map (External Energy) using Sobel filters
    cv::Mat grad_x, grad_y, edge_map;
    cv::Sobel(img, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(img, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, edge_map);
    
    // Normalize edge map to 0.0 - 1.0 range
    double minVal, maxVal;
    cv::minMaxLoc(edge_map, &minVal, &maxVal);
    if (maxVal > 0) edge_map = edge_map / maxVal;

    // 3. Setup Contour Points
    std::vector<cv::Point> contour;
    for (const auto& p : initial_points) {
        contour.push_back(cv::Point(p.first, p.second));
    }

    int n = contour.size();
    int window_size = 3; // 3x3 search neighborhood
    int offset = window_size / 2;

    // ==========================================
    // --- 4. GREEDY ALGORITHM EVOLUTION LOOP ---
    // ==========================================
    for (int iter = 0; iter < iterations; iter++) {
        
        // Calculate the current average distance between points
        float avg_dist = 0;
        for (int i = 0; i < n; i++) {
            cv::Point p1 = contour[i];
            cv::Point p2 = contour[(i + 1) % n];
            avg_dist += std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
        }
        avg_dist /= n;

        int points_moved = 0;

        for (int i = 0; i < n; i++) {
            cv::Point prev = contour[(i - 1 + n) % n];
            cv::Point curr = contour[i];
            cv::Point next = contour[(i + 1) % n];

            float min_energy = std::numeric_limits<float>::max();
            cv::Point best_point = curr;

            // Arrays to hold neighborhood energies for normalization
            std::vector<float> e_cont(9, 0), e_smooth(9, 0), e_ext(9, 0);
            std::vector<cv::Point> neighbors(9);

            float max_c = 0.0001f, max_s = 0.0001f, max_e = 0.0001f;
            float min_c = 99999.f, min_s = 99999.f, min_e = 99999.f;

            int idx = 0;
            // First Pass: Calculate raw energies in the 3x3 window
            for (int dy = -offset; dy <= offset; dy++) {
                for (int dx = -offset; dx <= offset; dx++) {
                    cv::Point v(curr.x + dx, curr.y + dy);
                    neighbors[idx] = v;

                    // Bounds checking
                    if (v.x < 0 || v.x >= img.cols || v.y < 0 || v.y >= img.rows) {
                        idx++; continue;
                    }

                    // E_cont: Continuity
                    float dist = std::sqrt(std::pow(prev.x - v.x, 2) + std::pow(prev.y - v.y, 2));
                    e_cont[idx] = std::abs(avg_dist - dist);
                    max_c = std::max(max_c, e_cont[idx]); min_c = std::min(min_c, e_cont[idx]);

                    // E_smooth: Curvature
                    e_smooth[idx] = std::pow(prev.x - 2 * v.x + next.x, 2) + std::pow(prev.y - 2 * v.y + next.y, 2);
                    max_s = std::max(max_s, e_smooth[idx]); min_s = std::min(min_s, e_smooth[idx]);

                    // E_ext: Image Edge Strength (Negative because we want minimum energy at strong edges)
                    e_ext[idx] = -edge_map.at<float>(v.y, v.x);
                    max_e = std::max(max_e, e_ext[idx]); min_e = std::min(min_e, e_ext[idx]);

                    idx++;
                }
            }

            // Second Pass: Normalize (0 to 1) and find the lowest total energy
            idx = 0;
            for (int dy = -offset; dy <= offset; dy++) {
                for (int dx = -offset; dx <= offset; dx++) {
                    cv::Point v = neighbors[idx];
                    if (v.x >= 0 && v.x < img.cols && v.y >= 0 && v.y < img.rows) {
                        
                        float n_cont = (e_cont[idx] - min_c) / (max_c - min_c + 1e-5);
                        float n_smooth = (e_smooth[idx] - min_s) / (max_s - min_s + 1e-5);
                        float n_ext = (e_ext[idx] - min_e) / (max_e - min_e + 1e-5);

                        float total_energy = (alpha * n_cont) + (beta * n_smooth) + (gamma * n_ext);

                        if (total_energy < min_energy) {
                            min_energy = total_energy;
                            best_point = v;
                        }
                    }
                    idx++;
                }
            }

            if (best_point != curr) {
                contour[i] = best_point;
                points_moved++;
            }
        }
        // Early stop if the snake has wrapped around the object and stopped moving
        if (points_moved == 0) break; 
    }

    // ==========================================
    // --- 5. METRICS CALCULATION ---
    // ==========================================
    
    // Calculate Area
    double area = 0.0;
    if (n > 2) {
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            area += (contour[i].x * contour[j].y) - (contour[j].x * contour[i].y);
        }
        area = std::abs(area) / 2.0;
    }

    // Calculate Perimeter
    double perimeter = 0.0;
    if (n > 1) {
        for (int i = 0; i < n; i++) {
            int j = (i + 1) % n;
            perimeter += cv::norm(contour[i] - contour[j]);
        }
    }

    // Calculate Freeman Chain Code (8-Directional)
    std::string chain_code = "";
    for (int i = 0; i < n; i++) {
        cv::Point curr = contour[i];
        cv::Point next = contour[(i + 1) % n];
        
        float dx = next.x - curr.x;
        float dy = next.y - curr.y;
        
        // Calculate angle and map it to 0-7 directions
        float ang_rad = std::atan2(dy, dx); 
        if (ang_rad < 0) ang_rad += 2 * CV_PI;
        int code = std::round(ang_rad / (CV_PI / 4.0));
        chain_code += std::to_string(code % 8);
    }

    // Convert to Python lists and return
    std::vector<std::pair<int, int>> final_points;
    for (const auto& p : contour) {
        final_points.push_back({p.x, p.y});
    }

    py::dict results;
    results["points"] = final_points;
    results["area"] = area;
    results["perimeter"] = perimeter;
    results["chain_code"] = chain_code;
    return results;
}