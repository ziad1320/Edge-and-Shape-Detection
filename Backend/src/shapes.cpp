#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace py = pybind11;

cv::Mat numpy_to_mat(py::array_t<unsigned char> input) {
    py::buffer_info buf = input.request();
    int channels = buf.ndim == 3 ? buf.shape[2] : 1;
    int type = channels == 3 ? CV_8UC3 : CV_8UC1;
    return cv::Mat(buf.shape[0], buf.shape[1], type, buf.ptr).clone();
}

py::array_t<unsigned char> mat_to_numpy(cv::Mat img) {
    py::array_t<unsigned char> result;
    if (img.channels() == 3) {
        result = py::array_t<unsigned char>({img.rows, img.cols, 3});
    } else {
        result = py::array_t<unsigned char>({img.rows, img.cols});
    }
    py::buffer_info buf = result.request();
    std::memcpy(buf.ptr, img.data, img.total() * img.elemSize());
    return result;
}

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ==========================================
// --- CANNY EDGE DETECTION (FROM SCRATCH) ---
// ==========================================
class CustomCanny {
public:
  static cv::Mat apply(const cv::Mat &src_gray, int threshold1, int threshold2) {
    cv::Mat blurred = applyGaussianBlur(src_gray);
    cv::Mat grad_mag, grad_dir;
    computeSobelGradients(blurred, grad_mag, grad_dir);
    cv::Mat nms;
    applyNonMaximumSuppression(grad_mag, grad_dir, nms);
    cv::Mat canny_edges;
    applyHysteresis(nms, threshold1, threshold2, canny_edges);
    return canny_edges;
  }
private:
  static cv::Mat applyGaussianBlur(const cv::Mat &src) {
    cv::Mat dst;
    src.convertTo(dst, CV_32F);
    float kernel[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
    float sum = 16.0f;
    cv::Mat result = cv::Mat::zeros(src.size(), CV_32F);
    for (int y = 1; y < src.rows - 1; ++y) {
      for (int x = 1; x < src.cols - 1; ++x) {
        float val = 0.0f;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            val += dst.at<float>(y + ky, x + kx) * (kernel[ky + 1][kx + 1] / sum);
          }
        }
        result.at<float>(y, x) = val;
      }
    }
    return result;
  }
  static void computeSobelGradients(const cv::Mat &blurred, cv::Mat &magnitude, cv::Mat &direction) {
    magnitude = cv::Mat::zeros(blurred.size(), CV_32F);
    direction = cv::Mat::zeros(blurred.size(), CV_32F);
    int gx_kernel[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int gy_kernel[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    for (int y = 1; y < blurred.rows - 1; ++y) {
      for (int x = 1; x < blurred.cols - 1; ++x) {
        float gx = 0, gy = 0;
        for (int ky = -1; ky <= 1; ++ky) {
          for (int kx = -1; kx <= 1; ++kx) {
            float p = blurred.at<float>(y + ky, x + kx);
            gx += p * gx_kernel[ky + 1][kx + 1];
            gy += p * gy_kernel[ky + 1][kx + 1];
          }
        }
        magnitude.at<float>(y, x) = std::sqrt(gx * gx + gy * gy);
        float angle = std::atan2(gy, gx) * 180.0f / M_PI;
        if (angle < 0) angle += 180.0f;
        direction.at<float>(y, x) = angle;
      }
    }
  }
  static void applyNonMaximumSuppression(const cv::Mat &magnitude, const cv::Mat &direction, cv::Mat &nms) {
    nms = cv::Mat::zeros(magnitude.size(), CV_32F);
    for (int y = 1; y < magnitude.rows - 1; ++y) {
      for (int x = 1; x < magnitude.cols - 1; ++x) {
        float q = 255.0f, r = 255.0f;
        float angle = direction.at<float>(y, x);
        if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) {
          q = magnitude.at<float>(y, x + 1); r = magnitude.at<float>(y, x - 1);
        } else if (22.5 <= angle && angle < 67.5) {
          q = magnitude.at<float>(y + 1, x + 1); r = magnitude.at<float>(y - 1, x - 1);
        } else if (67.5 <= angle && angle < 112.5) {
          q = magnitude.at<float>(y + 1, x); r = magnitude.at<float>(y - 1, x);
        } else if (112.5 <= angle && angle < 157.5) {
          q = magnitude.at<float>(y - 1, x + 1); r = magnitude.at<float>(y + 1, x - 1);
        }
        float mag = magnitude.at<float>(y, x);
        if (mag >= q && mag >= r) nms.at<float>(y, x) = mag;
        else nms.at<float>(y, x) = 0.0f;
      }
    }
  }
  static void applyHysteresis(const cv::Mat &nms, int lowThreshold, int highThreshold, cv::Mat &edges) {
    edges = cv::Mat::zeros(nms.size(), CV_8U);
    float highT = std::max(lowThreshold, highThreshold);
    float lowT = std::min(lowThreshold, highThreshold);
    std::vector<cv::Point> strong_edges;
    for (int y = 1; y < nms.rows - 1; ++y) {
      for (int x = 1; x < nms.cols - 1; ++x) {
        if (nms.at<float>(y, x) >= highT) {
          edges.at<uchar>(y, x) = 255;
          strong_edges.push_back(cv::Point(x, y));
        }
      }
    }
    int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    while (!strong_edges.empty()) {
      cv::Point pt = strong_edges.back();
      strong_edges.pop_back();
      for (int i = 0; i < 8; ++i) {
        int nx = pt.x + dx[i], ny = pt.y + dy[i];
        if (nx >= 1 && nx < nms.cols - 1 && ny >= 1 && ny < nms.rows - 1) {
          if (edges.at<uchar>(ny, nx) == 0 && nms.at<float>(ny, nx) >= lowT) {
            edges.at<uchar>(ny, nx) = 255;
            strong_edges.push_back(cv::Point(nx, ny));
          }
        }
      }
    }
  }
};

// ==========================================
// --- SHAPE DETECTION (WITH SIZE LIMITS) ---
// ==========================================
class ShapeDetector {
public:
  struct Line { float rho; float theta; int votes; };
  struct Circle { int x; int y; int r; int votes; };
  struct Ellipse { int x0; int y0; int a; int b; float alpha; int votes; };

  static std::vector<Line> detectLines(const cv::Mat &edges, int min_len, int max_len) {
    std::vector<Line> lines;
    int max_dist = std::hypot(edges.cols, edges.rows);
    int num_theta = 180;
    std::vector<std::vector<int>> accumulator(num_theta, std::vector<int>(2 * max_dist, 0));

    std::vector<float> sins(num_theta), coss(num_theta);
    for (int t = 0; t < num_theta; ++t) {
      float theta = t * M_PI / 180.0f;
      sins[t] = std::sin(theta); coss[t] = std::cos(theta);
    }

    int margin = std::min(5, std::min(edges.rows, edges.cols) / 10);
    for (int y = margin; y < edges.rows - margin; ++y) {
      const uchar *row = edges.ptr<uchar>(y);
      for (int x = margin; x < edges.cols - margin; ++x) {
        if (row[x] > 0) {
          for (int t = 0; t < num_theta; ++t) {
            int rho = std::round(x * coss[t] + y * sins[t]);
            accumulator[t][rho + max_dist]++;
          }
        }
      }
    }

    std::vector<Line> raw_lines;
    int real_min = std::min(min_len, max_len);
    int real_max = std::max(min_len, max_len);

    for (int t = 0; t < num_theta; ++t) {
      for (int r = 0; r < 2 * max_dist; ++r) {
        int v = accumulator[t][r];
        // Only accept lines whose length/votes fall exactly within the user's sliders!
        if (v >= real_min && v <= real_max) {
          raw_lines.push_back({(float)(r - max_dist), (float)(t * M_PI / 180.0f), v});
        }
      }
    }

    std::sort(raw_lines.begin(), raw_lines.end(), [](const Line &a, const Line &b) { return a.votes > b.votes; });
    for (const auto &cand : raw_lines) {
      bool keep = true;
      for (const auto &existing : lines) {
        float d_rho = std::abs(cand.rho - existing.rho);
        float d_theta = std::abs(cand.theta - existing.theta);
        d_theta = std::min(d_theta, (float)(M_PI - d_theta));
        if (d_rho < 10.0f && d_theta < 0.08f) { keep = false; break; }
      }
      if (keep) {
        lines.push_back(cand);
        if (lines.size() >= 50) break;
      }
    }
    return lines;
  }

  static std::vector<Circle> detectCircles(const cv::Mat &edges, int min_radius, int max_radius) {
    std::vector<Circle> raw_circles;
    int width = edges.cols, height = edges.rows;
    std::vector<int> acc(width * height, 0);

    // Filter math by user constraints
    int real_min = std::max(5, std::min(min_radius, max_radius));
    int real_max = std::min(std::min(width, height) / 2, std::max(min_radius, max_radius));
    if (real_min > real_max) return raw_circles;

    std::vector<std::pair<int, int>> edge_pts;
    for (int y = 0; y < height; ++y) {
      const uchar *row = edges.ptr<uchar>(y);
      for (int x = 0; x < width; ++x) {
        if (row[x] > 0) edge_pts.push_back({x, y});
      }
    }

    for (int r = real_min; r <= real_max; r += 2) {
      std::fill(acc.begin(), acc.end(), 0); 
      std::vector<std::pair<int, int>> circle_pts;
      int x_c = r, y_c = 0, err = 0;
      while (x_c >= y_c) {
        circle_pts.push_back({x_c, y_c}); circle_pts.push_back({y_c, x_c});
        circle_pts.push_back({-x_c, y_c}); circle_pts.push_back({-y_c, x_c});
        circle_pts.push_back({x_c, -y_c}); circle_pts.push_back({y_c, -x_c});
        circle_pts.push_back({-x_c, -y_c}); circle_pts.push_back({-y_c, -x_c});
        y_c += 1; err += 1 + 2 * y_c;
        if (2 * (err - x_c) + 1 > 0) { x_c -= 1; err += 1 - 2 * x_c; }
      }
      std::sort(circle_pts.begin(), circle_pts.end());
      circle_pts.erase(std::unique(circle_pts.begin(), circle_pts.end()), circle_pts.end());

      for (const auto &ep : edge_pts) {
        for (const auto &pt : circle_pts) {
          int a = ep.first - pt.first, b = ep.second - pt.second;
          if (a >= 0 && a < width && b >= 0 && b < height) acc[b * width + a]++;
        }
      }

      float ratio = 0.65f - 0.20f * ((float)(r - real_min) / (real_max - real_min + 1));
      ratio = std::max(0.45f, ratio);
      int min_v = std::max(40, (int)(circle_pts.size() * ratio));

      for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
          int v = acc[y * width + x];
          if (v >= min_v) {
            bool is_max = true;
            for (int dy = -1; dy <= 1; ++dy) {
              for (int dx = -1; dx <= 1; ++dx) {
                if (dx == 0 && dy == 0) continue;
                int ny = y + dy, nx = x + dx;
                if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                  int neighbor_v = acc[ny * width + nx];
                  if (neighbor_v > v || (neighbor_v == v && (dy < 0 || (dy == 0 && dx < 0)))) {
                    is_max = false; break;
                  }
                }
              }
              if (!is_max) break;
            }
            if (is_max) raw_circles.push_back({x, y, r, v});
          }
        }
      }
    }

    std::sort(raw_circles.begin(), raw_circles.end(), [](const Circle &a, const Circle &b) { return a.votes > b.votes; });
    std::vector<Circle> circles;
    for (const auto &c : raw_circles) {
      bool keep = true;
      for (const auto &existing : circles) {
        float dist = std::hypot(c.x - existing.x, c.y - existing.y);
        if (dist < 20.0f && std::abs(c.r - existing.r) < 15) { keep = false; break; }
      }
      if (keep) {
        circles.push_back(c);
        if (circles.size() >= 50) break;
      }
    }
    return circles;
  }

  static std::vector<Ellipse> detectEllipses(const cv::Mat &edges, int min_a, int max_a) {
    std::vector<Ellipse> raw_ellipses;
    std::vector<cv::Point> edge_pts;
    for (int y = 0; y < edges.rows; ++y) {
      const uchar *row = edges.ptr<uchar>(y);
      for (int x = 0; x < edges.cols; ++x) {
        if (row[x] > 0) edge_pts.push_back(cv::Point(x, y));
      }
    }
    int N = edge_pts.size();
    if (N < 20) return raw_ellipses;

    int real_minA = std::max(10, std::min(min_a, max_a));
    int real_maxA = std::max(min_a, max_a);

    srand(42);
    int max_attempts = 100000;
    int attempts = 0;

    while (attempts < max_attempts) {
      attempts++;
      int i = rand() % N; int j = rand() % N;
      if (i == j) continue;

      const auto &p1 = edge_pts[i], &p2 = edge_pts[j];
      float d_sq = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y);
      float a = std::sqrt(d_sq) / 2.0f;
      
      // Filter out ellipses that fall outside user's boundaries
      if (a < real_minA || a > real_maxA || a > edges.cols / 2.0f) continue;

      float x0 = (p1.x + p2.x) / 2.0f, y0 = (p1.y + p2.y) / 2.0f;
      float alpha = std::atan2(p2.y - p1.y, p2.x - p1.x);

      int max_b = std::round(a);
      if (max_b < 10) continue; 

      std::vector<int> b_acc(max_b + 1, 0);
      for (int k = 0; k < N; k++) {
        if (k == i || k == j) continue;
        const auto &p3 = edge_pts[k];
        float d_p3_sq = (p3.x - x0) * (p3.x - x0) + (p3.y - y0) * (p3.y - y0);
        float d_p3 = std::sqrt(d_p3_sq);
        if (d_p3 > a) continue;

        float d_focus = std::abs((p2.y - p1.y) * p3.x - (p2.x - p1.x) * p3.y + p2.x * p1.y - p2.y * p1.x) / (2.0f * a);
        if (d_focus == 0 && d_p3 > 0) continue;
        float denom = std::abs(a * a - (d_p3_sq - d_focus * d_focus)) + 1e-4f;
        float b_sq = (a * a * d_focus * d_focus) / denom;

        if (b_sq > 0) {
          int b = std::round(std::sqrt(b_sq));
          if (b > 0 && b <= max_b) b_acc[b]++;
        }
      }

      int max_votes = 0, best_b = 0;
      for (int b = 1; b <= max_b; ++b) {
        if (b_acc[b] > max_votes) { max_votes = b_acc[b]; best_b = b; }
      }

      int dynamic_thresh = std::max(15, (int)(a * 0.35f));
      if (max_votes > dynamic_thresh && best_b >= std::max(10, (int)(a * 0.15f))) {
        float expected_perimeter = M_PI * (3 * (a + best_b) - std::sqrt((3 * a + best_b) * (a + 3 * best_b)));
        int total_pts = std::round(expected_perimeter);
        int hit_count = 0;

        for (int step = 0; step < total_pts; step++) {
            float theta = (2.0f * M_PI * step) / total_pts;
            int ex = std::round(x0 + a * std::cos(theta) * std::cos(alpha) - best_b * std::sin(theta) * std::sin(alpha));
            int ey = std::round(y0 + a * std::cos(theta) * std::sin(alpha) + best_b * std::sin(theta) * std::cos(alpha));
            
            if (ex >= 0 && ex < edges.cols && ey >= 0 && ey < edges.rows) {
                if (edges.at<uchar>(ey, ex) > 0) hit_count++;
                else {
                    bool found = false;
                    for(int dy = -1; dy <= 1 && !found; dy++) {
                        for(int dx = -1; dx <= 1 && !found; dx++) {
                            int nx = ex + dx, ny = ey + dy;
                            if (nx >= 0 && nx < edges.cols && ny >= 0 && ny < edges.rows) {
                                if (edges.at<uchar>(ny, nx) > 0) found = true;
                            }
                        }
                    }
                    if (found) hit_count++;
                }
            }
        }
        float ratio = 0.70f - 0.20f * (a / (edges.cols / 2.0f));
        ratio = std::max(0.50f, ratio);
        if (hit_count > total_pts * ratio) {
            raw_ellipses.push_back({(int)x0, (int)y0, (int)a, best_b, alpha, hit_count});
        }
      }
    }

    std::sort(raw_ellipses.begin(), raw_ellipses.end(), [](const Ellipse &e1, const Ellipse &e2) { return e1.votes > e2.votes; });
    std::vector<Ellipse> ellipses;
    for (const auto &e : raw_ellipses) {
      bool keep = true;
      for (const auto &existing : ellipses) {
        float dist = std::hypot(e.x0 - existing.x0, e.y0 - existing.y0);
        float da = std::abs(e.a - existing.a);
        float db = std::abs(e.b - existing.b);
        if (dist < 30.0f && da < 30.0f && db < 30.0f) { keep = false; break; }
      }
      if (keep) { ellipses.push_back(e); if (ellipses.size() > 50) break; }
    }
    return ellipses;
  }
};

// ==========================================
// --- PYBIND11 WRAPPER EXPORT            ---
// ==========================================

py::array_t<unsigned char> apply_canny_wrapper(py::array_t<unsigned char> img, int threshold1, int threshold2) {
  auto src = numpy_to_mat(img);
  cv::Mat gray;
  if (src.channels() == 3) cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
  else gray = src.clone();
  cv::Mat edges = CustomCanny::apply(gray, threshold1, threshold2);
  return mat_to_numpy(edges);
}

py::array_t<unsigned char> detect_lines_wrapper(py::array_t<unsigned char> img_edges, py::array_t<unsigned char> img_orig, int min_len, int max_len) {
  auto edges = numpy_to_mat(img_edges);
  auto orig = numpy_to_mat(img_orig);
  cv::Mat result;
  if (orig.channels() == 1) cv::cvtColor(orig, result, cv::COLOR_GRAY2BGR);
  else result = orig.clone();

  auto lines = ShapeDetector::detectLines(edges, min_len, max_len);
  for (const auto &l : lines) {
    float a = std::cos(l.theta), b = std::sin(l.theta);
    float x0 = a * l.rho, y0 = b * l.rho;
    cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
    cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
    cv::line(result, pt1, pt2, cv::Scalar(0, 0, 255), 2);
  }
  return mat_to_numpy(result);
}

py::array_t<unsigned char> detect_circles_wrapper(py::array_t<unsigned char> img_edges, py::array_t<unsigned char> img_orig, int min_r, int max_r) {
  auto edges = numpy_to_mat(img_edges);
  auto orig = numpy_to_mat(img_orig);
  cv::Mat result;
  if (orig.channels() == 1) cv::cvtColor(orig, result, cv::COLOR_GRAY2BGR);
  else result = orig.clone();

  auto circles = ShapeDetector::detectCircles(edges, min_r, max_r);
  for (const auto &c : circles) {
    cv::circle(result, cv::Point(c.x, c.y), c.r, cv::Scalar(0, 255, 0), 2);
    cv::circle(result, cv::Point(c.x, c.y), 2, cv::Scalar(255, 0, 0), 3);
  }
  return mat_to_numpy(result);
}

py::array_t<unsigned char> detect_ellipses_wrapper(py::array_t<unsigned char> img_edges, py::array_t<unsigned char> img_orig, int min_a, int max_a) {
  auto edges = numpy_to_mat(img_edges);
  auto orig = numpy_to_mat(img_orig);
  cv::Mat result;
  if (orig.channels() == 1) cv::cvtColor(orig, result, cv::COLOR_GRAY2BGR);
  else result = orig.clone();

  auto ellipses = ShapeDetector::detectEllipses(edges, min_a, max_a);
  for (const auto &e : ellipses) {
    cv::ellipse(result, cv::Point(e.x0, e.y0), cv::Size(e.a, e.b), e.alpha * 180.0 / M_PI, 0, 360, cv::Scalar(255, 0, 255), 2);
    cv::circle(result, cv::Point(e.x0, e.y0), 2, cv::Scalar(255, 0, 0), 3);
  }
  return mat_to_numpy(result);
}