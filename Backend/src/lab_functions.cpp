#include <opencv2/opencv.hpp>

// Helper to convert NumPy arrays from Python into OpenCV Mats
// (We will implement the conversion math here when we write the functions)

// Task 1a: Change spatial resolution (Sampling/Interpolation)
// input: OpenCV Mat, new width, new height
cv::Mat changeSpatialResolution(const cv::Mat& inputImage, int newWidth, int newHeight) {
    cv::Mat output;
    if (inputImage.empty()) return inputImage;
    
    // placeholder: just copy for now
    inputImage.copyTo(output); 
    // real OpenCV resize logic goes here later...
    return output;
}

// Task 1b: Change intensity resolution (Quantization)
// input: OpenCV Mat, number of bits (e.g., 1-bit to 7-bit quantization)
cv::Mat quantizeImage(const cv::Mat& inputImage, int numBits) {
    cv::Mat output;
    if (inputImage.empty()) return inputImage;
    inputImage.copyTo(output); 
    // real quantization math goes here later...
    return output;
}

// Task 2: Convert to Grayscale
// input: BGR color image
cv::Mat convertToGrayscale(const cv::Mat& inputImage) {
    cv::Mat output;
    if (inputImage.empty()) return inputImage;
    
    // We can use a real OpenCV function here as a test!
    if (inputImage.channels() == 3) {
        cv::cvtColor(inputImage, output, cv::COLOR_BGR2GRAY);
    } else {
        inputImage.copyTo(output); // Already gray
    }
    return output;
}

// Task 3: Bit-plane slicing
// input: grayscale image, plane index (0-7)
cv::Mat getBitPlane(const cv::Mat& inputImage, int planeIndex) {
    cv::Mat output;
    if (inputImage.empty() || inputImage.channels() > 1) return inputImage;
    inputImage.copyTo(output); 
    // real slicing math goes here later...
    return output;
}