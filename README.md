# 👁️ Computer Vision Studio: Edge & Shape Detection

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![C++](https://img.shields.io/badge/C++-17-00599C.svg)
![PyQt6](https://img.shields.io/badge/UI-PyQt6-41CD52.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8.svg)
![Pybind11](https://img.shields.io/badge/Binding-Pybind11-red.svg)

A high-performance, mixed-language desktop application for advanced image processing. This project features a sleek, dark-themed **PyQt6 frontend** connected to a lightning-fast, custom-built **C++ backend**. 

**Note:** To demonstrate mathematical understanding, core computer vision algorithms (such as Canny edge detection, Hough transforms, and Active Contours) were implemented mathematically from scratch in C++, bypassing high-level OpenCV library functions.

---

## ✨ Features

### 🛠️ Interactive UI
* **Studio-Grade Interface:** A responsive, dark-themed GUI inspired by professional editing software.
* **Smart Canvas:** Clickable image labels with dynamic aspect-ratio scaling and precise pixel mapping.
* **Live Terminal:** Built-in scrolling text console for real-time algorithm metrics and outputs.
* **Dynamic Sliders:** Real-time parameter tuning for all mathematical thresholds and limits.

### 📐 Task 1: Shape Detection (Hough Transform)
* **Custom Canny Edge Detector:** Implements Gaussian blur, Sobel gradients, Non-Maximum Suppression, and Hysteresis thresholding from scratch.
* **Line Detection:** Tracks accumulator votes to superimpose dominant linear structures.
* **Circle Detection:** Uses a 1D flat accumulator and the Midpoint Circle Algorithm to detect radial shapes within customizable radius bounds.
* **Ellipse Detection:** Implements randomized RANSAC-style pair sampling (inspired by Xie & Ji) to detect complex elliptical orbits and shapes.

### 🐍 Task 2: Active Contour Model (Greedy Snake)
* **Interactive Initialization:** Draw a custom bounding shape directly onto the image canvas using the mouse.
* **Greedy Evolution:** Calculates Continuity, Smoothness, and Image Gradient energies to snap the contour to the object's boundaries.
* **Data Extraction:** Automatically calculates the **Area** (Shoelace Formula) and **Perimeter** of the final detected object.
* **Chain Code Generation:** Converts the final 2D vector boundary into an 8-directional Freeman Chain Code string.

---

## 🏗️ Architecture
This project utilizes **Pybind11** to bridge Python and C++. 
* **`Frontend/`**: Contains the `front.py` PyQt6 application.
* **`Backend/src/`**: Contains the raw C++ mathematics and Pybind11 wrapper modules.
* **`run.py`**: A custom launcher script that dynamically links the compiled C++ `.pyd` binaries to the Python path at runtime.

---

## 🚀 Installation & Setup

### Prerequisites
* Python 3.8+
* C++ Compiler (MinGW64 recommended for Windows)
* CMake (3.10+)

### 1. Install Python Dependencies
It is recommended to use a virtual environment.
```bash
pip install PyQt6 numpy opencv-python
```

### 2. Compile the C++ Backend
Navigate to the backend folder, configure CMake, and build the module.
```bash
cd Backend
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
```

### 3. Run the Application
Return to the root directory and launch the app using the wrapper script.
```bash
cd ../..
python run.py
```