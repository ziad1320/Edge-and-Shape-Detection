import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QFileDialog,
                             QTextEdit, QSplitter, QSlider, QSizePolicy, QScrollArea)
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtCore import Qt

try:
    import image_lab_backend # type: ignore
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"Backend load error: {e}")

class ViewerLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("ImageDisplay")
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.original_pixmap = None

    def set_image(self, img_array):
        h, w = img_array.shape[:2]
        c = img_array.shape[2] if len(img_array.shape) == 3 else 1
        bytes_per_line = c * w
        if c == 3:
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            qimg = QImage(img_array.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        
        self.original_pixmap = QPixmap.fromImage(qimg)
        self.setPixmap(self.original_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        if self.original_pixmap and not self.original_pixmap.isNull():
            self.setPixmap(self.original_pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        super().resizeEvent(event)

class ClickableImageLabel(ViewerLabel):
    def __init__(self):
        super().__init__()
        self.image_np = None
        self.points = []

    def set_image(self, img_array):
        self.image_np = img_array.copy()
        self.points = []
        super().set_image(self.image_np)

    def mousePressEvent(self, event):
        if self.image_np is None or event.button() != Qt.MouseButton.LeftButton: return
        lw, lh = self.width(), self.height()
        ih, iw = self.image_np.shape[:2]
        scale = min(lw / iw, lh / ih)
        dw, dh = iw * scale, ih * scale
        dx, dy = (lw - dw) / 2, (lh - dh) / 2
        mx, my = event.pos().x(), event.pos().y()
        
        if mx < dx or mx > dx + dw or my < dy or my > dy + dh: return
        x = int((mx - dx) / scale)
        y = int((my - dy) / scale)
        
        self.points.append((x, y))
        display_np = self.image_np.copy()
        for i, pt in enumerate(self.points):
            cv2.circle(display_np, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
            if i > 0: cv2.line(display_np, self.points[i-1], pt, (0, 255, 0), 3, cv2.LINE_AA)
        if len(self.points) > 2:
            cv2.line(display_np, self.points[-1], self.points[0], (0, 255, 0), 3, cv2.LINE_AA)
        super().set_image(display_np)

class ModernCVStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Computer Vision Studio")
        self.setGeometry(100, 100, 1400, 900)
        self.current_image_path = None
        self.current_edges = None

        self.apply_pro_theme()
        self.init_ui()

    def apply_pro_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #18181b; color: #e4e4e7; font-family: "Segoe UI"; }
            QSplitter::handle { background-color: #27272a; }
            QLabel#HeaderLabel { font-size: 18px; font-weight: bold; color: #38bdf8; padding-bottom: 8px; border-bottom: 2px solid #27272a; margin-top: 25px; margin-bottom: 5px;}
            QLabel#SliderLabel { font-size: 14px; color: #a1a1aa; }
            QPushButton { background-color: #27272a; color: #e4e4e7; border: 1px solid #3f3f46; border-radius: 6px; padding: 14px; font-size: 14px; text-align: left; }
            QPushButton:hover { background-color: #38bdf8; color: #0f172a; border: 1px solid #38bdf8; }
            QPushButton#PrimaryAction { background-color: #0284c7; color: white; font-weight: bold; text-align: center; border: none; margin-bottom: 15px; padding: 16px; font-size: 15px;}
            QPushButton#PrimaryAction:hover { background-color: #38bdf8; }
            QTextEdit { background-color: #09090b; color: #10b981; border: 1px solid #27272a; border-radius: 4px; font-family: "Consolas", monospace; padding: 10px; }
            QLabel#ImageDisplay { background-color: #09090b; border: 1px dashed #3f3f46; border-radius: 6px; }
            QSlider::groove:horizontal { border: 1px solid #3f3f46; height: 6px; background: #27272a; border-radius: 3px; }
            QSlider::handle:horizontal { background: #38bdf8; border: 1px solid #1e1e2e; width: 14px; margin: -5px 0; border-radius: 7px; }
            QScrollArea { border: none; background-color: transparent; }
            QScrollBar:vertical { background: #18181b; width: 10px; }
            QScrollBar::handle:vertical { background: #3f3f46; border-radius: 5px; }
        """)

    def create_slider(self, name, min_val, max_val, default_val, is_float=False):
        layout = QHBoxLayout()
        lbl_name = QLabel(name)
        lbl_name.setObjectName("SliderLabel")
        lbl_name.setFixedWidth(80)

        display_val = f"{default_val / 10.0:.1f}" if is_float else str(default_val)
        lbl_val = QLabel(display_val)
        lbl_val.setObjectName("SliderLabel")
        lbl_val.setFixedWidth(30)
        lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default_val)

        if is_float: slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(f"{v / 10.0:.1f}"))
        else: slider.valueChanged.connect(lambda v, l=lbl_val: l.setText(str(v)))

        layout.addWidget(lbl_name)
        layout.addWidget(slider)
        layout.addWidget(lbl_val)
        return layout, slider

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ==================================
        # 1. LEFT SIDEBAR (SCROLLABLE)
        # ==================================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(380)
        scroll_area.setMaximumWidth(420)
        
        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(15, 15, 25, 15)

        self.btn_load = QPushButton("📂 Select Input Image")
        self.btn_load.setObjectName("PrimaryAction")
        self.btn_load.clicked.connect(self.load_image)
        sidebar_layout.addWidget(self.btn_load)

        self.btn_canny = QPushButton("Detect Canny edges")
        self.btn_canny.clicked.connect(self.run_canny)
        sidebar_layout.addWidget(self.btn_canny)

        layout_t1, self.slider_t1 = self.create_slider("threshold 1", 0, 500, 100)
        layout_t2, self.slider_t2 = self.create_slider("threshold 2", 0, 500, 200)
        sidebar_layout.addLayout(layout_t1)
        sidebar_layout.addLayout(layout_t2)

        # Line Controls
        self.btn_lines = QPushButton("📏 Detect Hough Lines")
        self.btn_lines.clicked.connect(self.run_lines)
        sidebar_layout.addWidget(self.btn_lines)
        layout_lmin, self.slider_line_min = self.create_slider("Min Length", 10, 100, 80)
        layout_lmax, self.slider_line_max = self.create_slider("Max Length", 100, 500, 200)
        sidebar_layout.addLayout(layout_lmin)
        sidebar_layout.addLayout(layout_lmax)

        # Circle Controls
        self.btn_circles = QPushButton("⭕ Detect Hough Circles")
        self.btn_circles.clicked.connect(self.run_circles)
        sidebar_layout.addWidget(self.btn_circles)
        layout_cmin, self.slider_circ_min = self.create_slider("Min Radius", 5, 20, 7)
        layout_cmax, self.slider_circ_max = self.create_slider("Max Radius", 20, 200, 150)
        sidebar_layout.addLayout(layout_cmin)
        sidebar_layout.addLayout(layout_cmax)

        # Ellipse Controls
        self.btn_ellipses = QPushButton("🪐 Detect Ellipses")
        self.btn_ellipses.clicked.connect(self.run_ellipses)
        sidebar_layout.addWidget(self.btn_ellipses)
        layout_emin, self.slider_ell_min = self.create_slider("Min Axis", 10, 50, 20)
        layout_emax, self.slider_ell_max = self.create_slider("Max Axis", 50, 200, 150)
        sidebar_layout.addLayout(layout_emin)
        sidebar_layout.addLayout(layout_emax)

        self.btn_run_snake = QPushButton("🐍 Run Active Contours")
        self.btn_run_snake.clicked.connect(self.run_snake)
        sidebar_layout.addWidget(self.btn_run_snake)

        layout_alpha, self.slider_alpha = self.create_slider("Alpha", 0, 50, 10, is_float=True)
        layout_beta, self.slider_beta = self.create_slider("Beta", 0, 50, 10, is_float=True)
        layout_gamma, self.slider_gamma = self.create_slider("Gamma", 0, 50, 12, is_float=True)
        layout_iter, self.slider_iterations = self.create_slider("Iterations", 1, 500, 100)

        sidebar_layout.addLayout(layout_alpha)
        sidebar_layout.addLayout(layout_beta)
        sidebar_layout.addLayout(layout_gamma)
        sidebar_layout.addLayout(layout_iter)
        sidebar_layout.addStretch()

        scroll_area.setWidget(sidebar_content)

        # ==================================
        # 2. RIGHT AREA
        # ==================================
        right_area = QWidget()
        right_layout = QVBoxLayout(right_area)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        right_splitter = QSplitter(Qt.Orientation.Vertical)

        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)
        images_layout.setContentsMargins(0, 0, 0, 0)

        orig_container = QWidget()
        orig_layout = QVBoxLayout(orig_container)
        lbl_title_orig = QLabel("Original Input")
        lbl_title_orig.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_orig = ClickableImageLabel()
        orig_layout.addWidget(lbl_title_orig)
        orig_layout.addWidget(self.lbl_img_orig, stretch=1)

        proc_container = QWidget()
        proc_layout = QVBoxLayout(proc_container)
        lbl_title_proc = QLabel("Processed Output")
        lbl_title_proc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_img_proc = ViewerLabel()
        proc_layout.addWidget(lbl_title_proc)
        proc_layout.addWidget(self.lbl_img_proc, stretch=1)

        images_layout.addWidget(orig_container)
        images_layout.addWidget(proc_container)

        console_container = QWidget()
        console_layout = QVBoxLayout(console_container)
        lbl_output = QLabel("TERMINAL OUTPUT")
        lbl_output.setObjectName("HeaderLabel")
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("font-size: 15px; font-weight: bold; line-height: 1.5;")
        console_layout.addWidget(lbl_output)
        console_layout.addWidget(self.console)

        right_splitter.addWidget(images_widget)
        right_splitter.addWidget(console_container)
        right_splitter.setSizes([350, 250]) 
        
        right_layout.addWidget(right_splitter)

        main_splitter.addWidget(scroll_area)
        main_splitter.addWidget(right_area)
        main_splitter.setSizes([300, 1000])
        main_layout.addWidget(main_splitter)

    # ==================================
    # --- LOGIC & ACTIONS            ---
    # ==================================
    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.current_image_path = file_path
            self.current_edges = None
            self.log_to_console(f"\n[SYSTEM] Loaded: {file_path.split('/')[-1]}")
            img_np = cv2.imread(file_path)
            self.lbl_img_orig.set_image(img_np)
            self.lbl_img_proc.set_image(np.zeros_like(img_np))

    def log_to_console(self, text):
        self.console.append(f"> {text}")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def run_canny(self):
        if not self.current_image_path or not BACKEND_AVAILABLE: return
        img = cv2.imread(self.current_image_path)
        t1, t2 = self.slider_t1.value(), self.slider_t2.value()
        self.log_to_console(f"Extracting Canny Edges (T1={t1}, T2={t2})...")
        QApplication.processEvents()
        try:
            self.current_edges = image_lab_backend.apply_custom_canny(img, t1, t2)
            self.lbl_img_proc.set_image(self.current_edges)
            self.log_to_console("SUCCESS: Edges extracted.")
        except Exception as e:
            self.log_to_console(f"ERROR: {e}")

    def get_edges_and_img(self):
        if self.current_edges is None: self.run_canny()
        return self.current_edges, cv2.imread(self.current_image_path)

    def run_lines(self):
        if not self.current_image_path or not BACKEND_AVAILABLE: return
        edges, img = self.get_edges_and_img()
        min_l, max_l = self.slider_line_min.value(), self.slider_line_max.value()
        self.log_to_console(f"Detecting Lines [Length: {min_l} to {max_l}]...")
        QApplication.processEvents()
        try:
            res = image_lab_backend.detect_lines(edges, img, min_l, max_l)
            self.lbl_img_proc.set_image(res)
            self.log_to_console("SUCCESS: Lines Superimposed.")
        except Exception as e:
            self.log_to_console(f"ERROR: {e}")

    def run_circles(self):
        if not self.current_image_path or not BACKEND_AVAILABLE: return
        edges, img = self.get_edges_and_img()
        min_r, max_r = self.slider_circ_min.value(), self.slider_circ_max.value()
        self.log_to_console(f"Detecting Circles [Radius: {min_r} to {max_r}]...")
        QApplication.processEvents()
        try:
            res = image_lab_backend.detect_circles(edges, img, min_r, max_r)
            self.lbl_img_proc.set_image(res)
            self.log_to_console("SUCCESS: Circles Superimposed.")
        except Exception as e:
            self.log_to_console(f"ERROR: {e}")

    def run_ellipses(self):
        if not self.current_image_path or not BACKEND_AVAILABLE: return
        edges, img = self.get_edges_and_img()
        min_a, max_a = self.slider_ell_min.value(), self.slider_ell_max.value()
        self.log_to_console(f"Detecting Ellipses [Major Axis: {min_a} to {max_a}]...")
        QApplication.processEvents()
        try:
            res = image_lab_backend.detect_ellipses(edges, img, min_a, max_a)
            self.lbl_img_proc.set_image(res)
            self.log_to_console("SUCCESS: Ellipses Superimposed.")
        except Exception as e:
            self.log_to_console(f"ERROR: {e}")

    def run_snake(self):
        if not BACKEND_AVAILABLE: return
        points = self.lbl_img_orig.points
        if len(points) < 3:
            self.log_to_console("WARNING: Click on the Left image to draw a shape first!")
            return
        alpha, beta, gamma = self.slider_alpha.value()/10.0, self.slider_beta.value()/10.0, self.slider_gamma.value()/10.0
        iters = self.slider_iterations.value()
        orig_img = self.lbl_img_orig.image_np
        gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        self.log_to_console(f"\\n--- EVOLVING SNAKE ---")
        QApplication.processEvents()
        try:
            results = image_lab_backend.apply_greedy_snake(gray_img, points, iters, alpha, beta, gamma)
            # results = image_lab_backend.apply_greedy_snake(gray_img, points, alpha, beta, gamma)
            res_img = orig_img.copy()
            final_points = results["points"]
            for i in range(len(final_points)):
                pt1, pt2 = final_points[i], final_points[(i + 1) % len(final_points)]
                cv2.circle(res_img, pt1, 4, (255, 100, 50), -1, cv2.LINE_AA)
                cv2.line(res_img, pt1, pt2, (255, 255, 0), 5, cv2.LINE_AA)
            self.lbl_img_proc.set_image(res_img)
            self.log_to_console(f"Area: {results['area']:.1f} | Perim: {results['perimeter']:.1f}")
            self.log_to_console(f"Chain Code: {results['chain_code']}")
        except Exception as e:
            self.log_to_console(f"ERROR: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ModernCVStudio()
    window.show()
    sys.exit(app.exec())