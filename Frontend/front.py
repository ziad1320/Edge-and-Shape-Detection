import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QComboBox,
                             QSlider, QSpinBox, QTabWidget, QGroupBox, QFileDialog,
                             QScrollArea, QSplitter, QFrame, QSizePolicy, QMessageBox)
from PyQt6.QtCore import Qt, QSize, QTimer, QEvent, pyqtSignal
from PyQt6.QtGui import QAction, QIcon, QFont, QColor, QPalette, QPixmap, QImage, QShortcut, QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
import cv2
import os

try:
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(r"C:\msys64\mingw64\bin")
    else:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../Backend/build")))
        
    import task2_backend
    import image_lab_backend # type: ignore
    
    BACKEND_AVAILABLE = True
except ImportError as e:
    BACKEND_AVAILABLE = False
    print(f"Backend load error: {e}")

def load_and_scale_image_np(file_path, max_width=800):
    img = cv2.imread(file_path)
    if img is not None and img.shape[1] > max_width:
        ratio = max_width / img.shape[1]
        new_dim = (max_width, int(img.shape[0] * ratio))
        img = cv2.resize(img, new_dim, interpolation=cv2.INTER_AREA)
    return img

def numpy_to_qpixmap(img_array):
    if img_array is None:
        return QPixmap()
    
    # Needs to be a contiguous unmanaged array copied into Qt context to avoid GC crashes.
    if len(img_array.shape) == 3:
        h, w, c = img_array.shape
        bytes_per_line = c * w
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg.copy())
    else:
        h, w = img_array.shape
        bytes_per_line = w
        img_gray = np.ascontiguousarray(img_array)
        qimg = QImage(img_gray.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
        return QPixmap.fromImage(qimg.copy())


# ==========================================
# --- CUSTOM WIDGETS ---
# ==========================================

class ImageLabel(QLabel):
    """A custom QLabel that automatically scales its pixmap to fit its size while preserving aspect ratio."""

    double_clicked = pyqtSignal()

    def __init__(self, text=""):
        super().__init__(text)
        self.original_pixmap = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- THE FIX ---
        # This stops the infinite growth loop by telling the layout to ignore the image's inherent size
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setMinimumSize(100, 100)  # Prevents the label from completely collapsing

    def set_image(self, file_path):
        img_np = load_and_scale_image_np(file_path)
        if img_np is not None:
            pixmap = numpy_to_qpixmap(img_np)
            self.original_pixmap = pixmap
            self.update_image()
        else:
            self.setText("❌ Failed to load image.")
            self.original_pixmap = None
            
    def set_pixmap_data(self, pixmap):
        if not pixmap.isNull():
            self.original_pixmap = pixmap
            self.update_image()

    def update_image(self):
        if self.original_pixmap and not self.original_pixmap.isNull():
            # Scale the image to fit the label's current boundaries
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled_pixmap)


class ClickableImageLabel(ImageLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.image_np = None
        self.points = []

    def set_image(self, file_path):
        # We need numpy for cv2 drawing, so load it
        img_np = load_and_scale_image_np(file_path)
        if img_np is not None:
            self.image_np = img_np.copy()
            self.points = []
            super().set_pixmap_data(numpy_to_qpixmap(self.image_np))
        else:
            super().set_image(file_path)

    def set_pixmap_data(self, pixmap, img_np=None):
        if img_np is not None:
            self.image_np = img_np.copy()
        super().set_pixmap_data(pixmap)

    def mousePressEvent(self, event):
        if self.image_np is None or event.button() != Qt.MouseButton.LeftButton:
            return
            
        # Calculate scaling to translate widget coordinates to image coordinates
        lw, lh = self.width(), self.height()
        
        # Pixmap could be null or empty
        if not self.original_pixmap or self.original_pixmap.isNull():
            return
            
        pw, ph = self.original_pixmap.width(), self.original_pixmap.height()
        ih, iw = self.image_np.shape[:2]
        
        # The scale used by KeepAspectRatio
        scale = min(lw / pw, lh / ph)
        dw, dh = pw * scale, ph * scale
        dx, dy = (lw - dw) / 2, (lh - dh) / 2
        
        mx, my = event.pos().x(), event.pos().y()
        
        if mx < dx or mx > dx + dw or my < dy or my > dy + dh: 
            return
            
        x = int(((mx - dx) / scale) * (iw / pw))
        y = int(((my - dy) / scale) * (ih / ph))
        
        self.points.append((x, y))
        display_np = self.image_np.copy()
        
        for i, pt in enumerate(self.points):
            cv2.circle(display_np, pt, 5, (0, 0, 255), -1, cv2.LINE_AA)
            if i > 0: 
                cv2.line(display_np, self.points[i-1], pt, (0, 255, 0), 3, cv2.LINE_AA)
                
        if len(self.points) > 2:
            cv2.line(display_np, self.points[-1], self.points[0], (0, 255, 0), 3, cv2.LINE_AA)
            
        super().set_pixmap_data(numpy_to_qpixmap(display_np))


# ==========================================
# --- STYLING CONSTANTS ---
# ==========================================

class AppStyle:
    # --- DARK THEME PALETTE ---
    DARK_STYLE = """
        QMainWindow { background-color: #1e1e1e; }
        QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 14px; }

        /* Tabs */
        QTabWidget::pane { border: 1px solid #3d3d3d; background: #252526; border-radius: 5px; }
        QTabBar::tab { background: #2d2d2d; color: #aaa; padding: 10px 20px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
        QTabBar::tab:selected { background: #3e3e42; color: #fff; border-bottom: 2px solid #007acc; }

        /* Groups / Cards */
        QGroupBox { 
            border: 1px solid #3e3e42; 
            border-radius: 8px; 
            margin-top: 20px; 
            background-color: #252526; 
            font-weight: bold;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #007acc; }

        /* Controls */
        QPushButton { 
            background-color: #007acc; 
            color: white; 
            border: none; 
            border-radius: 4px; 
            padding: 8px 16px; 
            font-weight: bold;
        }
        QPushButton:hover { background-color: #005f9e; }
        QPushButton:pressed { background-color: #003e66; }
        QPushButton#SecondaryBtn { background-color: #3e3e42; color: #eee; }
        QPushButton#SecondaryBtn:hover { background-color: #4e4e52; }

        QComboBox, QSpinBox { 
            background-color: #333337; 
            border: 1px solid #3e3e42; 
            border-radius: 4px; 
            padding: 5px; 
            color: white; 
        }

        /* Image Display Area */
        QLabel#ImageDisplay { 
            border: 2px dashed #3e3e42; 
            background-color: #1e1e1e; 
            color: #555;
            font-size: 16px;
        }

        /* Plot Tab Buttons */
        QPushButton#PlotTabBtn { 
            background-color: #2d2d2d; 
            color: #aaa; 
            border: 1px solid #3e3e42;
            border-bottom: none;
            border-top-left-radius: 4px; 
            border-top-right-radius: 4px; 
            padding: 5px 15px; 
        }
        QPushButton#PlotTabBtn:hover { background-color: #3e3e42; }
        QPushButton#PlotTabBtn:checked { 
            background-color: #007acc; 
            color: white; 
            border-color: #007acc;
        }
    """


# ==========================================
# --- MAIN APPLICATION ---
# ==========================================

class ComputerVisionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CV Toolkit Pro")

        # State mapping
        self.current_image_np = None
        self.undo_stack_np = []
        self.redo_stack_np = []
        
        self.current_plot_mode = 'hist'

        # UI Initialization
        self.init_ui()
        self.init_main_tab()
        self.apply_theme()
        
        # Install event filter to prevent scroll wheel changing input values
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            # Ignore wheel events on these types unless they explicitly have focus
            if isinstance(obj, (QComboBox, QSpinBox, QSlider)) and not obj.hasFocus():
                event.ignore()
                return True
        return super().eventFilter(obj, event)

    def init_ui(self):
        # Keyboard Shortcuts
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo_action)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self).activated.connect(self.redo_action)
        QShortcut(QKeySequence("Ctrl+Y"), self).activated.connect(self.redo_action)

        # Main Layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Top Bar for Global Buttons (like Download Result)
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(10, 5, 10, 5)
        
        self.btn_download_main = QPushButton("💾 Download Result Image")
        self.btn_download_main.setObjectName("SecondaryBtn")
        self.btn_download_main.setMinimumHeight(35)
        self.btn_download_main.clicked.connect(lambda: self.download_image(self.current_image_np, "Result"))
        self.btn_download_main.setVisible(False)  # Hidden until we have an image
        
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(self.btn_download_main)
        
        main_layout.addWidget(top_bar)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.init_task2_tab()
        self.init_snake_tab()
        
        # Hide the global download button/topbar if not on the Image Processor tab
        self.tabs.currentChanged.connect(lambda index: top_bar.setVisible(index == 0))

    def init_main_tab(self):
        # Canvas
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(250)

    def init_task2_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Controls Side (Left Sidebar)
        controls = QWidget()
        controls.setMinimumWidth(380)
        controls.setMaximumWidth(430)
        
        c_layout = QVBoxLayout(controls)
        c_layout.setSpacing(20)
        
        # We need a state list to store paths for batch processing
        self.task2_batch_files = []
        self.task2_current_idx = -1

        grp_load = self.create_group_box("Batch Loader (Task 2)")
        l_load = QVBoxLayout()
        btn_load_folder = QPushButton("📂 Select Multiple Images")
        btn_load_folder.clicked.connect(self.load_batch_images)
        self.lbl_batch_status = QLabel("0 images loaded")
        
        # Navigation
        nav_layout = QHBoxLayout()
        self.btn_prev_img = QPushButton("⬅ Prev")
        self.btn_prev_img.setObjectName("SecondaryBtn")
        self.btn_prev_img.clicked.connect(self.prev_batch_image)
        self.btn_next_img = QPushButton("Next ➡")
        self.btn_next_img.setObjectName("SecondaryBtn")
        self.btn_next_img.clicked.connect(self.next_batch_image)
        nav_layout.addWidget(self.btn_prev_img)
        nav_layout.addWidget(self.btn_next_img)
        
        l_load.addWidget(btn_load_folder)
        l_load.addWidget(self.lbl_batch_status)
        l_load.addLayout(nav_layout)
        grp_load.setLayout(l_load)
        c_layout.addWidget(grp_load)

        # To support stacked controls, let's add a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        controls_inner = QWidget()
        l_opts = QVBoxLayout(controls_inner)
        l_opts.setContentsMargins(0, 0, 0, 0)
        
        # Section 1: Canny Edge Detection
        grp_canny = self.create_group_box("Section 1: Canny Edge Detection")
        l_canny = QVBoxLayout(grp_canny)
        
        canny_t1_layout, self.spin_c_t1, _ = self.create_slider_widget("Threshold 1:", 0, 500, 100)
        canny_t2_layout, self.spin_c_t2, _ = self.create_slider_widget("Threshold 2:", 0, 500, 200)
        
        btn_run_canny = QPushButton("Detect Edges")
        btn_run_canny.clicked.connect(self.run_canny_only)
        
        l_canny.addLayout(canny_t1_layout)
        l_canny.addWidget(self.spin_c_t1)
        l_canny.addLayout(canny_t2_layout)
        l_canny.addWidget(self.spin_c_t2)
        l_canny.addWidget(btn_run_canny)
        l_opts.addWidget(grp_canny)
        
        # Section 2: Line Detection
        grp_lines = self.create_group_box("Section 2: Line Detection")
        l_lines = QVBoxLayout(grp_lines)
        
        l_lines.addWidget(QLabel("Dynamically calculates voting thresholds based on image scale."))
        
        btn_run_lines = QPushButton("Detect Lines")
        btn_run_lines.clicked.connect(self.run_lines_only)
        
        l_lines.addWidget(btn_run_lines)
        l_opts.addWidget(grp_lines)
        
        # Section 3: Circle Detection
        grp_circles = self.create_group_box("Section 3: Circle Detection")
        l_circles = QVBoxLayout(grp_circles)
        
        l_circles.addWidget(QLabel("Dynamically calculates radius bounds and thresholds."))
        
        btn_run_circles = QPushButton("Detect Circles")
        btn_run_circles.clicked.connect(self.run_circles_only)
        
        l_circles.addWidget(btn_run_circles)
        l_opts.addWidget(grp_circles)
        
        # Section 4: Ellipse Detection
        grp_ellipses = self.create_group_box("Section 4: Ellipse Detection")
        l_ellipses = QVBoxLayout(grp_ellipses)
        l_ellipses.addWidget(QLabel(
            "Uses Xie & Ji 1D Accumulator algorithm.\n"
            "Automatically infers parameters from edge pairs."
        ))
        
        btn_run_ellipses = QPushButton("Detect Ellipses")
        btn_run_ellipses.clicked.connect(self.run_ellipses_only)
        l_ellipses.addWidget(btn_run_ellipses)
        l_opts.addWidget(grp_ellipses)
        
        # --- FIX 1: Removed the duplicated block assigning scroll_area to c_layout ---
        l_opts.addStretch()
        scroll_area.setWidget(controls_inner)
        c_layout.addWidget(scroll_area)
        # -----------------------------------------------------------------------------

        # Display Side
        display = QWidget()
        d_layout = QHBoxLayout(display)

        self.lbl_t2_orig = ImageLabel("Original Image\n(Select Batch first)")
        self.lbl_t2_orig.setObjectName("ImageDisplay")
        
        self.lbl_t2_res = ImageLabel("Detected Shapes (Task 2)")
        self.lbl_t2_res.setObjectName("ImageDisplay")
        self.lbl_t2_res.setStyleSheet("border: 2px solid #007acc;")

        d_layout.addWidget(self.lbl_t2_orig, stretch=1)
        d_layout.addWidget(self.lbl_t2_res, stretch=1)

        layout.addWidget(controls)
        layout.addWidget(display)

        self.tabs.addTab(tab, "Canny & Shapes")

    def init_snake_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)

        # Controls Side (Left Sidebar)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumWidth(380)
        scroll_area.setMaximumWidth(430)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        controls = QWidget()
        c_layout = QVBoxLayout(controls)
        c_layout.setSpacing(10)
        
        self.btn_load_snake = QPushButton("📂 Select Input Image")
        self.btn_load_snake.setObjectName("SecondaryBtn")
        self.btn_load_snake.clicked.connect(lambda: self.handle_image_upload(self.lbl_snake_orig))
        c_layout.addWidget(self.btn_load_snake)
        
        self.btn_clear_snake = QPushButton("🗑️ Clear Contour Points")
        self.btn_clear_snake.setObjectName("SecondaryBtn")
        self.btn_clear_snake.clicked.connect(self.clear_snake_points)
        c_layout.addWidget(self.btn_clear_snake)
        
        grp_snake = self.create_group_box("Active Contours (Greedy Snake)")
        l_snake = QVBoxLayout(grp_snake)
        
        alpha_layout, self.slider_alpha, _ = self.create_slider_widget("Alpha (Continuity):", 0, 50, 10, value_formatter=lambda v: f"{v / 10.0:.1f}")
        beta_layout, self.slider_beta, _ = self.create_slider_widget("Beta (Curvature):", 0, 50, 10, value_formatter=lambda v: f"{v / 10.0:.1f}")
        gamma_layout, self.slider_gamma, _ = self.create_slider_widget("Gamma (Image):", 0, 50, 12, value_formatter=lambda v: f"{v / 10.0:.1f}")
        thresh_layout, self.slider_threshold, _ = self.create_slider_widget("Stop Threshold:", 0, 100, 5, value_formatter=lambda v: f"{v / 10.0:.1f}")
        
        l_snake.addLayout(alpha_layout)
        l_snake.addWidget(self.slider_alpha)
        l_snake.addLayout(beta_layout)
        l_snake.addWidget(self.slider_beta)
        l_snake.addLayout(gamma_layout)
        l_snake.addWidget(self.slider_gamma)
        l_snake.addLayout(thresh_layout)
        l_snake.addWidget(self.slider_threshold)
        
        self.btn_run_snake = QPushButton("🐍 Evolve Snake")
        self.btn_run_snake.clicked.connect(self.run_snake)
        l_snake.addWidget(self.btn_run_snake)
        
        c_layout.addWidget(grp_snake)
        
        c_layout.addStretch()
        scroll_area.setWidget(controls)
        
        # Display Side
        display = QWidget()
        d_layout = QVBoxLayout(display)
        d_layout.setContentsMargins(0, 0, 0, 0)
        
        display_splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Images top
        images_widget = QWidget()
        images_layout = QHBoxLayout(images_widget)
        images_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_snake_orig = ClickableImageLabel("Initial Contour\n(Double-Click to Load, then Click to add points)")
        self.lbl_snake_orig.setObjectName("ImageDisplay")
        self.lbl_snake_orig.double_clicked.connect(lambda: self.handle_image_upload(self.lbl_snake_orig))
        
        self.lbl_snake_proc = ImageLabel("Animated Result")
        self.lbl_snake_proc.setObjectName("ImageDisplay")
        self.lbl_snake_proc.setStyleSheet("border: 2px solid #007acc;")
        
        images_layout.addWidget(self.lbl_snake_orig, stretch=1)
        images_layout.addWidget(self.lbl_snake_proc, stretch=1)
        
        # Console bottom
        console_widget = QWidget()
        console_layout = QVBoxLayout(console_widget)
        lbl_console = QLabel("Output Console / Metrics")
        lbl_console.setStyleSheet("font-weight: bold; color: #007acc;")
        from PyQt6.QtWidgets import QTextEdit
        self.snake_console = QTextEdit()
        self.snake_console.setReadOnly(True)
        self.snake_console.setStyleSheet("background-color: #1e1e1e; color: #4ade80; font-family: 'Consolas', monospace;")
        console_layout.addWidget(lbl_console)
        console_layout.addWidget(self.snake_console)
        
        display_splitter.addWidget(images_widget)
        display_splitter.addWidget(console_widget)
        display_splitter.setSizes([700, 200])
        
        d_layout.addWidget(display_splitter)
        
        layout.addWidget(scroll_area)
        layout.addWidget(display)
        
        self.tabs.addTab(tab, "Active Contours & Lab")

    # ==========================================
    # --- HELPER FUNCTIONS ---
    # ==========================================

    def create_slider_widget(self, label_text, min_val, max_val, default_val, step=1, value_formatter=str):
        layout = QHBoxLayout()
        layout.addWidget(QLabel(label_text))
        lbl_val = QLabel(value_formatter(default_val))
        layout.addWidget(lbl_val)
        layout.addStretch()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setSingleStep(step)
        slider.setValue(default_val)
        slider.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        slider.installEventFilter(self)
        
        slider.valueChanged.connect(lambda val, l=lbl_val, f=value_formatter: l.setText(f(val)))
        
        return layout, slider, lbl_val

    def load_batch_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Images for Task 2", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if files:
            self.task2_batch_files = files
            self.task2_current_idx = 0
            self.lbl_batch_status.setText(f"{len(files)} images loaded. Showing 1/{len(files)}")
            self.display_current_batch_image()

    def display_current_batch_image(self):
        if 0 <= self.task2_current_idx < len(self.task2_batch_files):
            file_path = self.task2_batch_files[self.task2_current_idx]
            self.lbl_t2_orig.set_image(file_path)
            self.lbl_t2_res.clear()
            self.lbl_t2_res.setText("Press Process...")
            self.lbl_batch_status.setText(f"Showing {self.task2_current_idx+1}/{len(self.task2_batch_files)}")

    def next_batch_image(self):
        if self.task2_batch_files and self.task2_current_idx < len(self.task2_batch_files) - 1:
            self.task2_current_idx += 1
            self.display_current_batch_image()

    def prev_batch_image(self):
        if self.task2_batch_files and self.task2_current_idx > 0:
            self.task2_current_idx -= 1
            self.display_current_batch_image()

    def _get_current_t2_image(self):
        if not self.task2_batch_files or self.task2_current_idx < 0:
            return None
        file_path = self.task2_batch_files[self.task2_current_idx]
        return load_and_scale_image_np(file_path)

    def run_canny_only(self):
        img_np = self._get_current_t2_image()
        if img_np is None: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.lbl_t2_res.setText("⏳ Extracting Edges...\nPlease wait")
        QApplication.processEvents()

        try:
            t1 = self.spin_c_t1.value()
            t2 = self.spin_c_t2.value()
            res_img = task2_backend.apply_custom_canny(img_np, t1, t2)
            
            # --- FIX 3: Convert directly to RGB (not BGR) for PyQt display consistency ---
            dis_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2RGB)
            self.lbl_t2_res.set_pixmap_data(numpy_to_qpixmap(dis_img))
        except Exception as e:
            self.lbl_t2_res.setText(f"❌ Error during edge detect:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def run_lines_only(self):
        img_np = self._get_current_t2_image()
        if img_np is None: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.lbl_t2_res.setText("⏳ Detecting Lines...\nPlease wait")
        QApplication.processEvents()

        try:
            t1 = self.spin_c_t1.value()
            t2 = self.spin_c_t2.value()
            edges = task2_backend.apply_custom_canny(img_np, t1, t2)
            
            res_img = task2_backend.detect_lines(edges, img_np)
            
            # --- FIX 3: Convert C++ output (BGR) to PyQt expected (RGB) ---
            if len(res_img.shape) == 3:
                res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                
            self.lbl_t2_res.set_pixmap_data(numpy_to_qpixmap(res_img))
        except Exception as e:
            self.lbl_t2_res.setText(f"❌ Error during line detect:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def run_circles_only(self):
        img_np = self._get_current_t2_image()
        if img_np is None: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.lbl_t2_res.setText("⏳ Detecting Circles...\nPlease wait")
        QApplication.processEvents()

        try:
            t1 = self.spin_c_t1.value()
            t2 = self.spin_c_t2.value()
            edges = task2_backend.apply_custom_canny(img_np, t1, t2)
            res_img = task2_backend.detect_circles(edges, img_np)
            
            # --- FIX 3: Convert C++ output (BGR) to PyQt expected (RGB) ---
            if len(res_img.shape) == 3:
                res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                
            self.lbl_t2_res.set_pixmap_data(numpy_to_qpixmap(res_img))
        except Exception as e:
            self.lbl_t2_res.setText(f"❌ Error during circle detect:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
            
    def run_ellipses_only(self):
        img_np = self._get_current_t2_image()
        if img_np is None: return
        
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.lbl_t2_res.setText("⏳ Detecting Ellipses\n(Xie & Ji Algorithm)...\nPlease wait")
        QApplication.processEvents()

        try:
            t1 = self.spin_c_t1.value()
            t2 = self.spin_c_t2.value()
            edges = task2_backend.apply_custom_canny(img_np, t1, t2)
            
            res_img = task2_backend.detect_ellipses(edges, img_np)
            
            # --- FIX 3: Convert C++ output (BGR) to PyQt expected (RGB) ---
            if len(res_img.shape) == 3:
                res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                
            self.lbl_t2_res.set_pixmap_data(numpy_to_qpixmap(res_img))
        except Exception as e:
            self.lbl_t2_res.setText(f"❌ Error during ellipse detect:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    # --- Snake Tab Logic ---
    def log_snake(self, text):
        self.snake_console.append(f"> {text}")
        scrollbar = self.snake_console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_snake_points(self):
        if hasattr(self.lbl_snake_orig, 'image_np') and self.lbl_snake_orig.image_np is not None:
            self.lbl_snake_orig.points = []
            self.lbl_snake_orig.set_pixmap_data(numpy_to_qpixmap(self.lbl_snake_orig.image_np))
            self.log_snake("Cleared contour points.")

    def run_snake(self):
        if not BACKEND_AVAILABLE: 
            self.log_snake("ERROR: image_lab_backend module not loaded!")
            return
            
        points = self.lbl_snake_orig.points
        if len(points) < 3:
            self.log_snake("WARNING: Click on the left image to draw at least 3 points first!")
            return
            
        alpha = self.slider_alpha.value() / 10.0
        beta = self.slider_beta.value() / 10.0
        gamma = self.slider_gamma.value() / 10.0
        thresh = self.slider_threshold.value() / 10.0
        
        orig_img = self.lbl_snake_orig.image_np
        
        # Provide grayscale image if it's BGR
        if len(orig_img.shape) == 3:
            gray_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = orig_img.copy()
            
        self.log_snake(f"\n--- EVOLVING SNAKE ---")
        self.log_snake(f"Params: [Alpha={alpha}, Beta={beta}, Gamma={gamma}, Stop Thresh={thresh}]")
        QApplication.processEvents()

        try:
            results = image_lab_backend.apply_greedy_snake(gray_img, points, thresh, alpha, beta, gamma)
            if not results:
                self.log_snake("Snake returned empty results.")
                return
                
            history = results.get("history", [results.get("points", [])])
            import time
            
            # Animate history
            for step_points in history:
                res_img = orig_img.copy()
                if len(res_img.shape) == 2:
                    res_img = cv2.cvtColor(res_img, cv2.COLOR_GRAY2RGB)
                elif len(res_img.shape) == 3:
                    res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
                    
                for i in range(len(step_points)):
                    pt1 = tuple(step_points[i])
                    pt2 = tuple(step_points[(i + 1) % len(step_points)])
                    cv2.circle(res_img, pt1, 4, (255, 100, 50), -1, cv2.LINE_AA)
                    cv2.line(res_img, pt1, pt2, (255, 255, 0), 3, cv2.LINE_AA)
                    
                self.lbl_snake_proc.set_pixmap_data(numpy_to_qpixmap(res_img))
                QApplication.processEvents()
                time.sleep(0.01)
                
            actual_iters = results.get("iterations_run", "Unknown")
            self.log_snake(f"CONVERGED IN: {actual_iters} iterations.")
            self.log_snake(f"Area: {results.get('area', 0):.1f} | Perim: {results.get('perimeter', 0):.1f}")
            self.log_snake(f"Chain Code: {results.get('chain_code', '')}")
            
        except Exception as e:
            self.log_snake(f"ERROR running snake: {e}")

    def create_group_box(self, title):
        group = QGroupBox(title)
        return group

    def apply_theme(self):
        self.setStyleSheet(AppStyle.DARK_STYLE)
        # Update Matplotlib colors for Dark Mode
        self.figure.patch.set_facecolor('#1e1e1e')
        self.figure.gca().set_facecolor('#252526')
        self.figure.gca().tick_params(colors='white')
        self.figure.gca().xaxis.label.set_color('white')
        self.figure.gca().yaxis.label.set_color('white')
        self.figure.gca().title.set_color('white')

        self.canvas.draw()

    def handle_image_upload(self, target_label):
        """Opens a file dialog, shows a loading state, and loads the image into the target label."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        if file_path:
            # Show the waiting cursor globally
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

            # Clear previous image and show loading text
            target_label.clear()
            target_label.original_pixmap = None
            target_label.setText("⏳ Loading...\nPlease wait")

            # Force the UI to process the text update immediately
            QApplication.processEvents()

            # Simulate a brief delay to show the loader, then load the image
            QTimer.singleShot(600, lambda: self.finalize_image_load(target_label, file_path))

    def finalize_image_load(self, target_label, file_path):
        """Sets the image on the label and removes the loading cursor."""
        target_label.set_image(file_path)
        QApplication.restoreOverrideCursor()
        
        img_np = load_and_scale_image_np(file_path)
        if img_np is None: return

        if hasattr(self, 'lbl_snake_orig') and target_label == self.lbl_snake_orig:
            # Special logic for snake since it's a ClickableImageLabel
            self.lbl_snake_orig.image_np = img_np
            self.lbl_snake_orig.points = []
            self.snake_console.clear()
            self.log_snake(f"Loaded image. Please click points to define initial contour.")
 
    def undo_action(self):
        if self.undo_stack_np:
            if self.current_image_np is not None:
                self.redo_stack_np.append(self.current_image_np.copy())
            self.current_image_np = self.undo_stack_np.pop()
            
            # Show on processed label (even if it's the original, just for visual feedback)
            qpixmap = numpy_to_qpixmap(self.current_image_np)
            self.lbl_proc.set_pixmap_data(qpixmap)
            self.update_histograms()

    def redo_action(self):
        if self.redo_stack_np:
            if self.current_image_np is not None:
                self.undo_stack_np.append(self.current_image_np.copy())
            self.current_image_np = self.redo_stack_np.pop()

            qpixmap = numpy_to_qpixmap(self.current_image_np)
            self.lbl_proc.set_pixmap_data(qpixmap)
            self.update_histograms()

    def download_image(self, img_np, image_name):
        """Helper to prompt for save location and save the current image."""
        if img_np is None:
            QMessageBox.warning(self, "No Image", f"Cannot download {image_name}. Please make sure an image is loaded and processed first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, f"Save {image_name}", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            cv2.imwrite(file_path, img_np)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Fusion style allows for better custom coloring
    window = ComputerVisionApp()
    window.showMaximized()
    sys.exit(app.exec())
    