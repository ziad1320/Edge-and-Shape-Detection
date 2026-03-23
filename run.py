import os
import sys

# --- PATH CONFIGURATION ---
OPENCV_DLL_PATH = r"C:/msys64/mingw64/bin"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_BUILD_DIR = os.path.join(ROOT_DIR, "Backend", "build")

# 1. Add DLL directory for Python on Windows
if sys.platform == 'win32':
    if os.path.exists(OPENCV_DLL_PATH):
        try:
            os.add_dll_directory(OPENCV_DLL_PATH)
        except Exception as e:
            pass

# 2. Add build directory to Python sys.path
if os.path.exists(BACKEND_BUILD_DIR):
    sys.path.append(BACKEND_BUILD_DIR)

# 3. Launch the new Frontend UI
try:
    sys.path.append(os.path.join(ROOT_DIR, "Frontend"))
    
    # --- THIS IS THE FIX! Pointing to front.py and ModernCVStudio ---
    from Frontend.front import QApplication, ModernCVStudio
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ModernCVStudio()
    window.show()
    sys.exit(app.exec())
    
except Exception as e:
    print(f"Failed to launch GUI! Error: {e}")