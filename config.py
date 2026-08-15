"""
=========================================
Configuration File
=========================================
"""

from pathlib import Path

# =========================================
# ROOT PATHS
# =========================================

BASE_DIR = Path(__file__).resolve().parent

# VIDEO_PATH = BASE_DIR / "videos" / "gemini_generated_video.mp4"
VIDEO_PATH = BASE_DIR / "videos" / "Video Project 8.mp4"

MODEL_PATH = BASE_DIR / "model" / "mobilenetv2_debris_classifier.keras"

CLASS_NAMES_PATH = BASE_DIR / "model" / "class_names.json"

# =========================================
# SEGFORMER
# =========================================

SEGFORMER_MODEL = BASE_DIR / "model" / "segformer_b0"

ROAD_CLASS_ID = 0

# =========================================
# VIDEO
# =========================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 360

SEGMENTATION_WIDTH = 224
SEGMENTATION_HEIGHT = 128

FRAME_SKIP = 5
FPS = 30

# =========================================
# ROI
# =========================================

ROI_TOP = 0.00
ROI_BOTTOM = 0.00

ROI_LEFT = 0.00
ROI_RIGHT = 0.00

# =========================================
# MODEL
# =========================================

INPUT_SIZE = 224

# =========================================
# MACHINE
# =========================================

MAX_BRUSH_RPM = 340
MAX_FAN_RPM = 3000

MAX_BRUSH_POWER = 1.20
MAX_FAN_POWER = 2.40
NORMAL_POWER = 3.20

MAX_POWER = 3.20