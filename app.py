import sys
import threading
import cv2
from pathlib import Path
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QProgressBar,
    QFrame,
    QSizePolicy,
)

# Ensure the package root is importable when running app.py directly.
ROOT_DIR = Path(__file__).resolve().parent
if ROOT_DIR.name == "AI_Street_Sweeper" and str(ROOT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR.parent))

from utils.video import VideoProcessor
from utils.predictor import DebrisPredictor
from utils.controller import Controller
from utils.power import PowerCalculator
from utils.segmentation import Segmentation


class StatCard(QFrame):
    def __init__(self, title: str, color: str, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame { background-color: #1F2A44; border-radius: 12px; }"
        )

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.title_label.setFont(QFont("Segoe UI", 12))

        self.value_label = QLabel("--")
        self.value_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
        self.value_label.setStyleSheet("color: white;")
        self.value_label.setTextFormat(Qt.RichText)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { background-color: #333333; border-radius: 6px; } "
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}"
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        layout.addWidget(self.progress)

    def update_stat(self, text: str, progress: float):
        self.value_label.setText(text)
        self.progress.setValue(min(max(int(progress * 100), 0), 100))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Street Sweeper Control System")
        self.setMinimumSize(1200, 720)

        self.video = VideoProcessor()
        self.predictor = DebrisPredictor()
        self.controller = Controller()
        self.power = PowerCalculator()
        self.segmenter = Segmentation()

        self.frame_counter = 0
        self.inference_lock = threading.Lock()
        self.model_results = {
            "prediction": "clean",
            "confidence": 0,
            "coverage": 0,
            "brush_rpm": 0,
            "fan_rpm": 0,
            "adaptive_power": 0,
            "saving": 0,
            "road_roi": None,
        }
        self.normal_power = self.power.max_power * 0.5
        self.ui_refresh_ms = 40
        self.segment_frame_interval = 30
        self.inference_thread = None

        self._build_ui()
        self._start_timer()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(16)
        main_layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("AI STREET SWEEPER CONTROL SYSTEM")
        header.setStyleSheet("color: white; font-size: 28px; font-weight: bold;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        video_layout = QHBoxLayout()
        video_layout.setSpacing(12)

        self.original_title = QLabel("Original Video")
        self.original_title.setAlignment(Qt.AlignCenter)
        self.original_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        self.original_video = QLabel()
        self.original_video.setAlignment(Qt.AlignCenter)
        self.original_video.setStyleSheet("background-color: #D5DCE5; border-radius: 12px;")
        self.original_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.original_video.setMinimumHeight(360)

        original_container = QVBoxLayout()
        original_container.addWidget(self.original_title)
        original_container.addWidget(self.original_video)

        self.road_title = QLabel("Segmented Road")
        self.road_title.setAlignment(Qt.AlignCenter)
        self.road_title.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
        self.road_video = QLabel()
        self.road_video.setAlignment(Qt.AlignCenter)
        self.road_video.setStyleSheet("background-color: #D5DCE5; border-radius: 12px;")
        self.road_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.road_video.setMinimumHeight(360)

        road_container = QVBoxLayout()
        road_container.addWidget(self.road_title)
        road_container.addWidget(self.road_video)

        video_layout.addLayout(original_container)
        video_layout.addLayout(road_container)
        main_layout.addLayout(video_layout)

        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(12)

        self.prediction_card = StatCard("DEBRIS LEVEL", "#A855F7")
        self.brush_card = StatCard("BRUSH", "#FF9F1C")
        self.fan_card = StatCard("FAN", "#33D9B2")
        self.normal_power_card = StatCard("NORMAL POWER", "#4DA3FF")
        self.adaptive_power_card = StatCard("ADAPTIVE POWER", "#FFD43B")
        self.saving_card = StatCard("ENERGY SAVING", "#5BE37D")

        stats_layout.addWidget(self.prediction_card)
        stats_layout.addWidget(self.brush_card)
        stats_layout.addWidget(self.fan_card)
        stats_layout.addWidget(self.normal_power_card)
        stats_layout.addWidget(self.adaptive_power_card)
        stats_layout.addWidget(self.saving_card)

        main_layout.addLayout(stats_layout)

    def _start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.ui_refresh_ms)

    def to_pixmap(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(image.copy())

    def overlay_roi(self, frame, mask):
        overlay = frame.copy()
        overlay[mask == 255] = (0, 255, 0)
        return cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    def update_frame(self):
        frame = self.video.get_frame()
        if frame is None:
            return

        roi, roi_mask = self.video.get_roi(frame)
        frame_with_roi = self.overlay_roi(frame, roi_mask)

        self.original_video.setPixmap(
            self.to_pixmap(frame_with_roi).scaled(
                self.original_video.size(),
                Qt.KeepAspectRatio,
                Qt.FastTransformation,
            )
        )
        self.frame_counter += 1

        if self.frame_counter % self.segment_frame_interval == 0:
            self.start_inference(frame, roi, roi_mask)

        with self.inference_lock:
            current_results = dict(self.model_results)

        if current_results["road_roi"] is not None:
            self.road_video.setPixmap(
                self.to_pixmap(current_results["road_roi"]).scaled(
                    self.road_video.size(),
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation,
                )
            )

        self.prediction_card.update_stat(
            f"{current_results['prediction'].upper()} ({current_results['confidence']:.0f}%)",
            {"clean": 0.25, "low": 0.50, "medium": 0.75, "high": 1.0}.get(current_results["prediction"].lower(), 0.0),
        )

        brush_percent = int(min(current_results["brush_rpm"] / 340, 1.0) * 100)
        fan_percent = int(min(current_results["fan_rpm"] / 3000, 1.0) * 100)

        self.brush_card.title_label.setText(f"BRUSH ({brush_percent}%)")
        self.fan_card.title_label.setText(f"FAN ({fan_percent}%)")

        self.brush_card.update_stat(
            f"{self.get_level_label(current_results['brush_rpm'], 'brush')} - {current_results['brush_rpm']} RPM",
            min(current_results["brush_rpm"] / 340, 1.0),
        )

        self.fan_card.update_stat(
            f"{self.get_level_label(current_results['fan_rpm'], 'fan')} - {current_results['fan_rpm']} RPM",
            min(current_results["fan_rpm"] / 3000, 1.0),
        )

        self.normal_power_card.update_stat(
            f"{self.normal_power:.2f} kW",
            min(self.normal_power / self.power.max_power, 1.0),
        )

        self.adaptive_power_card.update_stat(
            f"{current_results['adaptive_power']:.2f} kW",
            min(current_results["adaptive_power"] / self.power.max_power, 1.0),
        )

        self.saving_card.update_stat(
            f"{current_results['saving']:.1f}%",
            min(current_results["saving"] / 100, 1.0),
        )

    def get_level_label(self, rpm: int, control_type: str) -> str:
        if control_type == "brush":
            if rpm < 180:
                return "LOW"
            if rpm < 270:
                return "MEDIUM"
            return "HIGH"

        if control_type == "fan":
            if rpm < 1800:
                return "LOW"
            if rpm < 2600:
                return "MEDIUM"
            return "HIGH"

        return "UNKNOWN"

    def run_inference(self, frame, roi, roi_mask):
        prediction_label, confidence_score = self.predictor.predict(roi)
        road_roi, road_mask, coverage_score = self.segmenter.process(frame, roi_mask)

        settings = self.controller.get_settings(prediction_label)
        brush_speed = settings["brush"]
        fan_speed = settings["fan"]

        power_result = self.power.calculate(brush_speed, fan_speed)

        with self.inference_lock:
            self.model_results.update(
                {
                    "prediction": prediction_label,
                    "confidence": confidence_score,
                    "coverage": coverage_score,
                    "brush_rpm": brush_speed,
                    "fan_rpm": fan_speed,
                    "adaptive_power": power_result["power"],
                    "saving": power_result["saving"],
                    "road_roi": road_roi,
                }
            )

    def start_inference(self, frame, roi, roi_mask):
        if self.inference_thread is None or not self.inference_thread.is_alive():
            self.inference_thread = threading.Thread(
                target=self.run_inference,
                args=(frame.copy(), roi.copy(), roi_mask.copy()),
                daemon=True,
            )
            self.inference_thread.start()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()