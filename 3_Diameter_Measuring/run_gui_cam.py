import os
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from measuring_pipeline import process_frame, PipelineConfig

from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
)


# Make sure we draw to the Pi display
os.environ["DISPLAY"] = ":0"


class SilkGUI(QWidget):
    def __init__(self):
        super().__init__()

        # ----- Pipeline configuration -----
        self.config = PipelineConfig(
            um_per_px=1.2,        # adjust to your calibration
            slice_height_mm=0.25, # height of analyzed slice
        )

        # ----- Camera setup (Picamera2) -----
        self.picam2 = Picamera2()
        config = self.picam2.create_video_configuration(
            main={"size": (1440, 1080), "format": "RGB888"},
            controls={"FrameDurationLimits": (8333, 8333)},  # ~120 FPS limit
        )
        self.picam2.configure(config)
        self.picam2.start()
        self.picam2.set_controls({"AeEnable": True})

        # ===== IMAGE AREA (LEFT) =====
        # Main full-frame view
        self.main_video_label = QLabel("Main view")
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.main_video_label.setStyleSheet("background-color: black;")
        self.main_video_label.setMinimumSize(640, 480)

        # Secondary view (for now: zoomed crop / ROI)
        self.crop_video_label = QLabel("Top slice view")
        self.crop_video_label.setAlignment(Qt.AlignCenter)
        self.crop_video_label.setStyleSheet("background-color: black;")
        self.crop_video_label.setMinimumHeight(240)

        # Stack images vertically on the left
        images_layout = QVBoxLayout()
        images_layout.addWidget(self.main_video_label, stretch=3)
        images_layout.addWidget(self.crop_video_label, stretch=1)

        # ===== METRICS + CONTROLS (RIGHT) =====
        self.status_label = QLabel("Status: waiting for first frame")
        self.diameter_label = QLabel("Diameter stats: -")
        self.composition_label = QLabel("Composition: -")
        self.area_label = QLabel("Band area: -")
        self.fps_label = QLabel("FPS: -")

        for lbl in (
            self.status_label,
            self.diameter_label,
            self.composition_label,
            self.area_label,
            self.fps_label,
        ):
            lbl.setAlignment(Qt.AlignLeft)
            lbl.setWordWrap(True)

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("font-size: 18px; padding: 8px;")
        self.stop_button.clicked.connect(self.close)

        # Metrics grid
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(self.status_label,     0, 0, 1, 1)
        metrics_layout.addWidget(self.diameter_label,   1, 0, 1, 1)
        metrics_layout.addWidget(self.composition_label,2, 0, 1, 1)
        metrics_layout.addWidget(self.area_label,       3, 0, 1, 1)
        metrics_layout.addWidget(self.fps_label,        4, 0, 1, 1)

        # Right side vertical layout: metrics on top, Stop at bottom
        right_layout = QVBoxLayout()
        right_layout.addLayout(metrics_layout)
        right_layout.addStretch(1)
        right_layout.addWidget(self.stop_button, alignment=Qt.AlignRight | Qt.AlignBottom)

        # Put images (left) and metrics+button (right) side by side
        main_layout = QHBoxLayout()
        main_layout.addLayout(images_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        self.setLayout(main_layout)
        self.setWindowTitle("Spider Silk Diameter Measurement")

        # You want full screen
        # Size here is largely irrelevant once we go fullscreen
        self.resize(1280, 720)

        # ----- Timer for periodic frame updates -----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(0)  # as fast as pipeline allows

        # FPS tracking
        self._last_ts = time.time()
        self._fps = 0.0

    # ------------------------------------------------------------------
    # Core update loop
    # ------------------------------------------------------------------
    def update_frame(self):
        # Get latest frame from Picamera2
        try:
            frame_rgb = self.picam2.capture_array("main")
        except Exception as e:
            self.status_label.setText(f"Status: camera error: {e}")
            self.timer.stop()
            return

        if frame_rgb is None or frame_rgb.size == 0:
            self.status_label.setText("Status: empty frame from camera")
            return

        # Convert RGB -> BGR for OpenCV / pipeline
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Run measurement pipeline
        try:
            results, vis = process_frame(frame_bgr, self.config)
        except Exception as e:
            self.status_label.setText(f"Status: pipeline error: {e}")
            return

        # ---- Update FPS ----
        now = time.time()
        dt = now - self._last_ts
        if dt > 0:
            if self._fps <= 0:
                self._fps = 1.0 / dt
            else:
                # simple low-pass filter
                self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        self._last_ts = now
        self.fps_label.setText(f"FPS: {self._fps:.1f}")

        # ---- Update metrics text ----
        dia = results.get("diameter_stats", {})
        comp_percent = results.get("band_composition", {}).get("percent", {})
        band_area = results.get("band_area_px", 0)
        valid_band = results.get("valid_band", band_area > 0)
        slice_height_px = results.get("slice_height_px", vis.shape[0])

        if not valid_band or band_area == 0:
            self.status_label.setText("Status: no valid band detected")
            self.diameter_label.setText("Diameter stats: -")
            self.composition_label.setText("Composition: -")
            self.area_label.setText("Band area: 0 px")
        else:
            mean_um = dia.get("mean_um", np.nan)
            std_um = dia.get("std_um", np.nan)
            min_um = dia.get("min_um", np.nan)
            max_um = dia.get("max_um", np.nan)
            n_points = dia.get("n_points", 0)

            pure = comp_percent.get("pure", 0.0)
            unc = comp_percent.get("uncertainty", 0.0)
            bg = comp_percent.get("background", 0.0)
            other = comp_percent.get("other", 0.0)

            self.status_label.setText("Status: band detected")
            self.diameter_label.setText(
                f"Diameter stats: mean={mean_um:.1f} µm, std={std_um:.1f} µm, "
                f"min={min_um:.1f} µm, max={max_um:.1f} µm, n={n_points}"
            )
            self.composition_label.setText(
                f"Composition: pure={pure:.1f}%, uncertainty={unc:.1f}%, "
                f"bg={bg:.1f}%, other={other:.1f}%"
            )
            self.area_label.setText(f"Band area: {band_area} px")

        # ---- Prepare images for display ----
        # Full-frame overlay
        full_vis = vis

        # Top slice view (cropped and then scaled by Qt)
        slice_height_px = max(1, min(slice_height_px, full_vis.shape[0]))
        crop_vis = full_vis[:slice_height_px, :, :]

        # Show both
        self.show_frame(full_vis, self.main_video_label)
        self.show_frame(crop_vis, self.crop_video_label)

    def show_frame(self, frame_bgr: np.ndarray, label: QLabel):
        """Convert BGR numpy image to QImage and show in a given QLabel."""
        if frame_bgr is None or frame_bgr.size == 0:
            return

        height, width, ch = frame_bgr.shape
        rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bytes_per_line = ch * width
        qimg = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(
            pixmap.scaled(
                label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    # ------------------------------------------------------------------
    # Clean shutdown on Stop button / window close
    # ------------------------------------------------------------------
    def closeEvent(self, event):
        """Handle window close: stop timer and camera."""
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.picam2.stop()
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication([])
    win = SilkGUI()
    # Full-screen on the Pi touchscreen
    win.showFullScreen()
    app.exec_()


if __name__ == "__main__":
    main()
