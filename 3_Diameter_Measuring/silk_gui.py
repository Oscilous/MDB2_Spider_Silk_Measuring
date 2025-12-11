import sys
import cv2
import numpy as np

from PyQt5.QtCore import QTimer, pyqtSignal, pyqtSlot, QObject, QThread
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QMainWindow,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QSizePolicy,
    QPushButton,
    QSlider,
)
from PyQt5.QtCore import Qt


class SilkGUI(QMainWindow):
    # Signal to request one processing step (handled in worker thread)
    request_process = pyqtSignal()

    """
    PyQt5 GUI:
      - Left: stacked image from pipeline (scaled down for display, ~50% for speed).
      - Right: metrics (timing, diameter, purity) as text.

    Full-resolution frames are processed by the pipeline; only display is scaled.
    """

    def __init__(self, camera, process_frame_fn, pipeline_config, display_scale=0.5, parent=None):
        super().__init__(parent)

        self.camera = camera
        self.process_frame_fn = process_frame_fn
        self.pipeline_config = pipeline_config
        self.display_scale = display_scale

        # ------------- ROOT LAYOUT (LEFT: IMAGE, RIGHT: INFO) -------------
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)

        # LEFT: image
        self.image_label = QLabel(self)
        # No stretching; keep 1:1 pixels
        self.image_label.setScaledContents(False)
        self.image_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.image_label)
        root_layout.addLayout(left_layout, stretch=0)

        # RIGHT: info (timings, diameters, purity, buttons, slider)
        right_layout = QVBoxLayout()
        self.info_label = QLabel(self)
        self.info_label.setWordWrap(True)
        right_layout.addWidget(self.info_label)

        # Spacing between info and settings
        right_layout.addSpacing(20)

        # Settings section header
        settings_header = QLabel("<h2>Settings</h2>", self)
        right_layout.addWidget(settings_header)

        # Crop height slider (slice_height_mm)
        self.crop_label = QLabel(f"Crop Height (mm): {self.pipeline_config.slice_height_mm:.2f}", self)
        right_layout.addWidget(self.crop_label)
        
        self.crop_slider = QSlider(Qt.Horizontal, self)
        self.crop_slider.setMinimum(1)      # 0.01 mm
        self.crop_slider.setMaximum(10)     # 0.1 mm
        self.crop_slider.setValue(int(self.pipeline_config.slice_height_mm * 100))
        self.crop_slider.setSingleStep(1)   # Only discrete values
        self.crop_slider.setPageStep(1)     # Page step also discrete
        self.crop_slider.setTickPosition(QSlider.TicksBelow)
        self.crop_slider.setTickInterval(1)
        # Make slider much thicker
        self.crop_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 30px;
                margin: 2px 0;
                background: #e0e0e0;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: 2px solid #0078d4;
                width: 40px;
                margin: -10px 0;
            }
        """)
        self.crop_slider.valueChanged.connect(self.on_crop_slider_moved)
        right_layout.addWidget(self.crop_slider)
        
        # Tick labels below slider
        tick_labels_layout = QHBoxLayout()
        for val in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            mm_val = val / 100.0
            tick_label = QLabel(f"{mm_val:.2f}", self)
            tick_label.setAlignment(Qt.AlignCenter)
            tick_labels_layout.addWidget(tick_label)
        right_layout.addLayout(tick_labels_layout)

        # Save button
        self.save_button = QPushButton("Save Settings", self)
        self.save_button.setMinimumHeight(60)
        self.save_button.clicked.connect(self.save_settings)
        right_layout.addWidget(self.save_button)

        # Stop button
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.setMinimumHeight(60)
        self.stop_button.clicked.connect(self.stop_gui)
        right_layout.addWidget(self.stop_button)

        # Placeholder stretch so you can drop more buttons later
        right_layout.addStretch(1)

        root_layout.addLayout(right_layout, stretch=1)

        self.setWindowTitle("Spider Silk Measuring")

        # ------------- Worker thread for processing -------------
        class Worker(QObject):
            result_ready = pyqtSignal(object, object)
            error = pyqtSignal(str)

            def __init__(self, camera, process_frame_fn, pipeline_config):
                super().__init__()
                self.camera = camera
                self.process_frame_fn = process_frame_fn
                self.pipeline_config = pipeline_config

            @pyqtSlot()
            def process_frame_slot(self):
                try:
                    frame_bgr = self.camera.capture_array("main")
                    if frame_bgr is None or frame_bgr.ndim != 3:
                        return
                    results, stacked = self.process_frame_fn(frame_bgr, self.pipeline_config)
                    # Emit results back to GUI thread
                    self.result_ready.emit(results, stacked)
                except Exception as e:
                    # emit error for GUI to show; avoid verbose console logging in production
                    self.error.emit(str(e))

        # Set up worker thread
        self._worker_thread = QThread(self)
        self._worker = Worker(self.camera, self.process_frame_fn, self.pipeline_config)
        self._worker.moveToThread(self._worker_thread)
        self._worker.result_ready.connect(self.on_worker_result)
        self._worker.error.connect(self.on_worker_error)
        self.request_process.connect(self._worker.process_frame_slot)
        self._worker_thread.start()

        # Timer to periodically request processing (non-blocking)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # request ~33 FPS

        # Waiting flag to avoid queuing multiple outstanding requests
        self._waiting_for_result = False

    # -------------------------------------------------
    # Main update loop
    # -------------------------------------------------
    def update_frame(self):
        """
        Called repeatedly by QTimer.
        1) Capture frame from camera.
        2) Run measuring pipeline.
        3) Show stacked image (original, band, regions).
        4) Update timing / metric text on the right.
        """
        # If a request is already outstanding, skip
        if self._waiting_for_result:
            return

        # Request the worker to process one frame
        try:
            self._waiting_for_result = True
            self.request_process.emit()
        except Exception as e:
            print("request_process emit failed:", e, file=sys.stderr)
            self._waiting_for_result = False
            return

    # -------------------------------------------------
    # Crop height slider
    # -------------------------------------------------
    def on_crop_slider_moved(self, value):
        """Update crop height when slider is moved."""
        # Convert slider value (1-10) to mm (0.01-0.10)
        slice_height_mm = value / 100.0
        self.pipeline_config.slice_height_mm = slice_height_mm
        self.crop_label.setText(f"Crop Height (mm): {slice_height_mm:.2f}")

    # -------------------------------------------------
    # Save settings
    # -------------------------------------------------
    def save_settings(self):
        """Save current settings to JSON file."""
        from settings_manager import save_settings
        try:
            save_settings(self.pipeline_config, self.display_scale)
            self.info_label.setText("✓ Settings saved!")
        except Exception as e:
            self.info_label.setText(f"✗ Error saving: {e}")

    # -------------------------------------------------
    # Worker result handlers
    # -------------------------------------------------
    def on_worker_result(self, results, stacked):
        """Handle results emitted from worker thread (runs in GUI thread)."""
        try:
            # ------------- Convert stacked BGR -> QImage (no scaling) -------------
            try:
                stacked_rgb = cv2.cvtColor(stacked, cv2.COLOR_BGR2RGB)
            except cv2.error:
                stacked_rgb = stacked

            h, w, ch = stacked_rgb.shape
            # Scale down for display (full-res already processed by pipeline)
            display_h = int(h * self.display_scale)
            display_w = int(w * self.display_scale)
            stacked_scaled = cv2.resize(stacked_rgb, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

            bytes_per_line = ch * display_w

            qimg = QImage(
                stacked_scaled.data,
                display_w,
                display_h,
                bytes_per_line,
                QImage.Format_RGB888,
            )

            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap)
            self.image_label.resize(pixmap.size())

            # Update info text
            timing = results.get("timing", {})
            dia_stats = results.get("diameter_stats", {})
            purity = results.get("purity", {})

            mean_um = dia_stats.get("mean_um", 0.0)
            std_um = dia_stats.get("std_um", 0.0)
            if not np.isfinite(mean_um):
                mean_um = 0.0
            if not np.isfinite(std_um):
                std_um = 0.0

            total_ms = timing.get("total_ms", 0.0)
            otsu2_ms = timing.get("otsu2_ms", 0.0)
            enforce_ms = timing.get("enforce_ms", 0.0)
            skel_ms = timing.get("skeleton_diameter_ms", 0.0)
            otsu_band_ms = timing.get("otsu_inside_band_ms", 0.0)

            pure_pct = purity.get("pure_pct", 0.0)
            unc_pct = purity.get("unc_pct", 0.0)

            if total_ms > 0:
                crop_height_mm = self.pipeline_config.slice_height_mm
                speed_mm_s = crop_height_mm / (total_ms / 1000.0)
                fps = 1000.0 / total_ms
            else:
                speed_mm_s = 0.0
                fps = 0.0

            info_text = (
                f"<h2>Measurement</h2>"
                f"Diameter: {mean_um:.2f} µm (±{std_um:.2f})<br>"
                f"Purity: {pure_pct:.1f}% | Uncertainty: {unc_pct:.1f}%<br>"
                f"<br>"
                f"<h2>Speed</h2>"
                f"FPS: {fps:.1f}<br>"
                f"Measuring: {speed_mm_s:.2f} mm/s<br>"
                f"<br>"
                f"<h2>Benchmark</h2>"
                f"Total: {total_ms:.1f}ms | Otsu2: {otsu2_ms:.1f}ms | Enforce: {enforce_ms:.1f}ms<br>"
                f"Skeleton: {skel_ms:.1f}ms | Otsu-band: {otsu_band_ms:.1f}ms"
            )

            self.info_label.setText(info_text)

        except Exception:
            # suppress detailed console logging in production; GUI will show an error if needed
            pass
        finally:
            # allow next frame request
            self._waiting_for_result = False

    def on_worker_error(self, err_msg):
        try:
            self.info_label.setText(f"Worker error: {err_msg}")
        except Exception:
            pass
        self._waiting_for_result = False

    # -------------------------------------------------
    # Stop function
    # -------------------------------------------------
    def stop_gui(self):
        """Stop the GUI and close the application."""
        self.close()

    # -------------------------------------------------
    # Proper cleanup when window closes
    # -------------------------------------------------
    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass

        # Stop worker thread cleanly
        try:
            if hasattr(self, "_worker_thread") and self._worker_thread.isRunning():
                self._worker_thread.quit()
                self._worker_thread.wait(1000)
        except Exception:
            pass

        event.accept()
