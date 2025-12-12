import sys
import cv2
import numpy as np
import os
from datetime import datetime
import csv
import psutil

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

        # Recording state
        self.is_recording = False
        self.measurements_buffer = []  # List of dicts with measurement data
        self.images_folder = None  # Folder for current recording session images
        # Calculate buffer size dynamically: 75% of available memory / ~320 bytes per record
        try:
            available_mb = psutil.virtual_memory().available / (1024 * 1024)
            bytes_per_record = 320  # Estimated size of one measurement record
            # Use 75% of available memory as buffer
            self.max_buffer_size = max(1000, int(available_mb * 1024 * 1024 * 0.75 / bytes_per_record))
        except Exception:
            # Fallback to conservative default if psutil fails
            self.max_buffer_size = 10000
        # Store data under the 3_Diameter_Measuring folder next to this file
        self.data_folder = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(self.data_folder, exist_ok=True)

        # Recording timing
        self.recording_start_time = None
        self.recording_frame_count = 0
        self.last_frame_time = None

        # I2C state (placeholder)
        self.i2c_connected = False

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

        # Spacing
        right_layout.addSpacing(20)

        # Data Recording Section (moved up before stretch)
        data_header = QLabel("<h2>Data Recording</h2>", self)
        right_layout.addWidget(data_header)

        self.recording_label = QLabel("Ready", self)
        right_layout.addWidget(self.recording_label)

        # Recording control buttons layout
        recording_buttons_layout = QHBoxLayout()
        
        self.toggle_recording_button = QPushButton("START RECORDING", self)
        self.toggle_recording_button.setMinimumHeight(38)
        self.toggle_recording_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.toggle_recording_button.clicked.connect(self.toggle_recording)
        recording_buttons_layout.addWidget(self.toggle_recording_button)

        self.discard_recording_button = QPushButton("DISCARD", self)
        self.discard_recording_button.setMinimumHeight(38)
        self.discard_recording_button.setEnabled(False)
        self.discard_recording_button.setStyleSheet("background-color: #ff9800; color: white; font-weight: bold;")
        self.discard_recording_button.clicked.connect(self.discard_recording)
        recording_buttons_layout.addWidget(self.discard_recording_button)

        right_layout.addLayout(recording_buttons_layout)

        self.save_measurements_button = QPushButton("SAVE CSV", self)
        self.save_measurements_button.setMinimumHeight(38)
        self.save_measurements_button.setEnabled(False)
        self.save_measurements_button.clicked.connect(self.save_measurements_csv)
        right_layout.addWidget(self.save_measurements_button)

        # Storage space display
        self.storage_label = QLabel("Storage: Calculating...", self)
        right_layout.addWidget(self.storage_label)
        self.update_storage_display()  # Initial update

        # I2C Communication Section
        right_layout.addSpacing(20)
        i2c_header = QLabel("<h2>I2C Communication</h2>", self)
        right_layout.addWidget(i2c_header)

        self.i2c_status_label = QLabel("Disconnected", self)
        right_layout.addWidget(self.i2c_status_label)

        i2c_buttons_layout = QHBoxLayout()

        self.i2c_connect_button = QPushButton("CONNECT", self)
        self.i2c_connect_button.setMinimumHeight(38)
        self.i2c_connect_button.clicked.connect(self.i2c_connect)
        i2c_buttons_layout.addWidget(self.i2c_connect_button)

        self.i2c_send_button = QPushButton("SEND DATA", self)
        self.i2c_send_button.setMinimumHeight(38)
        self.i2c_send_button.setEnabled(False)
        self.i2c_send_button.clicked.connect(self.i2c_send_data)
        i2c_buttons_layout.addWidget(self.i2c_send_button)

        right_layout.addLayout(i2c_buttons_layout)

        # Growing stretch to push Settings to bottom
        right_layout.addStretch(1)

        # ======= Settings Section (ALWAYS AT BOTTOM) =======
        right_layout.addSpacing(20)
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

        # Save Settings and Exit buttons side by side
        settings_buttons_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Settings", self)
        self.save_button.setMinimumHeight(45)
        self.save_button.clicked.connect(self.save_settings)
        settings_buttons_layout.addWidget(self.save_button)

        self.exit_button = QPushButton("Exit", self)
        self.exit_button.setMinimumHeight(45)
        self.exit_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
        self.exit_button.clicked.connect(self.stop_gui)
        settings_buttons_layout.addWidget(self.exit_button)

        right_layout.addLayout(settings_buttons_layout)

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

            # Get extra data for display
            centroid_x = results.get("centroid_x", np.nan)
            centroid_y = results.get("centroid_y", np.nan)
            
            # Format centroid display
            if np.isfinite(centroid_x) and np.isfinite(centroid_y):
                centroid_str = f"X:{centroid_x:.0f} Y:{centroid_y:.0f}"
            else:
                centroid_str = "No silk"

            info_text = (
                f"<h2>Measurement & Speed</h2>"
                f"<table width='100%'>"
                f"<tr>"
                f"<td><b>Diameter:</b> {mean_um:.2f} µm (±{std_um:.2f})</td>"
                f"<td align='right'><b>FPS:</b> {fps:.1f}</td>"
                f"</tr>"
                f"<tr>"
                f"<td><b>Purity:</b> {pure_pct:.1f}% | Unc: {unc_pct:.1f}%</td>"
                f"<td align='right'><b>Measuring:</b> {speed_mm_s:.2f} mm/s</td>"
                f"</tr>"
                f"<tr>"
                f"<td><b>Position:</b> {centroid_str}</td>"
                f"<td></td>"
                f"</tr>"
                f"</table>"
                f"<br>"
                f"<h2>Benchmark (ms)</h2>"
                f"Total: {total_ms:.1f} | Splitting BG: {otsu2_ms:.1f} | Filling Gaps: {enforce_ms:.1f}<br>"
                f"Centerline: {skel_ms:.1f} | Finding Uncertainties: {otsu_band_ms:.1f}"
            )

            # Update recording buffer if recording is active
            if self.is_recording:
                current_time = datetime.now()
                time_since_start_ms = (current_time - self.recording_start_time).total_seconds() * 1000.0
                
                # Calculate frame duration
                if self.last_frame_time is not None:
                    frame_duration_ms = (current_time - self.last_frame_time).total_seconds() * 1000.0
                else:
                    frame_duration_ms = 0.0
                
                self.last_frame_time = current_time
                self.recording_frame_count += 1
                
                # Determine measurement quality
                n_pts = dia_stats.get("n_points", 0)
                valid = results.get("valid_band", False)
                if not valid or n_pts < 5:
                    quality = "low_confidence"
                elif n_pts < 20:
                    quality = "degraded"
                else:
                    quality = "good"
                
                centroid_x = results.get("centroid_x", np.nan)
                centroid_y = results.get("centroid_y", np.nan)
                
                meas_record = {
                    "timestamp": current_time.isoformat(),
                    "frame_number": self.recording_frame_count,
                    "time_since_start_ms": time_since_start_ms,
                    "frame_duration_ms": frame_duration_ms,
                    "diameter_um": mean_um,
                    "diameter_std_um": std_um,
                    "purity_pct": pure_pct,
                    "speed_mm_s": speed_mm_s,
                    "crop_height_mm": self.pipeline_config.slice_height_mm,
                    "n_points": n_pts,
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "measurement_quality": quality,
                }
                self.measurements_buffer.append(meas_record)
                count = len(self.measurements_buffer)
                percent_filled = int(100 * count / self.max_buffer_size)
                self.recording_label.setText(f"Recording: {count}/{self.max_buffer_size} ({percent_filled}%)")
                
                # Save images for this frame
                if self.images_folder is not None:
                    try:
                        frame_num = self.recording_frame_count
                        original_img = results.get("original_crop")
                        uncertainty_img = results.get("uncertainty_overlay")
                        
                        if original_img is not None:
                            original_path = os.path.join(self.images_folder, f"frame_{frame_num:06d}_original.png")
                            cv2.imwrite(original_path, original_img)
                        
                        if uncertainty_img is not None:
                            uncertainty_path = os.path.join(self.images_folder, f"frame_{frame_num:06d}_uncertainty.png")
                            cv2.imwrite(uncertainty_path, uncertainty_img)
                    except Exception as e:
                        print(f"Error saving images for frame {frame_num}: {e}")

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
    # Data Recording
    # -------------------------------------------------
    def update_storage_display(self):
        """Update available storage space display."""
        try:
            import shutil
            stat = shutil.disk_usage(self.data_folder)
            available_mb = stat.free / (1024 * 1024)
            available_gb = available_mb / 1024
            if available_gb > 1:
                self.storage_label.setText(f"Storage: {available_gb:.1f} GB available")
            else:
                self.storage_label.setText(f"Storage: {available_mb:.0f} MB available")
        except Exception:
            self.storage_label.setText("Storage: Unknown")

    def toggle_recording(self):
        """Toggle recording on/off."""
        if not self.is_recording:
            # Start recording
            self.is_recording = True
            self.measurements_buffer = []
            self.recording_start_time = datetime.now()
            self.last_frame_time = None
            self.recording_frame_count = 0
            
            # Create timestamped images folder
            timestamp_str = self.recording_start_time.strftime("%Y-%m-%d_%H%M%S")
            self.images_folder = os.path.join(self.data_folder, f"images_{timestamp_str}")
            os.makedirs(self.images_folder, exist_ok=True)
            
            self.recording_label.setText("Recording: 0 measurements")
            self.toggle_recording_button.setText("STOP RECORDING")
            self.toggle_recording_button.setStyleSheet("background-color: #f44336; color: white; font-weight: bold;")
            self.discard_recording_button.setEnabled(True)
            self.save_measurements_button.setEnabled(False)
        else:
            # Stop recording (images folder kept for saving)
            self.is_recording = False
            count = len(self.measurements_buffer)
            self.recording_label.setText(f"Stopped: {count} measurements (ready to save)")
            self.toggle_recording_button.setText("START RECORDING")
            self.toggle_recording_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
            self.discard_recording_button.setEnabled(True)
            self.save_measurements_button.setEnabled(count > 0)

    def discard_recording(self):
        """Discard current recording buffer and delete images."""
        self.measurements_buffer = []
        self.is_recording = False
        
        # Delete images folder if it exists
        if self.images_folder and os.path.exists(self.images_folder):
            import shutil
            try:
                shutil.rmtree(self.images_folder)
            except Exception as e:
                print(f"Error deleting images folder: {e}")
        self.images_folder = None
        
        self.recording_label.setText("Discarded - Ready")
        self.toggle_recording_button.setText("START RECORDING")
        self.toggle_recording_button.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.discard_recording_button.setEnabled(False)
        self.save_measurements_button.setEnabled(False)

    def save_measurements_csv(self):
        """Save buffered measurements to CSV file with decimal formatting."""
        if not self.measurements_buffer:
            self.recording_label.setText("No measurements to save")
            return

        try:
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            csv_filename = os.path.join(self.data_folder, f"measurements_{timestamp_str}.csv")

            # Write CSV with all tracked fields
            fieldnames = [
                "timestamp",
                "frame_number",
                "time_since_start_ms",
                "frame_duration_ms",
                "diameter_um",
                "diameter_std_um",
                "purity_pct",
                "speed_mm_s",
                "crop_height_mm",
                "n_points",
                "centroid_x",
                "centroid_y",
                "measurement_quality",
            ]

            # Numeric fields that should be formatted with decimal_places
            numeric_fields = {
                "diameter_um", "diameter_std_um", "purity_pct", "speed_mm_s",
                "crop_height_mm", "centroid_x", "centroid_y"
            }
            decimal_format = f"{{:.{self.pipeline_config.decimal_places}f}}"

            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Format numeric values with proper decimal places
                for row in self.measurements_buffer:
                    formatted_row = row.copy()
                    for field in numeric_fields:
                        if field in formatted_row and formatted_row[field] is not None:
                            try:
                                formatted_row[field] = decimal_format.format(float(formatted_row[field]))
                            except (ValueError, TypeError):
                                pass  # Keep original if formatting fails
                    writer.writerow(formatted_row)

            count = len(self.measurements_buffer)
            images_msg = f" & {count} image pairs" if self.images_folder else ""
            self.recording_label.setText(f"✓ Saved {count} measurements{images_msg} to {os.path.basename(csv_filename)}")
            self.measurements_buffer = []
            self.images_folder = None
            self.save_measurements_button.setEnabled(False)

        except Exception as e:
            self.recording_label.setText(f"✗ Save error: {e}")

    # -------------------------------------------------
    # I2C Communication (placeholder)
    # -------------------------------------------------
    def i2c_connect(self):
        """Connect to I2C device."""
        if not self.i2c_connected:
            # TODO: Implement actual I2C connection
            self.i2c_connected = True
            self.i2c_status_label.setText("✓ Connected (address: 0x50)")
            self.i2c_connect_button.setText("DISCONNECT")
            self.i2c_send_button.setEnabled(True)
        else:
            # TODO: Implement actual I2C disconnection
            self.i2c_connected = False
            self.i2c_status_label.setText("Disconnected")
            self.i2c_connect_button.setText("CONNECT")
            self.i2c_send_button.setEnabled(False)

    def i2c_send_data(self):
        """Send current measurement via I2C."""
        if not self.i2c_connected:
            self.i2c_status_label.setText("Not connected")
            return

        try:
            # TODO: Implement actual I2C data send
            # For now, just update status
            self.i2c_status_label.setText("✓ Data sent successfully")
        except Exception as e:
            self.i2c_status_label.setText(f"✗ Send failed: {e}")

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
