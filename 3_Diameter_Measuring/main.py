import os
import sys

from PyQt5.QtWidgets import QApplication
from picamera2 import Picamera2

from measuring_pipeline import process_frame, PipelineConfig
from silk_gui import SilkGUI
from settings_manager import get_default_config_and_scale


def main():
    # Make sure we draw to the Pi display
    os.environ["DISPLAY"] = ":0"

    # ----- Load or use default pipeline configuration -----
    pipeline_config, display_scale = get_default_config_and_scale()

    # ----- Camera setup (Picamera2) -----
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(
        main={"size": (1440, 1080), "format": "RGB888"},
        controls={"FrameDurationLimits": (8333, 8333)},  # ~120 FPS limit
    )
    picam2.configure(video_config)
    picam2.start()
    picam2.set_controls({"AeEnable": True})

    # ----- Qt Application + GUI -----
    app = QApplication(sys.argv)

    # Inject camera + pipeline into the GUI
    win = SilkGUI(
        camera=picam2,
        process_frame_fn=process_frame,
        pipeline_config=pipeline_config,
        display_scale=display_scale,
    )

    # Full-screen on the Pi touchscreen
    win.showFullScreen()

    # Run event loop
    exit_code = app.exec_()

    # Just in case GUI didn't already stop the camera
    try:
        picam2.stop()
    except Exception:
        pass

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
