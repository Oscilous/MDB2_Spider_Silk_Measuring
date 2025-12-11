from picamera2 import Picamera2
import cv2
import os
import time

from measuring_pipeline import process_frame, PipelineConfig

# Make sure we draw to the Pi display
os.environ["DISPLAY"] = ":0"


def main():
    # --- Pipeline config (tune later if needed) ---
    cfg = PipelineConfig(
        um_per_px=1.2,        # adjust to your calibration
        slice_height_mm=0.25, # vertical slice to analyze
    )

    # --- Camera setup (same idea as your working code) ---
    picam2 = Picamera2()

    config = picam2.create_video_configuration(
        main={"size": (1440, 1080), "format": "RGB888"},
        controls={"FrameDurationLimits": (8333, 8333)},  # ~120 FPS limit, real FPS will be lower
    )

    picam2.configure(config)
    picam2.start()

    # Auto exposure on (you had this for debugging)
    picam2.set_controls({"AeEnable": True})

    # --- OpenCV window on Pi touchscreen ---
    cv2.namedWindow("Silk analysis", cv2.WINDOW_NORMAL)

    # Optional: maximize window to screen size
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        root.destroy()
    except Exception:
        screen_w, screen_h = 1280, 720

    taskbar_margin = 50
    cv2.resizeWindow("Silk analysis", screen_w, screen_h - taskbar_margin)
    cv2.moveWindow("Silk analysis", 0, 0)

    print("Press 'q' in the OpenCV window to quit.")

    last_print = time.time()

    while True:
        # Get newest frame from camera (RGB)
        frame_rgb = picam2.capture_array("main")

        # Convert RGB -> BGR for OpenCV + pipeline
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # --- Run your analysis pipeline ---
        results, vis = process_frame(frame_bgr, cfg)

        # Extract some key metrics to show
        dia = results["diameter_stats"]
        comp = results["band_composition"]["percent"]

        if not results["valid_band"]:
            text1 = "No valid band detected"
            text2 = ""
        else:
            text1 = f"mean={dia['mean_um']:.1f} µm  std={dia['std_um']:.1f} µm"
            text2 = f"pure={comp['pure']:.1f}%  unc={comp['uncertainty']:.1f}%  bg={comp['background']:.1f}%"


        # Simple text overlay with mean diameter and composition
        text1 = f"mean={dia['mean_um']:.1f} µm  std={dia['std_um']:.1f} µm"
        text2 = f"pure={comp['pure']:.1f}%  unc={comp['uncertainty']:.1f}%  bg={comp['background']:.1f}%"

        cv2.putText(vis, text1, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis, text2, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Show post-processed overlay on touchscreen
        cv2.imshow("Silk analysis", vis)

        # Needed for GUI refresh + key handling
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Optional: also print to terminal once per second
        now = time.time()
        if now - last_print > 1.0:
            last_print = now
            print(text1, "|", text2)

    picam2.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
