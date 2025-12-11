from picamera2 import Picamera2
import cv2
import os
import time
import sys
import select

os.environ["DISPLAY"] = ":0"

picam2 = Picamera2()

config = picam2.create_video_configuration(
    main={"size": (1440, 1080), "format": "RGB888"},
    controls={"FrameDurationLimits": (8333, 8333)},
)

picam2.configure(config)
picam2.start()

# Auto-exposure enabled for debugging
picam2.set_controls({"AeEnable": True})

print("Press 'c' in the SSH terminal to capture an image.")
print("Press 'q' in the SSH terminal to quit.")

# Create a resizable window
cv2.namedWindow("Live Feed", cv2.WINDOW_NORMAL)

# Get desktop resolution using tkinter (reliable on Raspberry Pi)
try:
    import tkinter as tk
    root = tk.Tk()
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
except:
    # fallback if tkinter is missing
    screen_w, screen_h = 1280, 720

# Leave room for the Pi taskbar (~40px)
taskbar_margin = 50

cv2.resizeWindow("Live Feed", screen_w, screen_h - taskbar_margin)
cv2.moveWindow("Live Feed", 0, 0)   # top-left corner

while True:
    # Acquire the most recent frame
    frame = picam2.capture_array("main")

    # Display on the Pi touchscreen
    cv2.imshow("Live Feed", frame)
    cv2.waitKey(1)       # only to refresh the GUI; not used for input

    # --- Terminal input handling (SSH) ---
    # select() checks if there is data to read without blocking
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        key = sys.stdin.read(1)

        if key == 'c':
            # Timestamp-based filename for reproducibility
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.png"
            cv2.imwrite(filename, frame)
            print(f"[INFO] Captured image saved as: {filename}")

        elif key == 'q':
            print("[INFO] Quitting.")
            break

picam2.stop()
cv2.destroyAllWindows()
