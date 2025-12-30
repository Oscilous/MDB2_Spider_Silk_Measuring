# Spider Silk Diameter Measurement System

A complete pipeline for measuring spider silk strand diameter using a Raspberry Pi with a camera module, real-time image processing, and post-processing analysis on PC.

This was built for an MSc project at university - it captures microscope images of spider silk as it's being drawn, measures the diameter in real-time using Otsu thresholding, and exports the data for further analysis.

## Hardware Setup

- **Raspberry Pi 5** with camera module (1440x1080 @ ~120 FPS)
- Microscope lens setup for magnification
- Motor-driven wheel system for drawing silk strand
- Touchscreen display on the Pi for live monitoring

## Project Structure

### 🔬 `1_Live_Feed/` — Camera Testing (Raspberry Pi)
Simple test script to verify the camera is working. Run this first to check your setup, adjust focus, etc. Captures images on keypress via SSH.

```bash
# On the Raspberry Pi
python test.py
```

### 📓 `2_X_Algorithm/` — Otsu Threshold Experiments (PC)
Jupyter notebooks for developing and testing the segmentation algorithms. These were used to figure out the best approach before porting to real-time.

- `2_1_Otsu3_Algorithm/` - Three-level Otsu thresholding approach
- `2_2_Double_Otsu2_Algorithm/` - Double two-level Otsu (what ended up working best)
- `2_3_Double_Otsu2_Algorithm_report_for_empty_too/` - Handles frames without silk visible

Open the `main.ipynb` in each folder to see the step-by-step image processing pipeline with example outputs.

### 📏 `3_Diameter_Measuring/` — Real-Time Measurement App (Raspberry Pi)
The main application that runs on the Pi. Full-screen PyQt5 GUI with live camera feed, real-time diameter measurement, and data recording.

```bash
# On the Raspberry Pi
python main.py
```

**Key files:**
- `main.py` — Entry point, camera initialization
- `silk_gui.py` — PyQt5 GUI with recording controls
- `measuring_pipeline.py` — Image processing (Otsu thresholding, morphology, diameter calculation)
- `settings.json` — Calibration values (µm/pixel, slice height, etc.)

Hit record to start collecting data. It saves:
- CSV with measurements (diameter, min/max, frame timing)
- Frame images with uncertainty visualization

### 📊 `4_Data_processing/` — Data Analysis (PC)
After collecting data on the Pi, copy it to your PC and run the analysis tools.

**Workflow:**
1. **`viewer.py`** — Browse through captured frames, mark "good" sections of the silk strand (avoiding breaks, tangles, etc.)
2. **`process.py`** — Process selected sections: filters outliers, calculates spatial positions based on motor speed, generates statistics

```bash
# Step 1: Select good sections
python viewer.py

# Step 2: Process and analyze
python process.py
```

Outputs include filtered CSVs, section metadata, and summary statistics.

### 📈 `5_Matlab_data/` — MATLAB Export
Contains `.mat` file export and a LiveScript for plotting in MATLAB if you prefer that over Python.

### 📁 `data/`
Where measurement CSVs and captured images are stored. Each recording session creates:
- `measurements_YYYY-MM-DD_HHMMSS.csv`
- `images_YYYY-MM-DD_HHMMSS/` folder with frame PNGs

---

## Getting Started

### On the Raspberry Pi

1. Install dependencies:
```bash
pip install -r requirements.txt
```
> Note: `picamera2` requires Raspberry Pi OS and may need: `sudo apt install python3-picamera2`

2. Clone this repo and navigate to `3_Diameter_Measuring/`

3. Edit `settings.json` to match your microscope calibration (µm per pixel)

4. Run:
```bash
python main.py
```

### On Your PC (for analysis)

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy the `data/` folder from the Pi to your PC

3. Use `viewer.py` to select good sections, then `process.py` to analyze

---

## Calibration

The key calibration value is `um_per_px` (micrometers per pixel) in `settings.json`. Measure this by imaging something with a known size under your microscope setup.

Default values assume:
- 1.2 µm/pixel
- 0.1 mm slice height for measurements

---

## Output Data

The measurement CSV contains per-frame data:
- `diameter_um` — Mean diameter in micrometers
- `min_diameter_um`, `max_diameter_um` — Range
- `timestamp_ms` — Frame timestamp
- `silk_visible` — Whether silk was detected

The processing step adds spatial position based on the strand speed (calculated from motor RPM, gear ratio, and wheel diameter).

---

## Notes

- The Pi runs headless most of the time — use SSH/VNC to start the app
- The touchscreen GUI is fullscreen; press the quit button to exit
- If Qt crashes on the Pi, check the `QT_QPA_PLATFORM_PLUGIN_PATH` env vars in `main.py`
- Frame images include an "uncertainty" visualization showing the confidence of the edge detection

---

## License

University project — use at your own risk. No warranty provided.
