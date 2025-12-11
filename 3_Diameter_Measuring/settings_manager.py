"""
Settings manager for the silk measuring pipeline.
Handles saving/loading settings to/from JSON.
"""

import json
import os
from pathlib import Path
from dataclasses import asdict
from measuring_pipeline import PipelineConfig


SETTINGS_FILE = os.path.join(os.path.dirname(__file__), "settings.json")


def save_settings(config: PipelineConfig, display_scale: float) -> None:
    """
    Save pipeline config and display settings to JSON file.
    Separates hardcoded calibration values from user-editable settings.
    """
    settings = {
        "calibration": {
            "um_per_px": config.um_per_px,  # Camera calibration
            "frame_stride": config.frame_stride,  # Frame processing stride
            "decimal_places": config.decimal_places,  # Decimal precision for CSV export
        },
        "user_settings": {
            "slice_height_mm": config.slice_height_mm,  # Editable: crop height
            "display_scale": display_scale,  # Editable: display scaling
        },
    }
    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings, f, indent=2)
    print(f"Settings saved to {SETTINGS_FILE}")


def load_settings() -> dict:
    """
    Load settings from JSON file if it exists.
    Returns dict with 'pipeline' and 'display_scale' keys.
    """
    if not os.path.exists(SETTINGS_FILE):
        return None
    
    try:
        with open(SETTINGS_FILE, "r") as f:
            settings = json.load(f)
        print(f"Settings loaded from {SETTINGS_FILE}")
        return settings
    except Exception as e:
        print(f"Error loading settings: {e}")
        return None


def get_default_config_and_scale() -> tuple:
    """
    Load settings from JSON if available, otherwise return defaults.
    Returns (PipelineConfig, display_scale).
    
    Settings are organized as:
    - calibration: hardcoded values (um_per_px, frame_stride, decimal_places)
    - user_settings: editable from UI (slice_height_mm, display_scale)
    """
    settings = load_settings()
    
    if settings is None:
        return PipelineConfig(), 0.4
    
    # Get calibration (hardcoded) values
    calibration = settings.get("calibration", {})
    um_per_px = calibration.get("um_per_px", 1.2)
    frame_stride = calibration.get("frame_stride", 1)
    decimal_places = calibration.get("decimal_places", 2)
    
    # Get user-editable values
    user_settings = settings.get("user_settings", {})
    slice_height_mm = user_settings.get("slice_height_mm", 0.25)
    display_scale = user_settings.get("display_scale", 0.4)
    
    # Reconstruct PipelineConfig
    config = PipelineConfig(
        um_per_px=um_per_px,
        slice_height_mm=slice_height_mm,
        frame_stride=frame_stride,
        decimal_places=decimal_places,
    )
    
    return config, display_scale
