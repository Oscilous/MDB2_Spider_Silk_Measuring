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
    """
    settings = {
        "pipeline": asdict(config),
        "display_scale": display_scale,
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
    """
    settings = load_settings()
    
    if settings is None:
        return PipelineConfig(), 0.4
    
    # Reconstruct PipelineConfig from saved dict
    pipeline_dict = settings.get("pipeline", {})
    config = PipelineConfig(
        um_per_px=pipeline_dict.get("um_per_px", 1.2),
        slice_height_mm=pipeline_dict.get("slice_height_mm", 0.25),
        frame_stride=pipeline_dict.get("frame_stride", 1),
    )
    
    display_scale = settings.get("display_scale", 0.4)
    
    return config, display_scale
