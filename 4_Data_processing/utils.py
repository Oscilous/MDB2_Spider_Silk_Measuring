"""
Utility functions for processing spider silk measurement data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from PIL import Image


def load_selection_config(config_path="selection_config.json"):
    """
    Load the section selection configuration file.
    
    Returns:
        dict: Dictionary mapping dataset names to lists of [start, end] frame pairs
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        return json.load(f)


def load_dataset(dataset_name, base_path=None):
    """
    Load measurement CSV and image folder for a dataset.
    
    Args:
        dataset_name: Timestamp string (e.g., "2025-12-12_150442")
        base_path: Path to data folder (defaults to ../data relative to this script)
    
    Returns:
        dict: {'df': DataFrame, 'images': Path, 'csv': Path}
    """
    if base_path is None:
        base = Path(__file__).resolve().parent.parent / "data"
    else:
        base = Path(base_path).resolve()  # Always resolve to absolute path
    
    csv_file = base / f"measurements_{dataset_name}.csv"
    
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: {csv_file}")
    
    # Try exact match first
    image_folder = base / f"images_{dataset_name}"
    
    # If not found, try to find closest matching folder
    if not image_folder.exists():
        date_part = dataset_name.split('_')[0]  # e.g., "2025-12-12"
        matching_folders = list(base.glob(f"images_{date_part}*"))
        
        if matching_folders:
            image_folder = matching_folders[0]
            print(f"Using image folder: {image_folder.name} for {dataset_name}")
        else:
            raise FileNotFoundError(f"No image folder found for {dataset_name}")
    
    df = pd.read_csv(csv_file)
    
    return {
        'df': df,
        'images': image_folder,
        'csv': csv_file
    }


def extract_sections(df, sections):
    """
    Extract selected sections from a DataFrame.
    
    Args:
        df: Full DataFrame with measurements
        sections: List of [start, end] frame indices (0-based)
    
    Returns:
        list: List of DataFrames, one per section
    """
    section_dfs = []
    for start, end in sections:
        section_df = df.iloc[start:end+1].copy()
        section_dfs.append(section_df)
    
    return section_dfs


def calculate_spatial_positions(df, strand_speed_mm_s=None):
    """
    Calculate the spatial position of each frame along the strand.
    
    Uses strand speed and time information to determine how far the strand moved.
    This uses the same calculation as image stitching for consistency.
    
    Args:
        df: DataFrame with 'time_since_start_ms' column
        strand_speed_mm_s: Strand speed in mm/s (from motor parameters).
                          If None, falls back to 'speed_mm_s' column in df.
    
    Returns:
        DataFrame with added 'position_mm' column
    """
    df = df.copy()
    
    # Get timing info
    times_ms = df['time_since_start_ms'].values
    
    # Calculate position: speed (mm/s) * time (s)
    # Use provided strand speed, or fall back to DataFrame column
    if strand_speed_mm_s is not None:
        # Use constant strand speed (same as stitching)
        positions_mm = [strand_speed_mm_s * (t / 1000.0) for t in times_ms]
    else:
        # Fall back to per-frame speed from CSV (may differ from motor speed)
        df['time_s'] = df['time_since_start_ms'] / 1000.0
        df['time_delta_s'] = df['time_s'].diff().fillna(0)
        df['displacement_mm'] = df['speed_mm_s'] * df['time_delta_s']
        positions_mm = df['displacement_mm'].cumsum().tolist()
    
    # Normalize to start from 0 within this section
    min_pos = min(positions_mm)
    df['position_mm'] = [p - min_pos for p in positions_mm]
    
    return df


def filter_outliers(df, column='diameter_um', method='iqr', threshold=1.5):
    """
    Filter outliers from measurements.
    
    Args:
        df: DataFrame
        column: Column to check for outliers
        method: 'iqr' (Interquartile Range) or 'std' (Standard Deviation)
        threshold: IQR multiplier or number of std deviations
    
    Returns:
        DataFrame with outliers marked in 'is_outlier' column
    """
    df = df.copy()
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    elif method == 'std':
        mean = df[column].mean()
        std = df[column].std()
        lower_bound = mean - threshold * std
        upper_bound = mean + threshold * std
        df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return df


def stitch_images_vertical(image_paths, output_path, section_df, strand_speed_mm_s, 
                           um_per_px=1.2, max_width=300):
    """
    Stitch images vertically to create a strand visualization.
    Uses timing data and strand speed to calculate proper spacing.
    
    Args:
        image_paths: List of Path objects to images
        output_path: Where to save the stitched image
        section_df: DataFrame with timing info (time_since_start_ms, crop_height_mm columns)
        strand_speed_mm_s: Calculated strand speed in mm/s
        um_per_px: Micrometers per pixel (camera calibration, default 1.2)
        max_width: Maximum width to resize images to
    
    Returns:
        Path to saved image
    """
    if not image_paths:
        return None
    
    # Get crop height from DataFrame (how much mm each image represents)
    crop_height_mm = section_df['crop_height_mm'].iloc[0] if 'crop_height_mm' in section_df.columns else 0.1
    
    # Calculate image height in pixels based on calibration
    # crop_height_mm -> µm -> pixels
    crop_height_um = crop_height_mm * 1000  # mm to µm
    image_height_px = int(crop_height_um / um_per_px)
    
    print(f"      Crop height: {crop_height_mm} mm = {crop_height_um} µm = {image_height_px} px")
    
    # Load and resize all images to match calibrated height
    images = []
    for img_path in image_paths:
        if not img_path.exists():
            images.append(None)
            continue
        img = Image.open(img_path)
        # Resize to calibrated dimensions
        aspect = img.width / img.height
        new_height = image_height_px
        new_width = int(new_height * aspect)
        if new_width > max_width:
            new_width = max_width
            new_height = int(new_width / aspect)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        images.append(img)
    
    # Filter out None images and get valid indices
    valid_images = [(i, img) for i, img in enumerate(images) if img is not None]
    if not valid_images:
        return None
    
    # Get timing info from DataFrame
    times_ms = section_df['time_since_start_ms'].values
    
    # Calculate positions based on strand speed and time
    # Position in mm = speed (mm/s) * time (s)
    positions_mm = []
    for t in times_ms:
        pos = strand_speed_mm_s * (t / 1000.0)  # Convert ms to s
        positions_mm.append(pos)
    
    # Normalize positions to start from 0
    min_pos = min(positions_mm)
    positions_mm = [p - min_pos for p in positions_mm]
    
    # Convert positions to pixels: mm -> µm -> pixels
    positions_px = [int(p * 1000 / um_per_px) for p in positions_mm]
    
    # Get image dimensions from first valid image
    img_width = valid_images[0][1].width
    img_height = valid_images[0][1].height
    
    # Calculate total height needed (last position + image height)
    max_position = max(positions_px) + img_height
    total_height = max_position
    
    print(f"      Total strand length: {max(positions_mm):.2f} mm = {total_height} px")
    
    # Create new image with black background
    stitched = Image.new('RGB', (img_width, total_height), color='black')
    
    # Paste images at calculated positions
    for idx, img in valid_images:
        if idx < len(positions_px):
            y_pos = positions_px[idx]
            stitched.paste(img, (0, y_pos))
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path)
    
    return output_path


def stitch_images_horizontal(image_paths, output_path, spacing_px=5, max_height=200):
    """
    Stitch images horizontally to create a strand visualization.
    
    Args:
        image_paths: List of Path objects to images
        output_path: Where to save the stitched image
        spacing_px: Pixels between images
        max_height: Maximum height to resize images to
    
    Returns:
        Path to saved image
    """
    images = []
    for img_path in image_paths:
        img = Image.open(img_path)
        # Resize maintaining aspect ratio
        aspect = img.width / img.height
        new_height = min(max_height, img.height)
        new_width = int(new_height * aspect)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        images.append(img)
    
    # Calculate total width
    total_width = sum(img.width for img in images) + spacing_px * (len(images) - 1)
    max_img_height = max(img.height for img in images)
    
    # Create new image
    stitched = Image.new('RGB', (total_width, max_img_height), color='white')
    
    # Paste images
    x_offset = 0
    for img in images:
        stitched.paste(img, (x_offset, 0))
        x_offset += img.width + spacing_px
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stitched.save(output_path)
    
    return output_path


def get_statistics(df, column='diameter_um'):
    """
    Calculate summary statistics for a measurement.
    
    Returns:
        dict: Statistics including mean, std, min, max, median, etc.
    """
    data = df[column].dropna()
    
    return {
        'mean': data.mean(),
        'std': data.std(),
        'min': data.min(),
        'max': data.max(),
        'median': data.median(),
        'q25': data.quantile(0.25),
        'q75': data.quantile(0.75),
        'count': len(data),
        'cv': (data.std() / data.mean() * 100) if data.mean() != 0 else 0  # Coefficient of variation
    }
