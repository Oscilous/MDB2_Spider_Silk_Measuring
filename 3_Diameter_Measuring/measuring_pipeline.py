"""
Real-time analysis pipeline for silk band / thread in microscope images.

This module is designed to be imported by a Raspberry Pi application that:
- Grabs frames from a camera (BGR, OpenCV),
- Calls process_frame(frame, config),
- Displays the returned overlay + metrics in a lightweight GUI.

Dependencies:
    - opencv-python
    - numpy
    - scikit-image
    - scipy

The code is based on the logic you developed in the Jupyter notebook,
but refactored into clean, reusable functions with no plotting or GUI.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_multiotsu
from skimage.morphology import (
    remove_small_objects,
    remove_small_holes,
    disk,
    binary_opening,
    binary_closing,
    skeletonize,
)
from skimage.measure import label
from scipy.ndimage import distance_transform_edt
from skimage import img_as_ubyte


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Physical / geometric
    um_per_px: float = 1.2         # micrometres per pixel (adjust to your calibration)
    slice_height_mm: float = 0.25  # height of the analyzed slice in mm

    # Mask cleaning parameters for the silk band
    min_silk_size_px: int = 200    # remove smaller connected components
    silk_hole_size_px: int = 200   # fill small holes inside silk
    silk_open_radius: int = 1      # radius for morphological opening
    silk_close_radius: int = 2     # radius for morphological closing

    # Optional: limit processing to every N-th frame (can be used later)
    frame_stride: int = 1


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def clean_silk_mask(
    mask: np.ndarray,
    min_size: int = 200,
    hole_size: int = 200,
    open_radius: int = 1,
    close_radius: int = 2,
) -> np.ndarray:
    """
    Clean a raw band/silk mask.

    Parameters
    ----------
    mask : 2D array-like
        Binary mask with silk ≈ 1.
    min_size : int
        Minimum connected-component area to keep (px).
    hole_size : int
        Maximum hole area to fill (px).
    open_radius : int
        Structuring element radius for opening.
    close_radius : int
        Structuring element radius for closing.

    Returns
    -------
    cleaned : 2D bool array
    """
    mask_bool = mask.astype(bool)

    # Remove tiny specks (dust, noise)
    if min_size is not None and min_size > 0:
        mask_bool = remove_small_objects(mask_bool, min_size=min_size)

    # Fill small holes inside silk band
    if hole_size is not None and hole_size > 0:
        mask_bool = remove_small_holes(mask_bool, area_threshold=hole_size)

    # Opening: remove thin protrusions
    if open_radius is not None and open_radius > 0:
        selem = disk(open_radius)
        mask_bool = binary_opening(mask_bool, selem)

    # Closing: smooth contours / bridge tiny gaps
    if close_radius is not None and close_radius > 0:
        selem = disk(close_radius)
        mask_bool = binary_closing(mask_bool, selem)

    # Optionally, keep only the largest connected component
    lbl = label(mask_bool)
    if lbl.max() > 1:
        counts = np.bincount(lbl.ravel())
        counts[0] = 0  # ignore background
        largest = counts.argmax()
        mask_bool = lbl == largest

    return mask_bool


def compute_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Compute a 1-pixel-wide skeleton for a binary mask using skimage.skeletonize.

    Parameters
    ----------
    mask : 2D bool or 0/1

    Returns
    -------
    skeleton : 2D bool
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)

    skel = skeletonize(m)
    return skel.astype(bool)


def diameters_from_skeleton(
    skeleton: np.ndarray,
    dist_map: np.ndarray,
    um_per_px: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute local diameters along a 1-pixel-wide skeleton using the
    Euclidean distance transform.

    Parameters
    ----------
    skeleton : 2D bool or 0/1
        Skeleton mask: True/1 on the centreline.
    dist_map : 2D float
        Euclidean distance transform of the *same* binary mask used
        to generate the skeleton. Each pixel = distance [px] to nearest
        background.
    um_per_px : float
        Micrometres per pixel.

    Returns
    -------
    diam_px : 1D float array
        Local diameters in pixels.
    diam_um : 1D float array
        Local diameters in micrometres.
    stats : dict
        {'mean_um', 'std_um', 'min_um', 'max_um', 'n_points'}
    """
    skel_bool = skeleton.astype(bool)

    if not skel_bool.any():
        diam_px = np.array([], dtype=float)
        diam_um = np.array([], dtype=float)
        stats = {
            "mean_um": np.nan,
            "std_um": np.nan,
            "min_um": np.nan,
            "max_um": np.nan,
            "n_points": 0,
        }
        return diam_px, diam_um, stats

    radii_px = dist_map[skel_bool]
    radii_px = radii_px[radii_px > 0]  # remove any zero artefacts

    if radii_px.size == 0:
        diam_px = np.array([], dtype=float)
        diam_um = np.array([], dtype=float)
        stats = {
            "mean_um": np.nan,
            "std_um": np.nan,
            "min_um": np.nan,
            "max_um": np.nan,
            "n_points": 0,
        }
        return diam_px, diam_um, stats

    diam_px = 2.0 * radii_px
    diam_um = diam_px * um_per_px

    stats = {
        "mean_um": float(diam_um.mean()),
        "std_um": float(diam_um.std()),
        "min_um": float(diam_um.min()),
        "max_um": float(diam_um.max()),
        "n_points": int(diam_um.size),
    }
    return diam_px, diam_um, stats


def analyze_band(
    band_mask: np.ndarray,
    otsu_regions: np.ndarray,
    treat_remainder: str = "bg",
) -> Dict[str, Any]:
    """
    Analyze a single band with respect to 3-class Otsu segmentation.

    Parameters
    ----------
    band_mask : 2D bool
        Mask selecting the detected band (True inside band).
    otsu_regions : 2D uint8
        Otsu classes with 3 levels:
            0 -> pure silk
            1 -> uncertainty (gray, possibly liquid)
            2 -> background
    treat_remainder : {'bg', 'other'}
        If 'bg', unlabeled pixels are added to background.
        If 'other', they are tracked separately.

    Returns
    -------
    result : dict
        {
            'band_area': int,
            'counts': {'pure', 'uncertainty', 'background', 'other'},
            'percent': same keys in percent of band area,
            'masks': {'pure', 'uncertainty', 'background', 'other'} boolean masks
        }
    """
    band = band_mask.astype(bool)
    regs = otsu_regions.astype(np.int32)

    if band.shape != regs.shape:
        raise ValueError("band_mask and otsu_regions must have same shape")

    band_area = int(band.sum())

    if band_area == 0:
        counts = {k: 0 for k in ["pure", "uncertainty", "background", "other"]}
        percent = {k: 0.0 for k in counts.keys()}
        masks = {k: np.zeros_like(band, dtype=bool) for k in counts.keys()}
        return {
            "band_area": 0,
            "counts": counts,
            "percent": percent,
            "masks": masks,
        }

    # Restrict Otsu regions to band
    pure_mask = band & (regs == 0)
    unc_mask = band & (regs == 1)
    bg_mask = band & (regs == 2)

    known_mask = pure_mask | unc_mask | bg_mask
    remainder_mask = band & (~known_mask)

    if treat_remainder == "bg":
        bg_mask = bg_mask | remainder_mask
        remainder_mask = np.zeros_like(bg_mask, dtype=bool)

    counts = {
        "pure": int(pure_mask.sum()),
        "uncertainty": int(unc_mask.sum()),
        "background": int(bg_mask.sum()),
        "other": int(remainder_mask.sum()),
    }

    # Percentages w.r.t. band area
    percent = {k: (v / band_area * 100.0) for k, v in counts.items()}

    masks = {
        "pure": pure_mask,
        "uncertainty": unc_mask,
        "background": bg_mask,
        "other": remainder_mask,
    }

    return {
        "band_area": band_area,
        "counts": counts,
        "percent": percent,
        "masks": masks,
    }


def make_overlay(
    gray_crop: np.ndarray,
    masks: Dict[str, np.ndarray],
    alpha: float = 0.6,
) -> np.ndarray:
    """
    Create a color overlay visualization for the cropped grayscale image.

    Colors (BGR):
        pure silk      -> green
        uncertainty    -> blue
        background     -> red
        other          -> yellow

    Parameters
    ----------
    gray_crop : 2D uint8
    masks : dict of bool masks
    alpha : float
        blending factor (0..1)

    Returns
    -------
    overlay_bgr : 3D uint8
    """
    gray_u8 = img_as_ubyte(gray_crop)
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    h, w = gray_crop.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    pure = masks.get("pure", np.zeros_like(gray_crop, bool))
    unc = masks.get("uncertainty", np.zeros_like(gray_crop, bool))
    bg = masks.get("background", np.zeros_like(gray_crop, bool))
    other = masks.get("other", np.zeros_like(gray_crop, bool))

    # Pure silk -> green (0,255,0)
    color[pure, 1] = 255
    # Uncertainty -> blue (255,0,0) in BGR
    color[unc, 0] = 255
    # Background -> red (0,0,255)
    color[bg, 2] = 255
    # Other -> yellow (0,255,255)
    color[other, 1] = 255
    color[other, 2] = 255

    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0.0)
    return overlay


# ---------------------------------------------------------------------------
# Main entry point for real-time use
# ---------------------------------------------------------------------------

def process_frame(
    frame_bgr: np.ndarray,
    config: PipelineConfig | None = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Main pipeline entry point.

    Parameters
    ----------
    frame_bgr : 3D uint8 array
        BGR image from OpenCV (camera frame).
    config : PipelineConfig or None
        Configuration. If None, defaults are used.

    Returns
    -------
    results : dict
        {
            'diameter_stats': {...},
            'band_composition': {...},
            'band_area_px': int,
            'slice_height_px': int
        }
    vis_image : 3D uint8 array
        BGR overlay image (same size as input frame) for display.
    """
    if config is None:
        config = PipelineConfig()

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image (H,W,3) uint8")

    h, w, _ = frame_bgr.shape

    # Compute slice height in pixels, clamp to image height
    slice_height_px = int(config.slice_height_mm * 1000.0 * config.um_per_px)
    slice_height_px = max(1, min(slice_height_px, h))

    # Convert to grayscale and crop top slice
        # Convert to grayscale and crop top slice
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_crop = gray[:slice_height_px, :]

    # ------------------------------------------------------------------
    # Robust 3-class Otsu segmentation on cropped gray
    # If the image is too flat (no contrast) or multi-otsu fails,
    # we treat the frame as "no valid band" instead of crashing.
    # ------------------------------------------------------------------
    # Quick contrast check
    if gray_crop.std() < 1.0:
        # Too little variation: treat everything as background (class 2)
        otsu_regions = np.full_like(gray_crop, 2, dtype=np.uint8)
    else:
        try:
            thresholds = threshold_multiotsu(gray_crop, classes=3)
            otsu_regions = np.digitize(gray_crop, bins=thresholds).astype(np.uint8)
        except ValueError:
            # e.g. "only 1 different value" after binning
            otsu_regions = np.full_like(gray_crop, 2, dtype=np.uint8)

    # Silk band = core + uncertainty
    core_mask = otsu_regions == 0
    unc_mask = otsu_regions == 1
    raw_band = core_mask | unc_mask

    # Clean band mask
    band_mask = clean_silk_mask(
        raw_band,
        min_size=config.min_silk_size_px,
        hole_size=config.silk_hole_size_px,
        open_radius=config.silk_open_radius,
        close_radius=config.silk_close_radius,
    )

    # Skeleton + distance transform on band
    skeleton = compute_skeleton(band_mask)
    dist_map = distance_transform_edt(band_mask.astype(np.uint8))

    diam_px, diam_um, dia_stats = diameters_from_skeleton(
        skeleton,
        dist_map,
        config.um_per_px,
    )

    # Band composition analysis
    band_analysis = analyze_band(
        band_mask=band_mask,
        otsu_regions=otsu_regions,
        treat_remainder="bg",
    )

    # Visualization overlay only on cropped area
    overlay_crop = make_overlay(gray_crop, band_analysis["masks"], alpha=0.6)

    # Embed cropped overlay back into full-size frame
    vis_full = frame_bgr.copy()
    vis_full[:slice_height_px, :] = overlay_crop

    valid_band = band_analysis["band_area"] > 0

    results = {
        "diameter_stats": dia_stats,
        "band_composition": band_analysis,
        "band_area_px": band_analysis["band_area"],
        "slice_height_px": slice_height_px,
        "valid_band": bool(valid_band),
    }


    return results, vis_full


# ---------------------------------------------------------------------------
# Convenience function for offline testing on still images
# ---------------------------------------------------------------------------

def analyze_image_path(
    img_path: str,
    config: PipelineConfig | None = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Convenience wrapper for testing on a single image from disk.
    """
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    return process_frame(img, config=config)