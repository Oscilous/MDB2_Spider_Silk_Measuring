"""
Real-time analysis pipeline for silk band / thread in microscope images.

This module is designed to be imported by a Raspberry Pi application that:
- Grabs frames from a camera (BGR, OpenCV),
- Calls process_frame(frame, config),
- Displays the returned overlay + metrics in a lightweight GUI.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple

import time
import sys
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
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage import img_as_ubyte


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    # Physical / geometric
    um_per_px: float = 1.2         # micrometres per pixel (adjust to your calibration)
    slice_height_mm: float = 0.25  # height of the analyzed slice in mm

    # Output formatting
    decimal_places: int = 2        # decimal places for saved data (0-6)

    # Optional: limit processing to every N-th frame (can be used later)
    frame_stride: int = 1


# ---------------------------------------------------------------------------
# Notebook-derived core functions
# ---------------------------------------------------------------------------

def enforce_two_bg_one_silk(core: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Enforce exactly two background regions and one connected silk region.
    If unable to find 2 backgrounds (e.g., silk touching edges), returns all-background.
    """
    # enforce_two_bg_one_silk: diagnostics removed for production
    band_raw = core.astype(bool)
    bg_orig = ~band_raw

    def _sorted_components(mask_bool: np.ndarray):
        lbl = label(mask_bool)
        comps = []
        for lab in range(1, lbl.max() + 1):
            comp = (lbl == lab)
            comps.append((int(comp.sum()), comp))
        comps.sort(key=lambda x: x[0], reverse=True)
        return comps

    # 1) Two background regions
    areas_o = _sorted_components(bg_orig)

    if len(areas_o) >= 2:
        bg1 = areas_o[0][1].copy()
        bg2 = areas_o[1][1].copy()
        final_bg = bg1 | bg2
    else:
        # Only 1 background component found - likely silk touching edge
        # Quick check: if background spans >80% of width, silk is definitely at edge
        if len(areas_o) == 1:
            bg_comp = areas_o[0][1]
            # Check horizontal extent
            cols_with_bg = np.any(bg_comp, axis=0)
            bg_width = cols_with_bg.sum()
            if bg_width > 0.8 * bg_comp.shape[1]:
                # silk touching edge -> treat as no-silk case
                all_bg = np.ones_like(core, dtype=np.uint8)
                no_silk = np.zeros_like(core, dtype=np.uint8)
                return no_silk, all_bg, (1, 0)
        
        # Try to grow background to get at least 2 components (rare case)
        bg = bg_orig.copy()
        radius = 1
        max_radius = min(max(bg.shape) // 4, 5)  # Reduced from 10 to 5 for speed
        found_two = False

        while radius <= max_radius:
            try:
                comps = _sorted_components(bg)
                if len(comps) >= 2:
                    bg1 = comps[0][1].copy()
                    bg2 = comps[1][1].copy()
                    final_bg = bg1 | bg2
                    found_two = True
                    break

                # Safety: if bg has very few pixels, abort
                if bg.sum() < 10:
                    break

                bg = binary_closing(bg, disk(radius))
                bg = binary_fill_holes(bg)
                radius += 1
            except Exception:
                # On unexpected failures, abort growth and fall back
                break

        if not found_two:
            # Can't find 2 backgrounds (silk touching edges)
            # Return all-background (no silk detected) -> measurements will be 0
            all_bg = np.ones_like(core, dtype=np.uint8)
            no_silk = np.zeros_like(core, dtype=np.uint8)
            return no_silk, all_bg, (1, 0)

    # 2) Silk = complement of chosen background, filled
    final_silk = ~final_bg
    final_silk = binary_fill_holes(final_silk)

    # 3) Ensure silk is a single component
    lbl_silk = label(final_silk)
    if lbl_silk.max() > 1:
        silk_comps = _sorted_components(final_silk)
        main_silk = silk_comps[0][1].copy()
        other_silk = np.zeros_like(final_silk, dtype=bool)
        for _, comp in silk_comps[1:]:
            other_silk |= comp

        # Distances to each background half
        dist_bg1 = distance_transform_edt(~bg1)
        dist_bg2 = distance_transform_edt(~bg2)

        assign_to_bg1 = (dist_bg1 < dist_bg2) & other_silk
        assign_to_bg2 = (dist_bg2 <= dist_bg1) & other_silk

        bg1 |= assign_to_bg1
        bg2 |= assign_to_bg2

        final_bg = bg1 | bg2
        final_silk = ~final_bg
        final_silk = binary_fill_holes(final_silk)

    # 4) Safety fallback
    if final_silk.sum() == 0:
        final_silk = binary_fill_holes(band_raw)
        if final_silk.sum() == 0:
            final_silk = band_raw

    n_bg_final = label(final_bg).max()
    n_silk_final = label(final_silk).max()

    # finished enforce
    return final_silk.astype(np.uint8), final_bg.astype(np.uint8), (n_bg_final, n_silk_final)


def compute_skeleton(
    mask: np.ndarray,
    method: str = "centerline",
    smooth_window: int = 11,
    width: int = 1,
    closing_height: int = 3,
) -> np.ndarray:
    """
    Compute a 1-pixel-wide skeleton for a binary mask.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.zeros_like(m, dtype=bool)

    if method == "skeletonize":
        return skeletonize(m).astype(bool)

    # "centerline" method
    h, w = m.shape
    xs = np.full(h, np.nan, dtype=float)

    # For each row, take median of foreground columns as center
    for y in range(h):
        cols = np.nonzero(m[y, :])[0]
        if cols.size:
            xs[y] = np.median(cols)

    # If no rows contained silk, return empty skeleton
    if not np.isfinite(xs).any():
        return np.zeros_like(m, dtype=bool)

    # Interpolate missing rows
    rows = np.arange(h)
    good = np.isfinite(xs)
    if not good.all():
        xs[~good] = np.interp(rows[~good], rows[good], xs[good])

    # Smooth with simple moving average
    if smooth_window is None or smooth_window <= 1:
        xs_smooth = xs
    else:
        k = np.ones(smooth_window, dtype=float) / float(smooth_window)
        xs_smooth = np.convolve(xs, k, mode="same")

    xs_int = np.clip(np.round(xs_smooth).astype(int), 0, w - 1)

    # Construct initial 1-pixel-per-row line
    line = np.zeros((h, w), dtype=bool)
    line[rows, xs_int] = True

    # Fill tiny vertical gaps with a narrow vertical closing
    try:
        from skimage.morphology import binary_closing, rectangle
        line = binary_closing(line, rectangle(closing_height, 1))
    except Exception:
        # crude fallback: interpolate isolated missing rows
        for y in range(1, h - 1):
            if not line[y].any() and line[y - 1].any() and line[y + 1].any():
                cx_prev = int(np.round(np.nonzero(line[y - 1])[0].mean()))
                cx_next = int(np.round(np.nonzero(line[y + 1])[0].mean()))
                cx = int(np.round((cx_prev + cx_next) / 2.0))
                if 0 <= cx < w:
                    line[y, cx] = True

    # Optionally widen horizontally
    if width is not None and width > 1:
        try:
            from skimage.morphology import dilation, rectangle
            line = dilation(line, rectangle(1, width))
        except Exception:
            pad = width // 2
            new = np.zeros_like(line)
            for y in range(h):
                xs_y = np.nonzero(line[y])[0]
                for x in xs_y:
                    x0 = max(0, x - pad)
                    x1 = min(w, x + pad + 1)
                    new[y, x0:x1] = True
            line = new

    # Final skeletonization to make it 1-pixel-wide, just in case
    skel = skeletonize(line)
    return skel.astype(bool)


def diameter_profile(mask_bin: np.ndarray, um_per_px: float) -> np.ndarray:
    """
    Directly compute diameters [µm] along the silk band.
    """
    if not mask_bin.astype(bool).any():
        return np.array([], dtype=float)

    dist = distance_transform_edt(mask_bin)
    skel = skeletonize(mask_bin.astype(bool))
    radii_px = dist[skel]
    radii_px = radii_px[radii_px > 0]
    diam_px = 2.0 * radii_px
    return diam_px * um_per_px


def diameters_from_skeleton(
    skeleton: np.ndarray,
    dist_map: np.ndarray,
    um_per_px: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Compute local diameters along a skeleton using the distance transform.
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
    radii_px = radii_px[radii_px > 0]

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

    # Use robust statistics: IQR-based stddev instead of sample std
    # This reduces sensitivity to outliers and noise when slice is thin
    if diam_um.size > 0:
        q75, q25 = np.percentile(diam_um, [75, 25])
        iqr = q75 - q25
        # IQR-based standard deviation (roughly equivalent to std for normal dist)
        robust_std = iqr / 1.35
    else:
        robust_std = np.nan

    stats = {
        "mean_um": float(np.median(diam_um)) if diam_um.size > 0 else np.nan,  # Use median (more robust)
        "std_um": float(robust_std),
        "min_um": float(diam_um.min()) if diam_um.size > 0 else np.nan,
        "max_um": float(diam_um.max()) if diam_um.size > 0 else np.nan,
        "n_points": int(diam_um.size),
    }
    return diam_px, diam_um, stats


def analyze_purity(regions: np.ndarray) -> Dict[str, float]:
    """
    Compute purity metrics from a labeled regions map.
    """
    regs = np.asarray(regions)

    pure_px = int((regs == 0).sum())
    unc_px = int((regs == 1).sum())
    total_px = pure_px + unc_px

    if total_px == 0:
        return {
            "pure_pct": 0.0,
            "unc_pct": 0.0,
            "pure_plus_unc_pct": 0.0,
        }

    pure_pct = 100.0 * pure_px / total_px
    unc_pct = 100.0 * unc_px / total_px

    return {
        "pure_pct": pure_pct,
        "unc_pct": unc_pct,
        "pure_plus_unc_pct": pure_pct + unc_pct,
    }


# ---------------------------------------------------------------------------
# Band composition analysis + overlay
# ---------------------------------------------------------------------------

def analyze_band(
    band_mask: np.ndarray,
    otsu_regions: np.ndarray,
    treat_remainder: str = "bg",
) -> Dict[str, Any]:
    """
    Analyze a single band with respect to 3-class labeling.
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

    # Restrict regions to band
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
    """
    gray_u8 = img_as_ubyte(gray_crop)
    base_bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

    h, w = gray_crop.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    pure = masks.get("pure", np.zeros_like(gray_crop, bool))
    unc = masks.get("uncertainty", np.zeros_like(gray_crop, bool))
    bg = masks.get("background", np.zeros_like(gray_crop, bool))
    other = masks.get("other", np.zeros_like(gray_crop, bool))

    # Pure silk -> green
    color[pure, 1] = 255
    # Uncertainty -> blue (BGR)
    color[unc, 0] = 255
    # Background -> red
    color[bg, 2] = 255
    # Other -> yellow
    color[other, 1] = 255
    color[other, 2] = 255

    overlay = cv2.addWeighted(base_bgr, 1.0 - alpha, color, alpha, 0.0)
    return overlay


# ---------------------------------------------------------------------------
# Main entry point for real-time use (with timing + stacked display)
# ---------------------------------------------------------------------------

def process_frame(
    frame_bgr: np.ndarray,
    config: PipelineConfig | None = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Main pipeline entry point.

    Returns
    -------
    results : dict
        {
            'diameter_stats': {...},
            'band_composition': {...},
            'purity': {...},
            'band_area_px': int,
            'slice_height_px': int,
            'valid_band': bool,
            'timing': {...}   # per-stage timings in ms
        }
    vis_image : 3D uint8 array
        BGR stacked image:
            [ original frame
              band mask (from pipeline)
              regions overlay ]
    """
    # process_frame: start
    
    if config is None:
        config = PipelineConfig()

    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr must be a BGR image (H,W,3) uint8")

    t0 = time.perf_counter()

    h, w, _ = frame_bgr.shape

    # Compute slice height in pixels, clamp to image height
    slice_height_px = int(config.slice_height_mm * 1000.0 * config.um_per_px)
    slice_height_px = max(1, min(slice_height_px, h))

    # Crop original BGR frame first
    frame_crop_bgr = frame_bgr[:slice_height_px, :]
    
    # Convert to grayscale and crop
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_crop = gray[:slice_height_px, :]

    t1 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1) 2-class Otsu on cropped grayscale (silk vs background)
    #    AND enforce_two_bg_one_silk() — timed separately
    # ------------------------------------------------------------------
    low_contrast = gray_crop.std() < 1.0

    core = np.zeros_like(gray_crop, dtype=bool)
    band_mask = np.zeros_like(gray_crop, dtype=bool)

    # --- Otsu2 timing ---
    t_otsu_start = time.perf_counter()
    if not low_contrast:
        try:
            thresholds_2 = threshold_multiotsu(gray_crop, classes=2)
            regions_2 = np.digitize(gray_crop, bins=thresholds_2).astype(np.uint8)
            core = regions_2 == 0    # silk candidate
        except Exception:
            core[:] = False
    t2 = time.perf_counter()

    # --- enforce_two_bg_one_silk timing ---
    if not low_contrast and core.sum() > 0:
        try:
            # calling enforce (debugging removed)
            final_silk, final_bg, _ = enforce_two_bg_one_silk(core)
            # enforce returned (debugging removed)
            # Defensive checks: ensure final_silk is valid
            if final_silk is None or final_silk.shape != core.shape:
                # invalid final_silk, setting to False
                band_mask[:] = False
            else:
                band_mask = final_silk.astype(bool)
        except Exception as e:
            # If enforce step fails for any reason, treat as no band
            # enforce exception (handled)
            import traceback
            traceback.print_exc(file=sys.stderr)
            band_mask[:] = False
    else:
        band_mask[:] = False
    t3 = time.perf_counter()

    # ------------------------------------------------------------------
    # 2) Skeleton + distance transform on band
    # ------------------------------------------------------------------
    if band_mask.any():
        skeleton = compute_skeleton(
            band_mask,
            method="centerline",
            smooth_window=11,
            width=1,
            closing_height=3,
        )
        dist_map = distance_transform_edt(band_mask.astype(np.uint8))
        diam_px, diam_um, dia_stats = diameters_from_skeleton(
            skeleton,
            dist_map,
            config.um_per_px,
        )
    else:
        skeleton = np.zeros_like(band_mask, dtype=bool)
        dist_map = np.zeros_like(band_mask, dtype=float)
        diam_px = np.array([], dtype=float)
        diam_um = np.array([], dtype=float)
        dia_stats = {
            "mean_um": 0.0,
            "std_um": 0.0,
            "min_um": 0.0,
            "max_um": 0.0,
            "n_points": 0,
        }

    t4 = time.perf_counter()

    # ------------------------------------------------------------------
    # 3) Second Otsu *inside band only* → pure / uncertainty / background
    # ------------------------------------------------------------------
    h_b, w_b = band_mask.shape
    gray_use = gray_crop[:h_b, :w_b]

    band_vals = gray_use[band_mask]
    if band_vals.size == 0 or band_vals.std() < 1e-3:
        # No meaningful variation inside band: treat entire band as pure
        pure = band_mask.copy()
        unc = np.zeros_like(band_mask, dtype=bool)
    else:
        try:
            thresholds_band = threshold_multiotsu(band_vals, classes=2)
            if np.ndim(thresholds_band) == 0:
                t_band = float(thresholds_band)
            else:
                t_band = float(thresholds_band[0])
            # Silk darker → intensities <= t_band
            pure = band_mask & (gray_use <= t_band)
            unc = band_mask & (~pure)
        except Exception:
            # If Otsu fails, treat entire band as pure
            pure = band_mask.copy()
            unc = np.zeros_like(band_mask, dtype=bool)

    bg = ~band_mask  # everything outside filled silk

    regions3 = np.zeros_like(gray_crop, dtype=np.uint8)
    regions3[pure] = 0
    regions3[unc] = 1
    regions3[bg] = 2

    t5 = time.perf_counter()

    # Band composition & purity analysis
    band_analysis = analyze_band(
        band_mask=band_mask,
        otsu_regions=regions3,
        treat_remainder="bg",
    )
    purity = analyze_purity(regions3)

    # Visualization overlay only on cropped area
    overlay_crop = make_overlay(gray_crop, band_analysis["masks"], alpha=0.6)

    t6 = time.perf_counter()

    # ------------------------------------------------------------------
    # 4) Build stacked display:
    #    [cropped original BGR | band mask | regions overlay]
    # ------------------------------------------------------------------
    # Band from pipeline (filled silk region) as 0/255 grayscale
    band_vis_gray = (band_mask.astype(np.uint8) * 255)
    band_vis_bgr = cv2.cvtColor(band_vis_gray, cv2.COLOR_GRAY2BGR)

    # Defensive: ensure all three images have same width and 3 channels
    try:
        imgs = [frame_crop_bgr, band_vis_bgr, overlay_crop]
        # Ensure dtype uint8
        imgs = [img.astype(np.uint8) for img in imgs]
        widths = [img.shape[1] for img in imgs]
        min_w = min(widths)
        if min_w <= 0:
            # nothing to stack
            stacked = np.zeros((1, 1, 3), dtype=np.uint8)
        else:
            imgs_cropped = [img[:, :min_w] if img.shape[1] > min_w else img for img in imgs]
            # Ensure each image has 3 channels
            imgs_rgb = []
            for img in imgs_cropped:
                if img.ndim == 2:
                    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 1:
                    img3 = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
                else:
                    img3 = img
                imgs_rgb.append(img3)
            stacked = np.vstack(imgs_rgb)
    except Exception as e:
        print("Error building stacked image:", e, file=sys.stderr)
        # Fallback: small blank image so GUI shows something
        stacked = np.zeros((10, 10, 3), dtype=np.uint8)

    t7 = time.perf_counter()

    band_area = int(band_mask.sum())
    valid_band = band_area > 0

    timing = {
        "total_ms": (t7 - t0) * 1000.0,
        "gray_and_crop_ms": (t1 - t0) * 1000.0,
        "otsu2_ms": (t2 - t_otsu_start) * 1000.0,
        "enforce_ms": (t3 - t2) * 1000.0,
        "skeleton_diameter_ms": (t4 - t3) * 1000.0,
        "otsu_inside_band_ms": (t5 - t4) * 1000.0,
        "analysis_overlay_ms": (t6 - t5) * 1000.0,
        "stacking_ms": (t7 - t6) * 1000.0,
    }

    # Calculate silk centroid (position in crop)
    centroid_x = np.nan
    centroid_y = np.nan
    if band_mask.any():
        y_coords, x_coords = np.where(band_mask)
        centroid_x = float(np.mean(x_coords))
        centroid_y = float(np.mean(y_coords))

    results = {
        "diameter_stats": dia_stats,
        "band_composition": band_analysis,
        "purity": purity,
        "band_area_px": band_area,
        "slice_height_px": slice_height_px,
        "valid_band": bool(valid_band),
        "timing": timing,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
    }

    # process_frame: end (debugging removed)
    return results, stacked


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
