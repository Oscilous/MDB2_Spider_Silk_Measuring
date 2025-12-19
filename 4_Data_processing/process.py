"""
Main processing script for spider silk measurement data.
Reads selection_config.json to determine which sections to process,
then filters, analyzes, and visualizes the selected data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import math
import tkinter as tk
from tkinter import ttk
from utils import (
    load_selection_config, 
    load_dataset, 
    extract_sections,
    calculate_spatial_positions,
    filter_outliers,
    stitch_images_vertical,
    stitch_images_horizontal,
    get_statistics
)


class StrandSpeedDialog:
    """Dialog to get strand speed parameters before processing."""
    
    def __init__(self, parent=None):
        self.result = None
        
        self.root = tk.Tk() if parent is None else tk.Toplevel(parent)
        self.root.title("Strand Speed Settings")
        self.root.geometry("400x300")
        self.root.resizable(False, False)
        
        # Make dialog modal
        self.root.grab_set()
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        ttk.Label(main_frame, text="Strand Speed Calculation", 
                  font=('Arial', 12, 'bold')).pack(pady=(0, 15))
        
        # RPM
        rpm_frame = ttk.Frame(main_frame)
        rpm_frame.pack(fill=tk.X, pady=5)
        ttk.Label(rpm_frame, text="Motor RPM:", width=20).pack(side=tk.LEFT)
        self.rpm_var = tk.StringVar(value="20")
        ttk.Entry(rpm_frame, textvariable=self.rpm_var, width=15).pack(side=tk.LEFT)
        
        # Gear ratio
        gear_frame = ttk.Frame(main_frame)
        gear_frame.pack(fill=tk.X, pady=5)
        ttk.Label(gear_frame, text="Gear ratio (1:X):", width=20).pack(side=tk.LEFT)
        self.gear_var = tk.StringVar(value="8")
        ttk.Entry(gear_frame, textvariable=self.gear_var, width=15).pack(side=tk.LEFT)
        
        # Wheel diameter
        wheel_frame = ttk.Frame(main_frame)
        wheel_frame.pack(fill=tk.X, pady=5)
        ttk.Label(wheel_frame, text="Wheel diameter (mm):", width=20).pack(side=tk.LEFT)
        self.diameter_var = tk.StringVar(value="50")
        ttk.Entry(wheel_frame, textvariable=self.diameter_var, width=15).pack(side=tk.LEFT)
        
        # um per pixel (camera calibration)
        umpx_frame = ttk.Frame(main_frame)
        umpx_frame.pack(fill=tk.X, pady=5)
        ttk.Label(umpx_frame, text="µm per pixel:", width=20).pack(side=tk.LEFT)
        self.umpx_var = tk.StringVar(value="1.2")
        ttk.Entry(umpx_frame, textvariable=self.umpx_var, width=15).pack(side=tk.LEFT)
        
        # Calculated speed display
        self.speed_label = ttk.Label(main_frame, text="", font=('Arial', 10))
        self.speed_label.pack(pady=15)
        
        # Bind updates
        self.rpm_var.trace('w', self.update_speed)
        self.gear_var.trace('w', self.update_speed)
        self.diameter_var.trace('w', self.update_speed)
        self.update_speed()
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="Process", command=self.on_ok).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Cancel", command=self.on_cancel).pack(side=tk.LEFT, padx=10)
        
    def calculate_speed(self):
        """Calculate strand speed in mm/s from parameters."""
        try:
            rpm = float(self.rpm_var.get())
            gear_ratio = float(self.gear_var.get())
            diameter = float(self.diameter_var.get())
            
            # Actual wheel RPM = motor RPM / gear ratio
            wheel_rpm = rpm / gear_ratio
            
            # Circumference in mm
            circumference = math.pi * diameter
            
            # Speed = circumference * RPM = mm/min
            speed_mm_min = circumference * wheel_rpm
            
            # Convert to mm/s
            speed_mm_s = speed_mm_min / 60.0
            
            return speed_mm_s
        except (ValueError, ZeroDivisionError):
            return 0.0
    
    def update_speed(self, *args):
        speed = self.calculate_speed()
        self.speed_label.config(text=f"Calculated strand speed: {speed:.2f} mm/s")
    
    def on_ok(self):
        try:
            self.result = {
                'rpm': float(self.rpm_var.get()),
                'gear_ratio': float(self.gear_var.get()),
                'diameter_mm': float(self.diameter_var.get()),
                'um_per_px': float(self.umpx_var.get()),
                'strand_speed_mm_s': self.calculate_speed()
            }
            self.root.destroy()
        except ValueError:
            self.speed_label.config(text="Invalid input! Please enter numbers.")
    
    def on_cancel(self):
        self.result = None
        self.root.destroy()
    
    def show(self):
        self.root.mainloop()
        return self.result


def process_selected_data(config_file="selection_config.json", 
                         output_dir=None,
                         base_data_path=None,
                         strand_settings=None):
    """
    Process all selected sections from the config file.
    
    Args:
        config_file: Path to selection_config.json
        output_dir: Directory to save processed results
        base_data_path: Path to data folder (defaults to ../data relative to this script)
        strand_settings: Dict with strand speed parameters (if None, shows dialog)
    """
    # Show settings dialog if not provided
    if strand_settings is None:
        dialog = StrandSpeedDialog()
        strand_settings = dialog.show()
        if strand_settings is None:
            print("Processing cancelled.")
            return
    
    strand_speed_mm_s = strand_settings['strand_speed_mm_s']
    um_per_px = strand_settings.get('um_per_px', 1.2)
    
    print(f"\nStrand speed: {strand_speed_mm_s:.2f} mm/s")
    print(f"Camera calibration: {um_per_px} µm/px")
    
    # Default data path is one level up from this script
    if base_data_path is None:
        base_data_path = Path(__file__).resolve().parent.parent / "data"
    
    # Setup output directory (default to ./outputs next to this script)
    if output_dir is None:
        output_path = Path(__file__).resolve().parent / "outputs"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load selections from per-dataset inputs folder
    print("\nLoading selection configuration from inputs/ ...")
    selections = {}
    inputs_dir = Path(__file__).resolve().parent / "inputs"
    strand_metadata = {}  # Track strand_end_frame per dataset
    if inputs_dir.exists():
        for sel_file in inputs_dir.glob("*/selection_config.json"):
            try:
                dataset_name = sel_file.parent.name
                with open(sel_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    selections[dataset_name] = data
                elif isinstance(data, dict):
                    # New format: has 'sections' key
                    if 'sections' in data and isinstance(data['sections'], list):
                        selections[dataset_name] = data['sections']
                        # Store strand_end_frame if present
                        if 'strand_end_frame' in data:
                            strand_metadata[dataset_name] = {'strand_end_frame': data['strand_end_frame']}
                    else:
                        # fallback: store empty or attempt to use dataset key
                        selections[dataset_name] = data.get(dataset_name, []) if isinstance(data.get(dataset_name, []), list) else []
                else:
                    selections[dataset_name] = []
                print(f"  Found selections for {dataset_name}: {len(selections[dataset_name])} sections")
                if dataset_name in strand_metadata:
                    print(f"    Strand marked as ended at frame {strand_metadata[dataset_name]['strand_end_frame'] + 1}")
            except Exception as e:
                print(f"  Failed to read {sel_file}: {e}")

    if not selections:
        print("No selections found in inputs/. Run viewer.py to create selection_config.json per dataset.")
        return
    
    # Process each dataset
    all_sections = []
    section_metadata = []
    
    for dataset_name, sections in selections.items():
        print(f"\n{'='*60}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # Load dataset
        dataset = load_dataset(dataset_name, base_data_path)
        df = dataset['df']
        image_folder = dataset['images']

        # Create per-dataset output folder structure
        dataset_output = output_path / dataset_name
        dataset_output.mkdir(parents=True, exist_ok=True)
        sections_dir = dataset_output / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate number of sections for consistent coloring
        num_sections = len(sections)
        
        # Process each section
        for section_idx, (start, end) in enumerate(sections, 1):
            print(f"\n  Section {section_idx}: frames {start+1} to {end+1}")
            
            # Extract section data
            section_df = df.iloc[start:end+1].copy()
            
            # Add spatial positions (use same speed as stitching for consistency)
            section_df = calculate_spatial_positions(section_df, strand_speed_mm_s=strand_speed_mm_s)
            
            # Filter outliers
            section_df = filter_outliers(section_df, column='diameter_um', method='iqr')
            n_outliers = section_df['is_outlier'].sum()
            print(f"    Frames: {len(section_df)}, Outliers detected: {n_outliers}")
            
            # Calculate statistics
            stats = get_statistics(section_df[~section_df['is_outlier']], 'diameter_um')
            print(f"    Diameter: {stats['mean']:.2f} ± {stats['std']:.2f} μm")
            print(f"    Range: {stats['min']:.2f} - {stats['max']:.2f} μm")
            print(f"    CV: {stats['cv']:.2f}%")
            
            # Store section info
            section_id = f"{dataset_name}_section{section_idx}"
            section_df['section_id'] = section_id
            section_df['dataset'] = dataset_name
            section_df['section_number'] = section_idx
            
            all_sections.append(section_df)
            section_metadata.append({
                'section_id': section_id,
                'dataset': dataset_name,
                'section_number': section_idx,
                'start_frame': start + 1,
                'end_frame': end + 1,
                'n_frames': len(section_df),
                'n_outliers': n_outliers,
                **{f'diameter_{k}': v for k, v in stats.items()}
            })
            
            # Create per-section folder
            section_folder = sections_dir / f"section_{section_idx}"
            section_folder.mkdir(parents=True, exist_ok=True)
            
            # Create stitched image for this section
            image_paths = [
                image_folder / f"frame_{frame_num:06d}_original.png"
                for frame_num in range(start + 1, end + 2)
            ]
            stitched_path = section_folder / f"sec{section_idx}_stitched.png"
            print(f"    Creating stitched image (vertical, speed-based spacing)...")
            
            # Use vertical stitching with speed-based spacing
            stitch_images_vertical(
                image_paths, 
                stitched_path, 
                section_df,
                strand_speed_mm_s=strand_speed_mm_s,
                um_per_px=um_per_px,
                max_width=300
            )
            print(f"    Saved stitched image to: {stitched_path}")
            
            # Also stitch uncertainty images
            uncertainty_paths = [
                image_folder / f"frame_{frame_num:06d}_uncertainty.png"
                for frame_num in range(start + 1, end + 2)
            ]
            uncertainty_stitched_path = section_folder / f"sec{section_idx}_stitched_uncertainty.png"
            print(f"    Creating stitched uncertainty image...")
            
            stitch_images_vertical(
                uncertainty_paths, 
                uncertainty_stitched_path, 
                section_df,
                strand_speed_mm_s=strand_speed_mm_s,
                um_per_px=um_per_px,
                max_width=300
            )
            print(f"    Saved stitched uncertainty image to: {uncertainty_stitched_path}")
            
            # Also save per-section CSV
            section_csv = section_folder / f"sec{section_idx}_measurements.csv"
            section_df.to_csv(section_csv, index=False)
            print(f"    Saved section CSV to: {section_csv}")
            
            # Create per-section visualization
            create_section_visualization(section_df, section_folder, strand_speed_mm_s, um_per_px, section_idx=section_idx-1, num_sections=num_sections, section_num=section_idx)
    
    # Combine all sections
    print(f"\n{'='*60}")
    print("Combining all sections...")
    combined_df = pd.concat(all_sections, ignore_index=True)

    # Get the first dataset's output folder (since we're combining, use that as base)
    # All datasets go to their own folders, but combined analysis can be at outputs root
    # Actually, let's put it in each dataset's folder for organization
    for dataset_name in selections.keys():
        dataset_output = output_path / dataset_name
        analysis_dir = dataset_output / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        # Load full dataset for strand length calculation
        dataset = load_dataset(dataset_name, base_data_path)
        full_df = dataset['df']
        
        # Filter combined_df for this dataset
        dataset_combined = combined_df[combined_df['dataset'] == dataset_name]
        dataset_metadata = pd.DataFrame([m for m in section_metadata if m['dataset'] == dataset_name])
        
        if len(dataset_combined) > 0:
            # Save filtered data
            output_csv = analysis_dir / "filtered_measurements.csv"
            dataset_combined.to_csv(output_csv, index=False)
            print(f"Saved filtered data to: {output_csv}")
            
            # Save metadata summary
            metadata_csv = analysis_dir / "section_metadata.csv"
            dataset_metadata.to_csv(metadata_csv, index=False)
            print(f"Saved section metadata to: {metadata_csv}")
            
            # Create visualizations for this dataset
            print(f"\nCreating visualizations for {dataset_name}...")
            strand_end = strand_metadata.get(dataset_name, {}).get('strand_end_frame', None)
            create_visualizations(
                dataset_combined, 
                dataset_metadata, 
                analysis_dir,
                full_df=full_df,
                section_ranges=selections[dataset_name],
                strand_speed_mm_s=strand_speed_mm_s,
                strand_end_frame=strand_end,
                um_per_px=um_per_px
            )
    
    # Save strand settings at dataset level
    for dataset_name in selections.keys():
        dataset_output = output_path / dataset_name
        settings_file = dataset_output / "processing_settings.json"
        with open(settings_file, 'w') as f:
            json.dump(strand_settings, f, indent=2)
        print(f"Saved processing settings to: {settings_file}")
        
        # Create README summary
        dataset_combined = combined_df[combined_df['dataset'] == dataset_name]
        if len(dataset_combined) > 0:
            readme_path = dataset_output / "README.txt"
            with open(readme_path, 'w') as f:
                f.write(f"Spider Silk Analysis Report\n")
                f.write(f"Dataset: {dataset_name}\n")
                f.write(f"{'='*60}\n\n")
                f.write(f"Processing Settings:\n")
                f.write(f"  Strand speed: {strand_speed_mm_s:.2f} mm/s\n")
                f.write(f"  Camera calibration: {um_per_px} µm/px\n")
                f.write(f"  Motor RPM: {strand_settings['rpm']}\n")
                f.write(f"  Gear ratio: 1:{strand_settings['gear_ratio']}\n")
                f.write(f"  Wheel diameter: {strand_settings['diameter_mm']} mm\n\n")
                f.write(f"Results Summary:\n")
                f.write(f"  Total sections: {len(selections[dataset_name])}\n")
                f.write(f"  Total frames: {len(dataset_combined)}\n")
                f.write(f"  Overall mean diameter: {dataset_combined[~dataset_combined['is_outlier']]['diameter_um'].mean():.2f} µm\n")
                f.write(f"  Overall std diameter: {dataset_combined[~dataset_combined['is_outlier']]['diameter_um'].std():.2f} µm\n\n")
                f.write(f"Output Structure:\n")
                f.write(f"  sections/          - Individual section folders with stitched images and measurements\n")
                f.write(f"                       - stitched.png (original images)\n")
                f.write(f"                       - stitched_uncertainty.png (uncertainty maps)\n")
                f.write(f"  analysis/          - Combined analysis and visualizations\n")
                f.write(f"                       - strand_coverage.png (sampling overview bar)\n")
                f.write(f"                       - diameter_vs_position.png (diameter along strand)\n")
                f.write(f"                       - uncertainty_vs_position.png (uncertainty along strand)\n")
                f.write(f"                       - strand_movement_fov.png (strand position in camera FOV)\n")
                f.write(f"  processing_settings.json - Processing parameters\n")
                f.write(f"\nNote: Selection configs are stored in inputs/<dataset>/selection_config.json\n")
                f.write(f"      You can safely delete outputs/ without losing your viewer selections.\n")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total datasets: {len(selections)}")
    print(f"Total sections processed: {len(section_metadata)}")
    print(f"Total frames: {len(combined_df)}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"\nStructure:")
    print(f"  inputs/                          <- Viewer selections (safe to keep)")
    for dataset_name in selections.keys():
        print(f"  └── {dataset_name}/")
        print(f"      └── selection_config.json")
    print(f"")
    print(f"  outputs/                         <- Processing results (can be regenerated)")
    for dataset_name in selections.keys():
        print(f"  └── {dataset_name}/")
        print(f"      ├── processing_settings.json")
        print(f"      ├── README.txt")
        print(f"      ├── sections/")
        print(f"      │   ├── section_1/")
        print(f"      │   │   ├── stitched.png")
        print(f"      │   │   ├── stitched_uncertainty.png")
        print(f"      │   │   └── measurements.csv")
        print(f"      │   └── ...")
        print(f"      └── analysis/")
        print(f"          ├── filtered_measurements.csv")
        print(f"          ├── section_metadata.csv")
        print(f"          ├── strand_coverage.png")
        print(f"          ├── diameter_vs_position.png")
        print(f"          └── uncertainty_vs_position.png")
    print(f"{'='*60}")


def create_visualizations(df, metadata_df, output_dir, full_df=None, section_ranges=None, strand_speed_mm_s=None, strand_end_frame=None, um_per_px=1.2):
    """
    Create analysis plots and visualizations.
    
    Args:
        df: DataFrame with filtered/selected measurements
        metadata_df: DataFrame with section metadata
        output_dir: Directory to save plots
        full_df: Full dataset DataFrame (for strand coverage plot)
        section_ranges: List of (start, end) tuples for selected sections
        strand_speed_mm_s: Strand speed for position calculation
        strand_end_frame: Frame index where strand ended (for xlim limit)
        um_per_px: Camera calibration (µm per pixel)
    """
    output_dir = Path(output_dir)
    
    if full_df is None or section_ranges is None or strand_speed_mm_s is None:
        print("  Warning: Missing data for visualizations (full_df, section_ranges, or strand_speed)")
        return
    
    # Calculate full strand positions
    times_ms = full_df['time_since_start_ms'].values
    full_positions_mm = [strand_speed_mm_s * (t / 1000.0) for t in times_ms]
    min_pos = min(full_positions_mm)
    full_positions_mm = [p - min_pos for p in full_positions_mm]
    total_strand_length = max(full_positions_mm)
    
    # Determine the actual end position (either strand end frame or total length)
    if strand_end_frame is not None and strand_end_frame < len(full_positions_mm):
        strand_end_position = full_positions_mm[strand_end_frame]
    else:
        strand_end_position = total_strand_length
    
    # Common color scheme for sections
    section_colors = plt.cm.tab10(np.linspace(0, 1, len(section_ranges)))
    
    # Calculate gap info (used in multiple figures)
    gap_info = []
    
    # Add initial gap from 0mm to first section
    first_section_start = full_positions_mm[section_ranges[0][0]]
    if first_section_start > 0:
        gap_info.append({
            'from_section': 0,
            'to_section': 1,
            'gap_mm': first_section_start,
            'gap_start': 0,
            'gap_end': first_section_start
        })
    
    # Add gaps between sections
    for i, (start, end) in enumerate(section_ranges):
        if i < len(section_ranges) - 1:
            next_start = section_ranges[i+1][0]
            gap_start = full_positions_mm[end]
            gap_end = full_positions_mm[next_start]
            gap_length = gap_end - gap_start
            gap_info.append({
                'from_section': i+1,
                'to_section': i+2,
                'gap_mm': gap_length,
                'gap_start': gap_start,
                'gap_end': gap_end
            })
    
    # Calculate coverage statistics
    total_sampled = sum(full_positions_mm[e] - full_positions_mm[s] 
                       for s, e in section_ranges)
    coverage_pct = (total_sampled / total_strand_length) * 100 if total_strand_length > 0 else 0
    total_samples = sum(e - s + 1 for s, e in section_ranges)
    
    # =========================================================================
    # Figure 1: Strand Coverage Bar (horizontal bar with section numbers and gaps)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 3))
    
    # Draw full strand as gray background
    ax.barh(0, total_strand_length, height=0.8, color='lightgray', 
            edgecolor='gray', label='Unsampled', alpha=0.5)
    
    # Draw sampled sections as colored bars
    for i, (start, end) in enumerate(section_ranges):
        start_pos = full_positions_mm[start]
        end_pos = full_positions_mm[end]
        section_length = end_pos - start_pos
        
        ax.barh(0, section_length, left=start_pos, height=0.8, 
               color=section_colors[i], edgecolor='black', linewidth=1,
               label=f'Section {i+1}: {section_length:.1f} mm')
        
        # Add section number label
        ax.text(start_pos + section_length/2, 0, f'{i+1}', 
               ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Add gap annotations below the bar
    for gap in gap_info:
        gap_center = (gap['gap_start'] + gap['gap_end']) / 2
        ax.annotate(f"{gap['gap_mm']:.1f}", 
                   xy=(gap_center, -0.65), ha='center', fontsize=9, 
                   color='darkred', fontweight='bold')
        # Draw gap indicator arrow
        ax.annotate('', xy=(gap['gap_end'], -0.5), xytext=(gap['gap_start'], -0.5),
                   arrowprops=dict(arrowstyle='<->', color='darkred', lw=1.5))
    
    ax.set_xlim(0, strand_end_position)
    ax.set_ylim(-1.0, 0.8)
    ax.set_xlabel('Strand Position (mm)', fontsize=12)
    ax.set_title(f'Strand Sampling Coverage - Total: {total_strand_length:.1f} mm | '
                f'Sampled: {total_sampled:.1f} mm ({coverage_pct:.1f}%) | '
                f'Samples: {total_samples}', 
                fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=8, ncol=len(section_ranges)+1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_dir / "strand_coverage.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: strand_coverage.png")
    
    # =========================================================================
    # Figure 2: Diameter vs Strand Position (with error bars)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 5))
    
    for i, (start, end) in enumerate(section_ranges):
        # Get actual strand positions from full_positions_mm
        section_positions = full_positions_mm[start:end+1]
        
        # Get diameter data from full_df (not filtered df, to match positions)
        section_diameters = full_df.iloc[start:end+1]['diameter_um'].values
        
        # Filter out outliers based on IQR (inline)
        q1, q3 = np.percentile(section_diameters, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        mask = (section_diameters >= lower_bound) & (section_diameters <= upper_bound)
        
        clean_positions = np.array(section_positions)[mask]
        clean_diameters = section_diameters[mask]
        clean_diameter_stds = full_df.iloc[start:end+1]['diameter_std_um'].values[mask]
        
        if len(clean_diameters) > 0:
            # Plot as scatter with connecting line
            ax.plot(clean_positions, clean_diameters, 'o-', color=section_colors[i], 
                   markersize=3, alpha=0.7, linewidth=1,
                   label=f'Section {i+1}')
    
    # Mark gaps with subtle shading
    for gap in gap_info:
        ax.axvspan(gap['gap_start'], gap['gap_end'], alpha=0.1, color='red')
    
    ax.set_xlim(0, strand_end_position)
    ax.set_xlabel('Strand Position (mm)', fontsize=12)
    ax.set_ylabel('Diameter (μm)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "diameter_vs_position.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: diameter_vs_position.png")
    
    # =========================================================================
    # Figure 3: Uncertainty vs Strand Position
    # =========================================================================
    if 'purity_pct' in full_df.columns:
        fig, ax = plt.subplots(figsize=(16, 5))
        
        for i, (start, end) in enumerate(section_ranges):
            # Get actual strand positions from full_positions_mm
            section_positions = full_positions_mm[start:end+1]
            
            # Get uncertainty data from full_df
            section_uncertainty = full_df.iloc[start:end+1]['purity_pct'].values
            
            if len(section_uncertainty) > 0:
                ax.plot(section_positions, section_uncertainty, 'o-', color=section_colors[i], 
                       markersize=3, alpha=0.7, linewidth=1,
                       label=f'Section {i+1}')
        
        # Mark gaps with subtle shading
        for gap in gap_info:
            ax.axvspan(gap['gap_start'], gap['gap_end'], alpha=0.1, color='red')
        
        ax.set_xlim(0, strand_end_position)
        ax.set_ylim(0, 105)  # Uncertainty is 0-100%
        ax.set_xlabel('Strand Position (mm)', fontsize=12)
        ax.set_ylabel('Uncertainty (%)', fontsize=12)
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "uncertainty_vs_position.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: uncertainty_vs_position.png")
    
    # =========================================================================
    # Figure 4: Strand Movement in Camera Field of View
    # =========================================================================
    # Get image width from settings or use default for Raspberry Pi GS camera
    image_width_px = 1440  # Default for RPi GS camera module
    
    create_strand_movement_visualization(
        df=df,
        full_df=full_df,
        section_ranges=section_ranges,
        output_dir=output_dir,
        strand_speed_mm_s=strand_speed_mm_s,
        um_per_px=um_per_px,
        image_width_px=image_width_px,
        strand_end_frame=strand_end_frame
    )


def create_strand_movement_visualization(df, full_df, section_ranges, output_dir, strand_speed_mm_s, um_per_px, 
                                         image_width_px=1440, strand_end_frame=None):
    """
    Create a plot showing strand movement in the camera field of view.
    
    X-axis: Strand position (mm along the strand length)
    Y-axis: Centroid X position in the frame (µm), showing horizontal movement
    At each point, a vertical line is drawn representing the diameter.
    
    This visualization shows when the strand is close to leaving the frame.
    
    Args:
        df: DataFrame with filtered/selected measurements
        full_df: Full dataset DataFrame
        section_ranges: List of (start, end) tuples for selected sections
        output_dir: Directory to save the plot
        strand_speed_mm_s: Strand speed for position calculation
        um_per_px: Camera calibration (µm per pixel)
        image_width_px: Width of camera frame in pixels (default: 1440 for RPi GS camera)
        strand_end_frame: Frame index where strand ended (for xlim limit)
    """
    output_dir = Path(output_dir)
    
    # Calculate maximum field of view width in µm
    max_fov_um = image_width_px * um_per_px
    
    # Calculate full strand positions
    times_ms = full_df['time_since_start_ms'].values
    full_positions_mm = [strand_speed_mm_s * (t / 1000.0) for t in times_ms]
    min_pos = min(full_positions_mm)
    full_positions_mm = [p - min_pos for p in full_positions_mm]
    total_strand_length = max(full_positions_mm)
    
    # Determine the actual end position
    if strand_end_frame is not None and strand_end_frame < len(full_positions_mm):
        strand_end_position = full_positions_mm[strand_end_frame]
    else:
        strand_end_position = total_strand_length
    
    # Color scheme for sections
    section_colors = plt.cm.tab10(np.linspace(0, 1, len(section_ranges)))
    
    # Calculate gap info for shading
    gap_info = []
    first_section_start = full_positions_mm[section_ranges[0][0]]
    if first_section_start > 0:
        gap_info.append({'gap_start': 0, 'gap_end': first_section_start})
    
    for i, (start, end) in enumerate(section_ranges):
        if i < len(section_ranges) - 1:
            next_start = section_ranges[i+1][0]
            gap_start = full_positions_mm[end]
            gap_end = full_positions_mm[next_start]
            gap_info.append({'gap_start': gap_start, 'gap_end': gap_end})
    
    # =========================================================================
    # Figure: Strand Movement in Camera Field of View
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Draw frame boundaries as horizontal lines
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='Frame boundary')
    ax.axhline(y=max_fov_um, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
    
    # Plot each section
    for i, (start, end) in enumerate(section_ranges):
        # Get positions along the strand
        section_positions = full_positions_mm[start:end+1]
        
        # Get centroid X coordinates (in pixels) and diameters
        section_data = full_df.iloc[start:end+1]
        centroid_x_px = section_data['centroid_x'].values
        diameters_um = section_data['diameter_um'].values
        
        # Convert centroid X from pixels to µm
        centroid_x_um = centroid_x_px * um_per_px
        
        # Filter out NaN values (low confidence measurements)
        valid_mask = ~np.isnan(centroid_x_px) & ~np.isnan(diameters_um) & (diameters_um > 0)
        
        if np.sum(valid_mask) == 0:
            continue
            
        valid_positions = np.array(section_positions)[valid_mask]
        valid_centroid_x = centroid_x_um[valid_mask]
        valid_diameters = diameters_um[valid_mask]
        
        # Draw vertical lines representing diameter at each position
        # The line extends from (centroid_x - diameter/2) to (centroid_x + diameter/2)
        for pos, cx, diam in zip(valid_positions, valid_centroid_x, valid_diameters):
            y_bottom = cx - diam / 2
            y_top = cx + diam / 2
            ax.plot([pos, pos], [y_bottom, y_top], color=section_colors[i], 
                   linewidth=1.5, alpha=0.7)
        
        # Also plot the centroid path as a line
        ax.plot(valid_positions, valid_centroid_x, '-', color=section_colors[i], 
               linewidth=1, alpha=0.5, label=f'Section {i+1} centroid path')
    
    # Mark gaps with subtle shading
    for gap in gap_info:
        ax.axvspan(gap['gap_start'], gap['gap_end'], alpha=0.1, color='gray')
    
    # Set axis limits
    ax.set_xlim(0, strand_end_position)
    ax.set_ylim(0, max_fov_um)
    
    # Labels and formatting
    ax.set_xlabel('Strand Position (mm)', fontsize=12)
    ax.set_ylabel('Horizontal Position in Frame (µm)', fontsize=12)
    
    # Create legend with unique entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=9)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "strand_movement_fov.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: strand_movement_fov.png")


def create_section_visualization(section_df, section_folder, strand_speed_mm_s, um_per_px, section_idx=0, num_sections=1, section_num=1):
    """
    Create per-section diameter and uncertainty plots with error bars.
    
    Args:
        section_df: DataFrame with section measurements
        section_folder: Path to section folder (e.g., sections/section_1/)
        strand_speed_mm_s: Strand speed for position calculation
        um_per_px: Camera calibration
        section_idx: Index of section (0-based) for color coding
        num_sections: Total number of sections for consistent coloring
        section_num: Section number (1-based) for filename prefix
    """
    section_folder = Path(section_folder)
    section_folder.mkdir(parents=True, exist_ok=True)
    
    if len(section_df) == 0:
        return
    
    # Calculate positions relative to section start
    times_ms = section_df['time_since_start_ms'].values
    positions_mm = [strand_speed_mm_s * (t / 1000.0) for t in times_ms]
    min_pos = min(positions_mm)
    positions_mm = [p - min_pos for p in positions_mm]
    max_pos = max(positions_mm)
    
    # Get section color from tab10 colormap using same normalization as global visualization
    section_color = plt.cm.tab10(section_idx / (num_sections - 1) if num_sections > 1 else 0)
    
    # =========================================================================
    # Diameter plot for this section
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 4))
    
    diameters = section_df['diameter_um'].values
    diameter_stds = section_df['diameter_std_um'].values
    
    # Plot points with error bars
    ax.plot(positions_mm, diameters, 'o-', color=section_color, 
           markersize=2, alpha=0.5, linewidth=1, label='Diameter')
    ax.errorbar(positions_mm, diameters, yerr=diameter_stds,
               fmt='none', color=section_color, alpha=1, linewidth=1,
               capsize=2, capthick=0.8)
    
    ax.set_xlim(0, max_pos)
    ax.set_xlabel('Section Position (mm)', fontsize=11)
    ax.set_ylabel('Diameter (μm)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(section_folder / f"sec{section_num}_diameter_vs_position.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # =========================================================================
    # Uncertainty plot for this section
    # =========================================================================
    if 'purity_pct' in section_df.columns:
        fig, ax = plt.subplots(figsize=(12, 4))
        
        uncertainty = section_df['purity_pct'].values
        
        ax.plot(positions_mm, uncertainty, 'o-', color='forestgreen', 
               markersize=2, alpha=0.5, linewidth=1, label='Uncertainty')
        
        ax.set_xlim(0, max_pos)
        ax.set_ylim(0, 105)
        ax.set_xlabel('Section Position (mm)', fontsize=11)
        ax.set_ylabel('Uncertainty (%)', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(section_folder / f"sec{section_num}_uncertainty_vs_position.png", dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Run the processing pipeline
    # Don't pass output_dir or base_data_path - let the function use correct defaults
    process_selected_data()
