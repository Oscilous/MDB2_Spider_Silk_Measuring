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
    
    # Load selections from per-dataset outputs folder
    print("\nLoading selection configuration from outputs/ ...")
    selections = {}
    outputs_dir = Path(__file__).resolve().parent / "outputs"
    if outputs_dir.exists():
        for sel_file in outputs_dir.glob("*/selection_config.json"):
            try:
                dataset_name = sel_file.parent.name
                with open(sel_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    selections[dataset_name] = data
                elif isinstance(data, dict):
                    # if dict, try to normalize
                    if 'sections' in data and isinstance(data['sections'], list):
                        selections[dataset_name] = data['sections']
                    else:
                        # fallback: store empty or attempt to use dataset key
                        selections[dataset_name] = data.get(dataset_name, []) if isinstance(data.get(dataset_name, []), list) else []
                else:
                    selections[dataset_name] = []
                print(f"  Found selections for {dataset_name}: {len(selections[dataset_name])} sections")
            except Exception as e:
                print(f"  Failed to read {sel_file}: {e}")

    if not selections:
        print("No selections found in outputs/. Run viewer.py to create selection_config.json per dataset.")
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
            stitched_path = section_folder / "stitched.png"
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
            
            # Also stitch purity images
            purity_paths = [
                image_folder / f"frame_{frame_num:06d}_purity.png"
                for frame_num in range(start + 1, end + 2)
            ]
            purity_stitched_path = section_folder / "stitched_purity.png"
            print(f"    Creating stitched purity image...")
            
            stitch_images_vertical(
                purity_paths, 
                purity_stitched_path, 
                section_df,
                strand_speed_mm_s=strand_speed_mm_s,
                um_per_px=um_per_px,
                max_width=300
            )
            print(f"    Saved stitched purity image to: {purity_stitched_path}")
            
            # Also save per-section CSV
            section_csv = section_folder / "measurements.csv"
            section_df.to_csv(section_csv, index=False)
            print(f"    Saved section CSV to: {section_csv}")
    
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
            create_visualizations(
                dataset_combined, 
                dataset_metadata, 
                analysis_dir,
                full_df=full_df,
                section_ranges=selections[dataset_name],
                strand_speed_mm_s=strand_speed_mm_s
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
                f.write(f"                       - stitched_purity.png (purity maps)\n")
                f.write(f"  analysis/          - Combined analysis and visualizations\n")
                f.write(f"                       - diameter_analysis.png (frame index vs diameter)\n")
                f.write(f"                       - diameter_analysis_by_position.png (strand position vs diameter)\n")
                f.write(f"                       - purity_analysis_by_position.png (strand position vs purity)\n")
                f.write(f"                       - strand_coverage.png (full strand with sampling gaps)\n")
                f.write(f"                       - diameter_comparison.png (statistics by section)\n")
                f.write(f"  selection_config.json - Your selected sections\n")
                f.write(f"  processing_settings.json - Processing parameters\n")
    
    print(f"\n{'='*60}")
    print("Processing complete!")
    print(f"Total datasets: {len(selections)}")
    print(f"Total sections processed: {len(section_metadata)}")
    print(f"Total frames: {len(combined_df)}")
    print(f"Output directory: {output_path.absolute()}")
    print(f"\nStructure:")
    print(f"  outputs/")
    for dataset_name in selections.keys():
        print(f"  └── {dataset_name}/")
        print(f"      ├── selection_config.json")
        print(f"      ├── processing_settings.json")
        print(f"      ├── README.txt")
        print(f"      ├── sections/")
        print(f"      │   ├── section_1/")
        print(f"      │   │   ├── stitched.png")
        print(f"      │   │   ├── stitched_purity.png")
        print(f"      │   │   └── measurements.csv")
        print(f"      │   └── ...")
        print(f"      └── analysis/")
        print(f"          ├── filtered_measurements.csv")
        print(f"          ├── section_metadata.csv")
        print(f"          ├── diameter_analysis.png")
        print(f"          ├── diameter_analysis_by_position.png")
        print(f"          ├── purity_analysis_by_position.png")
        print(f"          ├── strand_coverage.png")
        print(f"          └── diameter_comparison.png")
    print(f"{'='*60}")


def create_visualizations(df, metadata_df, output_dir, full_df=None, section_ranges=None, strand_speed_mm_s=None):
    """
    Create analysis plots and visualizations.
    
    Args:
        df: DataFrame with filtered/selected measurements
        metadata_df: DataFrame with section metadata
        output_dir: Directory to save plots
        full_df: Full dataset DataFrame (for strand coverage plot)
        section_ranges: List of (start, end) tuples for selected sections
        strand_speed_mm_s: Strand speed for position calculation
    """
    output_dir = Path(output_dir)
    
    # Figure 1: Diameter over time for all sections
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: All sections with boundaries
    ax = axes[0]
    sections = df['section_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sections)))
    
    for i, section_id in enumerate(sections):
        section_data = df[df['section_id'] == section_id]
        clean_data = section_data[~section_data['is_outlier']]
        outlier_data = section_data[section_data['is_outlier']]
        
        # Plot clean data
        ax.plot(clean_data.index, clean_data['diameter_um'], 
               'o-', color=colors[i], label=section_id, markersize=4, alpha=0.7)
        
        # Plot outliers
        if len(outlier_data) > 0:
            ax.plot(outlier_data.index, outlier_data['diameter_um'], 
                   'x', color='red', markersize=6, alpha=0.5)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Diameter (μm)', fontsize=12)
    ax.set_title('Diameter Measurements Across All Sections', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of diameters per section
    ax = axes[1]
    section_data_list = []
    section_labels = []
    
    for section_id in sections:
        section_data = df[df['section_id'] == section_id]
        clean_data = section_data[~section_data['is_outlier']]['diameter_um']
        section_data_list.append(clean_data)
        section_labels.append(section_id)
    
    bp = ax.boxplot(section_data_list, labels=section_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_xlabel('Section', fontsize=12)
    ax.set_ylabel('Diameter (μm)', fontsize=12)
    ax.set_title('Diameter Distribution by Section', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "diameter_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3: Diameter vs Strand Position (spatial reconstruction)
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for i, section_id in enumerate(sections):
        section_data = df[df['section_id'] == section_id]
        clean_data = section_data[~section_data['is_outlier']]
        outlier_data = section_data[section_data['is_outlier']]
        
        # Plot clean data - using position_mm for x-axis
        ax.plot(clean_data['position_mm'], clean_data['diameter_um'], 
               'o-', color=colors[i], label=section_id, markersize=4, alpha=0.7)
        
        # Plot outliers
        if len(outlier_data) > 0:
            ax.plot(outlier_data['position_mm'], outlier_data['diameter_um'], 
                   'x', color='red', markersize=6, alpha=0.5)
    
    ax.set_xlabel('Strand Position (mm)', fontsize=12)
    ax.set_ylabel('Diameter (μm)', fontsize=12)
    ax.set_title('Diameter vs Strand Position (Spatial Reconstruction)', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "diameter_analysis_by_position.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 3b: Purity vs Strand Position
    if 'purity_pct' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        for i, section_id in enumerate(sections):
            section_data = df[df['section_id'] == section_id]
            clean_data = section_data[~section_data['is_outlier']]
            
            # Plot purity - using position_mm for x-axis
            ax.plot(clean_data['position_mm'], clean_data['purity_pct'], 
                   'o-', color=colors[i], label=section_id, markersize=4, alpha=0.7)
        
        ax.set_xlabel('Strand Position (mm)', fontsize=12)
        ax.set_ylabel('Measurement Purity (%)', fontsize=12)
        ax.set_title('Measurement Purity vs Strand Position', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / "purity_analysis_by_position.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Figure 4: Statistics summary
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metadata_df))
    width = 0.35
    
    ax.bar(x - width/2, metadata_df['diameter_mean'], width, 
           yerr=metadata_df['diameter_std'], label='Mean ± Std', 
           alpha=0.7, capsize=5)
    
    ax.set_xlabel('Section', fontsize=12)
    ax.set_ylabel('Diameter (μm)', fontsize=12)
    ax.set_title('Mean Diameter by Section', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metadata_df['section_id'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "diameter_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 5: Strand Coverage Map - shows full strand length with sampled regions
    if full_df is not None and section_ranges is not None and strand_speed_mm_s is not None:
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), gridspec_kw={'height_ratios': [1, 2]})
        
        # Calculate full strand positions
        times_ms = full_df['time_since_start_ms'].values
        full_positions_mm = [strand_speed_mm_s * (t / 1000.0) for t in times_ms]
        min_pos = min(full_positions_mm)
        full_positions_mm = [p - min_pos for p in full_positions_mm]
        total_strand_length = max(full_positions_mm)
        
        # Top plot: Strand coverage timeline
        ax = axes[0]
        
        # Draw full strand as gray background
        ax.barh(0, total_strand_length, height=0.8, color='lightgray', 
                edgecolor='gray', label='Unsampled', alpha=0.5)
        
        # Draw sampled sections as colored bars
        section_colors = plt.cm.tab10(np.linspace(0, 1, len(section_ranges)))
        gap_info = []
        
        for i, (start, end) in enumerate(section_ranges):
            # Get positions for this section
            start_pos = full_positions_mm[start]
            end_pos = full_positions_mm[end]
            section_length = end_pos - start_pos
            
            ax.barh(0, section_length, left=start_pos, height=0.8, 
                   color=section_colors[i], edgecolor='black', linewidth=1,
                   label=f'Section {i+1}: {section_length:.1f} mm')
            
            # Add section label
            ax.text(start_pos + section_length/2, 0, f'{i+1}', 
                   ha='center', va='center', fontsize=10, fontweight='bold', color='white')
            
            # Calculate gap to next section
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
        
        ax.set_xlim(0, total_strand_length)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlabel('Strand Position (mm)', fontsize=12)
        ax.set_title(f'Strand Sampling Coverage - Total Length: {total_strand_length:.1f} mm', 
                    fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=8, ncol=min(len(section_ranges)+1, 4))
        ax.grid(True, alpha=0.3, axis='x')
        
        # Bottom plot: Detailed view with gaps and sample points
        ax = axes[1]
        
        # Plot all potential sample positions as small gray ticks
        ax.scatter(full_positions_mm, [0.5]*len(full_positions_mm), 
                  s=2, c='lightgray', alpha=0.3, label='All frames')
        
        # Plot sampled positions colored by section
        for i, (start, end) in enumerate(section_ranges):
            section_positions = full_positions_mm[start:end+1]
            section_diameters = full_df.iloc[start:end+1]['diameter_um'].values
            
            # Normalize diameter for y-axis
            y_values = [1.0] * len(section_positions)  # Place at y=1
            
            ax.scatter(section_positions, y_values, s=20, c=[section_colors[i]], 
                      alpha=0.8, edgecolors='black', linewidths=0.5,
                      label=f'Section {i+1} ({len(section_positions)} samples)')
        
        # Mark gaps with red shading
        for gap in gap_info:
            ax.axvspan(gap['gap_start'], gap['gap_end'], alpha=0.2, color='red')
            gap_center = (gap['gap_start'] + gap['gap_end']) / 2
            ax.annotate(f"Gap: {gap['gap_mm']:.1f} mm", 
                       xy=(gap_center, 0.75), ha='center', fontsize=9, 
                       color='darkred', fontweight='bold')
        
        # Calculate and show statistics
        total_sampled = sum(full_positions_mm[e] - full_positions_mm[s] 
                           for s, e in section_ranges)
        coverage_pct = (total_sampled / total_strand_length) * 100 if total_strand_length > 0 else 0
        total_samples = sum(e - s + 1 for s, e in section_ranges)
        
        stats_text = (f"Total strand: {total_strand_length:.1f} mm | "
                     f"Sampled: {total_sampled:.1f} mm ({coverage_pct:.1f}%) | "
                     f"Samples: {total_samples} | "
                     f"Gaps: {len(gap_info)}")
        
        ax.set_xlim(0, total_strand_length)
        ax.set_ylim(0, 1.5)
        ax.set_xlabel('Strand Position (mm)', fontsize=12)
        ax.set_ylabel('Sample Points', fontsize=12)
        ax.set_title(stats_text, fontsize=11)
        ax.set_yticks([0.5, 1.0])
        ax.set_yticklabels(['All frames', 'Selected'])
        ax.legend(loc='upper right', fontsize=8, ncol=min(len(section_ranges)+1, 4))
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(output_dir / "strand_coverage.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved: diameter_analysis.png")
    print(f"  Saved: diameter_analysis_by_position.png")
    print(f"  Saved: purity_analysis_by_position.png")
    print(f"  Saved: diameter_comparison.png")
    if full_df is not None:
        print(f"  Saved: strand_coverage.png")


if __name__ == "__main__":
    # Run the processing pipeline
    # Don't pass output_dir or base_data_path - let the function use correct defaults
    process_selected_data()
