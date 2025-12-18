"""
Interactive viewer for selecting good sections of spider silk measurements.
Click 'Start Section' at the beginning of a good section, browse through images,
then click 'End Section' to save it.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import json
import os
from pathlib import Path
import pandas as pd


class SilkDataViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Spider Silk Data Section Selector")
        self.root.geometry("1200x900")
        
        # Data variables - data folder is one level up from this script's directory
        self.base_path = Path(__file__).resolve().parent.parent / "data"
        print(f"Looking for data in: {self.base_path}")
        self.datasets = self.load_datasets()
        self.current_dataset = None
        self.current_frame = 0
        self.sections = {}  # {dataset_name: [(start, end), ...]}
        self.section_in_progress = None  # (start_frame, dataset_name)
        
        # Load existing selections if available
        self.config_file = Path(__file__).parent / "selection_config.json"
        self.load_selections()
        
        # Setup UI first
        self.setup_ui()
        
        # Then set initial dataset in dropdown (but don't auto-load)
        if self.datasets:
            self.dataset_var.set(list(self.datasets.keys())[0])
            # Show message to load
            self.image_label_original.config(text="Click 'Load Dataset' to begin")
            self.image_label_uncertainty.config(text="Click 'Load Dataset' to begin")
        
    def load_datasets(self):
        """Find all measurement datasets in the data folder"""
        datasets = {}
        
        # Get all CSV files
        csv_files = sorted(list(self.base_path.glob("measurements_*.csv")))
        
        # Get all image folders
        image_folders = sorted(list(self.base_path.glob("images_*")))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {self.base_path}")
            return datasets
        
        if not image_folders:
            print(f"Warning: No image folders found in {self.base_path}")
            return datasets
        
        print(f"Found {len(csv_files)} CSV files and {len(image_folders)} image folders")
        
        # Extract timestamps from image folders
        folder_timestamps = []
        for folder in image_folders:
            ts = folder.name.replace("images_", "")
            folder_timestamps.append((ts, folder))
        
        # Match each CSV to closest image folder by timestamp
        for csv_file in csv_files:
            csv_timestamp = csv_file.stem.replace("measurements_", "")
            
            # Try exact match first
            exact_match = self.base_path / f"images_{csv_timestamp}"
            if exact_match.exists():
                datasets[csv_timestamp] = {
                    'csv': csv_file,
                    'images': exact_match,
                    'df': pd.read_csv(csv_file)
                }
                print(f"Exact match: {csv_timestamp}")
            else:
                # Find closest timestamp - images are taken BEFORE measurements
                # So image folder timestamp should be slightly earlier than CSV timestamp
                csv_time = csv_timestamp.replace("-", "").replace("_", "")
                
                best_match = None
                best_diff = float('inf')
                
                for folder_ts, folder in folder_timestamps:
                    folder_time = folder_ts.replace("-", "").replace("_", "")
                    try:
                        diff = int(csv_time) - int(folder_time)
                        # Image folder should be created before CSV (diff > 0)
                        # And should be the smallest positive difference
                        if 0 <= diff < best_diff:
                            best_diff = diff
                            best_match = folder
                    except ValueError:
                        continue
                
                if best_match:
                    datasets[csv_timestamp] = {
                        'csv': csv_file,
                        'images': best_match,
                        'df': pd.read_csv(csv_file)
                    }
                    print(f"Matched: {csv_timestamp} -> {best_match.name}")
                else:
                    print(f"Warning: No matching images for {csv_file.name}")
        
        return datasets
    
    def load_selections(self):
        """Load previously saved section selections.

        Loads per-dataset selection files from `inputs/<dataset>/selection_config.json`.
        Format: {"sections": [[start, end], ...], "strand_end_frame": N or null}
        Backwards-compatible: if old list format, wraps it in new dict format.
        """
        self.sections = {}

        inputs_dir = Path(__file__).resolve().parent / "inputs"
        if inputs_dir.exists():
            for sel_file in inputs_dir.glob("*/selection_config.json"):
                try:
                    dataset_name = sel_file.parent.name
                    with open(sel_file, 'r') as f:
                        data = json.load(f)
                    # Handle both old list format and new dict format
                    if isinstance(data, dict):
                        # New format: already has sections key
                        if 'sections' in data:
                            self.sections[dataset_name] = data
                        else:
                            # Old dict format without sections key, wrap it
                            self.sections[dataset_name] = {'sections': data}
                    elif isinstance(data, list):
                        # Old format: just a list, wrap in new format
                        self.sections[dataset_name] = {'sections': data}
                    else:
                        self.sections[dataset_name] = {'sections': []}
                    print(f"Loaded selections for {dataset_name} from {sel_file}")
                except Exception:
                    print(f"Failed to load selections from {sel_file}")

        # Backwards compatibility: also check outputs/ for old structure
        outputs_dir = Path(__file__).resolve().parent / "outputs"
        if outputs_dir.exists():
            for sel_file in outputs_dir.glob("*/selection_config.json"):
                try:
                    dataset_name = sel_file.parent.name
                    if dataset_name not in self.sections:  # Don't override inputs/
                        with open(sel_file, 'r') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            self.sections[dataset_name] = data
                        print(f"Loaded legacy selections for {dataset_name} from {sel_file}")
                except Exception:
                    pass

        # Backwards compatibility: top-level config
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # merge into sections
                    for k, v in data.items():
                        self.sections.setdefault(k, v)
                else:
                    self.sections.setdefault('default', data)
                print(f"Loaded legacy selection_config.json")
            except Exception:
                print("Failed to load legacy selection_config.json")
    
    def save_selections(self):
        """Save section selections.

        Saves per-dataset selection files to `inputs/<dataset>/selection_config.json`.
        Saves in format: {"sections": [[start, end], ...], "strand_end_frame": N or null}
        """
        inputs_dir = Path(__file__).resolve().parent / "inputs"
        inputs_dir.mkdir(parents=True, exist_ok=True)

        # Save per-dataset files to inputs/ (not outputs/)
        for dataset_name, dataset_data in self.sections.items():
            dataset_dir = inputs_dir / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            sel_file = dataset_dir / "selection_config.json"
            try:
                with open(sel_file, 'w') as f:
                    json.dump(dataset_data, f, indent=2)
                print(f"Saved selections for {dataset_name} -> {sel_file}")
            except Exception as e:
                print(f"Failed to save selections for {dataset_name}: {e}")

        # also update in-memory config file path for compatibility
        try:
            # write a small index file mapping datasets to selection files
            index_file = outputs_dir / "index.json"
            index = {k: f"outputs/{k}/selection_config.json" for k in self.sections.keys()}
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
        except Exception:
            pass
    
    def setup_ui(self):
        """Create the user interface"""
        # Top frame - Dataset selection
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="Dataset:").pack(side=tk.LEFT, padx=5)
        self.dataset_var = tk.StringVar()
        dataset_combo = ttk.Combobox(top_frame, textvariable=self.dataset_var, 
                                     values=list(self.datasets.keys()), state='readonly', width=30)
        dataset_combo.pack(side=tk.LEFT, padx=5)
        
        # Store reference to combo for later initialization
        self.dataset_combo = dataset_combo
        
        # Add Load button to explicitly load the selected dataset
        ttk.Button(top_frame, text="Load Dataset", command=self.load_dataset).pack(side=tk.LEFT, padx=5)
        
        # Show total frames for current dataset
        self.dataset_info_label = ttk.Label(top_frame, text="")
        self.dataset_info_label.pack(side=tk.LEFT, padx=10)
        
        # Middle frame - Image display
        image_frame = ttk.Frame(self.root, padding="10")
        image_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create frame for both images
        images_container = ttk.Frame(image_frame)
        images_container.pack(expand=True)
        
        # Original image
        original_frame = ttk.LabelFrame(images_container, text="Original", padding="5")
        original_frame.pack(side=tk.LEFT, padx=5)
        self.image_label_original = ttk.Label(original_frame, text="Select a dataset")
        self.image_label_original.pack()
        
        # Uncertainty image
        uncertainty_frame = ttk.LabelFrame(images_container, text="Uncertainty Map", padding="5")
        uncertainty_frame.pack(side=tk.LEFT, padx=5)
        self.image_label_uncertainty = ttk.Label(uncertainty_frame, text="Select a dataset")
        self.image_label_uncertainty.pack()
        
        # Info frame - Display measurement info
        info_frame = ttk.LabelFrame(self.root, text="Measurement Info", padding="10")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(info_frame, height=4, width=80, state='disabled')
        self.info_text.pack()
        
        # Section status frame
        status_frame = ttk.LabelFrame(self.root, text="Section Status", padding="10")
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_label = ttk.Label(status_frame, text="No section in progress", 
                                      font=('Arial', 10, 'bold'))
        self.status_label.pack()
        
        # Navigation frame
        nav_frame = ttk.Frame(self.root, padding="10")
        nav_frame.pack(fill=tk.X)
        
        ttk.Button(nav_frame, text="◀◀ First", command=self.first_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="◀ Previous", command=self.prev_frame).pack(side=tk.LEFT, padx=5)
        
        self.frame_label = ttk.Label(nav_frame, text="Frame: 0/0")
        self.frame_label.pack(side=tk.LEFT, padx=20)
        
        ttk.Button(nav_frame, text="Next ▶", command=self.next_frame).pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="Last ▶▶", command=self.last_frame).pack(side=tk.LEFT, padx=5)
        
        # Section control frame
        section_frame = ttk.Frame(self.root, padding="10")
        section_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(section_frame, text="🟢 Start Section", 
                                    command=self.start_section, style='Green.TButton')
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.end_btn = ttk.Button(section_frame, text="🔴 End Section", 
                                  command=self.end_section, state='disabled', style='Red.TButton')
        self.end_btn.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(section_frame, text="⏹️ Mark Strand End (Broken)", 
                  command=self.mark_strand_end).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(section_frame, text="💾 Save Selections", 
                  command=self.save_selections).pack(side=tk.LEFT, padx=10)
        
        # Sections display frame
        sections_frame = ttk.LabelFrame(self.root, text="Saved Sections", padding="10")
        sections_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.sections_text = tk.Text(sections_frame, height=6, width=80)
        self.sections_text.pack(fill=tk.BOTH, expand=True)
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_frame())
        self.root.bind('<Right>', lambda e: self.next_frame())
        self.root.bind('<Home>', lambda e: self.first_frame())
        self.root.bind('<End>', lambda e: self.last_frame())
        self.root.bind('<space>', lambda e: self.toggle_section())
        
    def load_dataset(self, event=None):
        """Load the selected dataset"""
        dataset_name = self.dataset_var.get()
        if not dataset_name or dataset_name not in self.datasets:
            messagebox.showwarning("No Dataset", "Please select a valid dataset")
            return
        
        # Reset section in progress if changing datasets
        if self.section_in_progress and self.current_dataset != dataset_name:
            response = messagebox.askyesno("Section in Progress", 
                                          "You have a section in progress. Discard it?")
            if response:
                self.section_in_progress = None
                self.start_btn.config(state='normal')
                self.end_btn.config(state='disabled')
                self.status_label.config(text="No section in progress")
            else:
                # Restore previous dataset - set combobox back
                if self.current_dataset:
                    self.dataset_var.set(self.current_dataset)
                return
        
        # Now switch to new dataset
        print(f"Loading dataset: {dataset_name}")
        self.current_dataset = dataset_name
        self.current_frame = 0
        
        # Update dataset info label
        dataset = self.datasets[dataset_name]
        total_frames = len(dataset['df'])
        self.dataset_info_label.config(text=f"({total_frames} frames)")
        
        # Force display update
        self.display_frame()
        self.update_sections_display()
        
        print(f"Dataset loaded: {dataset_name}, Total frames: {total_frames}")
        
    def display_frame(self):
        """Display the current frame"""
        if not self.current_dataset:
            return
        
        dataset = self.datasets[self.current_dataset]
        df = dataset['df']
        
        # Ensure current_frame is within bounds
        if self.current_frame >= len(df):
            self.current_frame = len(df) - 1
        if self.current_frame < 0:
            self.current_frame = 0
        
        # Get image paths
        frame_num = self.current_frame + 1  # Frame numbers start at 1
        image_path_original = dataset['images'] / f"frame_{frame_num:06d}_original.png"
        image_path_uncertainty = dataset['images'] / f"frame_{frame_num:06d}_uncertainty.png"
        
        # Load and display original image
        if image_path_original.exists():
            img_orig = Image.open(image_path_original)
            # Resize if too large
            max_width, max_height = 500, 400
            img_orig.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            photo_orig = ImageTk.PhotoImage(img_orig)
            
            self.image_label_original.config(image=photo_orig, text="")
            self.image_label_original.image = photo_orig  # Keep a reference
        else:
            self.image_label_original.config(text=f"Not found:\n{image_path_original.name}", image="")
            self.image_label_original.image = None
        
        # Load and display uncertainty image
        if image_path_uncertainty.exists():
            img_unc = Image.open(image_path_uncertainty)
            # Resize if too large
            max_width, max_height = 500, 400
            img_unc.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            photo_unc = ImageTk.PhotoImage(img_unc)
            
            self.image_label_uncertainty.config(image=photo_unc, text="")
            self.image_label_uncertainty.image = photo_unc  # Keep a reference
        else:
            self.image_label_uncertainty.config(text=f"Not found:\n{image_path_uncertainty.name}", image="")
            self.image_label_uncertainty.image = None
        
        # Update frame label
        total_frames = len(df)
        self.frame_label.config(text=f"Frame: {frame_num}/{total_frames}")
        
        # Update info text
        if self.current_frame < len(df):
            row = df.iloc[self.current_frame]
            info = f"Frame: {int(row['frame_number'])} | "
            info += f"Diameter: {row['diameter_um']:.2f} ± {row['diameter_std_um']:.2f} μm | "
            info += f"Speed: {row['speed_mm_s']:.2f} mm/s | "
            info += f"Purity: {row['purity_pct']:.1f}% | "
            info += f"Quality: {row['measurement_quality']}"
            
            self.info_text.config(state='normal')
            self.info_text.delete(1.0, tk.END)
            self.info_text.insert(1.0, info)
            self.info_text.config(state='disabled')
    
    def first_frame(self):
        """Go to first frame"""
        self.current_frame = 0
        self.display_frame()
    
    def last_frame(self):
        """Go to last frame"""
        if self.current_dataset:
            dataset = self.datasets[self.current_dataset]
            self.current_frame = len(dataset['df']) - 1
            self.display_frame()
    
    def prev_frame(self):
        """Go to previous frame"""
        if self.current_frame > 0:
            self.current_frame -= 1
            self.display_frame()
    
    def next_frame(self):
        """Go to next frame"""
        if self.current_dataset:
            dataset = self.datasets[self.current_dataset]
            if self.current_frame < len(dataset['df']) - 1:
                self.current_frame += 1
                self.display_frame()
    
    def toggle_section(self):
        """Toggle section start/end with spacebar"""
        if self.section_in_progress:
            self.end_section()
        else:
            self.start_section()
    
    def start_section(self):
        """Mark the start of a good section"""
        if not self.current_dataset:
            messagebox.showwarning("No Dataset", "Please select a dataset first")
            return
        
        self.section_in_progress = (self.current_frame, self.current_dataset)
        self.start_btn.config(state='disabled')
        self.end_btn.config(state='normal')
        self.status_label.config(text=f"Section started at frame {self.current_frame + 1}", 
                                foreground='green')
    
    def end_section(self):
        """Mark the end of a good section and save it"""
        if not self.section_in_progress:
            return
        
        start_frame, dataset_name = self.section_in_progress
        end_frame = self.current_frame
        
        # Validate
        if end_frame < start_frame:
            messagebox.showerror("Invalid Section", 
                               "End frame must be after start frame")
            return
        
        # Save section - ensure dataset entry exists in new format
        if dataset_name not in self.sections:
            self.sections[dataset_name] = {'sections': []}
        
        # Ensure it's in the new dict format
        if isinstance(self.sections[dataset_name], list):
            self.sections[dataset_name] = {'sections': self.sections[dataset_name]}
        
        self.sections[dataset_name]['sections'].append([start_frame, end_frame])
        
        # Reset state
        self.section_in_progress = None
        self.start_btn.config(state='normal')
        self.end_btn.config(state='disabled')
        self.status_label.config(text=f"Section saved: frames {start_frame + 1}-{end_frame + 1}", 
                                foreground='blue')
        
        self.update_sections_display()
        
        # Auto-save
        self.save_selections()
    
    def mark_strand_end(self):
        """Mark that the strand has broken/ended and no more data should be recorded"""
        if not self.current_dataset:
            messagebox.showwarning("No Dataset", "Please select a dataset first")
            return
        
        # Initialize dataset entry if needed
        if self.current_dataset not in self.sections:
            self.sections[self.current_dataset] = {'sections': []}
        
        # Ensure it's in the new format
        if isinstance(self.sections[self.current_dataset], list):
            self.sections[self.current_dataset] = {'sections': self.sections[self.current_dataset]}
        
        # Mark the strand end frame
        self.sections[self.current_dataset]['strand_end_frame'] = self.current_frame
        
        self.status_label.config(text=f"Strand marked as ended at frame {self.current_frame + 1}", 
                                foreground='orange')
        
        self.update_sections_display()
        
        # Auto-save
        self.save_selections()
    
    def update_sections_display(self):
        """Update the display of saved sections"""
        self.sections_text.delete(1.0, tk.END)
        
        if not self.sections:
            self.sections_text.insert(1.0, "No sections saved yet")
            return
        
        for dataset_name, dataset_data in self.sections.items():
            self.sections_text.insert(tk.END, f"\n📁 {dataset_name}:\n")
            
            # Handle both old list format and new dict format
            if isinstance(dataset_data, list):
                sections_list = dataset_data
                strand_end = None
            else:
                sections_list = dataset_data.get('sections', [])
                strand_end = dataset_data.get('strand_end_frame', None)
            
            for i, (start, end) in enumerate(sections_list, 1):
                n_frames = end - start + 1
                self.sections_text.insert(tk.END, 
                    f"  Section {i}: Frames {start + 1}-{end + 1} ({n_frames} frames)\n")
            
            if strand_end is not None:
                self.sections_text.insert(tk.END, f"  ⏹️ Strand ended at frame {strand_end + 1}\n")


def main():
    root = tk.Tk()
    app = SilkDataViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
