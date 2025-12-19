Spider Silk Analysis Report
Dataset: 2025-12-12_151402
============================================================

Processing Settings:
  Strand speed: 6.54 mm/s
  Camera calibration: 1.2 µm/px
  Motor RPM: 20.0
  Gear ratio: 1:8.0
  Wheel diameter: 50.0 mm

Results Summary:
  Total sections: 5
  Total frames: 275
  Overall mean diameter: 510.07 µm
  Overall std diameter: 135.28 µm

Output Structure:
  sections/          - Individual section folders with stitched images and measurements
                       - stitched.png (original images)
                       - stitched_purity.png (purity maps)
  analysis/          - Combined analysis and visualizations
                       - strand_coverage.png (sampling overview bar)
                       - diameter_vs_position.png (diameter along strand)
                       - purity_vs_position.png (purity along strand)
                       - strand_movement_fov.png (strand position in camera FOV)
  processing_settings.json - Processing parameters

Note: Selection configs are stored in inputs/<dataset>/selection_config.json
      You can safely delete outputs/ without losing your viewer selections.
