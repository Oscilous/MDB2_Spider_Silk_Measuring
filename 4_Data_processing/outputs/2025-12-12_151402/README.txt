Spider Silk Analysis Report
Dataset: 2025-12-12_151402
============================================================

Processing Settings:
  Strand speed: 11.78 mm/s
  Camera calibration: 1.2 µm/px
  Motor RPM: 20.0
  Gear ratio: 1:8.0
  Wheel diameter: 90.0 mm

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
                       - diameter_analysis.png (frame index vs diameter)
                       - diameter_analysis_by_position.png (strand position vs diameter)
                       - purity_analysis_by_position.png (strand position vs purity)
                       - strand_coverage.png (full strand with sampling gaps)
                       - diameter_comparison.png (statistics by section)
  selection_config.json - Your selected sections
  processing_settings.json - Processing parameters
