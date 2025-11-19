#!/usr/bin/env python3
"""Filter data directory down to first 1000 logs."""

import shutil
from pathlib import Path


def filter_logs(data_dir: str = "./data", output_dir: str = "./data_1000", num_logs: int = 800):
    """
    Copy first N log files to a new directory.
    
    Args:
        data_dir: Source directory containing CSV log files
        output_dir: Destination directory for filtered logs
        num_logs: Number of logs to keep (default: 1000)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    # Get all CSV files and sort numerically by filename
    csv_files = sorted(data_path.glob("*.csv"), key=lambda p: int(p.stem))
    
    # Take first N files
    selected_files = csv_files[:num_logs]
    
    print(f"Found {len(csv_files)} total log files")
    print(f"Filtering to first {len(selected_files)} logs")
    print(f"Range: {selected_files[0].stem} to {selected_files[-1].stem}")
    
    # Create output directory
    output_path.mkdir(exist_ok=True)
    
    # Copy files
    for i, src_file in enumerate(selected_files, 1):
        dest_file = output_path / src_file.name
        shutil.copy2(src_file, dest_file)
        
        if i % 100 == 0:
            print(f"Copied {i}/{len(selected_files)} files...")
    
    print(f"\nDone! Copied {len(selected_files)} files to {output_path}")


if __name__ == "__main__":
    filter_logs()

