import os
import pandas as pd
from pathlib import Path
import shutil
import argparse

def extract_segments(src_dir, dest_dir, v_min, v_max, n_segments):
    src_path = Path(src_dir)
    dest_path = Path(dest_dir)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    found_count = 0
    files = sorted(src_path.glob("*.csv"))
    
    print(f"Searching for {n_segments} segments where v_ego is always between {v_min} and {v_max} m/s...")
    
    for f in files:
        if found_count >= n_segments:
            break
            
        try:
            df = pd.read_csv(f)
            if 'vEgo' not in df.columns:
                continue
            v_ego = df['vEgo']
            
            # Check if all values in vEgo are within the range
            if (v_ego >= v_min).all() and (v_ego <= v_max).all():
                dest_file = dest_path / f.name
                shutil.copy(f, dest_file)
                found_count += 1
                print(f"[{found_count}/{n_segments}] Copied {f.name} (v_range: {v_ego.min():.2f} - {v_ego.max():.2f})")
        except Exception as e:
            print(f"Error processing {f.name}: {e}")

    if found_count < n_segments:
        print(f"Warning: Only found {found_count} segments matching the criteria.")
    else:
        print(f"Successfully found and copied {found_count} segments to {dest_dir}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract data segments based on velocity range.")
    parser.add_argument("--src", type=str, default="data", help="Source directory containing CSVs")
    parser.add_argument("--dest", type=str, default="tuning_data/speed_30_40", help="Destination directory")
    parser.add_argument("--vmin", type=float, default=30.0, help="Minimum velocity")
    parser.add_argument("--vmax", type=float, default=40.0, help="Maximum velocity")
    parser.add_argument("--n", type=int, default=10, help="Number of segments to extract")
    
    args = parser.parse_args()
    
    extract_segments(args.src, args.dest, args.vmin, args.vmax, args.n)