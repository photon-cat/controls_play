#!/usr/bin/env python3
import subprocess
import time

# List of speed groups in the order we want to tune them (High to Low is usually better)
SPEED_GROUPS = [
    "speed_36_40",
    "speed_32_36",
    "speed_28_32",
    "speed_24_28",
    "speed_20_24",
    "speed_16_20",
    "speed_12_16",
    "speed_8_12",
    "speed_4_9",
    "speed_0_4"
]

def run_tuning():
    print("="*80)
    print("STARTING FULL SYSTEM AUTO-TUNE")
    print("Settings: 30 Iterations | Up to 10 Files per group | Hybrid Method")
    print("="*80)
    
    start_time = time.time()
    
    for group in SPEED_GROUPS:
        print(f"\n\n>>> TUNING GROUP: {group} <<<")
        print("-" * 40)
        
        # Build the command
        cmd = [
            "python3", "autotune_speed.py",
            "--speed_group", group,
            "--max_total_iter", "100",
            "--num_files", "10",
            "--method", "Hybrid"
        ]
        
        try:
            # Run the command and stream output
            process = subprocess.Popen(cmd, stdout=None, stderr=None)
            process.wait()
            
            if process.returncode != 0:
                print(f"Warning: Tuning for {group} exited with code {process.returncode}")
                
        except KeyboardInterrupt:
            print("\nFull tune interrupted by user. Moving to next group or exiting...")
            # We allow the user to skip a group by hitting Ctrl-C once
            continue
            
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    print("\n" + "="*80)
    print(f"FULL AUTO-TUNE COMPLETE in {duration:.1f} minutes!")
    print("All best gains have been saved to controllers/pid_ff_scheduled_tune.py")
    print("="*80)

if __name__ == "__main__":
    run_tuning()


