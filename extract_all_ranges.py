import subprocess

ranges = [
    (0, 4),
    (4, 10),
    (10, 16),
    (16, 22),
    (22, 28),
    (28, 34),
    (34, 40),
    (40, 46)
]

for vmin, vmax in ranges:
    dest = f"tune/speed_{vmin}_{vmax}"
    print(f"\n--- Extracting {vmin}-{vmax} m/s ---")
    subprocess.run([
        "python3", "extract_segments.py",
        "--src", "data",
        "--dest", dest,
        "--vmin", str(vmin),
        "--vmax", str(vmax),
        "--n", "10"
    ])
