import numpy as np
import re

def smooth(arr, window=3):
    # Padding to handle edges
    res = np.convolve(arr, np.ones(window)/window, mode='same')
    # Keep edges as they were to avoid too much distortion
    res[0] = arr[0]
    res[-1] = arr[-1]
    return res

CONTROLLER_PATH = "controllers/pid_ff_scheduled_tune.py"

with open(CONTROLLER_PATH, 'r') as f:
    content = f.read()

names = ['p_points', 'i_points', 'd_points', 'k_ff_points', 'preview_points']

for name in names:
    pattern = rf'(self\.{name}\s*=\s*np\.array\(\[)(.*?)(\]\))'
    match = re.search(pattern, content)
    if match:
        prefix = match.group(1)
        values_str = match.group(2)
        suffix = match.group(3)
        values = np.array([float(v.strip()) for v in values_str.split(',')])
        
        # Smooth twice for extra smoothness
        smoothed = smooth(smooth(values))
        
        new_values_str = ", ".join([f"{v:.4f}" for v in smoothed])
        content = re.sub(pattern, f"{prefix}{new_values_str}{suffix}", content)

with open(CONTROLLER_PATH, 'w') as f:
    f.write(content)

print("Smoothed all gain tables in pid_ff_scheduled_tune.py")


