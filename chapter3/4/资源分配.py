import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# Set random seed for reproducibility
np.random.seed(42)

# Time steps (in minutes)
time_steps = 60  # 1 hour with 1-minute intervals
time_array = np.arange(time_steps)


# Function to generate bandwidth for a path with some pattern and randomness
def generate_bandwidth(base_value, amplitude, frequency, noise_level):
    # Create a sine wave pattern with given frequency
    time = np.arange(time_steps)
    pattern = base_value + amplitude * np.sin(2 * np.pi * frequency * time / time_steps)

    # Add some random noise
    noise = np.random.normal(0, noise_level, time_steps)

    # Combine pattern and noise, ensure all values are positive
    bandwidth = pattern + noise
    bandwidth = np.maximum(bandwidth, 0)

    return bandwidth


# Generate bandwidths for 5 paths with different characteristics
path_bandwidths = {
    'path1': generate_bandwidth(base_value=80, amplitude=15, frequency=2, noise_level=5),
    'path2': generate_bandwidth(base_value=60, amplitude=20, frequency=1.5, noise_level=7),
    'path3': generate_bandwidth(base_value=40, amplitude=10, frequency=3, noise_level=4),
    'path4': generate_bandwidth(base_value=30, amplitude=8, frequency=2.5, noise_level=3),
    'path5': generate_bandwidth(base_value=20, amplitude=5, frequency=1, noise_level=2)
}

# Create DataFrame for easier handling
df = pd.DataFrame(path_bandwidths)

# Calculate cumulative bandwidths
df['cum_path1'] = df['path1']
df['cum_path2'] = df['path1'] + df['path2']
df['cum_path3'] = df['path1'] + df['path2'] + df['path3']
df['cum_path4'] = df['path1'] + df['path2'] + df['path3'] + df['path4']
df['cum_path5'] = df['path1'] + df['path2'] + df['path3'] + df['path4'] + df['path5']

# Create custom colormap for paths
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
path_cmap = LinearSegmentedColormap.from_list('path_colors', colors)

# Create figure
plt.figure(figsize=(14, 10))

# Plot 1: Individual path bandwidths over time
plt.subplot(2, 1, 1)
for i, path_name in enumerate(path_bandwidths.keys()):
    plt.plot(time_array, df[path_name], label=path_name, color=colors[i], linewidth=2)

plt.title('Individual Path Bandwidths Over Time', fontsize=14)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Bandwidth (Mbps)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

# Plot 2: Stacked area chart showing cumulative bandwidths
plt.subplot(2, 1, 2)

# Plot the filled areas between cumulative lines
plt.fill_between(time_array, 0, df['cum_path1'], color=colors[0], alpha=0.8, label='path1')
plt.fill_between(time_array, df['cum_path1'], df['cum_path2'], color=colors[1], alpha=0.8, label='path2')
plt.fill_between(time_array, df['cum_path2'], df['cum_path3'], color=colors[2], alpha=0.8, label='path3')
plt.fill_between(time_array, df['cum_path3'], df['cum_path4'], color=colors[3], alpha=0.8, label='path4')
plt.fill_between(time_array, df['cum_path4'], df['cum_path5'], color=colors[4], alpha=0.8, label='path5')

# Plot the cumulative lines
plt.plot(time_array, df['cum_path1'], color='black', linewidth=1)
plt.plot(time_array, df['cum_path2'], color='black', linewidth=1)
plt.plot(time_array, df['cum_path3'], color='black', linewidth=1)
plt.plot(time_array, df['cum_path4'], color='black', linewidth=1)
plt.plot(time_array, df['cum_path5'], color='black', linewidth=1)

plt.title('Cumulative Path Bandwidths Over Time', fontsize=14)
plt.xlabel('Time (minutes)', fontsize=12)
plt.ylabel('Cumulative Bandwidth (Mbps)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('multipath_routing_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

