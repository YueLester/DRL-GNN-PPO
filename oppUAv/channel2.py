import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
D_max = 100  # Maximum distance in meters
height = 20  # Drone height in meters
positions = np.linspace(-D_max, D_max, 200)  # Drone positions along path
noise_power = 1.0  # Noise power in mW

# Calculate distances from ground station at each position
distances = np.sqrt(positions ** 2 + height ** 2)

# Path loss model: signal power decreases with square of distance
# Using formula: received power = transmitted power / (distance^2)
# Assume transmitted power = 100 mW
transmitted_power = 100  # mW
received_powers = transmitted_power / (distances ** 2)

# Calculate Signal to Noise Ratio (SNR)
snr = received_powers / noise_power

# Calculate data rate using Shannon's formula: C = B * log2(1 + SNR)
# Assume bandwidth B = 10 MHz
bandwidth = 10  # MHz
data_rates = bandwidth * np.log2(1 + snr)

# Create figure for static plot
plt.figure(figsize=(10, 6))

# Plot data rate vs position
plt.subplot(2, 1, 1)
plt.plot(positions, data_rates, 'b-', linewidth=2)
plt.grid(True)
plt.xlabel('Drone Position (m)')
plt.ylabel('Data Rate (Mbps)')
plt.title('Data Rate vs Drone Position During Uniform Flight')

# Plot received power vs position
plt.subplot(2, 1, 2)
plt.plot(positions, 10 * np.log10(received_powers), 'r-', linewidth=2)  # Convert to dBm
plt.grid(True)
plt.xlabel('Drone Position (m)')
plt.ylabel('Received Signal Power (dBm)')
plt.tight_layout()

# Save the static plot
plt.savefig('drone_data_rate_static.png')

# Create animation figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle('Drone Flight Data Rate Simulation', fontsize=16)

# Line for data rate plot
line1, = ax1.plot([], [], 'b-', linewidth=2)
point1, = ax1.plot([], [], 'ro', markersize=8)
ax1.set_xlim(-D_max, D_max)
ax1.set_ylim(0, max(data_rates) * 1.1)
ax1.grid(True)
ax1.set_xlabel('Drone Position (m)')
ax1.set_ylabel('Data Rate (Mbps)')
ax1.set_title('Data Rate vs Position')

# Line for signal power plot
line2, = ax2.plot([], [], 'r-', linewidth=2)
point2, = ax2.plot([], [], 'bo', markersize=8)
ax2.set_xlim(-D_max, D_max)
ax2.set_ylim(min(10 * np.log10(received_powers)) * 1.1, max(10 * np.log10(received_powers)) * 1.1)
ax2.grid(True)
ax2.set_xlabel('Drone Position (m)')
ax2.set_ylabel('Received Signal Power (dBm)')
ax2.set_title('Signal Power vs Position')

# Ground station marker
ground_station = plt.Line2D([0], [0], marker='s', color='green',
                            markersize=10, linestyle='None',
                            markerfacecolor='green')
ax2.add_line(ground_station)


# Actual drone flight animation
def update(frame):
    # Update data for data rate plot
    line1.set_data(positions[:frame], data_rates[:frame])
    point1.set_data(positions[frame - 1], data_rates[frame - 1])

    # Update data for signal power plot
    line2.set_data(positions[:frame], 10 * np.log10(received_powers[:frame]))
    point2.set_data(positions[frame - 1], 10 * np.log10(received_powers[frame - 1]))

    # Draw the drone position above the ground station
    ax2.set_title(f'Signal Power vs Position - Drone at {positions[frame - 1]:.1f}m')

    return line1, point1, line2, point2


# Create animation
ani = FuncAnimation(fig, update, frames=len(positions), interval=50, blit=True)

# Save animation (requires ffmpeg)
# ani.save('drone_flight_simulation.mp4', writer='ffmpeg', fps=30)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

# Additional visualization: 3D plot showing drone path and signal strength
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Create 3D coordinates for drone path
x = np.zeros_like(positions)  # Drone flies along y-axis
y = positions
z = np.ones_like(positions) * height

# Plot drone path
ax.plot(x, y, z, 'b-', linewidth=2, label='Drone Path')

# Plot ground station
ax.scatter([0], [0], [0], color='green', s=100, label='Ground Station')

# Plot signal strength at various points
# We'll use a sample of points to avoid cluttering
sample_indices = np.linspace(0, len(positions) - 1, 20, dtype=int)
for i in sample_indices:
    # Signal strength represented by line width and color
    signal_strength = received_powers[i]
    normalized_strength = (signal_strength - min(received_powers)) / (max(received_powers) - min(received_powers))

    # Line connecting drone to ground station
    ax.plot([0, x[i]], [0, y[i]], [0, z[i]],
            color=plt.cm.jet(normalized_strength),
            linewidth=1 + 3 * normalized_strength,
            alpha=0.6)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Visualization of Drone Path and Signal Strength')
ax.legend()

plt.savefig('drone_path_3d.png')
plt.show()

print("Simulation completed. Maximum data rate:", np.max(data_rates), "Mbps")
print("Position of maximum data rate:", positions[np.argmax(data_rates)], "m")

