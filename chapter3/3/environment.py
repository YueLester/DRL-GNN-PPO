import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh

# Simulation parameters
simulation_time = 100  # seconds
time_step = 0.1  # seconds
num_links = 3
avg_hops = 6

# Time array
time = np.arange(0, simulation_time, time_step)
num_samples = len(time)

# Channel model parameters
# Path loss model: PL(d) = PL(d0) + 10*n*log10(d/d0) + X_sigma
# Where:
# - PL(d0) is the path loss at reference distance d0
# - n is the path loss exponent
# - X_sigma is a zero-mean Gaussian random variable with standard deviation sigma

# Parameters for path loss model
pl_d0 = 40  # dB, path loss at reference distance
d0 = 1  # meter, reference distance
n_values = [2.7, 3.0, 3.3]  # path loss exponents for different links
sigma = 4  # dB, shadow fading standard deviation
distances = [500, 600, 700]  # meters, distances for each link

# Calculate static path loss component for each link
static_path_loss = np.array([pl_d0 + 10 * n * np.log10(d / d0) for n, d in zip(n_values, distances)])

# Generate time-varying components
# 1. Shadow fading (slow fading)
shadow_fading = np.array([sigma * np.random.randn(num_samples) for _ in range(num_links)])

# 2. Fast fading (Rayleigh)
rayleigh_scale = 0.5
fast_fading = np.array([rayleigh.rvs(scale=rayleigh_scale, size=num_samples) for _ in range(num_links)])

# 3. Periodic interference (simulating environmental changes)
interference_amplitude = 3  # dB
interference_frequencies = [0.05, 0.08, 0.12]  # Hz, different for each link
interference = np.array([interference_amplitude * np.sin(2 * np.pi * freq * time) for freq in interference_frequencies])

# Total path loss over time for each link
total_path_loss = np.zeros((num_links, num_samples))
for i in range(num_links):
    total_path_loss[i] = static_path_loss[i] + shadow_fading[i] + fast_fading[i] + interference[i]

# Bandwidth model (inverse relationship with path loss)
base_bandwidth = 55  # MHz
bandwidth_amplitude = 5  # MHz
bandwidth = np.zeros((num_links, num_samples))
for i in range(num_links):
    # Normalized path loss for bandwidth calculation
    normalized_pl = (total_path_loss[i] - np.min(total_path_loss[i])) / (np.max(total_path_loss[i]) - np.min(total_path_loss[i]))
    bandwidth[i] = base_bandwidth + bandwidth_amplitude * np.sin(2 * np.pi * 0.03 * time + i*2*np.pi/3)

# Packet loss model (function of path loss)
packet_loss = np.zeros((num_links, num_samples))
for i in range(num_links):
    # Simplified packet loss model: higher path loss -> higher packet loss
    normalized_pl = (total_path_loss[i] - np.min(total_path_loss)) / (np.max(total_path_loss) - np.min(total_path_loss))
    packet_loss[i] = 0.1 + 0.01 * normalized_pl * total_path_loss[i] / 10  # Base 10% + variable component

# Energy consumption model (based on hops and path loss)
# Energy per bit = basic_energy * hops * f(path_loss)
basic_energy = 1e-6  # Joules per bit
energy_consumption = np.zeros((num_links, num_samples))
for i in range(num_links):
    # Each link has a random number of hops around the average
    hops = np.random.poisson(avg_hops, num_samples)
    # Energy increases with path loss and number of hops
    energy_consumption[i] = basic_energy * hops * (1 + total_path_loss[i] / 100)

# Plotting
plt.figure(figsize=(14, 10))

# Plot 1: Bandwidth vs Time
plt.subplot(2, 1, 1)
for i in range(num_links):
    plt.plot(time, bandwidth[i], label=f'Link {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Bandwidth (MHz)')
plt.title('Bandwidth vs Time for Multiple Links')
plt.legend()
plt.grid(True)

# Plot 2: Packet Loss vs Time
plt.subplot(2, 1, 2)
for i in range(num_links):
    plt.plot(time, packet_loss[i], label=f'Link {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Packet Loss Rate')
plt.title('Packet Loss vs Time for Multiple Links')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Print average values
print("\nAverage Values:")
for i in range(num_links):
    print(f"Link {i+1}:")
    print(f"  Average Path Loss: {np.mean(total_path_loss[i]):.2f} dB")
    print(f"  Average Bandwidth: {np.mean(bandwidth[i]):.2f} MHz")
    print(f"  Average Packet Loss: {np.mean(packet_loss[i]):.4f}")
    print(f"  Average Energy Consumption: {np.mean(energy_consumption[i]):.9f} J/bit")

# Save simulation data for further analysis
simulation_data = {
    'time': time,
    'path_loss': total_path_loss,
    'bandwidth': bandwidth,
    'packet_loss': packet_loss,
    'energy': energy_consumption
}

# Additional function to calculate throughput based on bandwidth and packet loss
def calculate_throughput(bw, pl):
    # Simplified throughput calculation: throughput = bandwidth * (1 - packet_loss)
    # Convert bandwidth from MHz to Mbps (assuming 1 Hz = 1 bit/s for simplicity)
    return bw * (1 - pl)

# Calculate and plot throughput
throughput = np.zeros((num_links, num_samples))
for i in range(num_links):
    throughput[i] = calculate_throughput(bandwidth[i], packet_loss[i])

plt.figure(figsize=(14, 5))
for i in range(num_links):
    plt.plot(time, throughput[i], label=f'Link {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Throughput (Mbps)')
plt.title('Throughput vs Time for Multiple Links')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Print average throughput
print("\nAverage Throughput:")
for i in range(num_links):
    print(f"Link {i+1}: {np.mean(throughput[i]):.2f} Mbps")

