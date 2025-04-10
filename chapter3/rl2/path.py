import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import random


class Path:
    """
    A class representing a communication path in maritime networks
    with time-varying bandwidth and packet loss characteristics.

    The class simulates continuous changes in network parameters over time,
    with fluctuation levels determining the volatility of the changes.
    """

    def __init__(self,
                 avg_bandwidth_series: List[float],  # Series of average bandwidth values (Mbps)
                 avg_packet_loss_series: List[float],  # Series of average packet loss rate values
                 fluctuation_level_series: List[int],  # Series of fluctuation levels (1-5)
                 time_slot_duration: int = 5,  # Duration of each time slot in minutes
                 points_per_slot: int = 60,  # Number of simulation points per time slot
                 fluctuation_pattern: str = 'random_walk'):  # Type of fluctuation pattern: 'random_walk', 'sine', 'mixed'
        """
        Initialize a path with time-varying parameters.

        Args:
            avg_bandwidth_series: List of average bandwidth values for each time slot (Mbps)
            avg_packet_loss_series: List of average packet loss rate values for each time slot (0-1)
            fluctuation_level_series: List of fluctuation levels for each time slot (1-5)
            time_slot_duration: Duration of each time slot in minutes
            points_per_slot: Number of simulation points within each time slot
            fluctuation_pattern: Pattern of fluctuations: 'random_walk', 'sine', or 'mixed'
        """
        # Validate input series are equal length
        if not (len(avg_bandwidth_series) == len(avg_packet_loss_series) == len(fluctuation_level_series)):
            raise ValueError("All input series must have the same length")

        self.avg_bandwidth_series = avg_bandwidth_series
        self.avg_packet_loss_series = avg_packet_loss_series
        self.fluctuation_level_series = [min(max(level, 1), 5) for level in
                                         fluctuation_level_series]  # Constrain to 1-5
        self.time_slot_duration = time_slot_duration
        self.points_per_slot = points_per_slot
        self.fluctuation_pattern = fluctuation_pattern

        # Derived parameters
        self.total_slots = len(avg_bandwidth_series)
        self.total_duration = self.total_slots * time_slot_duration  # Total duration in minutes
        self.total_points = self.total_slots * points_per_slot

        # Initialize time series
        self.time_points = np.linspace(0, self.total_duration, self.total_points)

        # Generated data
        self.bandwidth_series = None
        self.packet_loss_series = None
        self.bandwidth_change_rates = None
        self.packet_loss_change_rates = None
        self.energy_consumption_series = None  # Energy consumption per byte at each time point

        # Generate the continuous time series
        self._generate_continuous_series()

    def _generate_continuous_series(self):
        """Generate continuous time series for bandwidth and packet loss."""
        # Initialize arrays
        self.bandwidth_series = np.zeros(self.total_points)
        self.packet_loss_series = np.zeros(self.total_points)

        # Generate data for each time slot, ensuring continuity between slots
        last_bw_value = None
        last_pl_value = None

        for slot_idx in range(self.total_slots):
            slot_start = slot_idx * self.points_per_slot
            slot_end = (slot_idx + 1) * self.points_per_slot

            # Get parameters for this slot
            avg_bw = self.avg_bandwidth_series[slot_idx]
            avg_pl = self.avg_packet_loss_series[slot_idx]
            fluct_level = self.fluctuation_level_series[slot_idx]

            # Generate data for this slot with continuity from previous slot
            self._generate_slot_data(slot_start, slot_end, avg_bw, avg_pl, fluct_level,
                                     last_bw_value, last_pl_value)

            # Store last values for next slot
            last_bw_value = self.bandwidth_series[slot_end - 1]
            last_pl_value = self.packet_loss_series[slot_end - 1]

        # Calculate change rates
        self._calculate_change_rates()

        # Calculate energy consumption
        self._calculate_energy_consumption()

    def _generate_slot_data(self, start_idx, end_idx, avg_bw, avg_pl, fluct_level,
                            last_bw_value=None, last_pl_value=None):
        """
        Generate data for a single time slot, ensuring continuity from previous slot.

        Args:
            start_idx: Start index for this slot
            end_idx: End index for this slot
            avg_bw: Target average bandwidth for this slot
            avg_pl: Target average packet loss for this slot
            fluct_level: Fluctuation level for this slot
            last_bw_value: Last bandwidth value from previous slot (None if first slot)
            last_pl_value: Last packet loss value from previous slot (None if first slot)
        """
        # Number of points in this slot
        num_points = end_idx - start_idx

        # Define standard deviations based on fluctuation level
        # Scale for bandwidth (maximum 4Mbps fluctuation as specified)
        bw_std_levels = {
            1: 0.2,  # < 1 std dev
            2: 0.5,  # ~ 1 std dev
            3: 1.0,  # 1-2 std dev
            4: 2.0,  # 2-3 std dev
            5: 4.0  # > 3 std dev
        }

        # Scale for packet loss (adjusted according to avg packet loss)
        pl_base_std = 0.02  # Base standard deviation
        pl_std_levels = {
            1: pl_base_std * 0.2,
            2: pl_base_std * 0.5,
            3: pl_base_std * 1.0,
            4: pl_base_std * 2.0,
            5: pl_base_std * 4.0
        }

        # Get standard deviations for this slot
        bw_std = bw_std_levels[fluct_level]
        pl_std = pl_std_levels[fluct_level]

        # Generate pattern based on selected fluctuation pattern
        if self.fluctuation_pattern == 'random_walk':
            # Random walk pattern (original implementation)
            bw_walk, pl_walk = self._generate_random_walk_pattern(num_points, bw_std, pl_std)
        elif self.fluctuation_pattern == 'sine':
            # Sine wave pattern
            bw_walk, pl_walk = self._generate_sine_wave_pattern(num_points, bw_std, pl_std)
        elif self.fluctuation_pattern == 'mixed':
            # Combine random walk and sine wave
            bw_walk, pl_walk = self._generate_mixed_pattern(num_points, bw_std, pl_std)
        else:
            # Default to random walk
            bw_walk, pl_walk = self._generate_random_walk_pattern(num_points, bw_std, pl_std)

        # Handle continuity from previous slot if this is not the first slot
        transition_points = min(int(self.points_per_slot * 0.2), 12)  # Use 20% of slot for transition, max 12 points

        if last_bw_value is not None and last_pl_value is not None:
            # Create a smooth transition from last value to the new average
            for i in range(transition_points):
                # Linear transition weight
                alpha = i / transition_points

                # Calculate transition values
                transition_bw = last_bw_value * (1 - alpha) + (avg_bw + bw_walk[i]) * alpha
                transition_pl = last_pl_value * (1 - alpha) + (avg_pl + pl_walk[i]) * alpha

                # Set values
                self.bandwidth_series[start_idx + i] = transition_bw
                self.packet_loss_series[start_idx + i] = np.clip(transition_pl, 0.001, 0.999)

            # Adjust the rest of the slot values
            rest_start = start_idx + transition_points
            rest_points = num_points - transition_points

            if rest_points > 0:
                # Adjust the mean of the remaining points to achieve the target average
                remaining_bw = avg_bw + bw_walk[transition_points:transition_points + rest_points]
                remaining_pl = avg_pl + pl_walk[transition_points:transition_points + rest_points]

                self.bandwidth_series[rest_start:end_idx] = np.clip(
                    remaining_bw,
                    max(0.1, avg_bw - 2 * bw_std),
                    avg_bw + 2 * bw_std
                )

                self.packet_loss_series[rest_start:end_idx] = np.clip(
                    remaining_pl,
                    max(0.001, avg_pl - 2 * pl_std),
                    min(0.999, avg_pl + 2 * pl_std)
                )
        else:
            # For first slot, just use the generated values
            self.bandwidth_series[start_idx:end_idx] = np.clip(
                avg_bw + bw_walk,
                max(0.1, avg_bw - 2 * bw_std),
                avg_bw + 2 * bw_std
            )

            self.packet_loss_series[start_idx:end_idx] = np.clip(
                avg_pl + pl_walk,
                max(0.001, avg_pl - 2 * pl_std),
                min(0.999, avg_pl + 2 * pl_std)
            )

        # Apply inverse relationship between bandwidth and packet loss
        # (introduces some anti-correlation)
        inverse_factor = 0.3  # Strength of inverse relationship
        for i in range(start_idx, end_idx):
            # Normalize to [-1, 1] range
            bw_norm = (self.bandwidth_series[i] - avg_bw) / max(1e-10, bw_std)

            # Apply inverse effect to packet loss (with some noise to keep it realistic)
            inverse_effect = -bw_norm * inverse_factor * pl_std
            self.packet_loss_series[i] = np.clip(
                self.packet_loss_series[i] + inverse_effect,
                0.001,  # Minimum packet loss
                0.999  # Maximum packet loss
            )

    def _generate_random_walk_pattern(self, num_points, bw_std, pl_std):
        """Generate random walk pattern for bandwidth and packet loss fluctuations."""
        # Generate random walk with controlled volatility
        # For bandwidth
        bw_changes = np.random.normal(0, bw_std / np.sqrt(self.points_per_slot / 10), num_points)
        bw_walk = np.cumsum(bw_changes)
        # Center and scale the walk
        bw_walk = bw_walk - np.mean(bw_walk)
        bw_walk = bw_walk * (bw_std / max(1e-10, np.std(bw_walk)))

        # For packet loss
        pl_changes = np.random.normal(0, pl_std / np.sqrt(self.points_per_slot / 10), num_points)
        pl_walk = np.cumsum(pl_changes)
        # Center and scale the walk
        pl_walk = pl_walk - np.mean(pl_walk)
        pl_walk = pl_walk * (pl_std / max(1e-10, np.std(pl_walk)))

        return bw_walk, pl_walk

    def _generate_sine_wave_pattern(self, num_points, bw_std, pl_std):
        """Generate sine wave pattern for bandwidth and packet loss fluctuations."""
        # Create time array for this slot
        t = np.linspace(0, 2 * np.pi, num_points)

        # Generate sine waves with random phase and frequency variations
        bw_freq = 1 + 0.5 * np.random.random()  # Random frequency component
        bw_phase = 2 * np.pi * np.random.random()  # Random phase
        bw_walk = bw_std * np.sin(bw_freq * t + bw_phase)

        # Slightly different parameters for packet loss to reduce correlation
        pl_freq = 1 + 0.5 * np.random.random()
        pl_phase = 2 * np.pi * np.random.random()
        pl_walk = pl_std * np.sin(pl_freq * t + pl_phase)

        return bw_walk, pl_walk

    def _generate_mixed_pattern(self, num_points, bw_std, pl_std):
        """Generate a mixed pattern combining random walk and sine wave."""
        # Generate both patterns
        bw_random, pl_random = self._generate_random_walk_pattern(num_points, bw_std * 0.5, pl_std * 0.5)
        bw_sine, pl_sine = self._generate_sine_wave_pattern(num_points, bw_std * 0.5, pl_std * 0.5)

        # Combine patterns
        bw_walk = bw_random + bw_sine
        pl_walk = pl_random + pl_sine

        # Normalize to maintain intended standard deviation
        bw_walk = bw_walk * (bw_std / max(1e-10, np.std(bw_walk)))
        pl_walk = pl_walk * (pl_std / max(1e-10, np.std(pl_walk)))

        return bw_walk, pl_walk

    def _calculate_change_rates(self):
        """Calculate the rate of change for bandwidth and packet loss."""
        # Time difference between consecutive points (in minutes)
        time_diff = self.total_duration / (self.total_points - 1)

        # Initialize arrays
        self.bandwidth_change_rates = np.zeros(self.total_points - 1)
        self.packet_loss_change_rates = np.zeros(self.total_points - 1)

        # Calculate change rates
        for i in range(self.total_points - 1):
            self.bandwidth_change_rates[i] = (self.bandwidth_series[i + 1] - self.bandwidth_series[i]) / time_diff
            self.packet_loss_change_rates[i] = (self.packet_loss_series[i + 1] - self.packet_loss_series[i]) / time_diff

    def _calculate_energy_consumption(self):
        """
        Calculate energy consumption per byte based on bandwidth.

        Energy consumption model: ~4J/300Byte at nominal bandwidth,
        with inverse relationship to bandwidth (lower bandwidth = higher energy usage)
        """
        # Calculate nominal energy (4J/300Byte)
        nominal_energy_per_byte = 4.0 / 300.0  # Joules per byte
        nominal_bandwidth = 10.0  # Mbps (reference point)

        # Initialize energy consumption array
        self.energy_consumption_series = np.zeros(self.total_points)

        # Calculate energy consumption as inversely proportional to bandwidth
        # with some randomness to model real-world variations
        for i in range(self.total_points):
            # Base energy is inversely proportional to bandwidth
            bandwidth_factor = nominal_bandwidth / max(0.1, self.bandwidth_series[i])

            # Add some random variation (±10%)
            random_factor = 1.0 + 0.1 * (2.0 * np.random.random() - 1.0)

            # Calculate energy per byte
            self.energy_consumption_series[i] = nominal_energy_per_byte * bandwidth_factor * random_factor

    def transmit_data(self, packet_size_bytes: int, time_point_idx: int) -> Dict:
        """
        Simulate transmission of a data packet at a specific time point.

        Args:
            packet_size_bytes: Size of the packet in bytes
            time_point_idx: Index of the time point for transmission

        Returns:
            Dictionary with transmission results:
                - success: Boolean indicating if packet was received
                - energy_consumed: Energy consumed for transmission attempt (J)
                - effective_throughput: Effective throughput (Mbps)
        """
        if time_point_idx < 0 or time_point_idx >= self.total_points:
            raise ValueError(f"Time point index must be between 0 and {self.total_points - 1}")

        # Get current channel parameters
        current_bandwidth = self.bandwidth_series[time_point_idx]  # Mbps
        current_packet_loss = self.packet_loss_series[time_point_idx]
        current_energy_per_byte = self.energy_consumption_series[time_point_idx]  # J/byte

        # Calculate transmission success based on packet loss probability
        success = np.random.random() > current_packet_loss

        # Calculate energy consumed for the transmission attempt
        # We consume energy regardless of success/failure
        energy_consumed = packet_size_bytes * current_energy_per_byte

        # Calculate effective throughput
        bits_transferred = packet_size_bytes * 8 / 1_000_000 if success else 0  # Mbits
        effective_throughput = current_bandwidth * (1 - current_packet_loss)

        return {
            "success": success,
            "energy_consumed": energy_consumed,
            "effective_throughput": effective_throughput
        }

    def transmit_bulk_data(self, total_bytes: int, time_point_idx: int, packet_size: int = 1500) -> Dict:
        """
        Simulate transmission of multiple packets at a specific time point (within 1 second).

        Args:
            total_bytes: Total amount of data to transmit in bytes
            time_point_idx: Index of the time point for transmission
            packet_size: Size of each packet in bytes (default: 1500 - standard MTU)

        Returns:
            Dictionary with transmission results:
                - packets_sent: Number of packets sent
                - packets_received: Number of packets successfully received
                - bytes_received: Number of bytes successfully received
                - energy_consumed: Total energy consumed for transmission (J)
                - success_rate: Percentage of successful transmissions
        """
        if time_point_idx < 0 or time_point_idx >= self.total_points:
            raise ValueError(f"Time point index must be between 0 and {self.total_points - 1}")

        # Calculate number of packets
        num_packets = int(np.ceil(total_bytes / packet_size))

        # Get current channel parameters
        current_bandwidth = self.bandwidth_series[time_point_idx]  # Mbps
        current_packet_loss = self.packet_loss_series[time_point_idx]
        current_energy_per_byte = self.energy_consumption_series[time_point_idx]  # J/byte

        # Calculate maximum packets transferable in 1 second with current bandwidth
        # bandwidth in Mbps = megabits per second
        bits_per_second = current_bandwidth * 1_000_000
        bytes_per_second = bits_per_second / 8
        max_packets_per_second = int(bytes_per_second / packet_size)

        # Limit packets to what can be sent in 1 second
        packets_to_send = min(num_packets, max_packets_per_second)

        # Simulate transmission of each packet
        packets_received = 0
        for _ in range(packets_to_send):
            if np.random.random() > current_packet_loss:
                packets_received += 1

        # Calculate bytes received and energy consumed
        bytes_received = packets_received * packet_size
        energy_consumed = packets_to_send * packet_size * current_energy_per_byte

        # Calculate success rate
        success_rate = (packets_received / packets_to_send) * 100 if packets_to_send > 0 else 0

        return {
            "packets_sent": packets_to_send,
            "packets_received": packets_received,
            "bytes_received": bytes_received,
            "energy_consumed": energy_consumed,
            "success_rate": success_rate
        }

    def get_bandwidth_at_time(self, time_in_minutes: float) -> float:
        """Get bandwidth at a specific time."""
        if time_in_minutes < 0 or time_in_minutes > self.total_duration:
            raise ValueError(f"Time must be between 0 and {self.total_duration} minutes")

        # Find closest index
        idx = int(round(time_in_minutes / self.total_duration * (self.total_points - 1)))
        return self.bandwidth_series[idx]

    def get_packet_loss_at_time(self, time_in_minutes: float) -> float:
        """Get packet loss at a specific time."""
        if time_in_minutes < 0 or time_in_minutes > self.total_duration:
            raise ValueError(f"Time must be between 0 and {self.total_duration} minutes")

        # Find closest index
        idx = int(round(time_in_minutes / self.total_duration * (self.total_points - 1)))
        return self.packet_loss_series[idx]

    def get_energy_at_time(self, time_in_minutes: float) -> float:
        """Get energy consumption per byte at a specific time."""
        if time_in_minutes < 0 or time_in_minutes > self.total_duration:
            raise ValueError(f"Time must be between 0 and {self.total_duration} minutes")

        # Find closest index
        idx = int(round(time_in_minutes / self.total_duration * (self.total_points - 1)))
        return self.energy_consumption_series[idx]

    def get_change_rates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the rate of change for bandwidth and packet loss.

        Returns:
            Tuple containing bandwidth change rates and packet loss change rates
        """
        return self.bandwidth_change_rates, self.packet_loss_change_rates

    def get_state_at_time(self, time_in_minutes: float) -> Dict:
        """
        Get complete channel state at a specific time.

        Args:
            time_in_minutes: Time point in minutes

        Returns:
            Dictionary with channel state parameters
        """
        if time_in_minutes < 0 or time_in_minutes > self.total_duration:
            raise ValueError("Time must be between 0 and {0} minutes".format(self.total_duration))

        # Find closest index
        idx = int(round(time_in_minutes / self.total_duration * (self.total_points - 1)))

        return {
            "time": time_in_minutes,
            "bandwidth": self.bandwidth_series[idx],
            "packet_loss": self.packet_loss_series[idx],
            "energy_per_byte": self.energy_consumption_series[idx]
        }

    def get_state_at_index(self, index: int) -> Dict:
        """
        Get complete channel state at a specific index.

        Args:
            index: Index of the time point

        Returns:
            Dictionary with channel state parameters
        """
        if index < 0 or index >= self.total_points:
            raise ValueError("Index must be between 0 and {0}".format(self.total_points - 1))

        time_in_minutes = self.time_points[index]

        return {
            "time": time_in_minutes,
            "bandwidth": self.bandwidth_series[index],
            "packet_loss": self.packet_loss_series[index],
            "energy_per_byte": self.energy_consumption_series[index]
        }

    def print_summary(self):
        """Print a summary of the path characteristics."""
        print(f"Total Duration: {self.total_duration} minutes")
        print(f"Time Slots: {self.total_slots} slots of {self.time_slot_duration} minutes each")
        print(f"Average Bandwidth: {np.mean(self.bandwidth_series):.2f} Mbps")
        print(f"Min/Max Bandwidth: {np.min(self.bandwidth_series):.2f}/{np.max(self.bandwidth_series):.2f} Mbps")
        print(f"Average Packet Loss: {np.mean(self.packet_loss_series) * 100:.2f}%")
        print(
            f"Min/Max Packet Loss: {np.min(self.packet_loss_series) * 100:.2f}%/{np.max(self.packet_loss_series) * 100:.2f}%")
        print(f"Average Energy Consumption: {np.mean(self.energy_consumption_series) * 1000:.2f} mJ/byte")
        print(f"Energy for 300 Bytes (avg): {np.mean(self.energy_consumption_series) * 300:.2f} J")

        # Calculate max change rates
        max_bw_change = np.max(np.abs(self.bandwidth_change_rates))
        max_pl_change = np.max(np.abs(self.packet_loss_change_rates)) * 100  # Convert to percentage

        print(f"Max Bandwidth Change Rate: {max_bw_change:.4f} Mbps/min")
        print(f"Max Packet Loss Change Rate: {max_pl_change:.4f} %/min")

        print(f"Time Slots: {self.total_slots} slots of {self.time_slot_duration} minutes each")
        print(f"Average Bandwidth: {np.mean(self.bandwidth_series):.2f} Mbps")
        print(f"Min/Max Bandwidth: {np.min(self.bandwidth_series):.2f}/{np.max(self.bandwidth_series):.2f} Mbps")
        print(f"Average Packet Loss: {np.mean(self.packet_loss_series) * 100:.2f}%")
        print(
            f"Min/Max Packet Loss: {np.min(self.packet_loss_series) * 100:.2f}%/{np.max(self.packet_loss_series) * 100:.2f}%")
        print(f"Average Energy Consumption: {np.mean(self.energy_consumption_series) * 1000:.2f} mJ/byte")
        print(f"Energy for 300 Bytes (avg): {np.mean(self.energy_consumption_series) * 300:.2f} J")

        # Calculate max change rates
        max_bw_change = np.max(np.abs(self.bandwidth_change_rates))
        max_pl_change = np.max(np.abs(self.packet_loss_change_rates)) * 100  # Convert to percentage

        print(f"Max Bandwidth Change Rate: {max_bw_change:.4f} Mbps/min")
        print(f"Max Packet Loss Change Rate: {max_pl_change:.4f} %/min")

    def plot(self, figsize=(12, 12)):
        """
        Plot the bandwidth, packet loss, and energy consumption over time.

        Args:
            figsize: Size of the figure as (width, height) in inches
        """
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=figsize)

        # Plot bandwidth
        ax1.plot(self.time_points, self.bandwidth_series, 'b-', linewidth=1.5)
        ax1.set_title('Bandwidth over Time')
        ax1.set_ylabel('Bandwidth (Mbps)')
        ax1.grid(True)

        # Highlight time slots with vertical lines
        for slot in range(1, self.total_slots):
            slot_time = slot * self.time_slot_duration
            ax1.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.5)

        # Plot packet loss
        ax2.plot(self.time_points, self.packet_loss_series * 100, 'r-', linewidth=1.5)
        ax2.set_title('Packet Loss over Time')
        ax2.set_ylabel('Packet Loss (%)')
        ax2.grid(True)

        # Highlight time slots
        for slot in range(1, self.total_slots):
            slot_time = slot * self.time_slot_duration
            ax2.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.5)

        # Plot energy consumption
        ax3.plot(self.time_points, self.energy_consumption_series * 1000, 'g-', linewidth=1.5)
        ax3.set_title('Energy Consumption per Byte over Time')
        ax3.set_ylabel('Energy (mJ/byte)')
        ax3.grid(True)

        # Plot bandwidth change rate
        time_points_rate = self.time_points[:-1] + (self.time_points[1] - self.time_points[0]) / 2
        ax4.plot(time_points_rate, self.bandwidth_change_rates, 'c-', linewidth=1.5)
        ax4.set_title('Bandwidth Change Rate over Time')
        ax4.set_ylabel('Change Rate (Mbps/min)')
        ax4.grid(True)

        # Plot packet loss change rate
        ax5.plot(time_points_rate, self.packet_loss_change_rates * 100, 'm-', linewidth=1.5)
        ax5.set_title('Packet Loss Change Rate over Time')
        ax5.set_xlabel('Time (minutes)')
        ax5.set_ylabel('Change Rate (%/min)')
        ax5.grid(True)

        plt.tight_layout()
        return fig

if __name__ == '__main__':
    # Path 1 parameters (more stable path)
    bw1_avg = [12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 11]
    pl1_avg = [0.03, 0.04, 0.05, 0.04, 0.03, 0.02, 0.03, 0.04, 0.05, 0.04, 0.03, 0.04]
    fluct1_level = [1, 2, 2, 1, 1, 2, 1, 2, 2, 1, 1, 2]  # Lower fluctuation

    # Path 2 parameters (less stable path)
    bw2_avg = [8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 6, 7]
    pl2_avg = [0.08, 0.09, 0.10, 0.12, 0.10, 0.09, 0.08, 0.07, 0.08, 0.09, 0.10, 0.09]
    fluct2_level = [3, 4, 5, 4, 3, 4, 3, 3, 4, 5, 4, 3]  # Higher fluctuation

    # Create paths with different fluctuation patterns
    path1 = Path(
        avg_bandwidth_series=bw1_avg,
        avg_packet_loss_series=pl1_avg,
        fluctuation_level_series=fluct1_level,
        time_slot_duration=5,  # 5 minutes per slot
        points_per_slot=300,  # 300 points per slot (1 point per second)
        fluctuation_pattern='sine'  # Use sine wave pattern for path 1
    )

    path2 = Path(
        avg_bandwidth_series=bw2_avg,
        avg_packet_loss_series=pl2_avg,
        fluctuation_level_series=fluct2_level,
        time_slot_duration=5,  # 5 minutes per slot
        points_per_slot=300,  # 300 points per slot (1 point per second)
        fluctuation_pattern='mixed'  # Use mixed pattern for path 2
    )
    path1.plot()