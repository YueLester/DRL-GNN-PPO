import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import random
import copy
from path import Path
from paths import Paths


class NetworkCodingOptimizer:
    """
    A class that optimizes network coding parameters (N, M, R) and allocation weights
    for maritime communication paths based on observed network conditions.
    """

    def __init__(self, maritime_paths: Paths, probe_interval: float = 5.0):
        """
        Initialize the network coding optimizer.

        Args:
            maritime_paths: Paths object containing maritime communication links
            probe_interval: Time interval between network probes (minutes)
        """
        self.maritime_paths = maritime_paths
        self.path_ids = maritime_paths.list_paths()
        self.num_paths = len(self.path_ids)
        self.probe_interval = probe_interval

        # Track last probe times for each path
        self.last_probe_times = {path_id: 0.0 for path_id in self.path_ids}
        self.probe_decisions = {path_id: [] for path_id in self.path_ids}

        # Network coding constraints
        self.max_data_packets = 128  # N: maximum number of data packets
        self.max_redundancy = 32  # R: maximum number of redundancy packets

        # Current network coding parameters
        self.N = 32  # Current number of data packets
        self.R = 8  # Current number of redundancy packets
        self.M = self.N + self.R  # Total number of packets

        # Current allocation weights
        self.weights = {path_id: 1.0 / self.num_paths for path_id in self.path_ids}

        # Track historical decisions for visualization
        self.coding_history = []
        self.weight_history = {path_id: [] for path_id in self.path_ids}
        self.time_points = []

    def get_current_network_state(self, current_time: float) -> Dict:
        """
        Get the current state of all network paths.

        Args:
            current_time: Current simulation time in minutes

        Returns:
            Dictionary with network state for each path
        """
        network_state = {}

        for path_id in self.path_ids:
            path = self.maritime_paths.get_path(path_id)
            state = path.get_state_at_time(current_time)
            network_state[path_id] = state

        return network_state

    def decide_probing(self, current_time: float, force_probe: bool = False) -> Dict[str, bool]:
        """
        Decide whether to probe each network path based on time interval
        or significant changes in network conditions.

        Args:
            current_time: Current simulation time in minutes
            force_probe: Force probing regardless of interval

        Returns:
            Dictionary with probing decisions for each path
        """
        probe_decisions = {}

        for path_id in self.path_ids:
            # Probe if enough time has passed since last probe
            time_since_last_probe = current_time - self.last_probe_times[path_id]

            # Default decision: probe if enough time has passed
            should_probe = (time_since_last_probe >= self.probe_interval) or force_probe

            # If probe interval is close and we detect high packet loss, probe more aggressively
            if time_since_last_probe >= 0.7 * self.probe_interval:
                path = self.maritime_paths.get_path(path_id)
                state = path.get_state_at_time(current_time)
                if state['packet_loss'] > 0.1:  # 10% packet loss threshold
                    should_probe = True

            probe_decisions[path_id] = should_probe

            # Update last probe time if probing
            if should_probe:
                self.last_probe_times[path_id] = current_time
                self.probe_decisions[path_id].append(current_time)

        return probe_decisions

    def optimize_coding_parameters(self, network_state: Dict) -> Tuple[int, int, int]:
        """
        Optimize network coding parameters (N, M, R) based on current network conditions.

        Args:
            network_state: Current state of all network paths

        Returns:
            Tuple of (N, M, R) values for network coding
        """
        # Calculate weighted average packet loss across all paths
        total_bandwidth = sum(state['bandwidth'] for state in network_state.values())

        weighted_packet_loss = 0.0
        for path_id, state in network_state.items():
            bandwidth_weight = state['bandwidth'] / total_bandwidth
            weighted_packet_loss += state['packet_loss'] * bandwidth_weight

        # Basic strategy:
        # 1. Higher packet loss -> more redundancy (R)
        # 2. Lower packet loss -> larger data blocks (N)

        # Adjust N based on packet loss (inversely proportional)
        # Mapping loss rate to N: 0% -> 128, 20%+ -> 16
        if weighted_packet_loss < 0.05:
            new_N = 128
        elif weighted_packet_loss < 0.10:
            new_N = 64
        elif weighted_packet_loss < 0.15:
            new_N = 32
        else:
            new_N = 16

        # Calculate redundancy rate based on packet loss
        # Higher loss -> higher redundancy
        # R/N ratio: minimum 0.1, maximum 1.0 (100% redundancy)
        redundancy_ratio = min(1.0, max(0.1, weighted_packet_loss * 5))  # Scale packet loss by 5
        new_R = min(self.max_redundancy, int(new_N * redundancy_ratio))

        # Ensure R is at least 1
        new_R = max(1, new_R)

        # Calculate M
        new_M = new_N + new_R

        return new_N, new_M, new_R

    def optimize_allocation_weights(self, network_state: Dict) -> Dict[str, float]:
        """
        Optimize allocation weights across paths based on bandwidth and packet loss.

        Args:
            network_state: Current state of all network paths

        Returns:
            Dictionary with optimized weights for each path
        """
        # Calculate effective bandwidth for each path: bandwidth * (1 - packet_loss)
        effective_bandwidths = {}
        for path_id, state in network_state.items():
            effective_bandwidths[path_id] = state['bandwidth'] * (1 - state['packet_loss'])

        # Total effective bandwidth
        total_effective_bandwidth = sum(effective_bandwidths.values())

        # Allocate weights proportional to effective bandwidth
        new_weights = {}
        for path_id in self.path_ids:
            if total_effective_bandwidth > 0:
                new_weights[path_id] = effective_bandwidths[path_id] / total_effective_bandwidth
            else:
                # Fallback to equal distribution if total effective bandwidth is zero
                new_weights[path_id] = 1.0 / len(self.path_ids)

        # Normalize weights to ensure they sum to 1
        weight_sum = sum(new_weights.values())
        for path_id in new_weights:
            new_weights[path_id] /= weight_sum

        return new_weights

    def calculate_packet_allocation(self) -> Dict[str, int]:
        """
        Calculate the actual number of packets to allocate to each path.

        Returns:
            Dictionary with number of packets allocated to each path
        """
        # Initial allocation based on weights
        packet_allocation = {path_id: self.weights[path_id] * self.M for path_id in self.path_ids}

        # Convert to integers with rounding
        rounded_allocation = {path_id: int(round(alloc)) for path_id, alloc in packet_allocation.items()}

        # Ensure the total exactly matches M by adjusting the largest allocation
        total_packets = sum(rounded_allocation.values())
        if total_packets != self.M:
            diff = self.M - total_packets
            # Sort paths by allocation (descending) to adjust largest first
            sorted_paths = sorted(self.path_ids, key=lambda x: rounded_allocation[x], reverse=True)

            # Add or subtract the difference from paths with largest allocations
            for path_id in sorted_paths:
                if diff > 0:
                    rounded_allocation[path_id] += 1
                    diff -= 1
                elif diff < 0:
                    if rounded_allocation[path_id] > 0:  # Ensure we don't go negative
                        rounded_allocation[path_id] -= 1
                        diff += 1

                if diff == 0:
                    break

        return rounded_allocation

    def update_network_view(self, current_time: float, force_update: bool = False) -> Dict:
        """
        Update network view and optimize parameters based on current conditions.

        Args:
            current_time: Current simulation time in minutes
            force_update: Force an update regardless of probing decisions

        Returns:
            Dictionary with updated network parameters and decisions
        """
        # Get current network state
        network_state = self.get_current_network_state(current_time)

        # Decide whether to probe each path
        probe_decisions = self.decide_probing(current_time, force_update)

        # Only update parameters if at least one path is probed
        update_needed = force_update or any(probe_decisions.values())

        if update_needed:
            # Optimize coding parameters
            self.N, self.M, self.R = self.optimize_coding_parameters(network_state)

            # Optimize allocation weights
            self.weights = self.optimize_allocation_weights(network_state)

            # Calculate packet allocation
            packet_allocation = self.calculate_packet_allocation()

            # Store history for visualization
            self.coding_history.append((self.N, self.M, self.R))
            for path_id in self.path_ids:
                self.weight_history[path_id].append(self.weights[path_id])
            self.time_points.append(current_time)

            return {
                "time": current_time,
                "probe_decisions": probe_decisions,
                "coding_parameters": {
                    "N": self.N,
                    "M": self.M,
                    "R": self.R
                },
                "allocation_weights": self.weights,
                "packet_allocation": packet_allocation,
                "updated": True
            }
        else:
            # Return current parameters without updating
            packet_allocation = self.calculate_packet_allocation()

            return {
                "time": current_time,
                "probe_decisions": probe_decisions,
                "coding_parameters": {
                    "N": self.N,
                    "M": self.M,
                    "R": self.R
                },
                "allocation_weights": self.weights,
                "packet_allocation": packet_allocation,
                "updated": False
            }

    def simulate_transmission(self, current_time: float, data_size_bytes: int) -> Dict:
        """
        Simulate a data transmission using current network coding parameters.

        Args:
            current_time: Current simulation time in minutes
            data_size_bytes: Size of the data to transmit

        Returns:
            Transmission results
        """
        # Update network view
        network_update = self.update_network_view(current_time)

        # Calculate packet allocation
        packet_allocation = network_update["packet_allocation"]

        # Calculate bytes per packet
        bytes_per_packet = data_size_bytes / self.N

        # Track overall transmission statistics
        total_packets_sent = 0
        total_packets_received = 0
        total_energy_consumed = 0.0

        # Simulate transmission on each path
        path_results = {}

        for path_id, packets in packet_allocation.items():
            if packets > 0:
                path = self.maritime_paths.get_path(path_id)

                # Calculate data volume for this path
                path_data_bytes = int(packets * bytes_per_packet)

                # Find time index
                idx = int(round(current_time / path.total_duration * (path.total_points - 1)))

                # Simulate transmission
                tx_result = path.transmit_bulk_data(path_data_bytes, idx)

                # Store results
                path_results[path_id] = tx_result

                # Update total statistics
                total_packets_sent += tx_result["packets_sent"]
                total_packets_received += tx_result["packets_received"]
                total_energy_consumed += tx_result["energy_consumed"]

        # Calculate transmission success
        # In network coding, we need at least N packets out of M to decode the data
        packets_needed = self.N
        transmission_success = total_packets_received >= packets_needed

        # Calculate received data percentage
        if transmission_success:
            received_percentage = 100.0
        else:
            # If we didn't receive enough packets to decode, calculate partial percentage
            received_percentage = (total_packets_received / packets_needed) * 100.0
            received_percentage = min(100.0, received_percentage)  # Cap at 100%

        return {
            "time": current_time,
            "data_size_bytes": data_size_bytes,
            "coding_parameters": {
                "N": self.N,
                "M": self.M,
                "R": self.R
            },
            "total_packets_sent": total_packets_sent,
            "total_packets_received": total_packets_received,
            "transmission_success": transmission_success,
            "received_percentage": received_percentage,
            "total_energy_consumed": total_energy_consumed,
            "path_results": path_results
        }

    def plot_coding_parameters_history(self, figsize=(12, 6)) -> plt.Figure:
        """
        Plot the history of network coding parameters over time.

        Args:
            figsize: Size of the figure (width, height) in inches

        Returns:
            Matplotlib Figure object
        """
        if not self.coding_history:
            raise ValueError("No coding history available to plot")

        # Extract parameters from history
        N_values = [params[0] for params in self.coding_history]
        M_values = [params[1] for params in self.coding_history]
        R_values = [params[2] for params in self.coding_history]

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(self.time_points, N_values, 'b-', label='N (Data Packets)', linewidth=2)
        ax.plot(self.time_points, R_values, 'r-', label='R (Redundancy)', linewidth=2)
        ax.plot(self.time_points, M_values, 'g--', label='M (Total Packets)', linewidth=1.5)

        ax.set_title('Network Coding Parameters Over Time')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Number of Packets')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_allocation_weights_history(self, figsize=(12, 6)) -> plt.Figure:
        """
        Plot the history of allocation weights over time.

        Args:
            figsize: Size of the figure (width, height) in inches

        Returns:
            Matplotlib Figure object
        """
        if not self.weight_history or not self.time_points:
            raise ValueError("No weight history available to plot")

        fig, ax = plt.subplots(figsize=figsize)

        for path_id in self.path_ids:
            path_weights = self.weight_history[path_id]
            ax.plot(self.time_points, path_weights, '-',
                    label=f'{path_id} Weight',
                    color=self.maritime_paths.path_colors[path_id],
                    linewidth=2)

        ax.set_title('Path Allocation Weights Over Time')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Allocation Weight')
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_network_metrics_with_probes(self, figsize=(15, 12)) -> plt.Figure:
        """
        Plot network metrics with probe decisions marked.

        Args:
            figsize: Size of the figure (width, height) in inches

        Returns:
            Matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot bandwidth for each path
        for path_id in self.path_ids:
            path = self.maritime_paths.get_path(path_id)
            ax1.plot(path.time_points, path.bandwidth_series,
                     color=self.maritime_paths.path_colors[path_id],
                     linestyle=self.maritime_paths.path_styles[path_id],
                     linewidth=1.5,
                     label=f"{path_id}")

            # Mark probe points
            probe_times = self.probe_decisions[path_id]
            if probe_times:
                # Get bandwidth values at probe times
                probe_bw_values = [path.get_bandwidth_at_time(t) for t in probe_times]
                ax1.plot(probe_times, probe_bw_values, 'o',
                         color=self.maritime_paths.path_colors[path_id],
                         markersize=6)

        ax1.set_title('Bandwidth with Network Probes')
        ax1.set_ylabel('Bandwidth (Mbps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Plot packet loss for each path
        for path_id in self.path_ids:
            path = self.maritime_paths.get_path(path_id)
            ax2.plot(path.time_points, path.packet_loss_series * 100,  # Convert to percentage
                     color=self.maritime_paths.path_colors[path_id],
                     linestyle=self.maritime_paths.path_styles[path_id],
                     linewidth=1.5,
                     label=f"{path_id}")

            # Mark probe points
            probe_times = self.probe_decisions[path_id]
            if probe_times:
                # Get packet loss values at probe times
                probe_pl_values = [path.get_packet_loss_at_time(t) * 100 for t in probe_times]
                ax2.plot(probe_times, probe_pl_values, 'o',
                         color=self.maritime_paths.path_colors[path_id],
                         markersize=6)

        ax2.set_title('Packet Loss with Network Probes')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Packet Loss (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        plt.tight_layout()
        return fig

    def run_simulation(self, total_duration: float, step_size: float = 1.0,
                       data_size_bytes: int = 100000) -> Dict:
        """
        Run a complete simulation over the specified duration.

        Args:
            total_duration: Total simulation duration in minutes
            step_size: Time step size in minutes
            data_size_bytes: Size of data to transmit at each step

        Returns:
            Dictionary with simulation results
        """
        # Results storage
        transmission_results = []
        network_updates = []
        current_time = 0.0

        # Force initial update
        initial_update = self.update_network_view(current_time, force_update=True)
        network_updates.append(initial_update)

        # Simulation loop
        while current_time <= total_duration:
            # Simulate transmission
            tx_result = self.simulate_transmission(current_time, data_size_bytes)
            transmission_results.append(tx_result)

            # Get network update (if any)
            network_update = self.update_network_view(current_time)
            if network_update["updated"]:
                network_updates.append(network_update)

            # Increment time
            current_time += step_size

        # Generate summary plots
        coding_params_fig = self.plot_coding_parameters_history()
        weights_fig = self.plot_allocation_weights_history()
        network_metrics_fig = self.plot_network_metrics_with_probes()

        return {
            "transmission_results": transmission_results,
            "network_updates": network_updates,
            "figures": {
                "coding_params": coding_params_fig,
                "weights": weights_fig,
                "network_metrics": network_metrics_fig
            }
        }


# Example usage
if __name__ == "__main__":
    # Create paths container with the maritime paths from the original example
    maritime_paths = Paths()

    # Create first maritime path (Satellite Link) with 12 time slots (1 hour total)
    avg_bandwidth_1 = [12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12]
    avg_packet_loss_1 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    fluctuation_level_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]

    satellite_path = Path(
        avg_bandwidth_series=avg_bandwidth_1,
        avg_packet_loss_series=avg_packet_loss_1,
        fluctuation_level_series=fluctuation_level_1,
        time_slot_duration=5,  # 5 minutes per slot
        points_per_slot=300  # 300 points per slot (1 point per second)
    )

    # Create second maritime path (Coastal Radio) with same time structure but different characteristics
    avg_bandwidth_2 = [8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6]
    avg_packet_loss_2 = [0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    fluctuation_level_2 = [3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4]

    coastal_radio_path = Path(
        avg_bandwidth_series=avg_bandwidth_2,
        avg_packet_loss_series=avg_packet_loss_2,
        fluctuation_level_series=fluctuation_level_2,
        time_slot_duration=5,
        points_per_slot=300
    )

    # Create third maritime path (HF Link) with more volatile characteristics
    avg_bandwidth_3 = [15, 8, 12, 5, 10, 15, 8, 12, 5, 10, 15, 8]
    avg_packet_loss_3 = [0.02, 0.09, 0.05, 0.12, 0.06, 0.02, 0.09, 0.05, 0.12, 0.06, 0.02, 0.09]
    fluctuation_level_3 = [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4]

    hf_link_path = Path(
        avg_bandwidth_series=avg_bandwidth_3,
        avg_packet_loss_series=avg_packet_loss_3,
        fluctuation_level_series=fluctuation_level_3,
        time_slot_duration=5,
        points_per_slot=300
    )

    # Add maritime paths to the container
    maritime_paths.add_path("Satellite Link", satellite_path, color="#1f77b4")
    maritime_paths.add_path("Coastal Radio", coastal_radio_path, color="#ff7f0e")
    maritime_paths.add_path("HF Link", hf_link_path, color="#2ca02c")

    # Create network coding optimizer
    optimizer = NetworkCodingOptimizer(maritime_paths, probe_interval=5.0)

    # Run simulation for 60 minutes
    simulation_results = optimizer.run_simulation(total_duration=60.0, step_size=1.0, data_size_bytes=100000)

    # Display figures
    plt.show()