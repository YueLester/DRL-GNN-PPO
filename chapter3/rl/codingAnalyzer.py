import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import copy
from path import Path
from paths import Paths


class MaritimeNetworkOptimizer:
    """
    A class for optimizing network coding parameters and transmission allocation
    over maritime communication paths with time-varying conditions.
    """

    def __init__(self, maritime_paths: Paths, decision_interval: float = 10.0):
        """
        Initialize the maritime network optimizer.

        Args:
            maritime_paths: Paths object containing the maritime communication paths
            decision_interval: Interval between transmission decisions (seconds)
        """
        self.paths = maritime_paths
        self.path_ids = ["Path1", "Path2"]  # Custom path names - reduced to 2 paths
        self.num_paths = len(self.path_ids)

        # Map custom path names to original path IDs
        original_path_ids = maritime_paths.list_paths()
        self.path_mapping = {self.path_ids[i]: original_path_ids[i] for i in
                             range(min(len(self.path_ids), len(original_path_ids)))}

        # Network coding parameters
        self.max_data_packets = 128
        self.max_redundancy = 32

        # Decision interval (seconds)
        self.decision_interval = decision_interval

        # Simulation parameters
        self.current_time = 0.0  # Current time in minutes

        # Last channel observation time for each path
        self.last_observation_times = {path_id: 0.0 for path_id in self.path_ids}

        # Record of probe decisions
        self.probe_decisions = {path_id: [] for path_id in self.path_ids}

        # History for visualization
        self.decision_times = []
        self.bandwidth_history = {path_id: [] for path_id in self.path_ids}
        self.packet_loss_history = {path_id: [] for path_id in self.path_ids}
        self.N_history = []
        self.R_history = []
        self.allocation_history = {path_id: [] for path_id in self.path_ids}

        # Initialize visualizations
        self.fig_bandwidth = None
        self.fig_packet_loss = None

    def get_path_state(self, path_id: str, time_min: float) -> Dict:
        """
        Get the state of a specific path at a specific time.

        Args:
            path_id: Path identifier
            time_min: Time in minutes

        Returns:
            Dictionary with path state information
        """
        original_path_id = self.path_mapping.get(path_id)
        if not original_path_id:
            raise ValueError(f"Path {path_id} not found in path mapping")

        path = self.paths.get_path(original_path_id)
        state = path.get_state_at_time(time_min)
        return state

    def should_probe(self, path_id: str, current_time: float) -> bool:
        """
        Decide whether to probe a specific path based on time since last observation
        and estimated channel volatility.

        Args:
            path_id: Path identifier
            current_time: Current time in minutes

        Returns:
            Boolean indicating whether to probe
        """
        # Get time since last observation
        time_since_last_observation = current_time - self.last_observation_times[path_id]

        # Get current path state
        state = self.get_path_state(path_id, current_time)

        # Define probe threshold based on packet loss and bandwidth volatility
        # Higher packet loss or greater bandwidth volatility -> probe more frequently
        packet_loss = state['packet_loss']

        # Base probe interval (minutes)
        base_interval = 2.0  # 2 minutes

        # Adjust interval based on packet loss
        # High packet loss -> probe more frequently
        loss_factor = 1.0 + 3.0 * (1.0 - packet_loss)  # 1.0 to 4.0
        probe_interval = base_interval * loss_factor

        # Get original path to check fluctuation level
        original_path_id = self.path_mapping.get(path_id)
        path = self.paths.get_path(original_path_id)

        # Find current time slot
        slot_idx = int(current_time / path.time_slot_duration)
        if slot_idx >= len(path.fluctuation_level_series):
            slot_idx = len(path.fluctuation_level_series) - 1

        # Get current fluctuation level (1-5)
        if hasattr(path, 'fluctuation_level_series'):
            fluctuation_level = path.fluctuation_level_series[slot_idx]
            # Adjust interval based on fluctuation level
            # Higher fluctuation -> probe more frequently
            fluctuation_factor = 1.0 / max(1.0, fluctuation_level / 3.0)
            probe_interval *= fluctuation_factor

        # Decision: probe if enough time has passed
        return time_since_last_observation >= probe_interval

    def optimize_network_coding(self, current_time: float) -> Dict:
        """
        Optimize network coding parameters and transmission allocation based on
        current path conditions.

        Args:
            current_time: Current time in minutes

        Returns:
            Dictionary with optimization results
        """
        # Decide whether to probe each path
        probe_decisions = {}
        for path_id in self.path_ids:
            should_probe_path = self.should_probe(path_id, current_time)
            probe_decisions[path_id] = should_probe_path

            if should_probe_path:
                self.last_observation_times[path_id] = current_time
                self.probe_decisions[path_id].append(current_time)

        # Get current network states
        network_states = {}
        for path_id in self.path_ids:
            network_states[path_id] = self.get_path_state(path_id, current_time)

        # Calculate weighted average packet loss
        total_bandwidth = sum(state['bandwidth'] for state in network_states.values())
        weighted_packet_loss = 0.0

        for path_id, state in network_states.items():
            bandwidth_weight = state['bandwidth'] / total_bandwidth if total_bandwidth > 0 else 1.0 / len(
                network_states)
            weighted_packet_loss += state['packet_loss'] * bandwidth_weight

        # Optimize N (data packets) based on packet loss
        # Lower packet loss -> more data packets
        if weighted_packet_loss < 0.05:
            N = 128
        elif weighted_packet_loss < 0.10:
            N = 64
        elif weighted_packet_loss < 0.15:
            N = 32
        else:
            N = 16

        # Optimize R (redundancy) based on packet loss
        # Higher packet loss -> more redundancy
        redundancy_ratio = min(1.0, max(0.1, weighted_packet_loss * 5))
        R = min(self.max_redundancy, int(N * redundancy_ratio))
        R = max(1, R)  # Ensure at least 1 redundancy packet

        # Calculate total packets (M)
        M = N + R

        # Optimize allocation weights based on effective bandwidth
        effective_bandwidths = {}
        for path_id, state in network_states.items():
            effective_bandwidths[path_id] = state['bandwidth'] * (1.0 - state['packet_loss'])

        total_effective_bandwidth = sum(effective_bandwidths.values())

        allocation_weights = {}
        for path_id in self.path_ids:
            if total_effective_bandwidth > 0:
                allocation_weights[path_id] = effective_bandwidths[path_id] / total_effective_bandwidth
            else:
                allocation_weights[path_id] = 1.0 / len(self.path_ids)

        # Calculate packet allocation for each path
        packet_allocation = {}
        for path_id, weight in allocation_weights.items():
            packet_allocation[path_id] = int(round(M * weight))

        # Adjust allocation to ensure sum equals M
        total_allocated = sum(packet_allocation.values())
        if total_allocated != M:
            # Sort paths by allocation (descending)
            sorted_paths = sorted(self.path_ids, key=lambda p: packet_allocation[p], reverse=True)

            diff = M - total_allocated
            for path_id in sorted_paths:
                if diff > 0:
                    packet_allocation[path_id] += 1
                    diff -= 1
                elif diff < 0:
                    if packet_allocation[path_id] > 0:
                        packet_allocation[path_id] -= 1
                        diff += 1

                if diff == 0:
                    break

        # Store history for visualization
        self.decision_times.append(current_time)
        self.N_history.append(N)
        self.R_history.append(R)

        for path_id in self.path_ids:
            self.bandwidth_history[path_id].append(network_states[path_id]['bandwidth'])
            self.packet_loss_history[path_id].append(network_states[path_id]['packet_loss'] * 100)  # To percentage
            self.allocation_history[path_id].append(packet_allocation[path_id])

        return {
            "time": current_time,
            "probe_decisions": probe_decisions,
            "network_states": network_states,
            "N": N,
            "R": R,
            "M": M,
            "redundancy_ratio": R / N,
            "allocation_weights": allocation_weights,
            "packet_allocation": packet_allocation
        }

    def run_simulation(self, duration_min: float) -> List[Dict]:
        """
        Run a simulation for the specified duration.

        Args:
            duration_min: Simulation duration in minutes

        Returns:
            List of optimization results at each decision point
        """
        # Convert decision interval from seconds to minutes
        decision_interval_min = self.decision_interval / 60.0

        # Reset simulation state
        self.current_time = 0.0
        self.last_observation_times = {path_id: 0.0 for path_id in self.path_ids}
        self.probe_decisions = {path_id: [] for path_id in self.path_ids}

        # Reset history
        self.decision_times = []
        self.bandwidth_history = {path_id: [] for path_id in self.path_ids}
        self.packet_loss_history = {path_id: [] for path_id in self.path_ids}
        self.N_history = []
        self.R_history = []
        self.allocation_history = {path_id: [] for path_id in self.path_ids}

        # Store optimization results
        results = []

        # Force initial observation of all paths
        for path_id in self.path_ids:
            self.probe_decisions[path_id].append(0.0)
            self.last_observation_times[path_id] = 0.0

        # Run simulation
        while self.current_time <= duration_min:
            # Optimize network coding at current time
            result = self.optimize_network_coding(self.current_time)
            results.append(result)

            # Move to next decision point
            self.current_time += decision_interval_min

        return results

    def plot_bandwidth_with_probes_and_allocation(self, figsize=(12, 8)) -> plt.Figure:
        """
        Plot bandwidth for each path with probe markers and packet allocation.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Matplotlib Figure object
        """
        if not self.decision_times:
            raise ValueError("No simulation data available")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

        # Plot bandwidth
        for path_id in self.path_ids:
            # Get original path ID for color
            original_path_id = self.path_mapping.get(path_id)
            color = self.paths.path_colors.get(original_path_id, 'blue')

            # Plot bandwidth
            line, = ax1.plot(self.decision_times, self.bandwidth_history[path_id], '-',
                             label=f'{path_id}', color=color, linewidth=2)

            # Add probe markers
            if self.probe_decisions[path_id]:
                # Get bandwidth values at probe times
                probe_bw_values = []
                for time in self.probe_decisions[path_id]:
                    # Find closest time point
                    idx = min(range(len(self.decision_times)), key=lambda i: abs(self.decision_times[i] - time))
                    if idx < len(self.bandwidth_history[path_id]):
                        probe_bw_values.append(self.bandwidth_history[path_id][idx])
                    else:
                        # Use the last value if index is out of range
                        probe_bw_values.append(self.bandwidth_history[path_id][-1])

                ax1.plot(self.probe_decisions[path_id], probe_bw_values, 'o',
                         color=color, markersize=6)

        ax1.set_title('Path Bandwidth with Network Probes')
        ax1.set_ylabel('Bandwidth (Mbps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Plot packet allocation
        for path_id in self.path_ids:
            original_path_id = self.path_mapping.get(path_id)
            color = self.paths.path_colors.get(original_path_id, 'blue')
            ax2.plot(self.decision_times, self.allocation_history[path_id], '-',
                     label=f'{path_id} Allocation', color=color, linewidth=2)

        ax2.set_title('Packet Allocation to Paths')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Packets Allocated')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        self.fig_bandwidth = fig
        return fig

    def plot_packet_loss_and_redundancy(self, figsize=(12, 8)) -> plt.Figure:
        """
        Plot packet loss for each path and network coding parameters.

        Args:
            figsize: Figure size (width, height) in inches

        Returns:
            Matplotlib Figure object
        """
        if not self.decision_times:
            raise ValueError("No simulation data available")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题
        plt.rcParams['font.family'] = 'SimHei'
        plt.rcParams['axes.unicode_minus'] = False  #

        # Plot packet loss
        for path_id in self.path_ids:
            original_path_id = self.path_mapping.get(path_id)
            color = self.paths.path_colors.get(original_path_id, 'blue')

            ax1.plot(self.decision_times, self.packet_loss_history[path_id], '-',
                     label=f'{path_id}', color=color, linewidth=2)

            # Add probe markers
            if self.probe_decisions[path_id]:
                # Get packet loss values at probe times
                probe_pl_values = []
                for time in self.probe_decisions[path_id]:
                    # Find closest time point
                    idx = min(range(len(self.decision_times)), key=lambda i: abs(self.decision_times[i] - time))
                    if idx < len(self.packet_loss_history[path_id]):
                        probe_pl_values.append(self.packet_loss_history[path_id][idx])
                    else:
                        # Use the last value if index is out of range
                        probe_pl_values.append(self.packet_loss_history[path_id][-1])

                ax1.plot(self.probe_decisions[path_id], probe_pl_values, 'o',
                         color=color, markersize=6)

        ax1.set_title('网络丢包率')
        ax1.set_ylabel('Packet Loss (%)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        # Plot redundancy ratio (R/N)
        redundancy_ratio = [r / n for r, n in zip(self.R_history, self.N_history)]
        ax2.plot(self.decision_times, redundancy_ratio, 'r-',
                 label='Redundancy Ratio (R/N)', linewidth=2)

        # Add N and R as text annotations
        ax2.set_title('Network Coding Parameters')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('R/N Ratio')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

        # Add N and R values as a second line with markers
        ax3 = ax2.twinx()
        ax3.plot(self.decision_times, self.N_history, 'b--', label='N (Data Packets)', linewidth=1.5)
        ax3.plot(self.decision_times, self.R_history, 'g--', label='R (Redundancy)', linewidth=1.5)
        ax3.set_ylabel('Packet Count')

        # Combine legends
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        self.fig_packet_loss = fig
        return fig


def run_simulation(duration_min=5.0, decision_interval_sec=10.0):
    """
    Run a simulation of the maritime network optimization.

    Args:
        duration_min: Simulation duration in minutes
        decision_interval_sec: Interval between optimization decisions in seconds

    Returns:
        MaritimeNetworkOptimizer object with simulation results
    """
    # Create paths container with maritime paths
    maritime_paths = Paths()

    # Create two maritime paths with different characteristics
    avg_bandwidth_1 = [12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12]
    avg_packet_loss_1 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    fluctuation_level_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]

    path1 = Path(
        avg_bandwidth_series=avg_bandwidth_1,
        avg_packet_loss_series=avg_packet_loss_1,
        fluctuation_level_series=fluctuation_level_1,
        time_slot_duration=5,
        points_per_slot=300
    )

    avg_bandwidth_2 = [8, 9, 10, 11, 12, 12, 11, 10, 9, 8, 7, 6]
    avg_packet_loss_2 = [0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    fluctuation_level_2 = [3, 3, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4]

    path2 = Path(
        avg_bandwidth_series=avg_bandwidth_2,
        avg_packet_loss_series=avg_packet_loss_2,
        fluctuation_level_series=fluctuation_level_2,
        time_slot_duration=5,
        points_per_slot=300
    )

    # Add maritime paths to the container
    maritime_paths.add_path("Satellite Link", path1, color="#1f77b4")
    maritime_paths.add_path("Coastal Radio", path2, color="#ff7f0e")

    # Create optimizer
    optimizer = MaritimeNetworkOptimizer(maritime_paths, decision_interval=decision_interval_sec)

    # Run simulation
    print(f"Running simulation for {duration_min} minutes with decision interval of {decision_interval_sec} seconds...")
    results = optimizer.run_simulation(duration_min)

    # Create visualizations
    print("Generating visualizations...")
    optimizer.plot_bandwidth_with_probes_and_allocation()
    optimizer.plot_packet_loss_and_redundancy()

    return optimizer


if __name__ == "__main__":
    # Run simulation for 5 minutes with decisions every 10 seconds
    optimizer = run_simulation(duration_min=5.0, decision_interval_sec=10.0)

    # Display plots
    plt.show()