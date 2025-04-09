import numpy as np
from typing import List, Dict, Tuple, Optional
import random

# Import the Paths class from paths module
from paths import Paths


class NetworkCodingDecisionMaker:
    def __init__(self, max_coding_packets=128, max_redundancy=32):
        """
        Initialize the network coding decision maker.

        Args:
            max_coding_packets: Maximum number of network coding packets (default: 128)
            max_redundancy: Maximum number of redundancy packets (default: 32)
        """
        self.max_coding_packets = max_coding_packets
        self.max_redundancy = max_redundancy
        self.network_view = {}  # Store current network view
        self.last_decision = None  # Store last decision result
        self.last_probe_time = {}  # Store last probe time for each path

    def observe_packet_loss(self, maritime_paths: Paths, time_point: float) -> Dict[str, float]:
        """
        Observe packet loss rates on each maritime path at the given time point.

        Args:
            maritime_paths: Paths object containing maritime communication paths
            time_point: Time point in minutes to observe packet loss

        Returns:
            Mapping from path ID to packet loss rate
        """
        packet_loss_rates = {}

        for path_id in maritime_paths.list_paths():
            path = maritime_paths.get_path(path_id)

            # Convert time point to index
            time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

            # Get packet loss rate at this time point
            packet_loss_rate = path.packet_loss_series[time_idx]
            packet_loss_rates[path_id] = packet_loss_rate

            # Update network view
            if path_id not in self.network_view:
                self.network_view[path_id] = {}

            self.network_view[path_id]['packet_loss_rate'] = packet_loss_rate
            self.network_view[path_id]['bandwidth'] = path.bandwidth_series[time_idx]

        return packet_loss_rates

    def calculate_coding_parameters(self, avg_packet_loss_rate: float) -> Tuple[int, int, int]:
        """
        Calculate network coding parameters N, M, R based on packet loss rate.

        Args:
            avg_packet_loss_rate: Average packet loss rate across all paths

        Returns:
            (N, M, R) - Data packet group count, total packet count, redundancy count
        """
        # Calculate N - Data packet group count, adapted to packet loss rate
        N = max(1, min(self.max_coding_packets, int(32 * (1 + avg_packet_loss_rate))))

        # Calculate R - Redundancy packet count, increases linearly with packet loss rate
        R = max(1, min(self.max_redundancy, int(self.max_redundancy * avg_packet_loss_rate * 2)))

        # Calculate M - Total packet count = N + R
        M = N + R

        return N, M, R

    def allocate_bandwidth(self, maritime_paths: Paths, time_point: float, M: int) -> Dict[str, float]:
        """
        Allocate transmission proportions on each maritime link.

        Args:
            maritime_paths: Paths object containing maritime communication paths
            time_point: Time point in minutes to observe path conditions
            M: Total packet count

        Returns:
            Mapping from path ID to transmission proportion, with sum = M
        """
        # Allocate based on available bandwidth and packet loss rate
        total_quality = 0
        path_qualities = {}

        for path_id in maritime_paths.list_paths():
            path = maritime_paths.get_path(path_id)

            # Convert time point to index
            time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

            # Get bandwidth and packet loss rate at this time point
            bandwidth = path.bandwidth_series[time_idx]
            packet_loss_rate = path.packet_loss_series[time_idx]

            # Path quality = bandwidth * (1 - packet_loss_rate)
            # This favors paths with high bandwidth and low packet loss
            quality = bandwidth * (1 - packet_loss_rate)
            path_qualities[path_id] = quality
            total_quality += quality

        # Allocate proportions
        allocations = {}
        remaining = M

        if total_quality > 0:
            # Allocate proportionally to quality
            for path_id, quality in path_qualities.items():
                # Initial allocation, rounded down
                allocation = int((quality / total_quality) * M)
                allocations[path_id] = allocation
                remaining -= allocation

        # Handle remaining allocation to ensure sum = M
        path_ids = list(path_qualities.keys())
        i = 0
        while remaining > 0 and path_ids:
            path_id = path_ids[i % len(path_ids)]
            allocations[path_id] += 1
            remaining -= 1
            i += 1

        return allocations

    def should_probe_path(self, path_id: str, time_point: float, base_interval: float) -> bool:
        """
        Determine if a specific path should be probed at this time point.

        Args:
            path_id: ID of the path to consider for probing
            time_point: Current time point in minutes
            base_interval: Base interval for probing in minutes

        Returns:
            Boolean indicating whether the path should be probed
        """
        # If this is the first time seeing this path, initialize last probe time
        if path_id not in self.last_probe_time:
            self.last_probe_time[path_id] = -base_interval  # Ensure it probes on first encounter

        # Calculate time since last probe
        time_since_last_probe = time_point - self.last_probe_time[path_id]

        # Decide whether to probe based on time interval
        # Add some randomness to avoid all paths being probed at exactly the same time
        interval_multiplier = 0.8 + 0.4 * random.random()  # Between 0.8 and 1.2
        adjusted_interval = base_interval * interval_multiplier

        if time_since_last_probe >= adjusted_interval:
            self.last_probe_time[path_id] = time_point
            return True

        return False

    def probe_network(self, maritime_paths: Paths, time_point: float, base_probe_interval: float) -> Dict[str, bool]:
        """
        Decide whether to perform network probing for each path and update network view.

        Args:
            maritime_paths: Paths object containing maritime communication paths
            time_point: Time point in minutes to perform probing
            base_probe_interval: Base interval for probing in minutes

        Returns:
            Dictionary mapping path IDs to boolean indicating if that path was probed
        """
        probed_paths = {}

        for path_id in maritime_paths.list_paths():
            # Determine if this path should be probed
            should_probe = self.should_probe_path(path_id, time_point, base_probe_interval)

            if should_probe:
                # Update network view for this path
                path = maritime_paths.get_path(path_id)

                # Convert time point to index
                time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

                if path_id not in self.network_view:
                    self.network_view[path_id] = {}

                # Update network view with current path conditions
                self.network_view[path_id]['packet_loss_rate'] = path.packet_loss_series[time_idx]
                self.network_view[path_id]['bandwidth'] = path.bandwidth_series[time_idx]
                self.network_view[path_id]['energy_consumption'] = path.energy_consumption_series[time_idx]

            probed_paths[path_id] = should_probe

        return probed_paths

    def calculate_decision_value(self) -> float:
        """
        Calculate network decision value to evaluate current decision effectiveness.

        Returns:
            Decision value score
        """
        if not self.last_decision:
            return 0

        N, M, R = self.last_decision['coding_parameters']
        allocations = self.last_decision['allocations']

        # Coding rate
        coding_rate = R / N if N > 0 else 0

        # Evaluate allocation balance
        if allocations:
            allocation_values = list(allocations.values())
            allocation_std = np.std(allocation_values) if len(allocation_values) > 1 else 0
            # Normalize standard deviation
            normalized_std = allocation_std / M if M > 0 else 0

            # Decision value = Coding rate adaptivity - Allocation imbalance
            # Weights can be adjusted based on actual requirements
            decision_value = coding_rate - normalized_std
            return decision_value

        return 0

    def make_decision(self, maritime_paths: Paths, time_point: float, probe_interval: float = 5.0) -> Dict:
        """
        Main decision function integrating all decision logic.

        Args:
            maritime_paths: Paths object containing maritime communication paths
            time_point: Time point in minutes to make decision
            probe_interval: Base interval for network probing in minutes

        Returns:
            Decision result including coding parameters and allocation proportions
        """
        # Step 1: Decide whether to perform network probing for each path
        probed_paths = self.probe_network(maritime_paths, time_point, probe_interval)

        # Step 2: Get packet loss rates
        packet_loss_rates = self.observe_packet_loss(maritime_paths, time_point)
        avg_packet_loss_rate = sum(packet_loss_rates.values()) / len(packet_loss_rates) if packet_loss_rates else 0

        # Step 3: Calculate network coding parameters
        N, M, R = self.calculate_coding_parameters(avg_packet_loss_rate)

        # Step 4: Allocate bandwidth
        allocations = self.allocate_bandwidth(maritime_paths, time_point, M)

        # Save decision result
        self.last_decision = {
            'coding_parameters': (N, M, R),
            'allocations': allocations,
            'probed_paths': probed_paths,
            'packet_loss_rates': packet_loss_rates,
            'time_point': time_point
        }

        # Step 5: Calculate decision value
        decision_value = self.calculate_decision_value()
        self.last_decision['decision_value'] = decision_value

        return self.last_decision
