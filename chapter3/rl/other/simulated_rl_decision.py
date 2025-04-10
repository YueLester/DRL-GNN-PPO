import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional
import random
import math

# Import the Paths class from paths module
from paths import Paths


class SimulatedRLNetworkCodingDecisionMaker:
    """
    A network coding decision maker that simulates reinforcement learning behavior
    using heuristic approaches for maritime networks.
    """

    def __init__(self, max_coding_packets=64, max_redundancy=24):
        """
        Initialize the simulated RL network coding decision maker.

        Args:
            max_coding_packets: Maximum number of network coding packets (default: 64)
            max_redundancy: Maximum number of redundancy packets (default: 24)
        """
        self.max_coding_packets = max_coding_packets
        self.max_redundancy = max_redundancy
        self.network_view = {}  # Store current network view
        self.last_decision = None  # Store last decision result
        self.last_probe_time = {}  # Store last probe time for each path
        self.exploration_probability = 0.2  # Exploration probability
        self.resource_block_size = 1000  # Resource block size in bytes
        self.packet_size = 1500  # Packet size in bytes

        # Simulation of learning progress parameters
        self.learning_progress = 0.0  # 0 to 1 - simulates learning over time
        self.experience_counter = 0  # Counter for "experience" gathered
        self.decision_history = []  # Track decision history
        self.reward_history = []  # Track reward history
        self.cumulative_reward = 0  # Track cumulative reward

        # Weather and channel parameters
        self.weather_factor = 1.0  # Weather normalization factor
        self.channel_erasure_probs = [0.1, 0.3, 0.5, 0.7]  # Predefined channel erasure probabilities

        # Decision adaptation parameters
        self.adaptation_rate = 0.1  # How quickly to adapt to changes
        self.communication_range = 250  # Communication range in meters

    def observe_network_state(self, maritime_paths: Paths, time_point: float) -> Dict:
        """
        Observe the current state of all maritime paths.

        Args:
            maritime_paths: Paths object containing maritime paths
            time_point: Current time point in minutes

        Returns:
            Dictionary containing network state information
        """
        network_state = {}

        for path_id in maritime_paths.list_paths():
            path = maritime_paths.get_path(path_id)

            # Convert time point to index
            time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

            # Get current path parameters
            bandwidth = path.bandwidth_series[time_idx]
            packet_loss = path.packet_loss_series[time_idx]
            energy_per_byte = path.energy_consumption_series[time_idx]

            # Calculate time since last probe
            time_since_probe = float('inf')
            if path_id in self.last_probe_time:
                time_since_probe = time_point - self.last_probe_time.get(path_id, 0)

            # Store path state
            if path_id not in self.network_view:
                self.network_view[path_id] = {}

            self.network_view[path_id]['bandwidth'] = bandwidth
            self.network_view[path_id]['packet_loss'] = packet_loss
            self.network_view[path_id]['energy_per_byte'] = energy_per_byte
            self.network_view[path_id]['time_since_probe'] = time_since_probe

            # Add state to network state dictionary
            network_state[path_id] = {
                'bandwidth': bandwidth,
                'packet_loss': packet_loss,
                'energy_per_byte': energy_per_byte,
                'time_since_probe': time_since_probe
            }

        return network_state

    def decide_probing(self, network_state: Dict, time_point: float) -> Dict[str, bool]:
        """
        Decide which paths to probe based on network state and simulated learning.

        Args:
            network_state: Current network state
            time_point: Current time point in minutes

        Returns:
            Dictionary mapping path IDs to probe decisions (True/False)
        """
        probe_decisions = {}

        for path_id, state in network_state.items():
            # Base probing strategy factors
            base_interval = 5.0  # Base probe interval

            # Adjust interval based on "learned" behavior
            if self.learning_progress > 0.3:
                # More adaptive probing as "learning" progresses
                packet_loss = state['packet_loss']
                bandwidth = state['bandwidth']

                # Probe more frequently for unstable paths (high packet loss or low bandwidth)
                stability_factor = 1.0 - (packet_loss * 0.5 + (1.0 - min(bandwidth, 15.0) / 15.0) * 0.5)
                adjusted_interval = base_interval * (0.5 + stability_factor)
            else:
                # Less sophisticated probing early in "learning"
                # Add random variation to simulate exploration
                adjusted_interval = base_interval * (0.8 + 0.4 * random.random())

            # Time-based decision
            time_since_probe = state['time_since_probe']
            should_probe = time_since_probe >= adjusted_interval

            # Add exploration randomness (simulate ε-greedy approach)
            if random.random() < self.exploration_probability * (1.0 - self.learning_progress):
                should_probe = random.random() < 0.3  # 30% chance to randomly probe

            probe_decisions[path_id] = should_probe

            # Update last probe time if probing
            if should_probe:
                self.last_probe_time[path_id] = time_point

        return probe_decisions

    def calculate_coding_parameters(self, network_state: Dict) -> Tuple[int, int, int]:
        """
        Calculate network coding parameters (N, M, R) based on network state
        and simulated learning experience.

        Args:
            network_state: Current network state

        Returns:
            Tuple (N, M, R) - data packets, total packets, redundancy packets
        """
        # Calculate average packet loss across all paths
        packet_losses = [state['packet_loss'] for state in network_state.values()]
        avg_packet_loss = sum(packet_losses) / len(packet_losses) if packet_losses else 0.1

        # Calculate packet loss variance - used for more sophisticated decisions
        packet_loss_variance = np.var(packet_losses) if len(packet_losses) > 1 else 0

        # Base calculation - simple heuristic approach
        if self.learning_progress < 0.3:
            # Simple linear relationship early in learning
            N = max(1, min(self.max_coding_packets, int(32 * (1 + avg_packet_loss))))
            R = max(1, min(self.max_redundancy, int(self.max_redundancy * avg_packet_loss * 2)))
        else:
            # More sophisticated approach after some "learning"
            # Use both average and variance in packet loss
            # Higher variance means we need more redundancy for reliability

            # Adjust N based on average packet loss
            base_N = 16 + int(40 * avg_packet_loss)

            # Adjust based on variance - higher variance means more coding
            variance_factor = min(1.0, packet_loss_variance * 10)
            N = max(1, min(self.max_coding_packets,
                           int(base_N * (1 + variance_factor * 0.5))))

            # Redundancy calculation
            # Higher packet loss and variance mean more redundancy
            base_R = self.max_redundancy * avg_packet_loss
            R = max(1, min(self.max_redundancy,
                           int(base_R * (1 + variance_factor))))

        # Total packets = Data packets + Redundancy packets
        M = N + R

        return N, M, R

    def calculate_path_quality(self, path_id: str, state: Dict) -> float:
        """
        Calculate the quality score for a path based on its state.
        Higher score means better quality for data transmission.

        Args:
            path_id: Path identifier
            state: Path state information

        Returns:
            Quality score for the path
        """
        bandwidth = state['bandwidth']
        packet_loss = state['packet_loss']
        energy_per_byte = state['energy_per_byte']

        # Basic quality formula: bandwidth * (1 - packet_loss)
        basic_quality = bandwidth * (1 - packet_loss)

        # Add energy efficiency factor if we've "learned" to consider it
        if self.learning_progress > 0.5:
            # Normalize energy per byte (assuming range 0.01-0.1 J/byte)
            normalized_energy = min(1.0, energy_per_byte / 0.1)
            energy_factor = 1.0 - normalized_energy * 0.5  # Energy efficiency factor

            # Include energy in quality calculation
            quality = basic_quality * energy_factor
        else:
            quality = basic_quality

        return quality

    def allocate_bandwidth(self, network_state: Dict, total_packets: int) -> Dict[str, float]:
        """
        Allocate packets to different paths based on path quality
        and simulated learning strategy.

        Args:
            network_state: Current network state
            total_packets: Total packets to allocate

        Returns:
            Dictionary mapping path IDs to packet allocations
        """
        path_qualities = {}
        path_ids = list(network_state.keys())

        # Calculate quality for each path
        for path_id, state in network_state.items():
            quality = self.calculate_path_quality(path_id, state)
            path_qualities[path_id] = quality

        # Allocation strategies based on learning progress
        if self.learning_progress < 0.3:
            # Simple proportional allocation based on quality
            allocations = self.allocate_proportional(path_qualities, total_packets)
        elif self.learning_progress < 0.7:
            # Water-filling algorithm - allocate more to better paths
            allocations = self.allocate_water_filling(path_qualities, total_packets)
        else:
            # Advanced allocation with reliability considerations
            allocations = self.allocate_advanced(network_state, path_qualities, total_packets)

        return allocations

    def allocate_proportional(self, path_qualities: Dict[str, float], total_packets: int) -> Dict[str, int]:
        """
        Simple proportional allocation based on path quality.

        Args:
            path_qualities: Dictionary mapping path IDs to quality scores
            total_packets: Total packets to allocate

        Returns:
            Dictionary mapping path IDs to packet allocations
        """
        total_quality = sum(path_qualities.values())
        allocations = {}

        if total_quality > 0:
            # Allocate proportionally to quality
            remaining = total_packets

            for path_id, quality in path_qualities.items():
                # Calculate proportion
                allocation = int((quality / total_quality) * total_packets)
                allocations[path_id] = allocation
                remaining -= allocation

            # Distribute remaining packets to highest quality paths
            sorted_paths = sorted(path_qualities.keys(),
                                  key=lambda p: path_qualities[p],
                                  reverse=True)

            i = 0
            while remaining > 0 and i < len(sorted_paths):
                allocations[sorted_paths[i]] += 1
                remaining -= 1
                i = (i + 1) % len(sorted_paths)
        else:
            # Equal allocation if all qualities are zero
            per_path = total_packets // len(path_qualities)
            remainder = total_packets % len(path_qualities)

            for i, path_id in enumerate(path_qualities.keys()):
                allocations[path_id] = per_path + (1 if i < remainder else 0)

        return allocations

    def allocate_water_filling(self, path_qualities: Dict[str, float], total_packets: int) -> Dict[str, int]:
        """
        Water-filling allocation algorithm that allocates more to better paths.

        Args:
            path_qualities: Dictionary mapping path IDs to quality scores
            total_packets: Total packets to allocate

        Returns:
            Dictionary mapping path IDs to packet allocations
        """
        # Initialize allocations
        allocations = {path_id: 0 for path_id in path_qualities}

        # Sort paths by quality
        sorted_paths = sorted(path_qualities.keys(),
                              key=lambda p: path_qualities[p],
                              reverse=True)

        # Water-filling algorithm
        remaining = total_packets

        # First pass: allocate minimum packets to all paths
        min_packets = 1
        for path_id in sorted_paths:
            allocations[path_id] = min_packets
            remaining -= min_packets

        # Second pass: allocate remaining packets to best paths first
        while remaining > 0:
            for path_id in sorted_paths:
                if remaining > 0:
                    allocations[path_id] += 1
                    remaining -= 1
                else:
                    break

        return allocations

    def allocate_advanced(self, network_state: Dict, path_qualities: Dict[str, float],
                          total_packets: int) -> Dict[str, int]:
        """
        Advanced allocation strategy that considers reliability and bandwidth efficiency.

        Args:
            network_state: Current network state
            path_qualities: Dictionary mapping path IDs to quality scores
            total_packets: Total packets to allocate

        Returns:
            Dictionary mapping path IDs to packet allocations
        """
        # Calculate effective capacity for each path
        effective_capacities = {}

        for path_id, state in network_state.items():
            bandwidth = state['bandwidth']  # Mbps
            packet_loss = state['packet_loss']

            # Convert bandwidth to packets per second
            # bandwidth (Mbps) * 1,000,000 / 8 = bytes per second
            # bytes per second / packet_size = packets per second
            bytes_per_second = bandwidth * 1000000 / 8
            packets_per_second = bytes_per_second / self.packet_size

            # Effective capacity = packets per second * (1 - packet_loss)
            effective_capacity = packets_per_second * (1 - packet_loss)
            effective_capacities[path_id] = effective_capacity

        # Normalize capacities to allocation weights
        total_capacity = sum(effective_capacities.values())

        if total_capacity > 0:
            allocation_weights = {
                path_id: capacity / total_capacity
                for path_id, capacity in effective_capacities.items()
            }
        else:
            # Equal weights if total capacity is zero
            allocation_weights = {
                path_id: 1.0 / len(effective_capacities)
                for path_id in effective_capacities
            }

        # Allocate based on weights
        allocations = {}
        remaining = total_packets

        for path_id, weight in allocation_weights.items():
            allocation = int(weight * total_packets)
            allocations[path_id] = allocation
            remaining -= allocation

        # Distribute remaining packets based on fractional parts
        fractional_parts = {
            path_id: allocation_weights[path_id] * total_packets - allocations[path_id]
            for path_id in allocation_weights
        }

        sorted_by_fraction = sorted(
            fractional_parts.keys(),
            key=lambda p: fractional_parts[p],
            reverse=True
        )

        i = 0
        while remaining > 0 and i < len(sorted_by_fraction):
            allocations[sorted_by_fraction[i]] += 1
            remaining -= 1
            i += 1

        return allocations

    def calculate_reward(self, network_state: Dict, decision: Dict,
                         transmission_result: Dict = None) -> float:
        """
        Calculate a reward for the current decision based on network state
        and transmission results.

        Args:
            network_state: Current network state
            decision: Current decision
            transmission_result: Results of transmission (if available)

        Returns:
            Calculated reward value
        """
        # Extract decision parameters
        N, M, R = decision['coding_parameters']

        # Base reward components
        coding_efficiency = N / max(1, M)  # Higher is better
        redundancy_ratio = R / max(1, N)  # Lower is better

        # Calculate probe cost
        probe_count = sum(1 for p in decision['probed_paths'].values() if p)
        probe_ratio = probe_count / max(1, len(decision['probed_paths']))

        # Basic reward calculation
        reward = (
                0.4 * coding_efficiency -  # Reward for efficient coding
                0.3 * redundancy_ratio -  # Penalty for excessive redundancy
                0.1 * probe_ratio  # Small penalty for probing overhead
        )

        # Add transmission result reward if available
        if transmission_result:
            # Calculate transmission efficiency
            success_rate = transmission_result.get('success_rate', 0) / 100.0
            energy_efficiency = 1.0 - min(1.0, transmission_result.get('energy_consumed', 0) / 10.0)

            # Add to reward
            reward += 0.4 * success_rate + 0.2 * energy_efficiency

        return reward

    def update_learning_progress(self, reward: float):
        """
        Update simulated learning progress based on experience and reward.

        Args:
            reward: Current reward value
        """
        # Increment experience counter
        self.experience_counter += 1

        # Update learning progress - simulates learning curve
        max_experience = 100  # Number of experiences to reach max learning
        self.learning_progress = min(1.0, self.experience_counter / max_experience)

        # Add noise to learning progress to simulate variability
        learning_noise = (random.random() - 0.5) * 0.1
        self.learning_progress = max(0.0, min(1.0, self.learning_progress + learning_noise))

        # Update cumulative reward
        self.cumulative_reward += reward

        # Store reward history
        self.reward_history.append(reward)

    def simulate_transmission(self, maritime_paths: Paths, time_point: float,
                              decision: Dict) -> Dict:
        """
        Simulate transmission using the current decision.

        Args:
            maritime_paths: Paths object containing maritime paths
            time_point: Current time point in minutes
            decision: Current decision

        Returns:
            Simulated transmission results
        """
        # Extract parameters
        allocations = decision['allocations']
        N, M, R = decision['coding_parameters']

        # Simulation variables
        bytes_received = 0
        energy_consumed = 0
        packets_sent = 0
        packets_received = 0

        # Simulate transmission on each path
        for path_id, allocation in allocations.items():
            path = maritime_paths.get_path(path_id)
            time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

            # Simulate bulk transmission for this path
            bulk_size = allocation * self.packet_size
            if bulk_size > 0:
                path_result = path.transmit_bulk_data(bulk_size, time_idx)

                # Accumulate results
                bytes_received += path_result['bytes_received']
                energy_consumed += path_result['energy_consumed']
                packets_sent += path_result['packets_sent']
                packets_received += path_result['packets_received']

        # Calculate success rate
        success_rate = (packets_received / max(1, packets_sent)) * 100

        # Simulate network coding effect
        # If we received at least N packets (out of N+R), we can reconstruct all data
        received_ratio = packets_received / max(1, packets_sent)
        coding_successful = packets_received >= N

        if coding_successful:
            # Successful decoding - we get all N packets worth of data
            decoded_packets = N
        else:
            # Only get what we received
            decoded_packets = packets_received

        transmission_result = {
            'bytes_received': bytes_received,
            'energy_consumed': energy_consumed,
            'packets_sent': packets_sent,
            'packets_received': packets_received,
            'decoded_packets': decoded_packets,
            'coding_successful': coding_successful,
            'success_rate': success_rate
        }

        return transmission_result

    def make_decision(self, maritime_paths: Paths, time_point: float,
                      flow_demand: float = 1.0) -> Dict:
        """
        Make network coding decision based on current network state.

        Args:
            maritime_paths: Paths object containing maritime paths
            time_point: Current time point in minutes
            flow_demand: Network flow demand (normalized)

        Returns:
            Decision result including coding parameters and allocation proportions
        """
        # Step 1: Observe current network state
        network_state = self.observe_network_state(maritime_paths, time_point)

        # Step 2: Decide which paths to probe
        probe_decisions = self.decide_probing(network_state, time_point)

        # Step 3: Calculate network coding parameters
        N, M, R = self.calculate_coding_parameters(network_state)

        # Step 4: Allocate bandwidth
        allocations = self.allocate_bandwidth(network_state, M)

        # Step 5: Simulate transmission
        decision = {
            'coding_parameters': (N, M, R),
            'probed_paths': probe_decisions,
            'allocations': allocations,
            'time_point': time_point
        }

        # Simulate the transmission
        transmission_result = self.simulate_transmission(maritime_paths, time_point, decision)
        decision['transmission_result'] = transmission_result

        # Step 6: Calculate reward
        reward = self.calculate_reward(network_state, decision, transmission_result)

        # Step 7: Update learning progress
        self.update_learning_progress(reward)

        # Complete decision result
        decision['reward'] = reward
        decision['cumulative_reward'] = self.cumulative_reward
        decision['learning_progress'] = self.learning_progress

        # Store decision in history
        self.decision_history.append(decision)
        self.last_decision = decision

        return decision