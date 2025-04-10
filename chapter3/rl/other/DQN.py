import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Import the Paths class from paths module
from paths import Paths


class DQN(nn.Module):
    """Deep Q-Network for network coding decision making"""

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)


class ReplayBuffer:
    """Experience replay buffer for storing and sampling transitions"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DRLNetworkCodingDecisionMaker:
    """
    Deep Reinforcement Learning based network coding decision maker
    for maritime communication networks.
    """

    def __init__(self, max_coding_packets=128, max_redundancy=32,
                 state_size=20, action_size=10, learning_rate=0.001,
                 gamma=0.95, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, memory_size=2000, batch_size=32):
        """
        Initialize the DRL network coding decision maker.

        Args:
            max_coding_packets: Maximum number of network coding packets
            max_redundancy: Maximum number of redundancy packets
            state_size: Size of the state space
            action_size: Size of the action space
            learning_rate: Learning rate for the neural network
            gamma: Discount factor for future rewards
            epsilon: Exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            memory_size: Size of the replay buffer
            batch_size: Batch size for training
        """
        self.max_coding_packets = max_coding_packets
        self.max_redundancy = max_redundancy
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        # Setup DQN and target networks
        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        # Setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Setup replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Current network state
        self.network_view = {}  # Store current network view
        self.last_decision = None  # Store last decision result
        self.last_probe_time = {}  # Store last probe time for each path
        self.last_state = None  # Last observed state
        self.last_action = None  # Last action taken
        self.training_mode = False  # Training mode flag
        self.cumulative_reward = 0  # Cumulative reward for evaluation
        self.rewards_history = []  # History of rewards

        # Environment factors
        self.weather_factor = 1.0  # Default weather normalization factor
        self.prev_transmission_result = None  # Previous transmission result

    def update_target_model(self):
        """Update target model with weights from the main model"""
        self.target_model.load_state_dict(self.model.state_dict())

    def train(self, enable=True):
        """Enable or disable training mode"""
        self.training_mode = enable
        if enable:
            self.model.train()
        else:
            self.model.eval()

    def observe_network_state(self, maritime_paths: Paths, time_point: float,
                              flow_demand: float = 1.0) -> np.ndarray:
        """
        Observe the current network state and convert to state vector.

        Args:
            maritime_paths: Paths object containing maritime paths
            time_point: Current time point in minutes
            flow_demand: Network flow demand (normalized between 0 and 1)

        Returns:
            State vector for the RL agent
        """
        path_ids = maritime_paths.list_paths()
        num_paths = len(path_ids)

        # Initialize state vector
        state = np.zeros(self.state_size)

        # Add flow demand to state
        state[0] = flow_demand

        # Add weather factor to state
        state[1] = self.weather_factor

        # Add decision time to state
        state[2] = time_point / 60.0  # Normalize by hour

        # Add previous transmission result if available
        if self.prev_transmission_result is not None:
            state[3] = self.prev_transmission_result.get('success_rate', 0) / 100.0

        # For each path, add current state information
        for i, path_id in enumerate(path_ids):
            if i >= (self.state_size - 4) // 4:  # Ensure we don't exceed state vector size
                break

            path = maritime_paths.get_path(path_id)
            time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

            # Get current path parameters
            bandwidth = path.bandwidth_series[time_idx]
            packet_loss = path.packet_loss_series[time_idx]
            energy_per_byte = path.energy_consumption_series[time_idx]

            # Calculate time since last probe
            time_since_probe = 0
            if path_id in self.last_probe_time:
                time_since_probe = time_point - self.last_probe_time.get(path_id, 0)

            # Add path parameters to state vector
            base_idx = 4 + i * 4
            state[base_idx] = bandwidth / 20.0  # Normalize assuming max bandwidth is 20 Mbps
            state[base_idx + 1] = packet_loss
            state[base_idx + 2] = energy_per_byte * 1000  # Convert to mJ/byte
            state[base_idx + 3] = min(time_since_probe / 10.0, 1.0)  # Normalize, cap at 1.0

            # Update network view
            if path_id not in self.network_view:
                self.network_view[path_id] = {}

            self.network_view[path_id]['bandwidth'] = bandwidth
            self.network_view[path_id]['packet_loss'] = packet_loss
            self.network_view[path_id]['energy_per_byte'] = energy_per_byte
            self.network_view[path_id]['time_since_probe'] = time_since_probe

        return state

    def select_action(self, state, maritime_paths: Paths):
        """
        Select an action using epsilon-greedy policy.

        Args:
            state: Current state vector
            maritime_paths: Paths object containing maritime paths

        Returns:
            Dictionary containing action parameters
        """
        path_ids = maritime_paths.list_paths()
        num_paths = len(path_ids)

        if self.training_mode and np.random.rand() <= self.epsilon:
            # Exploration: random action
            coding_n = np.random.randint(1, self.max_coding_packets // 2)
            coding_r = np.random.randint(1, min(coding_n, self.max_redundancy))

            # Generate probe decisions
            probe_decisions = {}
            for path_id in path_ids:
                probe_decisions[path_id] = np.random.random() < 0.2  # 20% chance to probe

            # Generate allocation proportions
            allocations = np.random.dirichlet(np.ones(num_paths))
            allocation_dict = {path_id: alloc for path_id, alloc in zip(path_ids, allocations)}

            # Action vector for memory
            action_vector = np.zeros(self.action_size)
            action_vector[0] = coding_n / self.max_coding_packets
            action_vector[1] = coding_r / self.max_redundancy
            for i, path_id in enumerate(path_ids):
                if i < (self.action_size - 2) // 2:
                    action_vector[2 + i] = allocation_dict[path_id]
                    action_vector[2 + num_paths + i] = 1 if probe_decisions[path_id] else 0

            action = {
                'coding_n': coding_n,
                'coding_r': coding_r,
                'probe_decisions': probe_decisions,
                'allocations': allocation_dict,
                'action_vector': action_vector
            }

        else:
            # Exploitation: use model prediction
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.model(state_tensor).numpy()[0]

            # Extract action parameters from Q-values
            coding_n = int(q_values[0] * self.max_coding_packets)
            coding_n = max(1, min(coding_n, self.max_coding_packets))

            coding_r = int(q_values[1] * self.max_redundancy)
            coding_r = max(1, min(coding_r, min(coding_n, self.max_redundancy)))

            # Extract allocation proportions and probe decisions
            allocations_raw = q_values[2:2 + num_paths]
            allocations_sum = np.sum(allocations_raw)
            if allocations_sum > 0:
                allocations = allocations_raw / allocations_sum
            else:
                allocations = np.ones(num_paths) / num_paths

            allocation_dict = {path_id: alloc for path_id, alloc in zip(path_ids, allocations)}

            probe_decisions = {}
            for i, path_id in enumerate(path_ids):
                if i < len(q_values) - (2 + num_paths):
                    probe_decisions[path_id] = q_values[2 + num_paths + i] > 0.5
                else:
                    probe_decisions[path_id] = False

            action = {
                'coding_n': coding_n,
                'coding_r': coding_r,
                'probe_decisions': probe_decisions,
                'allocations': allocation_dict,
                'action_vector': q_values
            }

        return action

    def calculate_reward(self, maritime_paths: Paths, time_point: float,
                         action: Dict, transmission_result: Dict) -> float:
        """
        Calculate reward for the current action.

        Args:
            maritime_paths: Paths object containing maritime paths
            time_point: Current time point in minutes
            action: Action taken
            transmission_result: Result of the transmission

        Returns:
            Calculated reward value
        """
        # Extract parameters
        coding_n = action['coding_n']
        coding_r = action['coding_r']
        probe_decisions = action['probe_decisions']

        # Calculate throughput reward
        throughput_reward = transmission_result.get('success_rate', 0) / 100.0

        # Calculate redundancy penalty
        redundancy_ratio = coding_r / max(1, coding_n)
        redundancy_penalty = -0.2 * redundancy_ratio

        # Calculate retransmission energy penalty
        energy_consumed = transmission_result.get('energy_consumed', 0)
        energy_penalty = -0.3 * min(energy_consumed / 10.0, 1.0)

        # Calculate probing overhead penalty
        probe_count = sum(1 for probe in probe_decisions.values() if probe)
        probe_penalty = -0.1 * (probe_count / len(probe_decisions))

        # Calculate weighted reward
        reward = (
                0.5 * throughput_reward +
                0.2 * redundancy_penalty +
                0.2 * energy_penalty +
                0.1 * probe_penalty
        )

        return reward

    def learn(self):
        """Train the model using experience replay"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # Calculate Q-values
        q_values = self.model(states)

        # Calculate target Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = torch.max(next_q_values, dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # Calculate loss
        loss = self.criterion(q_values.gather(1, actions.long().unsqueeze(1)).squeeze(), target_q)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_probe_times(self, action: Dict, time_point: float):
        """
        Update the last probe times based on the action.

        Args:
            action: Action taken
            time_point: Current time point in minutes
        """
        probe_decisions = action['probe_decisions']

        for path_id, probe in probe_decisions.items():
            if probe:
                self.last_probe_time[path_id] = time_point

    def calculate_packet_allocations(self, action: Dict, total_packets: int) -> Dict[str, int]:
        """
        Calculate the actual packet allocations based on proportions.

        Args:
            action: Action containing allocation proportions
            total_packets: Total number of packets to allocate

        Returns:
            Dictionary mapping path IDs to packet counts
        """
        allocations = action['allocations']
        path_ids = list(allocations.keys())

        # Calculate initial allocation based on proportions
        initial_allocation = {path_id: int(allocations[path_id] * total_packets)
                              for path_id in path_ids}

        # Handle remaining packets
        allocated = sum(initial_allocation.values())
        remaining = total_packets - allocated

        if remaining > 0:
            # Sort paths by fractional part of allocation in descending order
            frac_parts = {path_id: allocations[path_id] * total_packets - initial_allocation[path_id]
                          for path_id in path_ids}
            sorted_paths = sorted(path_ids, key=lambda p: frac_parts[p], reverse=True)

            # Distribute remaining packets
            for i in range(remaining):
                initial_allocation[sorted_paths[i % len(sorted_paths)]] += 1

        return initial_allocation

    def make_decision(self, maritime_paths: Paths, time_point: float,
                      flow_demand: float = 1.0, simulate_transmission: bool = False) -> Dict:
        """
        Make network coding and allocation decisions using DRL.

        Args:
            maritime_paths: Paths object containing maritime paths
            time_point: Current time point in minutes
            flow_demand: Network flow demand (normalized between 0 and 1)
            simulate_transmission: Whether to simulate transmission

        Returns:
            Decision result including coding parameters and allocation proportions
        """
        # Observe current network state
        current_state = self.observe_network_state(maritime_paths, time_point, flow_demand)

        # Select action
        action = self.select_action(current_state, maritime_paths)

        # Update probe times
        self.update_probe_times(action, time_point)

        # Calculate packet counts
        coding_n = action['coding_n']
        coding_r = action['coding_r']
        coding_m = coding_n + coding_r

        # Calculate actual packet allocations
        packet_allocations = self.calculate_packet_allocations(action, coding_m)

        # Simulate transmission if requested
        transmission_result = None
        if simulate_transmission:
            # Simple transmission simulation
            path_ids = maritime_paths.list_paths()
            bytes_received = 0
            energy_consumed = 0
            packets_sent = 0
            packets_received = 0

            for path_id in path_ids:
                path = maritime_paths.get_path(path_id)
                time_idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

                # Simulate bulk transmission for this path
                bulk_size = packet_allocations[path_id] * 1500  # Assuming 1500 bytes per packet
                if bulk_size > 0:
                    path_result = path.transmit_bulk_data(bulk_size, time_idx)

                    # Accumulate results
                    bytes_received += path_result['bytes_received']
                    energy_consumed += path_result['energy_consumed']
                    packets_sent += path_result['packets_sent']
                    packets_received += path_result['packets_received']

            # Calculate success rate
            success_rate = (packets_received / max(1, packets_sent)) * 100

            transmission_result = {
                'bytes_received': bytes_received,
                'energy_consumed': energy_consumed,
                'packets_sent': packets_sent,
                'packets_received': packets_received,
                'success_rate': success_rate
            }

            # Calculate reward
            reward = self.calculate_reward(maritime_paths, time_point, action, transmission_result)
            self.cumulative_reward += reward
            self.rewards_history.append(reward)

            # Store experience in replay buffer if in training mode
            if self.training_mode and self.last_state is not None:
                self.memory.add(
                    self.last_state,
                    self.last_action['action_vector'],
                    reward,
                    current_state,
                    False  # Done flag
                )

                # Train the model
                self.learn()

            # Update previous transmission result
            self.prev_transmission_result = transmission_result

        # Save current state and action
        self.last_state = current_state
        self.last_action = action

        # Prepare decision result
        decision_result = {
            'coding_parameters': (coding_n, coding_m, coding_r),
            'allocations': packet_allocations,
            'allocation_proportions': action['allocations'],
            'probed_paths': action['probe_decisions'],
            'transmission_result': transmission_result,
            'time_point': time_point,
            'cumulative_reward': self.cumulative_reward
        }

        # Store last decision
        self.last_decision = decision_result

        return decision_result

    def save_model(self, path):
        """Save model weights to file"""
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        """Load model weights from file"""
        self.model.load_state_dict(torch.load(path))
        self.update_target_model()

    def update_weather_factor(self, factor):
        """Update weather normalization factor"""
        self.weather_factor = factor

    def plot_reward_history(self, figsize=(10, 6)):
        """Plot the reward history"""
        plt.figure(figsize=figsize)
        plt.plot(self.rewards_history)
        plt.title('Reward History')
        plt.xlabel('Decision Steps')
        plt.ylabel('Reward')
        plt.grid(True)
        return plt.gcf()