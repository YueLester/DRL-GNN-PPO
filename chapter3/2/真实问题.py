import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from collections import deque
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)


class NetworkEnvironment:
    """Environment for the network resource allocation problem"""

    def __init__(self, num_nodes, resource_types=3, max_steps=100):
        """
        Initialize the environment

        Args:
            num_nodes: Number of nodes in the network
            resource_types: Number of resource types (default: 3 - CPU, memory, bandwidth)
            max_steps: Maximum number of steps per episode
        """
        self.num_nodes = num_nodes
        self.resource_types = resource_types
        self.max_steps = max_steps

        # Define action space: each node can get resources in discrete levels (0-10)
        self.action_levels = 11  # 0-10 levels for each resource
        self.flat_action_space = self.action_levels ** (num_nodes * resource_types)

        # For hierarchical approach: first select node, then allocate resources
        self.hier_action_space_level1 = num_nodes  # Select which node to modify
        self.hier_action_space_level2 = self.action_levels ** resource_types  # Allocate resources to that node

        # State space: current allocation + network metrics
        self.state_dim = num_nodes * resource_types + num_nodes

        # Resource constraints
        self.total_resources = [num_nodes * 7 for _ in range(resource_types)]  # Average of 7 units per node

        # Initialize state
        self.reset()

        # Define node interdependencies (more connections = more complex interactions)
        self.dependencies = self._create_dependencies()

        # Track history for visualization
        self.performance_history = []

    def _create_dependencies(self):
        """Create interdependencies between nodes"""
        # Higher number of nodes = more complex interdependencies
        dependencies = np.zeros((self.num_nodes, self.num_nodes))

        # Create sparse dependency matrix with increasing density as num_nodes increases
        density = min(0.3 + (self.num_nodes / 20), 0.8)  # More nodes = more interconnected

        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j and random.random() < density:
                    # Random influence strength -0.3 to 0.3
                    dependencies[i][j] = (random.random() * 0.6) - 0.3

        return dependencies

    def reset(self):
        """Reset environment to initial state"""
        # Initial allocation: random but valid
        self.allocation = np.zeros((self.num_nodes, self.resource_types))

        # Randomly initialize allocation within constraints
        for res_type in range(self.resource_types):
            remaining = self.total_resources[res_type]
            for node in range(self.num_nodes - 1):
                # Ensure we leave some resources for the last node
                max_alloc = min(10, remaining - (self.num_nodes - node - 1))
                alloc = random.randint(0, max_alloc)
                self.allocation[node, res_type] = alloc
                remaining -= alloc
            # Allocate remaining to last node
            self.allocation[self.num_nodes - 1, res_type] = remaining

        # Initial performance
        self.node_performance = self._calculate_performance()

        # Reset step counter
        self.current_step = 0

        # Reset history
        self.performance_history = []

        return self._get_state()

    def _calculate_performance(self):
        """Calculate performance of each node based on resource allocation and dependencies"""
        base_performance = np.zeros(self.num_nodes)

        # Base performance from resource allocation
        for node in range(self.num_nodes):
            # Weighted combination of resources with diminishing returns
            cpu_perf = np.sqrt(self.allocation[node, 0])
            mem_perf = np.log1p(self.allocation[node, 1])
            bw_perf = 0.8 * self.allocation[node, 2]

            # Combined performance normalized to 0-10 range
            base_performance[node] = (cpu_perf + mem_perf + bw_perf) / 3.0

        # Apply interdependency effects
        final_performance = base_performance.copy()
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i != j:
                    # Node j affects node i's performance
                    final_performance[i] += base_performance[j] * self.dependencies[i, j]

        # Ensure performance is non-negative
        final_performance = np.maximum(final_performance, 0.0)

        return final_performance

    def _get_state(self):
        """Return current state representation"""
        # Concatenate allocation and performance metrics
        return np.concatenate([self.allocation.flatten(), self.node_performance])

    def _decode_flat_action(self, action_idx):
        """Convert flat action index to node-resource allocation action"""
        allocations = []
        temp = action_idx

        for _ in range(self.num_nodes * self.resource_types):
            allocations.append(temp % self.action_levels)
            temp //= self.action_levels

        # Reshape to node x resource_type
        return np.array(allocations).reshape(self.num_nodes, self.resource_types)

    def _decode_hierarchical_action(self, node, resource_action):
        """Convert hierarchical action to node-resource allocation"""
        allocation_change = np.zeros((self.num_nodes, self.resource_types))

        # Decode resource_action for the selected node
        temp = resource_action
        for res in range(self.resource_types):
            allocation_change[node, res] = temp % self.action_levels
            temp //= self.action_levels

        return allocation_change

    def step_flat(self, action_idx):
        """Take a step using flat action space"""
        # Decode action
        new_allocation = self._decode_flat_action(action_idx)

        # Check resource constraints
        for res in range(self.resource_types):
            total_allocated = np.sum(new_allocation[:, res])
            if total_allocated > self.total_resources[res]:
                # Reduce allocations proportionally to meet constraint
                scale_factor = self.total_resources[res] / total_allocated
                new_allocation[:, res] = np.floor(new_allocation[:, res] * scale_factor)

        # Update allocation
        self.allocation = new_allocation

        # Calculate new performance
        self.node_performance = self._calculate_performance()

        # Calculate reward
        reward = np.mean(self.node_performance)

        # Update step counter
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Record performance
        self.performance_history.append(reward)

        return self._get_state(), reward, done, {}

    def step_hierarchical(self, node, resource_action):
        """Take a step using hierarchical action space"""
        # Decode action for the selected node
        new_node_allocation = self._decode_hierarchical_action(node, resource_action)

        # Apply the change to just the selected node
        self.allocation = new_node_allocation

        # Check resource constraints
        for res in range(self.resource_types):
            total_allocated = np.sum(self.allocation[:, res])
            if total_allocated > self.total_resources[res]:
                # Reduce allocations proportionally to meet constraint
                scale_factor = self.total_resources[res] / total_allocated
                self.allocation[:, res] = np.floor(self.allocation[:, res] * scale_factor)

        # Calculate new performance
        self.node_performance = self._calculate_performance()

        # Calculate reward
        reward = np.mean(self.node_performance)

        # Update step counter
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.max_steps

        # Record performance
        self.performance_history.append(reward)

        return self._get_state(), reward, done, {}


class DQNAgent:
    """Flat DQN Agent for resource allocation"""

    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self):
        """Neural network model for DQN"""
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        act_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        """Train model using experience replay"""
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Q(s,a) and Q(s',a')
        targets = self.model.predict(states, verbose=0)
        next_q_values = self.model.predict(next_states, verbose=0)

        # Update targets for actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train model
        self.model.fit(states, targets, epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


class HierarchicalDQNAgent:
    """Hierarchical DQN Agent with two-level action selection"""

    def __init__(self, state_dim, node_action_dim, resource_action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.node_action_dim = node_action_dim  # Level 1: which node to modify
        self.resource_action_dim = resource_action_dim  # Level 2: how to allocate resources

        # Two memory buffers for each level
        self.node_memory = deque(maxlen=10000)
        self.resource_memory = deque(maxlen=10000)

        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate

        # Two separate networks for hierarchical decisions
        self.node_model = self._build_node_model()
        self.resource_model = self._build_resource_model()

    def _build_node_model(self):
        """Model for selecting which node to modify"""
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_dim, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.node_action_dim, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _build_resource_model(self):
        """Model for selecting resource allocation for a node"""
        # Input: state + one-hot encoded node
        state_input = Input(shape=(self.state_dim,))
        node_input = Input(shape=(self.node_action_dim,))

        # Combined inputs
        combined = Concatenate()([state_input, node_input])

        # Hidden layers
        x = Dense(128, activation='relu')(combined)
        x = Dense(128, activation='relu')(x)

        # Output layer: resource allocation action
        outputs = Dense(self.resource_action_dim, activation='linear')(x)

        # Create model
        model = Model(inputs=[state_input, node_input], outputs=outputs)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember_node(self, state, action, reward, next_state, done):
        """Store node selection experience"""
        self.node_memory.append((state, action, reward, next_state, done))

    def remember_resource(self, state, node, action, reward, next_state, done):
        """Store resource allocation experience"""
        # Include the selected node as part of the state
        node_one_hot = np.zeros(self.node_action_dim)
        node_one_hot[node] = 1
        self.resource_memory.append((state, node_one_hot, action, reward, next_state, done))

    def act_node(self, state):
        """Select which node to modify"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.node_action_dim)
        act_values = self.node_model.predict(np.array([state]), verbose=0)
        return np.argmax(act_values[0])

    def act_resource(self, state, node):
        """Select resource allocation for chosen node"""
        # Create one-hot encoding for node
        node_one_hot = np.zeros(self.node_action_dim)
        node_one_hot[node] = 1

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.resource_action_dim)

        # Predict with both state and node
        act_values = self.resource_model.predict([np.array([state]), np.array([node_one_hot])], verbose=0)
        return np.argmax(act_values[0])

    def replay_node(self, batch_size):
        """Train node selection model"""
        if len(self.node_memory) < batch_size:
            return

        minibatch = random.sample(self.node_memory, batch_size)
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])

        # Q(s,a) and Q(s',a')
        targets = self.node_model.predict(states, verbose=0)
        next_q_values = self.node_model.predict(next_states, verbose=0)

        # Update targets for actions taken
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])

        # Train model
        self.node_model.fit(states, targets, epochs=1, verbose=0)

    def replay_resource(self, batch_size):
        """Train resource allocation model"""
        if len(self.resource_memory) < batch_size:
            return

        minibatch = random.sample(self.resource_memory, batch_size)
        states = np.array([transition[0] for transition in minibatch])
        nodes = np.array([transition[1] for transition in minibatch])
        actions = np.array([transition[2] for transition in minibatch])
        rewards = np.array([transition[3] for transition in minibatch])
        next_states = np.array([transition[4] for transition in minibatch])
        dones = np.array([transition[5] for transition in minibatch])

        # Current Q-values
        targets = self.resource_model.predict([states, nodes], verbose=0)

        # For each transition, get the max Q-value for the next state
        for i in range(batch_size):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # For the next state, need to find best node and then best resource allocation
                next_node = np.argmax(self.node_model.predict(np.array([next_states[i]]), verbose=0)[0])
                next_node_one_hot = np.zeros(self.node_action_dim)
                next_node_one_hot[next_node] = 1

                next_q_values = self.resource_model.predict(
                    [np.array([next_states[i]]), np.array([next_node_one_hot])],
                    verbose=0
                )[0]

                targets[i, actions[i]] = rewards[i] + self.gamma * np.max(next_q_values)

        # Train model
        self.resource_model.fit([states, nodes], targets, epochs=1, verbose=0)

    def replay(self, batch_size):
        """Train both models"""
        self.replay_node(batch_size)
        self.replay_resource(batch_size)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def run_experiment(node_counts):
    """Run experiment comparing flat vs hierarchical RL across different node counts"""
    num_episodes = 1000
    max_steps = 50
    batch_size = 32

    # Store results
    results = {}

    for num_nodes in node_counts:
        print(f"Running experiment with {num_nodes} nodes...")

        # Initialize environment
        env = NetworkEnvironment(num_nodes=num_nodes, max_steps=max_steps)

        # Initialize flat DQN agent
        # For large networks, we need to use a smaller action space approximation
        flat_action_dim = min(10000, env.flat_action_space)
        flat_agent = DQNAgent(env.state_dim, flat_action_dim)

        # Initialize hierarchical DQN agent
        hier_agent = HierarchicalDQNAgent(
            env.state_dim,
            env.hier_action_space_level1,
            env.hier_action_space_level2
        )

        # Store rewards for each episode
        flat_rewards = []
        hier_rewards = []

        # Train flat agent
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                # Choose action
                action = flat_agent.act(state)

                # Take action
                next_state, reward, done, _ = env.step_flat(action)

                # Remember experience
                flat_agent.remember(state, action, reward, next_state, done)

                # Update state and reward
                state = next_state
                total_reward += reward

                if done:
                    break

            # Train agent
            flat_agent.replay(batch_size)

            # Record episode reward
            flat_rewards.append(total_reward)

            if (episode + 1) % 50 == 0:
                print(f"Flat RL - Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.4f}")

        # Train hierarchical agent
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                # Choose node to modify
                node = hier_agent.act_node(state)

                # Choose resource allocation for that node
                resource_action = hier_agent.act_resource(state, node)

                # Take action
                next_state, reward, done, _ = env.step_hierarchical(node, resource_action)

                # Remember experience
                hier_agent.remember_node(state, node, reward, next_state, done)
                hier_agent.remember_resource(state, node, resource_action, reward, next_state, done)

                # Update state and reward
                state = next_state
                total_reward += reward

                if done:
                    break

            # Train agent
            hier_agent.replay(batch_size)

            # Record episode reward
            hier_rewards.append(total_reward)

            if (episode + 1) % 50 == 0:
                print(f"Hierarchical RL - Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.4f}")

        # Store results for this node count
        results[num_nodes] = {
            'flat': flat_rewards,
            'hierarchical': hier_rewards
        }

    return results


def plot_results(results, node_counts):
    """Plot convergence results"""
    plt.figure(figsize=(12, 8))

    # Colors for different node counts
    colors = ['blue', 'orange', 'green']

    # Apply smoothing to make trends clearer
    window_size = 20

    for i, nodes in enumerate(node_counts):
        # Get data
        flat_rewards = results[nodes]['flat']
        hier_rewards = results[nodes]['hierarchical']

        # Smooth data
        flat_smoothed = np.convolve(flat_rewards, np.ones(window_size) / window_size, mode='valid')
        hier_smoothed = np.convolve(hier_rewards, np.ones(window_size) / window_size, mode='valid')

        # Compute confidence intervals (use standard deviation for simplicity)
        flat_std = np.std(flat_rewards) * 0.2  # Reduce for clarity
        hier_std = np.std(hier_rewards) * 0.2

        x = np.arange(len(flat_smoothed))

        # Plot flat RL with confidence interval
        plt.plot(x, flat_smoothed, linestyle='--', color=colors[i], label=f'Flat RL ({nodes} nodes)')
        plt.fill_between(x, flat_smoothed - flat_std, flat_smoothed + flat_std, color=colors[i], alpha=0.1)

        # Plot hierarchical RL with confidence interval
        plt.plot(x, hier_smoothed, linestyle='-', color=colors[i], label=f'Hierarchical RL ({nodes} nodes)')
        plt.fill_between(x, hier_smoothed - hier_std, hier_smoothed + hier_std, color=colors[i], alpha=0.1)

    plt.title('Reinforcement Learning Convergence for Network Resource Allocation', fontsize=14)
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward (Network Performance)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Add annotation
    plt.text(50, min(flat_smoothed) - flat_std,
             'Solid lines: Hierarchical RL\nDashed lines: Flat RL\nShaded areas: Performance variation',
             fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig('rl_convergence_comparison.png', dpi=300)
    plt.show()


# Main execution
if __name__ == "__main__":
    # Run experiment with different network sizes
    node_counts = [3, 5, 8]  # Small, Medium, Large networks
    results = run_experiment(node_counts)

    # Plot results
    plot_results(results, node_counts)

