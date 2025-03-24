import random

import numpy as np
import matplotlib.pyplot as plt

def calculate_moving_average(data, window_size):
    """Calculate moving average with the specified window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')

epochs=1000
window_size=50
try:
    np.random.seed(42)
    rewards = []
    base_reward = 20  # Higher starting reward
    convergence_point = 850  # Later convergence point

    # Generate rewards for each epoch
    for i in range(epochs):
        progress = i / convergence_point if i < convergence_point else 1

        if i < convergence_point:
            # Pre-convergence: larger noise and learning rate
            noise_scale = 0.4 * (1 - progress ** 0.5)
            learning_rate = 0.015 * (1 - progress ** 0.3)
        else:
            # Post-convergence: maintain moderate fluctuation
            noise_scale = 0.15
            learning_rate = 0.005

        # Add periodic noise
        periodic_noise = np.sin(i * 0.1) * 5 * noise_scale

        # Generate random noise
        random_noise = noise_scale * (np.random.random() - 0.5) * 60

        # Target value with fluctuation after convergence
        if i < convergence_point:
            target = 100  # Higher target value
        else:
            target = 100 + np.random.normal(0, 3)

        # Update base reward
        base_reward = base_reward + learning_rate * (target - base_reward)

        # Combine all noise and clip to valid range
        reward = np.clip(base_reward + random_noise + periodic_noise, 0, 120)
        rewards.append(reward)

    # Calculate moving average
    moving_avg = calculate_moving_average(rewards, window_size)
plt.figure(figsize=(15, 8))

plt.plot(range(window_size - 1, epochs), moving_avg, 'r-',
         label='reward', linewidth=2)


plt.xlabel("flowNum")  # ratio
plt.ylabel("aoi(s)")  # 纵坐标名字
plt.legend(loc="best")  # 图例
plt.show()