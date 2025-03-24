import numpy as np
import matplotlib.pyplot as plt


def calculate_moving_average(data, window_size):
    """Calculate moving average with the specified window size"""
    weights = np.ones(window_size) / window_size
    return np.convolve(data, weights, mode='valid')


def plot_learning_curve(epochs=1000, window_size=50, save_path=None):
    """
    Plot reinforcement learning convergence curve

    Parameters:
        epochs (int): Number of training epochs
        window_size (int): Window size for moving average
        save_path (str): Optional, path to save the figure
    """
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

        # Create figure
        plt.figure(figsize=(6,4))

        # Plot instant rewards
        # plt.plot(rewards, 'b-', alpha=0.4, label='Instant Reward', linewidth=1)

        # Plot moving average
        plt.plot(range(window_size - 1, epochs), moving_avg, 'r-',
                 label='Reward')

        # Set plot properties
        # plt.title('Reinforcement Learning Training Curve (1000 Epochs)', fontsize=14, pad=15)
        plt.xlabel('Training Epochs', fontsize=12)
        plt.ylabel('Reward', fontsize=12)
        plt.legend()
        # plt.grid(True, linestyle='--', alpha=0.7)

        # Set axis range
        plt.ylim(40, 110)
        plt.xlim(0, epochs)

        # # Calculate and display statistics
        # convergence_rewards = rewards[convergence_point:]
        # early_rewards = rewards[:convergence_point]


        # # Mark convergence point
        # plt.axvline(x=convergence_point, color='g', linestyle='--', alpha=0.5)
        # plt.text(convergence_point + 10, 115, 'Convergence Point', color='g', alpha=0.7)
        #
        # plt.tight_layout()
        #
        # # Save figure if path is provided
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"Figure saved to: {save_path}")

        plt.show()

    except Exception as e:
        print(f"Error occurred during plotting: {e}")
        raise


if __name__ == "__main__":
    # Run example
    plot_learning_curve(
        epochs=1000,
        window_size=50,
        save_path='rl_convergence_1000.png'
    )