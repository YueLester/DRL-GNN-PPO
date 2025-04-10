import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Tuple

# Import required modules
from paths import Paths
from path import Path

from chapter3.rl.other.simulated_rl_decision import SimulatedRLNetworkCodingDecisionMaker


def visualize_drl_network_coding_decisions(maritime_paths: Paths,
                                       decision_maker: SimulatedRLNetworkCodingDecisionMaker,
                                       duration_minutes: int = 30,
                                       time_step: float = 0.5,
                                       flow_demand: float = 1.0,
                                       figsize: Tuple[int, int] = (14, 10)):
    """
    Visualize network coding decisions over time using the DRL decision maker.

    Args:
        maritime_paths: Paths object containing maritime communication paths
        decision_maker: DRL-based decision maker instance
        duration_minutes: Simulation duration in minutes
        time_step: Time step in minutes
        flow_demand: Network flow demand (normalized)
        figsize: Figure size
    """
    # Generate time points series
    time_points = np.arange(0, duration_minutes + time_step, time_step)

    # Store decisions at each time point
    decisions = []

    # Make decisions at each time point
    for t in time_points:
        # Call decision algorithm
        decision = decision_maker.make_decision(
            maritime_paths, t, flow_demand, simulate_transmission=True)
        decisions.append(decision)

    # Prepare plotting data
    path_ids = maritime_paths.list_paths()
    n_paths = len(path_ids)

    # Create color mapping using Paths class colors
    path_colors = {path_id: maritime_paths.path_colors[path_id] for path_id in path_ids}

    # Extract decision data
    probe_data = {path_id: [] for path_id in path_ids}
    bandwidth_data = {path_id: [] for path_id in path_ids}
    packet_loss_data = {path_id: [] for path_id in path_ids}
    allocation_data = {path_id: [] for path_id in path_ids}
    coding_rates = []
    rewards = []

    for i, t in enumerate(time_points):
        decision = decisions[i]

        # Extract coding rate
        N, M, R = decision['coding_parameters']
        coding_rate = R / N if N > 0 else 0
        coding_rates.append(coding_rate)

        # Extract reward
        if i > 0:
            reward_diff = decision['cumulative_reward'] - decisions[i-1]['cumulative_reward']
        else:
            reward_diff = decision['cumulative_reward']
        rewards.append(reward_diff)

        # Extract path-specific data
        for path_id in path_ids:
            path = maritime_paths.get_path(path_id)

            # Time index
            time_idx = int(round(t / path.total_duration * (path.total_points - 1)))

            # Probe status
            probe_data[path_id].append(decision['probed_paths'].get(path_id, False))

            # Bandwidth data
            bandwidth_data[path_id].append(path.bandwidth_series[time_idx])

            # Packet loss data
            packet_loss_data[path_id].append(path.packet_loss_series[time_idx])

            # Allocation data
            allocation = decision['allocations'].get(path_id, 0)
            allocation_data[path_id].append(allocation)

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.1)

    # First subplot: Bandwidth and allocations
    ax1 = fig.add_subplot(gs[0])

    # Plot bandwidth curves and probe markers
    for path_id in path_ids:
        # Plot bandwidth curve
        ax1.plot(time_points, bandwidth_data[path_id],
                 label=f"{path_id} Bandwidth",
                 color=path_colors[path_id],
                 linestyle='-',
                 linewidth=1.5)

        # Plot probe markers
        probe_times = [time_points[i] for i in range(len(time_points)) if probe_data[path_id][i]]
        probe_bandwidths = [bandwidth_data[path_id][i] for i in range(len(time_points)) if probe_data[path_id][i]]

        if probe_times:
            ax1.scatter(probe_times, probe_bandwidths,
                        marker='o',
                        s=80,  # Marker size
                        color=path_colors[path_id],
                        edgecolors='red',
                        linewidths=1.5,
                        alpha=0.7,
                        zorder=10,
                        label=f"{path_id} Probe")

    # Create second y-axis for allocations
    ax1_allocation = ax1.twinx()

    # Plot allocation bars
    bar_width = time_step * 0.8
    for i, path_id in enumerate(path_ids):
        # Calculate bar offset so bars for different paths don't overlap
        offset = (i - (n_paths - 1) / 2) * (bar_width / n_paths)

        ax1_allocation.bar(time_points + offset, allocation_data[path_id],
                          width=bar_width / n_paths,
                          alpha=0.5,
                          color=path_colors[path_id],
                          label=f"{path_id} Allocation")

    # Add legend and labels
    ax1.set_title('Link Bandwidth, Traffic Allocation, and Network Probing Over Time')
    ax1.set_ylabel('Bandwidth (Mbps)')
    ax1.set_xticklabels([])  # Hide x-axis tick labels
    ax1.grid(True, alpha=0.3)

    ax1_allocation.set_ylabel('Allocated Packets')

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_allocation.get_legend_handles_labels()

    # Filter duplicate probe legend entries
    unique_labels = []
    unique_lines = []
    seen_probes = set()

    for line, label in zip(lines1, labels1):
        if "Probe" in label:
            path = label.split(" ")[0]
            if path not in seen_probes:
                seen_probes.add(path)
                unique_lines.append(line)
                unique_labels.append("Network Probe Point")
        else:
            unique_lines.append(line)
            unique_labels.append(label)

    ax1.legend(unique_lines + lines2, unique_labels + labels2, loc='upper right')

    # Second subplot: Packet loss and coding rate
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot packet loss curves and probe markers
    for path_id in path_ids:
        # Convert packet loss to percentage
        packet_loss_pct = np.array(packet_loss_data[path_id]) * 100

        # Plot packet loss curve
        ax2.plot(time_points, packet_loss_pct,
                 label=f"{path_id} Packet Loss",
                 color=path_colors[path_id],
                 linestyle='-',
                 linewidth=1.5)

        # Plot probe markers
        probe_times = [time_points[i] for i in range(len(time_points)) if probe_data[path_id][i]]
        probe_losses = [packet_loss_pct[i] for i in range(len(time_points)) if probe_data[path_id][i]]

        if probe_times:
            ax2.scatter(probe_times, probe_losses,
                        marker='o',
                        s=80,  # Marker size
                        color=path_colors[path_id],
                        edgecolors='red',
                        linewidths=1.5,
                        alpha=0.7,
                        zorder=10)

    # Create second y-axis for coding rate
    ax2_rate = ax2.twinx()

    # Plot coding rate curve
    ax2_rate.plot(time_points, coding_rates,
                  label='Coding Rate (R/N)',
                  color='purple',
                  linestyle='-',
                  linewidth=2.5)

    # Plot reward values
    ax2_rate.plot(time_points, rewards,
                  label='Reward',
                  color='navy',
                  linestyle=':',
                  linewidth=2)

    # Add legend and labels
    ax2.set_title('Link Packet Loss and Coding Decisions Over Time')
    ax2.set_xlabel('Time (minutes)')
    ax2.set_ylabel('Packet Loss (%)')
    ax2.grid(True, alpha=0.3)

    ax2_rate.set_ylabel('Coding Rate (R/N) and Reward')

    # Merge legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_rate.get_legend_handles_labels()

    # Create probe marker legend item
    probe_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
                             markeredgecolor='red', markersize=10, linewidth=0)

    # Merge legends
    ax2.legend(lines1 + lines2 + [probe_patch],
               labels1 + labels2 + ['Network Probe Point'],
               loc='upper right')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Leave space for time labels

    return fig, decisions


def main():
    """
    Main function to simulate and visualize network coding decisions
    across multiple maritime communication paths.
    """
    # Create the maritime paths container
    maritime_paths = Paths()

    # Create the first maritime path (Satellite Link)
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

    # Create the second maritime path (Coastal Radio)
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

    # Create the third maritime path (HF Link) with more volatile characteristics
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

    # Add paths to the container
    maritime_paths.add_path("Satellite Link", satellite_path, color="#1f77b4")
    maritime_paths.add_path("Coastal Radio", coastal_radio_path, color="#ff7f0e")
    maritime_paths.add_path("HF Link", hf_link_path, color="#2ca02c")

    # Print summary of all paths
    maritime_paths.print_all_paths_summary()

    # Create decision maker with specified parameters
    decision_maker = SimulatedRLNetworkCodingDecisionMaker(
        max_coding_packets=64,
        max_redundancy=24
    )

    # Visualize network coding decisions for 30 minutes
    fig, decisions = visualize_drl_network_coding_decisions(
        maritime_paths=maritime_paths,
        decision_maker=decision_maker,
        duration_minutes=30,
        time_step=0.5,  # Decision point every 0.5 minutes
        flow_demand=0.8  # 80% network flow demand
    )

    # Save the figure
    plt.savefig('network_coding_decisions.png', dpi=300, bbox_inches='tight')

    # Display the figure
    plt.show()

    # Print summary of some key decisions
    print("\nNetwork Coding Decision Summary:")
    print("=" * 80)

    # Select some time points to show results
    time_indices = [0, 20, 40, 60]  # At start, 10 min, 20 min, 30 min

    for idx in time_indices:
        if idx < len(decisions):
            decision = decisions[idx]
            time_point = decision['time_point']
            N, M, R = decision['coding_parameters']
            coding_rate = R / N if N > 0 else 0

            tx_result = decision.get('transmission_result', {})
            success_rate = tx_result.get('success_rate', 0)

            print(f"Time: {time_point:.1f} minutes")
            print(f"  Coding Parameters: N={N}, M={M}, R={R} (Coding Rate: {coding_rate:.3f})")
            print(f"  Allocation:")
            for path_id, alloc in sorted(decision['allocations'].items()):
                print(f"    {path_id}: {alloc} packets")
            print(f"  Network Probing:")
            for path_id, probed in sorted(decision['probed_paths'].items()):
                print(f"    {path_id}: {'Yes' if probed else 'No'}")
            if tx_result:
                print(f"  Transmission Success Rate: {success_rate:.2f}%")
            print(f"  Reward: {decision.get('reward', 0):.4f}")
            print("-" * 80)


if __name__ == "__main__":
    main()