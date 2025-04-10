import numpy as np
import matplotlib.pyplot as plt
from path import Path
from paths import Paths
from network_coding import NetworkCodingOptimizer
from visualization import NetworkCodingVisualizer


def run_basic_demo():
    """
    Run a basic demonstration of the maritime network coding algorithm.
    """
    print("Running basic maritime network coding demonstration...")

    # Create paths container with maritime paths
    maritime_paths = Paths()

    # Create three maritime paths with different characteristics
    avg_bandwidth_1 = [12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12]
    avg_packet_loss_1 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    fluctuation_level_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]

    satellite_path = Path(
        avg_bandwidth_series=avg_bandwidth_1,
        avg_packet_loss_series=avg_packet_loss_1,
        fluctuation_level_series=fluctuation_level_1,
        time_slot_duration=5,
        points_per_slot=300
    )

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

    # Print summary of maritime paths
    maritime_paths.print_all_paths_summary()

    # Create network coding optimizer
    optimizer = NetworkCodingOptimizer(maritime_paths, probe_interval=5.0)

    # Print initial network state and optimization parameters
    time_point = 0.0
    initial_state = optimizer.get_current_network_state(time_point)
    initial_update = optimizer.update_network_view(time_point, force_update=True)

    print("\nInitial Network State:")
    for path_id, state in initial_state.items():
        print(f"  {path_id}:")
        print(f"    Bandwidth: {state['bandwidth']:.2f} Mbps")
        print(f"    Packet Loss: {state['packet_loss'] * 100:.2f}%")
        print(f"    Energy: {state['energy_per_byte'] * 1000:.2f} mJ/byte")

    print("\nInitial Network Coding Parameters:")
    print(f"  N (Data Packets): {initial_update['coding_parameters']['N']}")
    print(f"  R (Redundancy): {initial_update['coding_parameters']['R']}")
    print(f"  M (Total Packets): {initial_update['coding_parameters']['M']}")
    print(
        f"  Redundancy Rate: {initial_update['coding_parameters']['R'] / initial_update['coding_parameters']['N']:.2f}")

    print("\nInitial Packet Allocation:")
    for path_id, packets in initial_update['packet_allocation'].items():
        print(f"  {path_id}: {packets} packets")

    # Simulate transmission at time 0
    tx_result = optimizer.simulate_transmission(time_point, 100000)  # 100 KB

    print("\nTransmission Results at time 0:")
    print(f"  Success: {tx_result['transmission_success']}")
    print(f"  Data Received: {tx_result['received_percentage']:.2f}%")
    print(f"  Energy Consumed: {tx_result['total_energy_consumed']:.2f} J")

    # Simulate transmission at specific time points of interest
    time_points = [15.0, 30.0, 45.0]

    for time in time_points:
        tx_result = optimizer.simulate_transmission(time, 100000)  # 100 KB

        print(f"\nTransmission Results at time {time}:")
        print(f"  Network Coding Parameters: N={tx_result['coding_parameters']['N']}, "
              f"R={tx_result['coding_parameters']['R']}, "
              f"M={tx_result['coding_parameters']['M']}")
        print(f"  Success: {tx_result['transmission_success']}")
        print(f"  Data Received: {tx_result['received_percentage']:.2f}%")
        print(f"  Energy Consumed: {tx_result['total_energy_consumed']:.2f} J")

        # Show per-path results
        print("  Per-path Results:")
        for path_id, path_result in tx_result['path_results'].items():
            print(f"    {path_id}:")
            print(f"      Packets Sent: {path_result['packets_sent']}")
            print(f"      Packets Received: {path_result['packets_received']}")
            print(f"      Success Rate: {path_result['success_rate']:.2f}%")


def run_visualization_demo():
    """
    Run a demo with visualization of the maritime network coding system.
    """
    print("Running maritime network coding visualization demo...")

    # Create paths container with maritime paths
    maritime_paths = Paths()

    # Create maritime paths (same as in basic demo)
    avg_bandwidth_1 = [12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12]
    avg_packet_loss_1 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    fluctuation_level_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]
    satellite_path = Path(
        avg_bandwidth_series=avg_bandwidth_1,
        avg_packet_loss_series=avg_packet_loss_1,
        fluctuation_level_series=fluctuation_level_1,
        time_slot_duration=5,
        points_per_slot=300
    )

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

    # Create visualizer
    visualizer = NetworkCodingVisualizer(optimizer, step_size=1.0)

    print("Running full simulation for 60 minutes...")
    # Run simulation for 60 minutes
    dashboard = visualizer.run_full_simulation(duration=60.0, data_size_bytes=100000)

    # Display dashboard
    plt.show()

    # Summary of simulation results
    print("\nSimulation completed!")
    print(f"Total time steps: {len(visualizer.time_history)}")

    # Calculate overall transmission success rate
    success_rate = sum(visualizer.transmission_success_history) / len(visualizer.transmission_success_history) * 100
    print(f"Overall transmission success rate: {success_rate:.2f}%")

    # Calculate average data received percentage
    avg_data_received = sum(visualizer.received_percentage_history) / len(visualizer.received_percentage_history)
    print(f"Average data received percentage: {avg_data_received:.2f}%")

    # Calculate total energy consumption
    total_energy = sum(visualizer.energy_consumption_history)
    print(f"Total energy consumption: {total_energy:.2f} J")

    # Plot coding parameters over time
    optimizer.plot_coding_parameters_history()
    plt.title("Network Coding Parameters Over Time")

    # Plot allocation weights over time
    optimizer.plot_allocation_weights_history()
    plt.title("Path Allocation Weights Over Time")

    # Show plots
    plt.show()


def compare_with_without_network_coding():
    """
    Compare performance with and without network coding.
    """
    print("Comparing performance with and without network coding...")

    # Create paths container with maritime paths
    maritime_paths = Paths()

    # Create maritime paths (same as in previous demos)
    avg_bandwidth_1 = [12, 11, 10, 9, 8, 7, 7, 8, 9, 10, 11, 12]
    avg_packet_loss_1 = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03]
    fluctuation_level_1 = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3, 4]
    satellite_path = Path(
        avg_bandwidth_series=avg_bandwidth_1,
        avg_packet_loss_series=avg_packet_loss_1,
        fluctuation_level_series=fluctuation_level_1,
        time_slot_duration=5,
        points_per_slot=300
    )

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

    # Simulation parameters
    duration = 60.0
    step_size = 1.0
    data_size = 100000  # 100 KB

    # With network coding (adaptive)
    optimizer = NetworkCodingOptimizer(maritime_paths, probe_interval=5.0)

    # Run simulation with network coding
    nc_success_history = []
    nc_received_percentage_history = []
    nc_energy_history = []

    current_time = 0.0
    while current_time <= duration:
        result = optimizer.simulate_transmission(current_time, data_size)

        nc_success_history.append(result["transmission_success"])
        nc_received_percentage_history.append(result["received_percentage"])
        nc_energy_history.append(result["total_energy_consumed"])

        current_time += step_size

    # Without network coding (fixed parameters)
    class SimpleTransmissionSimulator:
        def __init__(self, maritime_paths):
            self.maritime_paths = maritime_paths
            self.path_ids = maritime_paths.list_paths()

        def simulate_transmission(self, current_time, data_size):
            """Simple transmission without network coding"""
            # Use equal allocation across all paths
            path_allocation = {path_id: 1.0 / len(self.path_ids) for path_id in self.path_ids}

            # Track transmission statistics
            total_packets_sent = 0
            total_packets_received = 0
            total_energy_consumed = 0.0

            # Simulate transmission on each path
            for path_id in self.path_ids:
                path = self.maritime_paths.get_path(path_id)

                # Calculate data for this path
                path_data_bytes = int(data_size * path_allocation[path_id])

                # Find time index
                idx = int(round(current_time / path.total_duration * (path.total_points - 1)))

                # Simulate transmission
                tx_result = path.transmit_bulk_data(path_data_bytes, idx)

                # Update total statistics
                total_packets_sent += tx_result["packets_sent"]
                total_packets_received += tx_result["packets_received"]
                total_energy_consumed += tx_result["energy_consumed"]

            # Calculate received percentage
            if total_packets_sent > 0:
                received_percentage = (total_packets_received / total_packets_sent) * 100
            else:
                received_percentage = 0.0

            # Define success threshold (arbitrary but reasonable)
            transmission_success = received_percentage >= 90.0

            return {
                "transmission_success": transmission_success,
                "received_percentage": received_percentage,
                "total_energy_consumed": total_energy_consumed
            }

    # Create simple transmission simulator
    simple_simulator = SimpleTransmissionSimulator(maritime_paths)

    # Run simulation without network coding
    simple_success_history = []
    simple_received_percentage_history = []
    simple_energy_history = []

    current_time = 0.0
    while current_time <= duration:
        result = simple_simulator.simulate_transmission(current_time, data_size)

        simple_success_history.append(result["transmission_success"])
        simple_received_percentage_history.append(result["received_percentage"])
        simple_energy_history.append(result["total_energy_consumed"])

        current_time += step_size

    # Compare results
    nc_success_rate = sum(nc_success_history) / len(nc_success_history) * 100
    simple_success_rate = sum(simple_success_history) / len(simple_success_history) * 100

    nc_avg_received = sum(nc_received_percentage_history) / len(nc_received_percentage_history)
    simple_avg_received = sum(simple_received_percentage_history) / len(simple_received_percentage_history)

    nc_total_energy = sum(nc_energy_history)
    simple_total_energy = sum(simple_energy_history)

    print("\nComparison of Results:")
    print(f"  With Network Coding:")
    print(f"    Success Rate: {nc_success_rate:.2f}%")
    print(f"    Average Data Received: {nc_avg_received:.2f}%")
    print(f"    Total Energy Consumed: {nc_total_energy:.2f} J")

    print(f"\n  Without Network Coding:")
    print(f"    Success Rate: {simple_success_rate:.2f}%")
    print(f"    Average Data Received: {simple_avg_received:.2f}%")
    print(f"    Total Energy Consumed: {simple_total_energy:.2f} J")

    print(f"\n  Improvement with Network Coding:")
    print(f"    Success Rate: {nc_success_rate - simple_success_rate:.2f}%")
    print(f"    Data Received: {nc_avg_received - simple_avg_received:.2f}%")
    print(
        f"    Energy Savings: {simple_total_energy - nc_total_energy:.2f} J ({(1 - nc_total_energy / simple_total_energy) * 100:.2f}%)")

    # Plot comparison
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    time_points = np.arange(0, duration + step_size, step_size)

    # Plot transmission success
    ax1.plot(time_points, nc_received_percentage_history, 'b-', label='With Network Coding', linewidth=2)
    ax1.plot(time_points, simple_received_percentage_history, 'r--', label='Without Network Coding', linewidth=2)
    ax1.set_title('Data Received Percentage')
    ax1.set_ylabel('Data Received (%)')
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot energy consumption
    ax2.plot(time_points, nc_energy_history, 'b-', label='With Network Coding', linewidth=2)
    ax2.plot(time_points, simple_energy_history, 'r--', label='Without Network Coding', linewidth=2)
    ax2.set_title('Energy Consumption')
    ax2.set_ylabel('Energy (Joules)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot cumulative energy consumption
    nc_cumulative_energy = np.cumsum(nc_energy_history)
    simple_cumulative_energy = np.cumsum(simple_energy_history)

    ax3.plot(time_points, nc_cumulative_energy, 'b-', label='With Network Coding', linewidth=2)
    ax3.plot(time_points, simple_cumulative_energy, 'r--', label='Without Network Coding', linewidth=2)
    ax3.set_title('Cumulative Energy Consumption')
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Energy (Joules)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.show()

    return {
        "with_network_coding": {
            "success_rate": nc_success_rate,
            "avg_received": nc_avg_received,
            "total_energy": nc_total_energy
        },
        "without_network_coding": {
            "success_rate": simple_success_rate,
            "avg_received": simple_avg_received,
            "total_energy": simple_total_energy
        }
    }


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("MARITIME NETWORK CODING DEMONSTRATION")
    print("=" * 50 + "\n")

    # Choose which demos to run
    run_basic = True
    run_visualization = True
    run_comparison = True

    if run_basic:
        print("\n" + "=" * 50)
        print("BASIC DEMO")
        print("=" * 50)
        run_basic_demo()

    if run_visualization:
        print("\n" + "=" * 50)
        print("VISUALIZATION DEMO")
        print("=" * 50)
        run_visualization_demo()

    if run_comparison:
        print("\n" + "=" * 50)
        print("COMPARISON DEMO")
        print("=" * 50)
        compare_with_without_network_coding()