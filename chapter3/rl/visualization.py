import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional
from path import Path
from paths import Paths
from network_coding import NetworkCodingOptimizer


class NetworkCodingVisualizer:
    """
    A class for visualizing network coding decisions and transmission performance
    in maritime networks.
    """

    def __init__(self, optimizer: NetworkCodingOptimizer, step_size: float = 1.0):
        """
        Initialize the visualizer.

        Args:
            optimizer: NetworkCodingOptimizer instance
            step_size: Time step size for animation (minutes)
        """
        self.optimizer = optimizer
        self.maritime_paths = optimizer.maritime_paths
        self.path_ids = optimizer.path_ids
        self.step_size = step_size

        # Current time in simulation
        self.current_time = 0.0

        # History of network states and decisions
        self.time_history = []
        self.bandwidth_history = {path_id: [] for path_id in self.path_ids}
        self.packet_loss_history = {path_id: [] for path_id in self.path_ids}
        self.coding_params_history = []
        self.allocation_history = {path_id: [] for path_id in self.path_ids}
        self.probe_history = {path_id: [] for path_id in self.path_ids}

        # Transmission statistics
        self.transmission_success_history = []
        self.received_percentage_history = []
        self.energy_consumption_history = []

    def run_simulation_step(self, data_size_bytes: int = 100000) -> Dict:
        """
        Run a single simulation step and collect data for visualization.

        Args:
            data_size_bytes: Size of data to transmit

        Returns:
            Dictionary with results from this step
        """
        # Get current network state
        network_state = self.optimizer.get_current_network_state(self.current_time)

        # Store network state
        self.time_history.append(self.current_time)
        for path_id, state in network_state.items():
            self.bandwidth_history[path_id].append(state['bandwidth'])
            self.packet_loss_history[path_id].append(state['packet_loss'] * 100)  # To percentage

        # Simulate transmission
        tx_result = self.optimizer.simulate_transmission(self.current_time, data_size_bytes)

        # Store transmission results
        self.transmission_success_history.append(tx_result['transmission_success'])
        self.received_percentage_history.append(tx_result['received_percentage'])
        self.energy_consumption_history.append(tx_result['total_energy_consumed'])

        # Store coding parameters
        self.coding_params_history.append((
            tx_result['coding_parameters']['N'],
            tx_result['coding_parameters']['M'],
            tx_result['coding_parameters']['R']
        ))

        # Store allocation
        packet_allocation = tx_result.get('path_results', {})
        for path_id in self.path_ids:
            if path_id in packet_allocation:
                # Store packets sent
                self.allocation_history[path_id].append(packet_allocation[path_id].get('packets_sent', 0))
            else:
                self.allocation_history[path_id].append(0)

        # Check for probing decisions
        network_update = self.optimizer.update_network_view(self.current_time)
        probe_decisions = network_update['probe_decisions']
        for path_id, was_probed in probe_decisions.items():
            self.probe_history[path_id].append(was_probed)

        # Increment time
        self.current_time += self.step_size

        return tx_result

    def create_dashboard(self, figsize=(15, 10)) -> Tuple[plt.Figure, List]:
        """
        Create a dashboard visualization of the network coding system.

        Args:
            figsize: Size of the figure (width, height) in inches

        Returns:
            Tuple of (Figure, list of Axes)
        """
        # Create figure
        fig = plt.figure(figsize=figsize)

        # Create grid layout - Python 3.6 compatible version with explicit subplot creation
        # Avoid using GridSpec which might be causing the empty sequence error

        # Create subplots in a 3x2 grid
        ax_bandwidth = plt.subplot2grid((3, 2), (0, 0))
        ax_packet_loss = plt.subplot2grid((3, 2), (0, 1))
        ax_coding = plt.subplot2grid((3, 2), (1, 0))
        ax_allocation = plt.subplot2grid((3, 2), (1, 1))
        ax_performance = plt.subplot2grid((3, 2), (2, 0))
        ax_energy = plt.subplot2grid((3, 2), (2, 1))

        # List of all axes
        axes = [ax_bandwidth, ax_packet_loss, ax_coding, ax_allocation, ax_performance, ax_energy]

        return fig, axes

    def update_dashboard(self, fig: plt.Figure, axes: List, clear: bool = True) -> None:
        """
        Update the dashboard with current simulation data.

        Args:
            fig: Matplotlib Figure object
            axes: List of Axes objects
            clear: Whether to clear axes before plotting
        """
        if not self.time_history:
            return

        # Unpack axes
        ax_bandwidth, ax_packet_loss, ax_coding, ax_allocation, ax_performance, ax_energy = axes

        # Clear axes if requested
        if clear:
            for ax in axes:
                ax.clear()

        # Plot bandwidth
        for path_id in self.path_ids:
            path_bw = self.bandwidth_history[path_id]
            line, = ax_bandwidth.plot(self.time_history, path_bw, '-',
                                      label=f'{path_id}',
                                      color=self.maritime_paths.path_colors[path_id],
                                      linewidth=2)

            # Add probe markers
            probe_times = []
            probe_values = []
            for i, time in enumerate(self.time_history):
                if i < len(self.probe_history[path_id]) and self.probe_history[path_id][i]:
                    probe_times.append(time)
                    probe_values.append(path_bw[i])

            if probe_times:
                ax_bandwidth.plot(probe_times, probe_values, 'o',
                                  color=line.get_color(),
                                  markersize=6)

        ax_bandwidth.set_title('Bandwidth with Network Probes')
        ax_bandwidth.set_ylabel('Bandwidth (Mbps)')
        ax_bandwidth.grid(True, alpha=0.3)
        ax_bandwidth.legend(loc='best')

        # Plot packet loss
        for path_id in self.path_ids:
            path_pl = self.packet_loss_history[path_id]
            line, = ax_packet_loss.plot(self.time_history, path_pl, '-',
                                        label=f'{path_id}',
                                        color=self.maritime_paths.path_colors[path_id],
                                        linewidth=2)

            # Add probe markers
            probe_times = []
            probe_values = []
            for i, time in enumerate(self.time_history):
                if i < len(self.probe_history[path_id]) and self.probe_history[path_id][i]:
                    probe_times.append(time)
                    probe_values.append(path_pl[i])

            if probe_times:
                ax_packet_loss.plot(probe_times, probe_values, 'o',
                                    color=line.get_color(),
                                    markersize=6)

        ax_packet_loss.set_title('Packet Loss with Network Probes')
        ax_packet_loss.set_ylabel('Packet Loss (%)')
        ax_packet_loss.grid(True, alpha=0.3)
        ax_packet_loss.legend(loc='best')

        # Plot coding parameters
        N_values = [params[0] for params in self.coding_params_history]
        M_values = [params[1] for params in self.coding_params_history]
        R_values = [params[2] for params in self.coding_params_history]

        ax_coding.plot(self.time_history, N_values, 'b-', label='N (Data Packets)', linewidth=2)
        ax_coding.plot(self.time_history, R_values, 'r-', label='R (Redundancy)', linewidth=2)
        ax_coding.plot(self.time_history, M_values, 'g--', label='M (Total Packets)', linewidth=1.5)

        ax_coding.set_title('Network Coding Parameters')
        ax_coding.set_ylabel('Number of Packets')
        ax_coding.grid(True, alpha=0.3)
        ax_coding.legend(loc='best')

        # Plot allocation weights
        for path_id in self.path_ids:
            ax_allocation.plot(self.time_history, self.allocation_history[path_id], '-',
                               label=f'{path_id}',
                               color=self.maritime_paths.path_colors[path_id],
                               linewidth=2)

        ax_allocation.set_title('Packet Allocation')
        ax_allocation.set_ylabel('Packets Allocated')
        ax_allocation.grid(True, alpha=0.3)
        ax_allocation.legend(loc='best')

        # Plot transmission performance
        ax_performance.plot(self.time_history, self.received_percentage_history, 'b-',
                            label='Data Received (%)', linewidth=2)

        # Add markers for successful transmissions
        success_times = []
        success_values = []
        for i, time in enumerate(self.time_history):
            if self.transmission_success_history[i]:
                success_times.append(time)
                success_values.append(self.received_percentage_history[i])

        if success_times:
            ax_performance.plot(success_times, success_values, 'go',
                                label='Successful Transmissions',
                                markersize=6)

        # Add markers for failed transmissions
        fail_times = []
        fail_values = []
        for i, time in enumerate(self.time_history):
            if not self.transmission_success_history[i]:
                fail_times.append(time)
                fail_values.append(self.received_percentage_history[i])

        if fail_times:
            ax_performance.plot(fail_times, fail_values, 'ro',
                                label='Failed Transmissions',
                                markersize=6)

        ax_performance.set_title('Transmission Performance')
        ax_performance.set_ylabel('Data Received (%)')
        ax_performance.set_ylim(0, 105)
        ax_performance.grid(True, alpha=0.3)
        ax_performance.legend(loc='best')

        # Plot energy consumption
        ax_energy.plot(self.time_history, self.energy_consumption_history, 'g-',
                       label='Energy Consumed (J)', linewidth=2)

        ax_energy.set_title('Energy Consumption')
        ax_energy.set_ylabel('Energy (Joules)')
        ax_energy.grid(True, alpha=0.3)
        ax_energy.legend(loc='best')

        # Add x-labels to bottom plots
        ax_performance.set_xlabel('Time (minutes)')
        ax_energy.set_xlabel('Time (minutes)')

        # Adjust layout
        fig.tight_layout()

    def create_animation(self, duration: float, data_size_bytes: int = 100000,
                         interval: int = 200) -> FuncAnimation:
        """
        Create an animation of the network coding simulation.

        Args:
            duration: Total duration to simulate (minutes)
            data_size_bytes: Size of data to transmit at each step
            interval: Animation interval in milliseconds

        Returns:
            FuncAnimation object
        """
        # Reset state
        self.current_time = 0.0
        self.time_history = []
        self.bandwidth_history = {path_id: [] for path_id in self.path_ids}
        self.packet_loss_history = {path_id: [] for path_id in self.path_ids}
        self.coding_params_history = []
        self.allocation_history = {path_id: [] for path_id in self.path_ids}
        self.probe_history = {path_id: [] for path_id in self.path_ids}
        self.transmission_success_history = []
        self.received_percentage_history = []
        self.energy_consumption_history = []

        # Create dashboard
        fig, axes = self.create_dashboard()

        # Define update function for animation
        def update(frame):
            if self.current_time <= duration:
                self.run_simulation_step(data_size_bytes)
                self.update_dashboard(fig, axes)
            return axes

        # Create animation
        ani = FuncAnimation(fig, update, frames=int(duration / self.step_size) + 1,
                            interval=interval, blit=False, repeat=False)

        return ani

    def run_full_simulation(self, duration: float, data_size_bytes: int = 100000) -> plt.Figure:
        """
        Run a full simulation and display final dashboard.

        Args:
            duration: Total duration to simulate (minutes)
            data_size_bytes: Size of data to transmit at each step

        Returns:
            Matplotlib Figure object with final dashboard
        """
        # Reset state
        self.current_time = 0.0
        self.time_history = []
        self.bandwidth_history = {path_id: [] for path_id in self.path_ids}
        self.packet_loss_history = {path_id: [] for path_id in self.path_ids}
        self.coding_params_history = []
        self.allocation_history = {path_id: [] for path_id in self.path_ids}
        self.probe_history = {path_id: [] for path_id in self.path_ids}
        self.transmission_success_history = []
        self.received_percentage_history = []
        self.energy_consumption_history = []

        # Create dashboard
        fig, axes = self.create_dashboard()

        # Run simulation
        while self.current_time <= duration:
            self.run_simulation_step(data_size_bytes)

        # Update dashboard
        self.update_dashboard(fig, axes)

        return fig


# Example usage
if __name__ == "__main__":
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

    # Create network coding optimizer
    optimizer = NetworkCodingOptimizer(maritime_paths, probe_interval=5.0)

    # Create visualizer
    visualizer = NetworkCodingVisualizer(optimizer, step_size=1.0)

    # Run simulation for 60 minutes
    dashboard = visualizer.run_full_simulation(duration=60.0, data_size_bytes=100000)

    # Show dashboard
    plt.show()