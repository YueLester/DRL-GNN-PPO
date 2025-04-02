import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
import random
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

# Import the Path class from path module
from path import Path as Path


class Paths:
    """
    A class for managing multiple maritime communication paths.
    Provides functionality to visualize and compare multiple maritime links.
    """

    def __init__(self):
        """Initialize an empty Paths container for maritime links."""
        self.paths: Dict[str, Path] = {}
        self.path_colors: Dict[str, str] = {}
        self.path_styles: Dict[str, str] = {}
        # Define a default colormap for maritime path visualization
        self.default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                               '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def add_path(self, path_id: str, path: Path, color: Optional[str] = None,
                 line_style: str = '-') -> None:
        """
        Add a maritime path to the collection.

        Args:
            path_id: Unique identifier for the maritime path
            path: Path object to add
            color: Color code for plotting this path (optional)
            line_style: Line style for plotting this path (defaults to solid)
        """
        if path_id in self.paths:
            raise ValueError(f"Path ID '{path_id}' already exists")

        self.paths[path_id] = path

        # Assign color if provided, otherwise use default color cycle
        if color:
            self.path_colors[path_id] = color
        else:
            color_idx = len(self.paths) - 1
            self.path_colors[path_id] = self.default_colors[color_idx % len(self.default_colors)]

        self.path_styles[path_id] = line_style

    def remove_path(self, path_id: str) -> None:
        """
        Remove a maritime path from the collection.

        Args:
            path_id: Identifier of the path to remove
        """
        if path_id not in self.paths:
            raise ValueError(f"Path ID '{path_id}' does not exist")

        del self.paths[path_id]
        del self.path_colors[path_id]
        del self.path_styles[path_id]

    def get_path(self, path_id: str) -> Path:
        """
        Get a specific maritime path by ID.

        Args:
            path_id: Identifier of the path to retrieve

        Returns:
            Path object
        """
        if path_id not in self.paths:
            raise ValueError(f"Path ID '{path_id}' does not exist")

        return self.paths[path_id]

    def list_paths(self) -> List[str]:
        """
        Get a list of all maritime path IDs.

        Returns:
            List of path identifiers
        """
        return list(self.paths.keys())

    def plot_bandwidth_comparison(self, figsize: Tuple[int, int] = (12, 6),
                                  path_ids: Optional[List[str]] = None,
                                  title: str = "Maritime Links Bandwidth Comparison") -> Figure:
        """
        Plot bandwidth comparison between multiple maritime paths.

        Args:
            figsize: Size of the figure as (width, height) in inches
            path_ids: List of path IDs to include (defaults to all paths)
            title: Title for the plot

        Returns:
            Matplotlib Figure object
        """
        if not self.paths:
            raise ValueError("No maritime paths available to plot")

        # Use all paths if path_ids not specified
        if path_ids is None:
            path_ids = list(self.paths.keys())

        # Validate path_ids
        for path_id in path_ids:
            if path_id not in self.paths:
                raise ValueError(f"Path ID '{path_id}' does not exist")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot bandwidth for each path
        for path_id in path_ids:
            path = self.paths[path_id]
            ax.plot(path.time_points, path.bandwidth_series,
                    color=self.path_colors[path_id],
                    linestyle=self.path_styles[path_id],
                    linewidth=1.5,
                    label=f"{path_id}")

        # Add vertical lines for time slots (using the first path's time slot duration)
        first_path = self.paths[path_ids[0]]
        for slot in range(1, first_path.total_slots):
            slot_time = slot * first_path.time_slot_duration
            ax.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Bandwidth (Mbps)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

    def plot_packet_loss_comparison(self, figsize: Tuple[int, int] = (12, 6),
                                    path_ids: Optional[List[str]] = None,
                                    title: str = "Maritime Links Packet Loss Comparison") -> Figure:
        """
        Plot packet loss comparison between multiple maritime paths.

        Args:
            figsize: Size of the figure as (width, height) in inches
            path_ids: List of path IDs to include (defaults to all paths)
            title: Title for the plot

        Returns:
            Matplotlib Figure object
        """
        if not self.paths:
            raise ValueError("No maritime paths available to plot")

        # Use all paths if path_ids not specified
        if path_ids is None:
            path_ids = list(self.paths.keys())

        # Validate path_ids
        for path_id in path_ids:
            if path_id not in self.paths:
                raise ValueError(f"Path ID '{path_id}' does not exist")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot packet loss for each path
        for path_id in path_ids:
            path = self.paths[path_id]
            ax.plot(path.time_points, path.packet_loss_series * 100,  # Convert to percentage
                    color=self.path_colors[path_id],
                    linestyle=self.path_styles[path_id],
                    linewidth=1.5,
                    label=f"{path_id}")

        # Add vertical lines for time slots (using the first path's time slot duration)
        first_path = self.paths[path_ids[0]]
        for slot in range(1, first_path.total_slots):
            slot_time = slot * first_path.time_slot_duration
            ax.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Packet Loss (%)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

    def plot_energy_consumption_comparison(self, figsize: Tuple[int, int] = (12, 6),
                                           path_ids: Optional[List[str]] = None,
                                           title: str = "Maritime Links Energy Consumption Comparison") -> Figure:
        """
        Plot energy consumption comparison between multiple maritime paths.

        Args:
            figsize: Size of the figure as (width, height) in inches
            path_ids: List of path IDs to include (defaults to all paths)
            title: Title for the plot

        Returns:
            Matplotlib Figure object
        """
        if not self.paths:
            raise ValueError("No maritime paths available to plot")

        # Use all paths if path_ids not specified
        if path_ids is None:
            path_ids = list(self.paths.keys())

        # Validate path_ids
        for path_id in path_ids:
            if path_id not in self.paths:
                raise ValueError(f"Path ID '{path_id}' does not exist")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot energy consumption for each path
        for path_id in path_ids:
            path = self.paths[path_id]
            ax.plot(path.time_points, path.energy_consumption_series * 1000,  # Convert to mJ/byte
                    color=self.path_colors[path_id],
                    linestyle=self.path_styles[path_id],
                    linewidth=1.5,
                    label=f"{path_id}")

        # Add vertical lines for time slots (using the first path's time slot duration)
        first_path = self.paths[path_ids[0]]
        for slot in range(1, first_path.total_slots):
            slot_time = slot * first_path.time_slot_duration
            ax.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.3)

        ax.set_title(title)
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Energy Consumption (mJ/byte)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        plt.tight_layout()
        return fig

    def plot_combined_metrics(self, figsize: Tuple[int, int] = (15, 12),
                              path_ids: Optional[List[str]] = None) -> Figure:
        """
        Create a comprehensive plot showing bandwidth, packet loss, and energy consumption
        for multiple maritime paths.

        Args:
            figsize: Size of the figure as (width, height) in inches
            path_ids: List of path IDs to include (defaults to all paths)

        Returns:
            Matplotlib Figure object
        """
        if not self.paths:
            raise ValueError("No maritime paths available to plot")

        # Use all paths if path_ids not specified
        if path_ids is None:
            path_ids = list(self.paths.keys())

        # Validate path_ids
        for path_id in path_ids:
            if path_id not in self.paths:
                raise ValueError(f"Path ID '{path_id}' does not exist")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize)

        # Plot bandwidth for each path
        for path_id in path_ids:
            path = self.paths[path_id]
            ax1.plot(path.time_points, path.bandwidth_series,
                     color=self.path_colors[path_id],
                     linestyle=self.path_styles[path_id],
                     linewidth=1.5,
                     label=f"{path_id}")

        ax1.set_title('Maritime Links Bandwidth Comparison')
        ax1.set_ylabel('Bandwidth (Mbps)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='best')

        # Plot packet loss for each path
        for path_id in path_ids:
            path = self.paths[path_id]
            ax2.plot(path.time_points, path.packet_loss_series * 100,  # Convert to percentage
                     color=self.path_colors[path_id],
                     linestyle=self.path_styles[path_id],
                     linewidth=1.5,
                     label=f"{path_id}")

        ax2.set_title('Maritime Links Packet Loss Comparison')
        ax2.set_ylabel('Packet Loss (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best')

        # Plot energy consumption for each path
        for path_id in path_ids:
            path = self.paths[path_id]
            ax3.plot(path.time_points, path.energy_consumption_series * 1000,  # Convert to mJ/byte
                     color=self.path_colors[path_id],
                     linestyle=self.path_styles[path_id],
                     linewidth=1.5,
                     label=f"{path_id}")

        ax3.set_title('Maritime Links Energy Consumption Comparison')
        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Energy Consumption (mJ/byte)')
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')

        # Add vertical lines for time slots (using the first path's time slot duration)
        first_path = self.paths[path_ids[0]]
        for slot in range(1, first_path.total_slots):
            slot_time = slot * first_path.time_slot_duration
            ax1.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.3)
            ax2.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.3)
            ax3.axvline(x=slot_time, color='gray', linestyle='--', alpha=0.3)

        plt.tight_layout()
        return fig

    def compare_path_transmission(self, data_size: int, time_point: float,
                                  path_ids: Optional[List[str]] = None) -> Dict[str, Dict]:
        """
        Compare transmission performance across multiple maritime paths for a given data size
        and time point.

        Args:
            data_size: Size of data to transmit in bytes
            time_point: Time point in minutes for the transmission
            path_ids: List of path IDs to include (defaults to all paths)

        Returns:
            Dictionary with transmission results for each path
        """
        if not self.paths:
            raise ValueError("No maritime paths available to compare")

        # Use all paths if path_ids not specified
        if path_ids is None:
            path_ids = list(self.paths.keys())

        # Validate path_ids
        for path_id in path_ids:
            if path_id not in self.paths:
                raise ValueError(f"Path ID '{path_id}' does not exist")

        results = {}

        for path_id in path_ids:
            path = self.paths[path_id]

            # Convert time to index
            idx = int(round(time_point / path.total_duration * (path.total_points - 1)))

            # Get transmission results
            tx_results = path.transmit_bulk_data(data_size, idx)

            # Store results
            results[path_id] = {
                "bandwidth": path.bandwidth_series[idx],
                "packet_loss": path.packet_loss_series[idx],
                "energy_per_byte": path.energy_consumption_series[idx],
                "transmission_results": tx_results
            }

        return results

    def plot_transmission_comparison(self, data_size: int, time_point: float,
                                     path_ids: Optional[List[str]] = None,
                                     figsize: Tuple[int, int] = (10, 6)) -> Figure:
        """
        Plot comparison of transmission performance across multiple maritime paths.

        Args:
            data_size: Size of data to transmit in bytes
            time_point: Time point in minutes for the transmission
            path_ids: List of path IDs to include (defaults to all paths)
            figsize: Size of the figure as (width, height) in inches

        Returns:
            Matplotlib Figure object
        """
        # Get transmission results
        results = self.compare_path_transmission(data_size, time_point, path_ids)

        # Create bar plot
        fig, ax = plt.subplots(figsize=figsize)

        path_ids = list(results.keys())
        x = np.arange(len(path_ids))
        width = 0.2

        # Plot success rate
        success_rates = [results[path_id]["transmission_results"]["success_rate"] for path_id in path_ids]
        ax.bar(x - width, success_rates, width, label='Success Rate (%)', color='#2ca02c')

        # Plot energy consumption (normalized to make it comparable)
        energy_values = [results[path_id]["transmission_results"]["energy_consumed"] for path_id in path_ids]
        max_energy = max(energy_values)
        norm_energy = [e / max_energy * 100 for e in energy_values]
        ax.bar(x, norm_energy, width, label='Relative Energy (%)', color='#d62728')

        # Plot bytes received (as percentage of data_size)
        bytes_received = [results[path_id]["transmission_results"]["bytes_received"] / data_size * 100
                          for path_id in path_ids]
        ax.bar(x + width, bytes_received, width, label='Data Received (%)', color='#1f77b4')

        # Set labels and title
        ax.set_ylabel('Percentage')
        ax.set_title(f'Maritime Links Transmission Comparison ({data_size / 1000:.1f} KB at {time_point} min)')
        ax.set_xticks(x)
        ax.set_xticklabels(path_ids)
        ax.legend()

        # Add value labels on the bars
        for i, v in enumerate(success_rates):
            ax.text(i - width, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)

        for i, v in enumerate(norm_energy):
            ax.text(i, v + 2, f"{energy_values[i]:.1f}J", ha='center', va='bottom', fontsize=8)

        for i, v in enumerate(bytes_received):
            ax.text(i + width, v + 2, f"{v:.1f}%", ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        return fig

    def create_path_heatmap(self, metric: str = 'bandwidth',
                            figsize: Tuple[int, int] = (12, 6),
                            title: Optional[str] = None) -> Figure:
        """
        Create a heatmap visualization comparing all maritime paths over time for a specific metric.

        Args:
            metric: Metric to visualize ('bandwidth', 'packet_loss', or 'energy')
            figsize: Size of the figure as (width, height) in inches
            title: Custom title for the plot (optional)

        Returns:
            Matplotlib Figure object
        """
        if not self.paths:
            raise ValueError("No maritime paths available to compare")

        # Validate metric
        valid_metrics = ['bandwidth', 'packet_loss', 'energy']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric '{metric}'. Must be one of {valid_metrics}")

        path_ids = list(self.paths.keys())

        # Get the maximum time duration across all paths
        max_duration = max(path.total_duration for path in self.paths.values())

        # Create a normalized time grid
        # (Use the first path's time points as a reference)
        first_path = self.paths[path_ids[0]]
        time_points = first_path.time_points

        # Create a matrix to hold the data
        data_matrix = np.zeros((len(path_ids), len(time_points)))

        # Fill the matrix with the appropriate metric
        for i, path_id in enumerate(path_ids):
            path = self.paths[path_id]

            if metric == 'bandwidth':
                data_matrix[i, :] = path.bandwidth_series
            elif metric == 'packet_loss':
                data_matrix[i, :] = path.packet_loss_series * 100  # Convert to percentage
            elif metric == 'energy':
                data_matrix[i, :] = path.energy_consumption_series * 1000  # Convert to mJ/byte

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Choose color map based on metric
        if metric == 'bandwidth':
            cmap = plt.cm.viridis
            vmin = np.min(data_matrix)
            vmax = np.max(data_matrix)
            label = 'Bandwidth (Mbps)'
        elif metric == 'packet_loss':
            cmap = plt.cm.Reds
            vmin = 0
            vmax = np.max(data_matrix)
            label = 'Packet Loss (%)'
        else:  # energy
            cmap = plt.cm.plasma
            vmin = np.min(data_matrix)
            vmax = np.max(data_matrix)
            label = 'Energy (mJ/byte)'

        # Create heatmap
        im = ax.imshow(data_matrix, aspect='auto', cmap=cmap, interpolation='nearest',
                       extent=[0, max_duration, -0.5, len(path_ids) - 0.5],
                       origin='lower', vmin=vmin, vmax=vmax)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label)

        # Set labels and title
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Maritime Path')
        ax.set_yticks(range(len(path_ids)))
        ax.set_yticklabels(path_ids)

        if title:
            ax.set_title(title)
        else:
            metric_title = {
                'bandwidth': 'Bandwidth',
                'packet_loss': 'Packet Loss',
                'energy': 'Energy Consumption'
            }
            ax.set_title(f"Maritime Links: {metric_title[metric]} Comparison")

        # Add vertical lines for time slots (using the first path's time slot duration)
        for slot in range(1, first_path.total_slots):
            slot_time = slot * first_path.time_slot_duration
            ax.axvline(x=slot_time, color='white', linestyle='--', alpha=0.5)

        plt.tight_layout()
        return fig

    def print_all_paths_summary(self) -> None:
        """Print a summary of all maritime paths in the collection."""
        if not self.paths:
            print("No maritime paths available.")
            return

        print(f"Maritime Paths Summary ({len(self.paths)} paths):")
        print("=" * 60)

        for path_id, path in self.paths.items():
            print(f"Maritime Path: {path_id}")
            print(f"  Duration: {path.total_duration} minutes ({path.total_slots} slots)")
            print(
                f"  Avg Bandwidth: {np.mean(path.bandwidth_series):.2f} Mbps (Min: {np.min(path.bandwidth_series):.2f}, Max: {np.max(path.bandwidth_series):.2f})")
            print(
                f"  Avg Packet Loss: {np.mean(path.packet_loss_series) * 100:.2f}% (Min: {np.min(path.packet_loss_series) * 100:.2f}%, Max: {np.max(path.packet_loss_series) * 100:.2f}%)")
            print(f"  Avg Energy: {np.mean(path.energy_consumption_series) * 1000:.2f} mJ/byte")
            print("-" * 60)

        print("=" * 60)


# Example usage
if __name__ == "__main__":
    # Create paths container for maritime links
    maritime_paths = Paths()

    # Create first maritime path (Satellite Link) with 12 time slots (1 hour total)
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

    # Create second maritime path (Coastal Radio) with same time structure but different characteristics
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

    # Create third maritime path (HF Link) with more volatile characteristics
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

    # Print summary of all maritime paths
    maritime_paths.print_all_paths_summary()

    # Plot bandwidth comparison
    fig_bw = maritime_paths.plot_bandwidth_comparison(
        title="Maritime Communication Links: Bandwidth Comparison"
    )

    # Plot packet loss comparison
    fig_pl = maritime_paths.plot_packet_loss_comparison(
        title="Maritime Communication Links: Packet Loss Comparison"
    )

    # Plot combined metrics
    fig_combined = maritime_paths.plot_combined_metrics()

    # Create heatmap visualization for bandwidth
    fig_heatmap = maritime_paths.create_path_heatmap(
        metric='bandwidth',
        title="Bandwidth Heatmap Across Maritime Communication Links"
    )

    # Compare transmission performance at 30 minutes
    tx_results = maritime_paths.compare_path_transmission(
        data_size=100000,  # 100 KB
        time_point=30
    )

    # Plot transmission comparison
    fig_tx = maritime_paths.plot_transmission_comparison(
        data_size=100000,
        time_point=30
    )

    # Show all plots
    plt.show()

