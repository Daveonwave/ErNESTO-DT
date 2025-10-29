import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import warnings
from datetime import datetime


class ParameterSpaceVisualizer:
    """
    Stateless visualizer for parameter space time evolution in real-time adaptation scenarios.
    Handles interactive plotting of parameter estimation over time with cluster assignments.
    """
    def __init__(self, 
                 param_names: List[str],
                 domain_vars: List[str] = None,
                 interactive: bool = True,
                 max_points: int = 1000):
        """
        Initialize the visualizer.
        
        Args:
            param_names: List of parameter names (e.g., ['r0', 'r1', 'c1'])
            domain_vars: List of domain variable names (kept for compatibility)
            interactive: Enable interactive mode for real-time updates
            max_points: Maximum number of points to keep in memory for performance
        """
        self.param_names = param_names
        self.domain_vars = domain_vars or ['soc', 'temperature']
        self.figsize = (12, 3 * len(param_names))
        self.interactive = interactive
        self.max_points = max_points
        
        # Plot state
        self._fig = None
        self._axes = None
        self._lines = {}
        self._original_plotted = False
        
        # Data storage for plotting
        self._time_data = []
        self._param_data = {param: [] for param in param_names}
        self._cluster_data = []
        self._outlier_flags = []
        
        # Color mapping for clusters
        self._cluster_colors = {
            'original': 'blue',
            'inlier': 'green', 
            'outlier': 'red',
            'new_cluster': 'orange'
        }
        
        # Time tracking
        self._start_time = None
        self._current_time = 0
        
        if self.interactive:
            plt.ion()
    
    def setup_plot(self, title: str = "Parameter Evolution Over Time") -> Tuple[plt.Figure, List[plt.Axes]]:
        """
        Setup the matplotlib figure for time series visualization.
        
        Args:
            title: Main title for the figure
            
        Returns:
            Tuple of (figure, axes)
        """
        if self._fig is not None:
            plt.close(self._fig)
        
        # Create subplots - one for each parameter
        self._fig, self._axes = plt.subplots(len(self.param_names), 1, figsize=self.figsize, sharex=True)
        self._fig.suptitle(title, fontsize=16)
        
        # Handle single parameter case
        if len(self.param_names) == 1:
            self._axes = [self._axes]
        
        # Setup each subplot
        for i, (ax, param_name) in enumerate(zip(self._axes, self.param_names)):
            ax.set_ylabel(f'{param_name.upper()}')
            ax.set_title(f'{param_name.upper()} Evolution')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 100)  # Initial time range, will auto-adjust
            
            # Initialize empty lines for different cluster types
            self._lines[param_name] = {
                'original': ax.plot([], [], 'o-', color=self._cluster_colors['original'], 
                                  label='Original Cluster', markersize=4, alpha=0.7)[0],
                'inlier': ax.plot([], [], 'o-', color=self._cluster_colors['inlier'], 
                                label='Inlier', markersize=4, alpha=0.7)[0],
                'outlier': ax.plot([], [], 'x', color=self._cluster_colors['outlier'], 
                                 label='Outlier', markersize=6, alpha=0.8)[0],
                'new_cluster': ax.plot([], [], 's-', color=self._cluster_colors['new_cluster'], 
                                     label='New Cluster', markersize=4, alpha=0.7)[0]
            }
            
            ax.legend(loc='upper right', fontsize=8)
        
        # Set x-label only for the bottom subplot
        self._axes[-1].set_xlabel('Time (simulation steps)')
        
        plt.tight_layout()
        
        if self.interactive:
            plt.show(block=False)

        return self._fig, self._axes
    
    def plot_original_clusters(self, 
                             cluster_data: pd.DataFrame,
                             cluster_name: str = "Original",
                             **kwargs) -> None:
        """
        Plot original cluster data as initial reference points.
        For time series, we'll show these as horizontal reference lines.
        
        Args:
            cluster_data: DataFrame with parameter columns
            cluster_name: Name of the cluster for legend
        """
        if self._fig is None:
            self.setup_plot()
        
        # Calculate mean values for each parameter from original cluster
        for i, param_name in enumerate(self.param_names):
            if param_name.lower() in cluster_data.columns:
                param_mean = cluster_data[param_name.lower()].mean()
                param_std = cluster_data[param_name.lower()].std()
                
                # Add horizontal reference lines for mean ± std
                self._axes[i].axhline(y=param_mean, color=self._cluster_colors['original'], 
                                    linestyle='--', alpha=0.5, label=f'{cluster_name} Mean')
                self._axes[i].fill_between([0, 1], param_mean - param_std, param_mean + param_std, 
                                         color=self._cluster_colors['original'], alpha=0.2, 
                                         label=f'{cluster_name} ±1σ')
        
        self._original_plotted = True
        self._update_legends()
        
        if self.interactive:
            plt.draw()
            plt.pause(0.01)
    
    def add_estimated_point(self,
                            parameters: List[float] = None,
                            is_outlier: bool = False,
                            time_step: int = None) -> None:
        """
        Add a new estimated point to the time series visualization.
        
        Args:
            soc: State of charge value (kept for compatibility, not used in time series)
            temperature: Temperature value (kept for compatibility, not used in time series)
            parameters: List of parameter values
            is_outlier: Whether the point is an outlier
            point_label: Optional label for the point
            time_step: Time step for x-axis (if None, auto-increment)
        """
        if self._fig is None:
            warnings.warn("Plot not initialized. Call setup_plot() first.")
            return
        
        if parameters is None or len(parameters) != len(self.param_names):
            warnings.warn(f"Parameters list must have {len(self.param_names)} values.")
            return
        
        # Handle time tracking
        if time_step is not None:
            self._current_time = time_step
        else:
            self._current_time += 1
        
        # Store data
        self._time_data.append(self._current_time)
        for i, param_name in enumerate(self.param_names):
            if i < len(parameters):
                self._param_data[param_name].append(parameters[i])
        
        # Determine cluster type
        cluster_type = 'outlier' if is_outlier else 'inlier'
        self._cluster_data.append(cluster_type)
        self._outlier_flags.append(is_outlier)
        
        # Limit data size for performance
        # self._limit_data_size()
        
        if self.interactive:
            # Update plots
            self._update_time_series_plots()
            plt.draw()
            plt.pause(0.01)
    
    def _limit_data_size(self):
        """Limit the size of stored data for performance."""
        if len(self._time_data) > self.max_points:
            # Keep only the most recent max_points
            excess = len(self._time_data) - self.max_points
            
            self._time_data = self._time_data[excess:]
            self._cluster_data = self._cluster_data[excess:]
            self._outlier_flags = self._outlier_flags[excess:]
            
            for param_name in self.param_names:
                self._param_data[param_name] = self._param_data[param_name][excess:]
    
    def _update_time_series_plots(self):
        """Update the time series plots with current data."""
        if not self._time_data:
            return
        
        # Group data by cluster type for efficient plotting
        cluster_groups = {cluster_type: {'time': [], 'params': {param: [] for param in self.param_names}} 
                         for cluster_type in self._cluster_colors.keys()}
        
        for i, cluster_type in enumerate(self._cluster_data):
            cluster_groups[cluster_type]['time'].append(self._time_data[i])
            for param_name in self.param_names:
                if i < len(self._param_data[param_name]):
                    cluster_groups[cluster_type]['params'][param_name].append(self._param_data[param_name][i])
        
        # Update each parameter subplot
        for j, param_name in enumerate(self.param_names):
            # Update line data for each cluster type
            for cluster_type, line in self._lines[param_name].items():
                time_vals = cluster_groups[cluster_type]['time']
                param_vals = cluster_groups[cluster_type]['params'][param_name]
                
                if time_vals and param_vals:
                    line.set_data(time_vals, param_vals)
            
            # Auto-adjust axes limits
            if self._time_data and self._param_data[param_name]:
                self._axes[j].set_xlim(max(0, min(self._time_data) - 5), max(self._time_data) + 5)
                
                all_param_vals = self._param_data[param_name]
                if all_param_vals:
                    param_min, param_max = min(all_param_vals), max(all_param_vals)
                    param_range = param_max - param_min
                    margin = param_range * 0.1 if param_range > 0 else 0.1
                    self._axes[j].set_ylim(param_min - margin, param_max + margin)
    
    def add_multiple_points(self, points_data: List[Dict[str, Any]]) -> None:
        """
        Add multiple estimated points at once.
        
        Args:
            points_data: List of dictionaries containing point data
                        Each dict should have: 'parameters', 'is_outlier', optional 'time_step'
        """
        for i, point in enumerate(points_data):
            self.add_estimated_point(
                parameters=point['parameters'],
                is_outlier=point.get('is_outlier', False),
                point_label=point.get('label', None),
                time_step=point.get('time_step', None)
            )
    
    def _update_legends(self) -> None:
        """Update legends for all subplots."""
        for ax in self._axes:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = []
            unique_handles = []
            
            for handle, label in zip(handles, labels):
                if label not in unique_labels:
                    unique_labels.append(label)
                    unique_handles.append(handle)
            
            ax.legend(unique_handles, unique_labels, loc='upper right', fontsize=8)
    
    def create_summary_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create a summary plot showing parameter statistics over time.
        
        Returns:
            Tuple of (figure, axes)
        """
        if not self._time_data:
            warnings.warn("No data available for summary plot.")
            return None, None
        
        # Create summary figure
        fig_summary, ax_summary = plt.subplots(1, 1, figsize=(12, 8))
        fig_summary.suptitle('Parameter Evolution Summary', fontsize=16)
        
        # Calculate moving averages and confidence intervals
        window_size = min(50, len(self._time_data) // 4)  # Adaptive window size
        
        for i, param_name in enumerate(self.param_names):
            param_vals = np.array(self._param_data[param_name])
            time_vals = np.array(self._time_data)
            
            if len(param_vals) >= window_size:
                # Calculate moving average
                moving_avg = np.convolve(param_vals, np.ones(window_size)/window_size, mode='valid')
                time_moving = time_vals[window_size-1:]
                
                # Plot with different colors for each parameter
                color = plt.cm.tab10(i)
                ax_summary.plot(time_moving, moving_avg, label=f'{param_name.upper()} (Moving Avg)', 
                              color=color, linewidth=2)
                
                # Add scatter points for outliers
                outlier_times = [t for t, is_out in zip(time_vals, self._outlier_flags) if is_out]
                outlier_vals = [p for p, is_out in zip(param_vals, self._outlier_flags) if is_out]
                
                if outlier_times:
                    ax_summary.scatter(outlier_times, outlier_vals, color=color, marker='x', 
                                     s=50, alpha=0.7, label=f'{param_name.upper()} Outliers')
        
        ax_summary.set_xlabel('Time (simulation steps)')
        ax_summary.set_ylabel('Parameter Values')
        ax_summary.grid(True, alpha=0.3)
        ax_summary.legend()
        
        plt.tight_layout()
        plt.show()
        
        return fig_summary, ax_summary
    
    def create_heatmap(self, *args, **kwargs):
        """
        For time series visualization, create a summary plot instead of heatmap.
        """
        return self.create_summary_plot()
    
    def save_plots(self, 
                   filepath: str,
                   dpi: int = 300,
                   format: str = 'png') -> None:
        """
        Save the current plots to file.
        
        Args:
            filepath: Path to save the figure
            dpi: Resolution for saved figure
            format: File format ('png', 'pdf', 'svg', etc.)
        """
        if self._fig is not None:
            self._fig.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
    
    def clear_estimated_points(self) -> None:
        """Clear all estimated points but keep original reference lines."""
        self._time_data.clear()
        self._cluster_data.clear()
        self._outlier_flags.clear()
        for param_name in self.param_names:
            self._param_data[param_name].clear()
        
        # Clear line data
        if self._lines:
            for param_name in self.param_names:
                for line in self._lines[param_name].values():
                    line.set_data([], [])
        
        if self.interactive and self._fig:
            plt.draw()
            plt.pause(0.01)
    
    def export_data(self, filepath: str) -> None:
        """
        Export the collected time series data to CSV.
        
        Args:
            filepath: Path to save the CSV file
        """
        if not self._time_data:
            warnings.warn("No data to export.")
            return
        
        # Create DataFrame
        data_dict = {'time': self._time_data, 'cluster_type': self._cluster_data, 'is_outlier': self._outlier_flags}
        for param_name in self.param_names:
            data_dict[param_name] = self._param_data[param_name]
        
        df = pd.DataFrame(data_dict)
        df.to_csv(filepath, index=False)
        print(f"Data exported to {filepath}")
    
    def close(self) -> None:
        """Close all plots and disable interactive mode."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None
            self._lines = {}
        
        if self.interactive:
            plt.ioff()


def plot_estimation_step(visualizer: ParameterSpaceVisualizer,
                         soc: float,
                         temperature: float,
                         parameters: List[float],
                         is_outlier: bool = False,
                         time_step: int = None) -> None:
    """
    Convenience function to add a single estimation step to the time series visualizer.
    
    Args:
        visualizer: ParameterSpaceVisualizer instance
        soc: State of charge (kept for compatibility)
        temperature: Temperature (kept for compatibility)
        parameters: Parameter values
        is_outlier: Whether point is an outlier
        time_step: Time step for x-axis
    """
    visualizer.add_estimated_point(soc, temperature, parameters, is_outlier, time_step=time_step)