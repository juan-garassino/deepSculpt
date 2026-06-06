"""
3D Visualization Tools for DeepSculpt
This module provides multiple visualization techniques for 3D voxel-based sculptures,
including static 3D plots, 2D cross-sections, interactive point clouds, and animated
rotations. It handles color mapping, file output, and view customization.

Key features:
- Multi-view 3D plots: Visualize sculptures from different angles
- Cross-sectional views: Display 2D slices of 3D structures
- Point cloud rendering: Interactive visualization using Plotly
- Animation: Create rotating views as animated GIFs
- File output: Save visualizations in various formats
- Directory sampling: Visualize random samples from a directory
- PyTorch tensor support: Automatic tensor-to-numpy conversion for visualization

Dependencies:
- logger.py: For operation tracking and status reporting
- utils.py: For array transformations and data preparation
- numpy: For array manipulation
- matplotlib: For 3D and 2D plotting
- plotly: For interactive point cloud visualization
- datetime: For timestamped file naming
- torch: For PyTorch tensor support
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from deepsculpt.core.utils.logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)


def _tensor_to_numpy(tensor_or_array: Union[np.ndarray, 'torch.Tensor']) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array for visualization compatibility.
    
    Args:
        tensor_or_array: Input tensor or array
        
    Returns:
        numpy array
    """
    if not TORCH_AVAILABLE:
        return tensor_or_array
    
    if torch.is_tensor(tensor_or_array):
        # Move to CPU if on GPU and convert to numpy
        if tensor_or_array.is_cuda:
            log_info("Moving tensor from GPU to CPU for visualization")
        return tensor_or_array.detach().cpu().numpy()
    else:
        return tensor_or_array


def _is_torch_tensor(obj: Any) -> bool:
    """Check if object is a PyTorch tensor."""
    return TORCH_AVAILABLE and torch.is_tensor(obj)


def _get_tensor_device(tensor: 'torch.Tensor') -> str:
    """Get the device of a PyTorch tensor."""
    if not TORCH_AVAILABLE or not torch.is_tensor(tensor):
        return "cpu"
    return str(tensor.device)


class Visualizer:
    """
    A class for visualizing 3D shapes and sculptures.
    """

    def __init__(
        self,
        figsize: int = 25,
        style: str = "#ffffff",
        dpi: int = 100,
        transparent: bool = False,
    ):
        """
        Initialize a new Visualizer instance.

        Args:
            figsize: Figure size for matplotlib plots
            style: Background color for plots
            dpi: DPI for raster images
            transparent: Whether to use transparent backgrounds
        """
        self.figsize = figsize
        self.style = style
        self.dpi = dpi
        self.transparent = transparent

    def plot_sections(
        self,
        structure: Union[np.ndarray, 'torch.Tensor'],
        title: str = "Structure Sections",
        cmap: str = "gray",
        show: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot 2D sections of a 3D structure.

        Args:
            structure: 3D numpy array or PyTorch tensor to visualize
            title: Title for the plot
            cmap: Colormap to use
            show: Whether to display the plot (default: False)
            save_path: Path to save the plot (if None, plot is not saved)

        Returns:
            The matplotlib figure object
        """
        # Convert PyTorch tensor to numpy if needed
        if _is_torch_tensor(structure):
            device = _get_tensor_device(structure)
            log_info(f"Converting PyTorch tensor from {device} to numpy for visualization")
            structure = _tensor_to_numpy(structure)
        
        begin_section(f"Plotting {structure.shape[0]} sections")

        try:
            # Calculate grid dimensions
            n_sections = structure.shape[0]
            n_cols = min(6, n_sections)
            n_rows = int(np.ceil(n_sections / n_cols))

            # Create figure and axes
            fig, axes = plt.subplots(
                ncols=n_cols,
                nrows=n_rows,
                figsize=(self.figsize, self.figsize),
                facecolor=self.style,
                dpi=self.dpi,
            )

            # Ensure axes is an array even when there's only one subplot
            if n_sections == 1:
                axes = np.array([axes])

            # Flatten the axes array for easier indexing
            axes = axes.ravel()

            # Plot each section
            for i in range(n_sections):
                axes[i].imshow(structure[i, :, :], cmap=cmap)
                axes[i].set_title(f"Section {i}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])

            # Hide any unused subplots
            for i in range(n_sections, len(axes)):
                axes[i].axis("off")

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved sections plot to {save_path}")

            # Show if requested
            if show:
                plt.show()
            else:
                plt.close(fig)

            log_success("Sections plot created successfully")
            end_section()

            return fig

        except Exception as e:
            log_error(f"Error plotting sections: {str(e)}")
            end_section("Section plotting failed")
            raise

    def plot_sculpture(
        self,
        structure: Union[np.ndarray, 'torch.Tensor'],
        colors: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        title: str = "3D Sculpture",
        angles: List[int] = [0, 1, 2, 3],
        hide_axis: bool = False,
        linewidth: float = 0.05,
        show: bool = False,
        save_path: Optional[str] = None,
        save_array: bool = False,
        save_dir: str = ".",
    ):
        """
        Plot a 3D sculpture from different angles.

        Args:
            structure: 3D numpy array or PyTorch tensor representing the sculpture
            colors: 3D numpy array or PyTorch tensor with color information (optional)
            title: Title for the plot
            angles: List of rotation angles to show (in 90° increments)
            hide_axis: Whether to hide the axes
            linewidth: Width of the edges
            show: Whether to display the plot (default: False)
            save_path: Path to save the plot (if None, plot is not saved)
            save_array: Whether to save the structure and color arrays
            save_dir: Directory to save arrays if save_array is True

        Returns:
            The matplotlib figure object
        """
        # Convert PyTorch tensors to numpy if needed
        original_structure = structure
        original_colors = colors
        
        if _is_torch_tensor(structure):
            device = _get_tensor_device(structure)
            log_info(f"Converting structure tensor from {device} to numpy for visualization")
            structure = _tensor_to_numpy(structure)
            
        if colors is not None and _is_torch_tensor(colors):
            device = _get_tensor_device(colors)
            log_info(f"Converting colors tensor from {device} to numpy for visualization")
            colors = _tensor_to_numpy(colors)
        
        # Handle tensor dimensions - ensure we have a 3D array
        if structure.ndim == 4:
            # Remove channel dimension if present [C, D, H, W] -> [D, H, W]
            if structure.shape[0] == 1:
                structure = structure.squeeze(0)
                log_info(f"Squeezed channel dimension: {structure.shape}")
            else:
                # Take first channel if multiple channels
                structure = structure[0]
                log_info(f"Took first channel: {structure.shape}")
        elif structure.ndim == 5:
            # Remove batch and channel dimensions [B, C, D, H, W] -> [D, H, W]
            structure = structure.squeeze()
            if structure.ndim > 3:
                structure = structure[0] if structure.ndim == 4 else structure[0, 0]
            log_info(f"Squeezed batch/channel dimensions: {structure.shape}")
        
        # Handle colors similarly
        if colors is not None:
            if colors.ndim == 4:
                if colors.shape[0] == 1:
                    colors = colors.squeeze(0)
                else:
                    colors = colors[0]
            elif colors.ndim == 5:
                colors = colors.squeeze()
                if colors.ndim > 3:
                    colors = colors[0] if colors.ndim == 4 else colors[0, 0]
        
        # Ensure we have a 3D structure
        if structure.ndim != 3:
            raise ValueError(f"Structure must be 3-dimensional after processing, got shape {structure.shape}")
        
        log_info(f"Final structure shape for visualization: {structure.shape}")
        
        begin_section(f"Plotting 3D sculpture with shape {structure.shape}")

        try:
            # Calculate grid dimensions
            n_views = len(angles)
            n_cols = min(2, n_views)
            n_rows = int(np.ceil(n_views / n_cols))

            # Create figure and axes
            fig, axes = plt.subplots(
                ncols=n_cols,
                nrows=n_rows,
                figsize=(self.figsize, self.figsize),
                facecolor=self.style,
                subplot_kw=dict(projection="3d"),
                dpi=self.dpi,
            )

            # Ensure axes is an array even when there's only one subplot
            if n_views == 1:
                axes = np.array([axes])

            # Flatten the axes array for easier indexing
            axes = axes.ravel()

            # Plot each angle
            for i, angle in enumerate(angles):
                if i < len(axes):
                    # Hide axis if requested
                    if hide_axis:
                        axes[i].set_axis_off()

                    # Rotate the structure
                    rotated_structure = np.rot90(structure, angle)

                    # Plot with colors if provided
                    if colors is not None:
                        rotated_colors = np.rot90(colors, angle)
                        axes[i].voxels(
                            rotated_structure,
                            facecolors=rotated_colors,
                            edgecolors="k",
                            linewidth=linewidth,
                        )
                        log_info(f"Plotted view {i} (rotation {angle*90}°) with colors")
                    else:
                        axes[i].voxels(
                            rotated_structure,
                            facecolors="#E87F2A",
                            edgecolors="#C06A20",
                            linewidth=linewidth,
                        )
                        log_info(
                            f"Plotted view {i} (rotation {angle*90}°) without colors"
                        )

                    axes[i].set_title(f"Rotation {angle*90}°")

            # Hide any unused subplots
            for i in range(n_views, len(axes)):
                axes[i].axis("off")

            plt.suptitle(title, fontsize=16)
            plt.tight_layout()

            # Save if requested
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved sculpture plot to {save_path}")

            # Save arrays if requested
            if save_array:
                timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

                # Create directories if they don't exist
                structure_dir = os.path.join(save_dir, "structure_array")
                os.makedirs(structure_dir, exist_ok=True)

                # Save structure array (use original tensor if available)
                structure_path = os.path.join(
                    structure_dir, f"structure_array_{timestamp}.npy"
                )
                if _is_torch_tensor(original_structure):
                    # Save as PyTorch tensor
                    torch_path = structure_path.replace('.npy', '.pt')
                    torch.save(original_structure, torch_path)
                    log_success(f"Saved structure tensor to {torch_path}")
                    # Also save numpy version for compatibility
                    np.save(structure_path, structure)
                    log_success(f"Saved structure array to {structure_path}")
                else:
                    np.save(structure_path, structure)
                    log_success(f"Saved structure array to {structure_path}")

                # Save color array if provided (use original tensor if available)
                if original_colors is not None:
                    colors_dir = os.path.join(save_dir, "colors_array")
                    os.makedirs(colors_dir, exist_ok=True)
                    colors_path = os.path.join(
                        colors_dir, f"colors_array_{timestamp}.npy"
                    )
                    if _is_torch_tensor(original_colors):
                        # Save as PyTorch tensor
                        torch_path = colors_path.replace('.npy', '.pt')
                        torch.save(original_colors, torch_path)
                        log_success(f"Saved colors tensor to {torch_path}")
                        # Also save numpy version for compatibility
                        np.save(colors_path, colors)
                        log_success(f"Saved colors array to {colors_path}")
                    else:
                        np.save(colors_path, colors)
                        log_success(f"Saved colors array to {colors_path}")

            # Show if requested
            if show:
                plt.show()
            else:
                plt.close(fig)

            log_success("Sculpture plot created successfully")
            end_section()

            return fig

        except Exception as e:
            log_error(f"Error plotting sculpture: {str(e)}")
            end_section("Sculpture plotting failed")
            raise

    def plot_single_view(
        self,
        structure: Union[np.ndarray, 'torch.Tensor'],
        colors: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        angle: int = 0,
        figsize: Optional[Tuple[int, int]] = None,
        hide_axis: bool = False,
        linewidth: float = 0.05,
        title: str = "3D View",
        show: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot a single view of a 3D sculpture.

        Args:
            structure: 3D numpy array or PyTorch tensor representing the sculpture
            colors: 3D numpy array or PyTorch tensor with color information (optional)
            angle: Rotation angle in 90° increments
            figsize: Figure size as (width, height) tuple
            hide_axis: Whether to hide the axes
            linewidth: Width of the edges
            title: Title for the plot
            show: Whether to display the plot (default: False)
            save_path: Path to save the plot (if None, plot is not saved)

        Returns:
            The matplotlib figure object
        """
        # Convert PyTorch tensors to numpy if needed
        if _is_torch_tensor(structure):
            device = _get_tensor_device(structure)
            log_info(f"Converting structure tensor from {device} to numpy for visualization")
            structure = _tensor_to_numpy(structure)
            
        if colors is not None and _is_torch_tensor(colors):
            device = _get_tensor_device(colors)
            log_info(f"Converting colors tensor from {device} to numpy for visualization")
            colors = _tensor_to_numpy(colors)
        
        begin_section(f"Plotting single 3D view (rotation {angle*90}°)")

        try:
            # Use default figsize if none provided
            if figsize is None:
                figsize = (self.figsize // 2, self.figsize // 2)

            # Create figure and axis
            fig = plt.figure(figsize=figsize, facecolor=self.style, dpi=self.dpi)
            ax = fig.add_subplot(111, projection="3d")

            # Hide axis if requested
            if hide_axis:
                ax.set_axis_off()

            # Rotate the structure
            rotated_structure = np.rot90(structure, angle)

            # Plot with colors if provided
            if colors is not None:
                rotated_colors = np.rot90(colors, angle)
                ax.voxels(
                    rotated_structure,
                    facecolors=rotated_colors,
                    edgecolors="k",
                    linewidth=linewidth,
                )
                log_info(f"Plotted with colors")
            else:
                ax.voxels(
                    rotated_structure,
                    facecolors="#E87F2A",
                    edgecolors="#C06A20",
                    linewidth=linewidth,
                )
                log_info(f"Plotted without colors")

            plt.title(title)

            # Save if requested
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved single view plot to {save_path}")

            # Show if requested
            if show:
                plt.show()
            else:
                plt.close(fig)

            log_success("Single view plot created successfully")
            end_section()

            return fig

        except Exception as e:
            log_error(f"Error plotting single view: {str(e)}")
            end_section("Single view plotting failed")
            raise

    @staticmethod
    def voxel_to_pointcloud(structure: Union[np.ndarray, 'torch.Tensor'], subdivision: int = 3) -> np.ndarray:
        """
        Convert a voxel grid to a point cloud.

        Args:
            structure: 3D numpy array or PyTorch tensor representing the voxel grid
            subdivision: Number of points to generate per voxel dimension

        Returns:
            Nx3 numpy array of point coordinates
        """
        # Convert PyTorch tensor to numpy if needed
        if _is_torch_tensor(structure):
            device = _get_tensor_device(structure)
            log_info(f"Converting structure tensor from {device} to numpy for point cloud conversion")
            structure = _tensor_to_numpy(structure)
        
        begin_section(f"Converting voxel grid to point cloud")

        try:
            n_x, n_y, n_z = structure.shape
            points = []

            # Create points for each filled voxel
            for i in range(n_x):
                for j in range(n_y):
                    for k in range(n_z):
                        if structure[i, j, k]:
                            # Create a grid of points within this voxel
                            x = np.linspace(i, i + 1, subdivision + 1)[:-1]
                            y = np.linspace(j, j + 1, subdivision + 1)[:-1]
                            z = np.linspace(k, k + 1, subdivision + 1)[:-1]

                            # Create all combinations of x, y, z coordinates
                            for xi in x:
                                for yi in y:
                                    for zi in z:
                                        points.append([xi, yi, zi])

            # Convert to numpy array and return
            points_array = np.array(points)

            log_success(f"Created point cloud with {len(points_array)} points")
            end_section()

            return points_array

        except Exception as e:
            log_error(f"Error converting to point cloud: {str(e)}")
            end_section("Point cloud conversion failed")
            raise

    def plot_pointcloud(
        self,
        points: Union[np.ndarray, 'torch.Tensor'],
        colors: Optional[Union[np.ndarray, 'torch.Tensor', Tuple[float, float, float]]] = None,
        size: float = 1.0,
        alpha: float = 1.0,
        title: str = "3D Point Cloud",
        show: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot a 3D point cloud using Plotly.

        Args:
            points: Nx3 numpy array or PyTorch tensor of point coordinates
            colors: Point colors (Nx3 array/tensor or single RGB tuple)
            size: Point size
            alpha: Point opacity
            title: Title for the plot
            show: Whether to display the plot (default: False)
            save_path: Path to save the plot (if None, plot is not saved)

        Returns:
            The plotly figure object
        """
        # Convert PyTorch tensors to numpy if needed
        if _is_torch_tensor(points):
            device = _get_tensor_device(points)
            log_info(f"Converting points tensor from {device} to numpy for visualization")
            points = _tensor_to_numpy(points)
            
        if colors is not None and _is_torch_tensor(colors):
            device = _get_tensor_device(colors)
            log_info(f"Converting colors tensor from {device} to numpy for visualization")
            colors = _tensor_to_numpy(colors)
        
        begin_section(f"Plotting point cloud with {len(points)} points")

        try:
            # Extract coordinates
            x, y, z = points[:, 0], points[:, 1], points[:, 2]

            # If colors is a single RGB tuple, convert it to the right format
            if colors is None:
                colors = (0, 0, 0)  # Default to black

            if not isinstance(colors, np.ndarray):
                color_str = f"rgba({colors[0]}, {colors[1]}, {colors[2]}, {alpha})"
                marker_dict = dict(size=size, color=color_str)
            else:
                # Assume colors is an array of RGB values for each point
                color_array = [f"rgba({r}, {g}, {b}, {alpha})" for r, g, b in colors]
                marker_dict = dict(size=size, color=color_array)

            # Create the scatter3d trace
            trace = go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="markers",
                marker=marker_dict,
                name="Points",
            )

            # Set the layout of the plot
            layout = go.Layout(
                title=title,
                scene=dict(
                    aspectratio=dict(x=1, y=1, z=1),
                    xaxis=dict(title="X"),
                    yaxis=dict(title="Y"),
                    zaxis=dict(title="Z"),
                ),
                width=1200,
                height=800,
                margin=dict(l=0, r=0, b=0, t=30),
            )

            # Create the figure
            fig = go.Figure(data=trace, layout=layout)

            # Save if requested
            if save_path:
                fig.write_html(save_path)
                log_success(f"Saved point cloud to {save_path}")

            # Show if requested
            if show:
                fig.show()

            log_success("Point cloud plot created successfully")
            end_section()

            return fig

        except Exception as e:
            log_error(f"Error plotting point cloud: {str(e)}")
            end_section("Point cloud plotting failed")
            raise

    def plot_animated_rotation(
        self,
        structure: Union[np.ndarray, 'torch.Tensor'],
        colors: Optional[Union[np.ndarray, 'torch.Tensor']] = None,
        n_frames: int = 36,
        fps: int = 10,
        title: str = "Rotating 3D Sculpture",
        hide_axis: bool = False,
        linewidth: float = 0.05,
        save_path: Optional[str] = None,
    ):
        """
        Create an animated rotation of a 3D sculpture.

        Args:
            structure: 3D numpy array or PyTorch tensor representing the sculpture
            colors: 3D numpy array or PyTorch tensor with color information (optional)
            n_frames: Number of frames in the animation
            fps: Frames per second
            title: Title for the animation
            hide_axis: Whether to hide the axes
            linewidth: Width of the edges
            save_path: Path to save the animation (if None, animation is not saved)

        Returns:
            The matplotlib animation object
        """
        # Convert PyTorch tensors to numpy if needed
        if _is_torch_tensor(structure):
            device = _get_tensor_device(structure)
            log_info(f"Converting structure tensor from {device} to numpy for animation")
            structure = _tensor_to_numpy(structure)
            
        if colors is not None and _is_torch_tensor(colors):
            device = _get_tensor_device(colors)
            log_info(f"Converting colors tensor from {device} to numpy for animation")
            colors = _tensor_to_numpy(colors)
        
        begin_section(f"Creating animated rotation")

        try:
            from matplotlib.animation import FuncAnimation

            # Create figure and axis
            fig = plt.figure(
                figsize=(self.figsize // 2, self.figsize // 2),
                facecolor=self.style,
                dpi=self.dpi,
            )
            ax = fig.add_subplot(111, projection="3d")

            # Hide axis if requested
            if hide_axis:
                ax.set_axis_off()

            plt.title(title)

            # Function to update the plot for each frame
            def update(frame):
                ax.clear()
                if hide_axis:
                    ax.set_axis_off()

                # Calculate the elevation and azimuth for this frame
                elev = 30
                azim = frame * (360 / n_frames)

                ax.view_init(elev=elev, azim=azim)

                # Plot with colors if provided
                if colors is not None:
                    voxel = ax.voxels(
                        structure,
                        facecolors=colors,
                        edgecolors="k",
                        linewidth=linewidth,
                    )
                else:
                    voxel = ax.voxels(
                        structure,
                        facecolors="#E87F2A",
                        edgecolors="#C06A20",
                        linewidth=linewidth,
                    )

                return (voxel,)

            # Create the animation
            anim = FuncAnimation(
                fig, update, frames=n_frames, interval=1000 / fps, blit=False
            )

            # Save if requested
            if save_path:
                anim.save(save_path, writer="pillow", fps=fps)
                log_success(f"Saved animation to {save_path}")
                plt.close(fig)
            else:
                plt.close(fig)

            log_success("Animation created successfully")
            end_section()

            return anim

        except Exception as e:
            log_error(f"Error creating animation: {str(e)}")
            end_section("Animation creation failed")
            raise

    def visualize_sample_from_files(
        self,
        structure_path: str,
        colors_path: str,
        title: Optional[str] = None,
        angles: List[int] = [0, 1, 2, 3],
        hide_axis: bool = True,
        save_path: Optional[str] = None,
        show: bool = False,
    ):
        """
        Load and visualize a sample from structure and colors files.

        Args:
            structure_path: Path to the structure .npy or .pt file
            colors_path: Path to the colors .npy or .pt file
            title: Title for the visualization (default: extracted from filename)
            angles: List of rotation angles to show (in 90° increments)
            hide_axis: Whether to hide the axes
            save_path: Path to save the visualization
            show: Whether to display the plot (default: False)

        Returns:
            The matplotlib figure object
        """
        begin_section(f"Visualizing sample from files")

        try:
            # Load structure and colors - support both .npy and .pt files
            if structure_path.endswith('.pt') and TORCH_AVAILABLE:
                structure = torch.load(structure_path, map_location='cpu')
                log_info(f"Loaded PyTorch tensor from {structure_path}")
            else:
                structure = np.load(structure_path, allow_pickle=True)
                log_info(f"Loaded numpy array from {structure_path}")
                
            if colors_path.endswith('.pt') and TORCH_AVAILABLE:
                colors = torch.load(colors_path, map_location='cpu')
                log_info(f"Loaded PyTorch tensor from {colors_path}")
            else:
                colors = np.load(colors_path, allow_pickle=True)
                log_info(f"Loaded numpy array from {colors_path}")

            # Extract sample ID from filename if title not provided
            if title is None:
                basename = os.path.basename(structure_path)
                # Extract sample number from filenames
                if "structure_" in basename:
                    sample_id = basename.replace("structure_", "").replace(".npy", "")
                    title = f"Sample {sample_id}"
                elif "volume_" in basename:
                    sample_id = basename.replace("volume_", "").replace(".npy", "")
                    title = f"Sample {sample_id}"
                else:
                    title = "Sample from files"

            # Visualize the sample
            fig = self.plot_sculpture(
                structure=structure,
                colors=colors,
                title=title,
                angles=angles,
                hide_axis=hide_axis,
                save_path=save_path,
                show=show,
            )

            log_success(f"Visualized sample from {structure_path}")
            end_section()

            return fig

        except Exception as e:
            log_error(f"Error visualizing sample from files: {str(e)}")
            end_section("Sample visualization failed")
            raise

    def visualize_samples_from_directory(
        self,
        directory: str,
        n_samples: int = 3,
        structure_pattern: str = "structure_*.npy",
        colors_pattern: str = "colors_*.npy",
        output_dir: Optional[str] = None,
        angles: List[int] = [0, 1, 2, 3],
        hide_axis: bool = True,
        show: bool = False,
    ):
        """
        Load and visualize random samples from a directory.

        Args:
            directory: Directory containing samples
            n_samples: Number of samples to visualize
            structure_pattern: Glob pattern for structure files (supports .npy and .pt)
            colors_pattern: Glob pattern for colors files (supports .npy and .pt)
            output_dir: Directory to save visualizations (if None, don't save)
            angles: List of rotation angles to show (in 90° increments)
            hide_axis: Whether to hide the axes
            show: Whether to display the plots (default: False)

        Returns:
            List of matplotlib figure objects
        """
        begin_section(f"Visualizing {n_samples} samples from directory {directory}")

        try:
            # Check for new directory structure
            structures_dir = os.path.join(directory, "structures")
            colors_dir = os.path.join(directory, "colors")

            # Determine directory structure and handle accordingly
            if os.path.exists(structures_dir) and os.path.exists(colors_dir):
                # New structure - search in subdirectories
                structure_files = glob.glob(
                    os.path.join(structures_dir, structure_pattern)
                )
                
                # Also search for PyTorch tensor files
                if TORCH_AVAILABLE:
                    pt_pattern = structure_pattern.replace('.npy', '.pt')
                    structure_files.extend(glob.glob(
                        os.path.join(structures_dir, pt_pattern)
                    ))

                # If no files found with new pattern, try legacy patterns
                if not structure_files:
                    structure_files = glob.glob(
                        os.path.join(structures_dir, "volume_*.npy")
                    )
                    if TORCH_AVAILABLE:
                        structure_files.extend(glob.glob(
                            os.path.join(structures_dir, "volume_*.pt")
                        ))
            else:
                # Legacy structure - search directly in directory
                structure_files = glob.glob(os.path.join(directory, structure_pattern))
                
                # Also search for PyTorch tensor files
                if TORCH_AVAILABLE:
                    pt_pattern = structure_pattern.replace('.npy', '.pt')
                    structure_files.extend(glob.glob(os.path.join(directory, pt_pattern)))

                # If no files found with new pattern, try legacy patterns
                if not structure_files:
                    structure_files = glob.glob(os.path.join(directory, "volume_*.npy"))
                    if TORCH_AVAILABLE:
                        structure_files.extend(glob.glob(os.path.join(directory, "volume_*.pt")))

            if not structure_files:
                log_warning(f"No structure files found in {directory}")
                end_section("No samples found")
                return []

            # Select random samples
            selected_files = random.sample(
                structure_files, min(n_samples, len(structure_files))
            )
            log_info(f"Selected {len(selected_files)} random samples")

            # Prepare output directory if needed
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Visualize each sample
            figures = []
            for i, structure_file in enumerate(selected_files):
                # Get sample ID from filename
                basename = os.path.basename(structure_file)
                file_ext = '.pt' if basename.endswith('.pt') else '.npy'
                
                if "structure_" in basename:
                    sample_id = basename.replace("structure_", "").replace(file_ext, "")
                    colors_file_pattern = f"colors_*{file_ext}"
                    expected_color_file = basename.replace("structure_", "colors_")
                elif "volume_" in basename:
                    sample_id = basename.replace("volume_", "").replace(file_ext, "")
                    colors_file_pattern = f"material_*{file_ext}"
                    expected_color_file = basename.replace("volume_", "material_")
                else:
                    # Try to extract a number if available
                    import re

                    match = re.search(r"\d+", basename)
                    sample_id = match.group(0) if match else str(i + 1)
                    colors_file_pattern = colors_pattern
                    if colors_file_pattern.endswith('.npy'):
                        colors_file_pattern = colors_file_pattern.replace('.npy', file_ext)
                    expected_color_file = None

                # Determine where to look for colors file
                if os.path.exists(structures_dir) and os.path.exists(colors_dir):
                    # Look in colors subdirectory
                    colors_search_dir = colors_dir
                else:
                    # Look in same directory as structure file
                    colors_search_dir = os.path.dirname(structure_file)

                # Find matching colors file
                colors_file = None

                # First try the expected matching file if known
                if expected_color_file:
                    potential_colors_file = os.path.join(
                        colors_search_dir, expected_color_file
                    )
                    if os.path.exists(potential_colors_file):
                        colors_file = potential_colors_file

                # If that fails, search for any matching file (both .npy and .pt)
                if not colors_file:
                    colors_files = glob.glob(
                        os.path.join(colors_search_dir, colors_file_pattern)
                    )
                    
                    # Also search for the other file type if not found
                    if not colors_files and TORCH_AVAILABLE:
                        alt_pattern = colors_file_pattern.replace('.pt', '.npy') if colors_file_pattern.endswith('.pt') else colors_file_pattern.replace('.npy', '.pt')
                        colors_files = glob.glob(
                            os.path.join(colors_search_dir, alt_pattern)
                        )
                    
                    for cf in colors_files:
                        if sample_id in os.path.basename(cf):
                            colors_file = cf
                            break

                if not colors_file:
                    log_warning(
                        f"Could not find matching colors file for {structure_file}"
                    )
                    continue

                # Prepare save path if output directory is provided
                save_path = None
                if output_dir:
                    save_path = os.path.join(output_dir, f"sample_{sample_id}.png")

                # Visualize this sample
                log_action(
                    f"Visualizing sample {sample_id} ({i+1}/{len(selected_files)})"
                )
                try:
                    fig = self.visualize_sample_from_files(
                        structure_path=structure_file,
                        colors_path=colors_file,
                        title=f"Sample {sample_id}",
                        angles=angles,
                        hide_axis=hide_axis,
                        save_path=save_path,
                        show=show,
                    )
                    figures.append(fig)

                    if save_path:
                        log_success(f"Saved visualization to {save_path}")
                except Exception as e:
                    log_warning(f"Error visualizing sample {sample_id}: {str(e)}")

            log_success(f"Visualized {len(figures)} samples from directory")
            end_section()

            return figures

        except Exception as e:
            log_error(f"Error visualizing samples from directory: {str(e)}")
            end_section("Sample visualization failed")
            raise


class PyTorchVisualizer(Visualizer):
    """
    Enhanced 3D visualization system with native PyTorch tensor support.
    
    This class extends the base Visualizer with additional PyTorch-specific
    functionality and optimizations for GPU tensors.
    """
    
    def __init__(
        self,
        backend: str = "matplotlib",  # "matplotlib", "plotly", "open3d"
        device: str = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu",
        figsize: int = 25,
        style: str = "#ffffff",
        dpi: int = 100,
        transparent: bool = False,
    ):
        """
        Initialize PyTorchVisualizer.
        
        Args:
            backend: Visualization backend to use
            device: Default device for tensor operations
            figsize: Figure size for matplotlib plots
            style: Background color for plots
            dpi: DPI for raster images
            transparent: Whether to use transparent backgrounds
        """
        super().__init__(figsize, style, dpi, transparent)
        self.backend = backend
        self.device = device
        
        if not TORCH_AVAILABLE:
            log_warning("PyTorch not available. PyTorchVisualizer will work with numpy arrays only.")
    
    def plot_training_progress(self, metrics: Dict[str, List[float]], **kwargs) -> Any:
        """
        Plot training progress metrics.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            **kwargs: Additional plotting arguments
            
        Returns:
            The matplotlib figure object
        """
        begin_section("Plotting training progress")
        
        try:
            fig, axes = plt.subplots(
                nrows=len(metrics), 
                ncols=1, 
                figsize=(12, 4 * len(metrics)),
                facecolor=self.style,
                dpi=self.dpi
            )
            
            if len(metrics) == 1:
                axes = [axes]
            
            for i, (metric_name, values) in enumerate(metrics.items()):
                axes[i].plot(values)
                axes[i].set_title(f"{metric_name.replace('_', ' ').title()}")
                axes[i].set_xlabel("Epoch/Step")
                axes[i].set_ylabel(metric_name)
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            save_path = kwargs.get('save_path')
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved training progress plot to {save_path}")
            
            show = kwargs.get('show', False)
            if show:
                plt.show()
            else:
                plt.close(fig)
            
            log_success("Training progress plot created successfully")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error plotting training progress: {str(e)}")
            end_section("Training progress plotting failed")
            raise
    
    def visualize_latent_space(self, model: 'torch.nn.Module', **kwargs) -> Any:
        """
        Visualize latent space of a model.
        
        Args:
            model: PyTorch model to analyze
            **kwargs: Additional visualization arguments
            
        Returns:
            The visualization figure object
        """
        begin_section("Visualizing latent space")
        
        try:
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch not available for latent space visualization")
            
            # This is a placeholder implementation
            # In practice, this would generate samples from the latent space
            # and visualize them using dimensionality reduction techniques
            
            log_info("Latent space visualization is a placeholder implementation")
            log_info("This would typically involve:")
            log_info("1. Sampling from the model's latent space")
            log_info("2. Generating outputs from latent codes")
            log_info("3. Using t-SNE or UMAP for dimensionality reduction")
            log_info("4. Creating interactive plots of the latent space")
            
            # Create a simple placeholder plot
            fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.style, dpi=self.dpi)
            ax.text(0.5, 0.5, "Latent Space Visualization\n(Placeholder Implementation)", 
                   ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_title("Model Latent Space")
            
            save_path = kwargs.get('save_path')
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved latent space plot to {save_path}")
            
            show = kwargs.get('show', False)
            if show:
                plt.show()
            else:
                plt.close(fig)
            
            log_success("Latent space visualization created successfully")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error visualizing latent space: {str(e)}")
            end_section("Latent space visualization failed")
            raise


# Main function for testing
if __name__ == "__main__":
    # Create a simple test structure
    void_dim = 10
    structure = np.zeros((void_dim, void_dim, void_dim))
    colors_array = np.empty(structure.shape, dtype=object)

    # Add some shapes to the structure
    # Add a plane at z=0
    structure[:, :, 0] = 1

    # Add a column in the center
    center = void_dim // 2
    structure[center, center, :] = 1

    # Add some random voxels
    for _ in range(20):
        x = np.random.randint(0, void_dim)
        y = np.random.randint(0, void_dim)
        z = np.random.randint(0, void_dim)
        structure[x, y, z] = 1

    # Test with numpy arrays
    print("Testing with numpy arrays...")
    visualizer = Visualizer()
    visualizer.plot_sculpture(structure, title="Test Sculpture (NumPy)")
    
    # Test with PyTorch tensors if available
    if TORCH_AVAILABLE:
        print("Testing with PyTorch tensors...")
        torch_structure = torch.from_numpy(structure).float()
        pytorch_visualizer = PyTorchVisualizer()
        pytorch_visualizer.plot_sculpture(torch_structure, title="Test Sculpture (PyTorch)")
        
        # Test GPU tensor if CUDA is available
        if torch.cuda.is_available():
            print("Testing with GPU tensors...")
            gpu_structure = torch_structure.cuda()
            pytorch_visualizer.plot_sculpture(gpu_structure, title="Test Sculpture (GPU)")
    else:
        print("PyTorch not available, skipping tensor tests")

    # Assign colors
    colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
    for i in range(void_dim):
        for j in range(void_dim):
            for k in range(void_dim):
                if structure[i, j, k] == 1:
                    colors_array[i, j, k] = np.random.choice(colors)

    # Create a visualizer
    visualizer = Visualizer(figsize=10, dpi=100)

    # Plot sections - save but don't show by default
    visualizer.plot_sections(
        structure=structure,
        title="Test Structure Sections",
        save_path="test_sections.png",
    )

    # Plot sculpture - save but don't show by default
    visualizer.plot_sculpture(
        structure=structure,
        colors=colors_array,
        title="Test 3D Sculpture",
        hide_axis=True,
        save_path="test_sculpture.png",
        save_array=False,
    )

    # Plot single view - save but don't show by default
    visualizer.plot_single_view(
        structure=structure,
        colors=colors_array,
        angle=1,
        hide_axis=True,
        title="Single View (90°)",
        save_path="test_single_view.png",
    )

    # Save sample to files for testing directory functions
    test_dir = "viz_test"
    os.makedirs(test_dir, exist_ok=True)
    np.save(os.path.join(test_dir, "volume_00001.npy"), structure)
    np.save(os.path.join(test_dir, "material_00001.npy"), colors_array)

    # Test loading from files
    visualizer.visualize_sample_from_files(
        structure_path=os.path.join(test_dir, "volume_00001.npy"),
        colors_path=os.path.join(test_dir, "material_00001.npy"),
        save_path=os.path.join(test_dir, "sample_from_files.png"),
    )

    # If there's a samples directory to test with, try loading from directory
    if os.path.exists("results"):
        # Look for any samples directory recursively
        samples_dirs = []
        for root, dirs, files in os.walk("results"):
            if "samples" in dirs:
                samples_dirs.append(os.path.join(root, "samples"))

        if samples_dirs:
            # Test visualizing from directory with the first samples directory found
            visualizer.visualize_samples_from_directory(
                directory=samples_dirs[0], n_samples=2, output_dir=test_dir
            )
            print(f"Visualized samples from {samples_dirs[0]}")

    # Convert to point cloud and plot - save but don't show by default
    points = visualizer.voxel_to_pointcloud(structure=structure, subdivision=2)

    # Pick a random color for each point
    point_colors = np.random.randint(0, 256, size=(len(points), 3))

    # Plot point cloud - save but don't show by default
    visualizer.plot_pointcloud(
        points=points,
        colors=point_colors,
        size=3.0,
        alpha=0.7,
        title="Test Point Cloud",
        save_path="test_pointcloud.html",
    )

    # Plot animated rotation - save only
    visualizer.plot_animated_rotation(
        structure=structure,
        colors=colors_array,
        n_frames=36,
        title="Rotating Test Sculpture",
        hide_axis=True,
        save_path="test_rotation.gif",
    )
