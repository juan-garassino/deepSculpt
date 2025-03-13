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

Dependencies:
- logger.py: For operation tracking and status reporting
- utils.py: For array transformations and data preparation
- numpy: For array manipulation
- matplotlib: For 3D and 2D plotting
- plotly: For interactive point cloud visualization
- datetime: For timestamped file naming
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
from logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)


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
        structure: np.ndarray,
        title: str = "Structure Sections",
        cmap: str = "gray",
        show: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot 2D sections of a 3D structure.

        Args:
            structure: 3D numpy array to visualize
            title: Title for the plot
            cmap: Colormap to use
            show: Whether to display the plot (default: False)
            save_path: Path to save the plot (if None, plot is not saved)

        Returns:
            The matplotlib figure object
        """
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
        structure: np.ndarray,
        colors: Optional[np.ndarray] = None,
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
            structure: 3D numpy array representing the sculpture
            colors: 3D numpy array with color information (optional)
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
                            edgecolors="k",
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

                # Save structure array
                structure_path = os.path.join(
                    structure_dir, f"structure_array_{timestamp}.npy"
                )
                np.save(structure_path, structure)
                log_success(f"Saved structure array to {structure_path}")

                # Save color array if provided
                if colors is not None:
                    colors_dir = os.path.join(save_dir, "colors_array")
                    os.makedirs(colors_dir, exist_ok=True)
                    colors_path = os.path.join(
                        colors_dir, f"colors_array_{timestamp}.npy"
                    )
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
        structure: np.ndarray,
        colors: Optional[np.ndarray] = None,
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
            structure: 3D numpy array representing the sculpture
            colors: 3D numpy array with color information (optional)
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
                    edgecolors="k",
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
    def voxel_to_pointcloud(structure: np.ndarray, subdivision: int = 3) -> np.ndarray:
        """
        Convert a voxel grid to a point cloud.

        Args:
            structure: 3D numpy array representing the voxel grid
            subdivision: Number of points to generate per voxel dimension

        Returns:
            Nx3 numpy array of point coordinates
        """
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
        points: np.ndarray,
        colors: Optional[Union[np.ndarray, Tuple[float, float, float]]] = None,
        size: float = 1.0,
        alpha: float = 1.0,
        title: str = "3D Point Cloud",
        show: bool = False,
        save_path: Optional[str] = None,
    ):
        """
        Plot a 3D point cloud using Plotly.

        Args:
            points: Nx3 numpy array of point coordinates
            colors: Point colors (Nx3 array or single RGB tuple)
            size: Point size
            alpha: Point opacity
            title: Title for the plot
            show: Whether to display the plot (default: False)
            save_path: Path to save the plot (if None, plot is not saved)

        Returns:
            The plotly figure object
        """
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
        structure: np.ndarray,
        colors: Optional[np.ndarray] = None,
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
            structure: 3D numpy array representing the sculpture
            colors: 3D numpy array with color information (optional)
            n_frames: Number of frames in the animation
            fps: Frames per second
            title: Title for the animation
            hide_axis: Whether to hide the axes
            linewidth: Width of the edges
            save_path: Path to save the animation (if None, animation is not saved)

        Returns:
            The matplotlib animation object
        """
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
                        edgecolors="k",
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
            structure_path: Path to the structure .npy file
            colors_path: Path to the colors .npy file
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
            # Load structure and colors
            structure = np.load(structure_path, allow_pickle=True)
            colors = np.load(colors_path, allow_pickle=True)

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
            structure_pattern: Glob pattern for structure files
            colors_pattern: Glob pattern for colors files
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

                # If no files found with new pattern, try legacy pattern
                if not structure_files:
                    structure_files = glob.glob(
                        os.path.join(structures_dir, "volume_*.npy")
                    )
            else:
                # Legacy structure - search directly in directory
                structure_files = glob.glob(os.path.join(directory, structure_pattern))

                # If no files found with new pattern, try legacy pattern
                if not structure_files:
                    structure_files = glob.glob(os.path.join(directory, "volume_*.npy"))

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
                if "structure_" in basename:
                    sample_id = basename.replace("structure_", "").replace(".npy", "")
                    colors_file_pattern = "colors_*.npy"
                    expected_color_file = basename.replace("structure_", "colors_")
                elif "volume_" in basename:
                    sample_id = basename.replace("volume_", "").replace(".npy", "")
                    colors_file_pattern = "material_*.npy"
                    expected_color_file = basename.replace("volume_", "material_")
                else:
                    # Try to extract a number if available
                    import re

                    match = re.search(r"\d+", basename)
                    sample_id = match.group(0) if match else str(i + 1)
                    colors_file_pattern = colors_pattern
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

                # If that fails, search for any matching file
                if not colors_file:
                    colors_files = glob.glob(
                        os.path.join(colors_search_dir, colors_file_pattern)
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
        z = np.random.randint(1, void_dim)  # Avoid z=0 as we already have a plane there
        structure[x, y, z] = 1

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
