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

Dependencies:
- logger.py: For operation tracking and status reporting
- utils.py: For array transformations and data preparation
- numpy: For array manipulation
- matplotlib: For 3D and 2D plotting
- plotly: For interactive point cloud visualization
- datetime: For timestamped file naming

Used by:
- sculptor.py: For visualizing individual sculptures
- collector.py: For displaying samples from generated datasets
- curator.py: For visualizing encoded and processed data

TODO:
- Add support for custom colormaps and styling
- Implement interactive 3D viewer with controls
- Add cross-section animation capabilities
- Improve performance for very large voxel arrays
- Add VR/AR export options
- Support textured rendering modes
- Add support for volumetric rendering
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, Union
from logger import begin_section, end_section, log_action, log_success, log_error, log_info, log_warning

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
        volumes: np.ndarray,
        title: str = "Volume Sections",
        cmap: str = "gray",
        show: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Plot 2D sections of a 3D volume.
        
        Args:
            volumes: 3D numpy array to visualize
            title: Title for the plot
            cmap: Colormap to use
            show: Whether to display the plot
            save_path: Path to save the plot (if None, plot is not saved)
            
        Returns:
            The matplotlib figure object
        """
        begin_section(f"Plotting {volumes.shape[0]} sections")
        
        try:
            # Calculate grid dimensions
            n_sections = volumes.shape[0]
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
                axes[i].imshow(volumes[i, :, :], cmap=cmap)
                axes[i].set_title(f"Section {i}")
                axes[i].set_xticks([])
                axes[i].set_yticks([])
            
            # Hide any unused subplots
            for i in range(n_sections, len(axes)):
                axes[i].axis('off')
            
            plt.suptitle(title, fontsize=16)
            plt.tight_layout()
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved sections plot to {save_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            log_success("Sections plot created successfully")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error plotting sections: {str(e)}")
            end_section("Section plotting failed")
            raise
    
    def plot_sculpture(
        self,
        volumes: np.ndarray,
        materials: Optional[np.ndarray] = None,
        title: str = "3D Sculpture",
        angles: List[int] = [0, 1, 2, 3],
        hide_axis: bool = False,
        linewidth: float = 0.05,
        show: bool = True,
        save_path: Optional[str] = None,
        save_array: bool = False,
        save_dir: str = ".",
    ):
        """
        Plot a 3D sculpture from different angles.
        
        Args:
            volumes: 3D numpy array representing the sculpture
            materials: 3D numpy array with color information (optional)
            title: Title for the plot
            angles: List of rotation angles to show (in 90° increments)
            hide_axis: Whether to hide the axes
            linewidth: Width of the edges
            show: Whether to display the plot
            save_path: Path to save the plot (if None, plot is not saved)
            save_array: Whether to save the volume and material arrays
            save_dir: Directory to save arrays if save_array is True
            
        Returns:
            The matplotlib figure object
        """
        begin_section(f"Plotting 3D sculpture with shape {volumes.shape}")
        
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
                    
                    # Rotate the volume
                    rotated_volumes = np.rot90(volumes, angle)
                    
                    # Plot with materials if provided
                    if materials is not None:
                        rotated_materials = np.rot90(materials, angle)
                        axes[i].voxels(
                            rotated_volumes,
                            facecolors=rotated_materials,
                            edgecolors="k",
                            linewidth=linewidth,
                        )
                        log_info(f"Plotted view {i} (rotation {angle*90}°) with materials")
                    else:
                        axes[i].voxels(
                            rotated_volumes,
                            edgecolors="k",
                            linewidth=linewidth,
                        )
                        log_info(f"Plotted view {i} (rotation {angle*90}°) without materials")
                    
                    axes[i].set_title(f"Rotation {angle*90}°")
            
            # Hide any unused subplots
            for i in range(n_views, len(axes)):
                axes[i].axis('off')
            
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
                volumes_dir = os.path.join(save_dir, "volume_array")
                os.makedirs(volumes_dir, exist_ok=True)
                
                # Save volume array
                volume_path = os.path.join(volumes_dir, f"volume_array_{timestamp}.npy")
                np.save(volume_path, volumes)
                log_success(f"Saved volume array to {volume_path}")
                
                # Save material array if provided
                if materials is not None:
                    materials_dir = os.path.join(save_dir, "material_array")
                    os.makedirs(materials_dir, exist_ok=True)
                    material_path = os.path.join(materials_dir, f"material_array_{timestamp}.npy")
                    np.save(material_path, materials)
                    log_success(f"Saved material array to {material_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            log_success("Sculpture plot created successfully")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error plotting sculpture: {str(e)}")
            end_section("Sculpture plotting failed")
            raise
    
    def plot_single_view(
        self,
        volumes: np.ndarray,
        materials: Optional[np.ndarray] = None,
        angle: int = 0,
        figsize: Optional[Tuple[int, int]] = None,
        hide_axis: bool = False,
        linewidth: float = 0.05,
        title: str = "3D View",
        show: bool = True,
        save_path: Optional[str] = None,
    ):
        """
        Plot a single view of a 3D sculpture.
        
        Args:
            volumes: 3D numpy array representing the sculpture
            materials: 3D numpy array with color information (optional)
            angle: Rotation angle in 90° increments
            figsize: Figure size as (width, height) tuple
            hide_axis: Whether to hide the axes
            linewidth: Width of the edges
            title: Title for the plot
            show: Whether to display the plot
            save_path: Path to save the plot (if None, plot is not saved)
            
        Returns:
            The matplotlib figure object
        """
        begin_section(f"Plotting single 3D view (rotation {angle*90}°)")
        
        try:
            # Use default figsize if none provided
            if figsize is None:
                figsize = (self.figsize//2, self.figsize//2)
            
            # Create figure and axis
            fig = plt.figure(figsize=figsize, facecolor=self.style, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Hide axis if requested
            if hide_axis:
                ax.set_axis_off()
            
            # Rotate the volume
            rotated_volumes = np.rot90(volumes, angle)
            
            # Plot with materials if provided
            if materials is not None:
                rotated_materials = np.rot90(materials, angle)
                ax.voxels(
                    rotated_volumes,
                    facecolors=rotated_materials,
                    edgecolors="k",
                    linewidth=linewidth,
                )
                log_info(f"Plotted with materials")
            else:
                ax.voxels(
                    rotated_volumes,
                    edgecolors="k",
                    linewidth=linewidth,
                )
                log_info(f"Plotted without materials")
            
            plt.title(title)
            
            # Save if requested
            if save_path:
                plt.savefig(save_path, transparent=self.transparent)
                log_success(f"Saved single view plot to {save_path}")
            
            # Show if requested
            if show:
                plt.show()
            
            log_success("Single view plot created successfully")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error plotting single view: {str(e)}")
            end_section("Single view plotting failed")
            raise
    
    @staticmethod
    def voxel_to_pointcloud(
        volumes: np.ndarray,
        subdivision: int = 3
    ) -> np.ndarray:
        """
        Convert a voxel grid to a point cloud.
        
        Args:
            volumes: 3D numpy array representing the voxel grid
            subdivision: Number of points to generate per voxel dimension
            
        Returns:
            Nx3 numpy array of point coordinates
        """
        begin_section(f"Converting voxel grid to point cloud")
        
        try:
            n_x, n_y, n_z = volumes.shape
            points = []
            
            # Create points for each filled voxel
            for i in range(n_x):
                for j in range(n_y):
                    for k in range(n_z):
                        if volumes[i, j, k]:
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
        show: bool = True,
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
            show: Whether to display the plot
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
                color_array = [
                    f"rgba({r}, {g}, {b}, {alpha})" 
                    for r, g, b in colors
                ]
                marker_dict = dict(size=size, color=color_array)
            
            # Create the scatter3d trace
            trace = go.Scatter3d(
                x=x, y=y, z=z,
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
        volumes: np.ndarray,
        materials: Optional[np.ndarray] = None,
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
            volumes: 3D numpy array representing the sculpture
            materials: 3D numpy array with color information (optional)
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
            fig = plt.figure(figsize=(self.figsize//2, self.figsize//2), facecolor=self.style, dpi=self.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
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
                
                # Plot with materials if provided
                if materials is not None:
                    voxel = ax.voxels(
                        volumes,
                        facecolors=materials,
                        edgecolors="k",
                        linewidth=linewidth,
                    )
                else:
                    voxel = ax.voxels(
                        volumes,
                        edgecolors="k",
                        linewidth=linewidth,
                    )
                
                return voxel,
            
            # Create the animation
            anim = FuncAnimation(
                fig,
                update,
                frames=n_frames,
                interval=1000/fps,
                blit=False
            )
            
            # Save if requested
            if save_path:
                anim.save(save_path, writer='pillow', fps=fps)
                log_success(f"Saved animation to {save_path}")
            
            log_success("Animation created successfully")
            end_section()
            
            return anim
            
        except Exception as e:
            log_error(f"Error creating animation: {str(e)}")
            end_section("Animation creation failed")
            raise

# Main function for testing
if __name__ == "__main__":
    # Create a simple test volume
    void_dim = 10
    void = np.zeros((void_dim, void_dim, void_dim))
    color_void = np.empty(void.shape, dtype=object)
    
    # Add some shapes to the void
    # Add a plane at z=0
    void[:, :, 0] = 1
    
    # Add a column in the center
    center = void_dim // 2
    void[center, center, :] = 1
    
    # Add some random voxels
    for _ in range(20):
        x = np.random.randint(0, void_dim)
        y = np.random.randint(0, void_dim)
        z = np.random.randint(1, void_dim)  # Avoid z=0 as we already have a plane there
        void[x, y, z] = 1
    
    # Assign colors
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
    for i in range(void_dim):
        for j in range(void_dim):
            for k in range(void_dim):
                if void[i, j, k] == 1:
                    color_void[i, j, k] = np.random.choice(colors)
    
    # Create a visualizer
    visualizer = Visualizer(figsize=10, dpi=100)
    
    # Plot sections
    visualizer.plot_sections(void, title="Test Volume Sections", show=True)
    
    # Plot sculpture
    visualizer.plot_sculpture(
        void, 
        color_void, 
        title="Test 3D Sculpture", 
        hide_axis=True, 
        show=True,
        save_array=False
    )
    
    # Plot single view
    visualizer.plot_single_view(
        void, 
        color_void, 
        angle=1, 
        hide_axis=True, 
        title="Single View (90°)",
        show=True
    )
    
    # Convert to point cloud and plot
    points = visualizer.voxel_to_pointcloud(void, subdivision=2)
    
    # Pick a random color for each point
    point_colors = np.random.randint(0, 256, size=(len(points), 3))
    
    # Plot point cloud
    visualizer.plot_pointcloud(
        points,
        colors=point_colors,
        size=3.0,
        alpha=0.7,
        title="Test Point Cloud",
        show=True
    )
    
    # Plot animated rotation 
    # (Uncomment to test, but be aware it may take a while to generate)
    # visualizer.plot_animated_rotation(
    #     void,
    #     color_void,
    #     n_frames=36,
    #     title="Rotating Test Sculpture",
    #     hide_axis=True,
    #     save_path="rotation.gif"
    # )