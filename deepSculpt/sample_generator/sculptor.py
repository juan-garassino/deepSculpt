"""
Sculpture Generation System for DeepSculpt
This module provides the central Sculptor class that creates complete 3D voxel-based
sculptures by combining various shape components. It manages the entire sculpture 
generation workflow including creation, visualization, and persistence.

Key features:
- Complete sculpture generation: Creates full 3D structures from components
- Shape composition: Combines edges, planes, pipes, and grids
- Parameter control: Configurable constraints for each component type
- Visualization integration: Built-in methods to view results
- Save/load functionality: Persistence for generated sculptures
- Method chaining: Fluent interface for operations

Dependencies:
- logger.py: For process tracking and status reporting
- shapes.py: For individual shape component generation
- utils.py: For array operations and validation
- visualization.py: For displaying the generated sculptures
- numpy: For array operations

Used by:
- collector.py: For batch generation of sculptures
- demo.py: For demonstration and testing

TODO:
- Add support for sculpture modification after generation
- Implement style transfer between sculptures
- Add symmetry constraints and options
- Improve space filling algorithms for denser structures
- Add generative constraints like growth patterns
- Implement evolutionary algorithms for sculpture optimization
- Support for importing external 3D models as seeds
"""

import numpy as np
import time
import os
from typing import Tuple, List, Dict, Any, Optional, Union
from enum import Enum

from logger import begin_section, end_section, log_action, log_success, log_error, log_info
from shapes import ShapeType, attach_edge, attach_plane, attach_pipe, attach_grid, attach_shape
from visualization import Visualizer

class Sculptor:
    """
    A class for creating 3D sculptures with various components.
    """
    
    def __init__(
        self,
        void_dim: int = 20,
        edges: Tuple[int, float, float] = (1, 0.3, 0.5),
        planes: Tuple[int, float, float] = (1, 0.3, 0.5),
        pipes: Tuple[int, float, float] = (1, 0.3, 0.5),
        grid: Tuple[int, int] = (1, 4),
        colors: Optional[Dict[str, Any]] = None,
        step: int = 1,
        verbose: bool = False,
    ):
        """
        Initialize a new Sculptor instance.
        
        Args:
            void_dim: The dimension of the void (cube size)
            edges: Tuple of (count, min_ratio, max_ratio) for edges
            planes: Tuple of (count, min_ratio, max_ratio) for planes
            pipes: Tuple of (count, min_ratio, max_ratio) for pipes
            grid: Tuple of (enable, step) for grid
            colors: Dictionary of colors for different shape types
            step: Step size for shape dimensions
            verbose: Whether to print detailed information
        """
        self.void_dim = void_dim
        self.volumes_void = np.zeros((void_dim, void_dim, void_dim))
        self.materials_void = np.empty(self.volumes_void.shape, dtype=object)
        
        self.edges = edges
        self.planes = planes
        self.pipes = pipes
        self.grid = grid
        
        # Default colors if not provided
        if colors is None:
            self.colors = {
                "edges": "red",
                "planes": "green",
                "pipes": ["blue", "cyan", "magenta"],
                "volumes": ["purple", "brown", "orange"]
            }
        else:
            self.colors = colors
        
        self.step = step
        self.verbose = verbose
        
        # Create a visualizer for this sculptor
        self.visualizer = Visualizer(figsize=15, dpi=100)
    
    def generate_sculpture(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a sculpture by attaching various components.
        
        Returns:
            Tuple of (volumes_void, materials_void) arrays
        """
        begin_section("Generating Sculpture")
        
        try:
            # Start timer
            start_time = time.time()
            
            # Create grid if enabled
            if self.grid[0] == 1:
                log_action("Adding grid structure")
                
                self.volumes_void, self.materials_void = attach_grid(
                    self.volumes_void,
                    self.materials_void,
                    step=self.grid[1],
                    colors=self.colors,
                    verbose=self.verbose
                )
            
            # Add edges
            for i in range(self.edges[0]):
                log_action(f"Adding edge {i+1}/{self.edges[0]}", is_last=(i==self.edges[0]-1))
                
                self.volumes_void, self.materials_void = attach_edge(
                    self.volumes_void,
                    self.materials_void,
                    element_edge_min_ratio=self.edges[1],
                    element_edge_max_ratio=self.edges[2],
                    step=self.step,
                    colors=self.colors,
                    verbose=self.verbose
                )
            
            # Add planes
            for i in range(self.planes[0]):
                log_action(f"Adding plane {i+1}/{self.planes[0]}", is_last=(i==self.planes[0]-1))
                
                self.volumes_void, self.materials_void = attach_plane(
                    self.volumes_void,
                    self.materials_void,
                    element_plane_min_ratio=self.planes[1],
                    element_plane_max_ratio=self.planes[2],
                    step=self.step,
                    colors=self.colors,
                    verbose=self.verbose
                )
            
            # Add pipes
            for i in range(self.pipes[0]):
                log_action(f"Adding pipe {i+1}/{self.pipes[0]}", is_last=(i==self.pipes[0]-1))
                
                self.volumes_void, self.materials_void = attach_pipe(
                    self.volumes_void,
                    self.materials_void,
                    element_volume_min_ratio=self.pipes[1],
                    element_volume_max_ratio=self.pipes[2],
                    step=self.step,
                    colors=self.colors,
                    verbose=self.verbose
                )
            
            # Log statistics
            filled_voxels = np.sum(self.volumes_void > 0)
            total_voxels = self.volumes_void.size
            fill_percentage = (filled_voxels / total_voxels) * 100
            
            log_info(f"Filled voxels: {filled_voxels}/{total_voxels} ({fill_percentage:.2f}%)")
            log_success(f"Sculpture generated in {time.time() - start_time:.2f} seconds")
            
            end_section()
            
            return self.volumes_void, self.materials_void
            
        except Exception as e:
            log_error(f"Error generating sculpture: {str(e)}")
            end_section("Sculpture generation failed")
            raise
    
    def visualize(
        self,
        title: str = "3D Sculpture",
        hide_axis: bool = True,
        save_path: Optional[str] = None,
        save_array: bool = False,
        save_dir: str = "output"
    ):
        """
        Visualize the generated sculpture.
        
        Args:
            title: Title for the visualization
            hide_axis: Whether to hide the axes
            save_path: Path to save the visualization
            save_array: Whether to save the arrays
            save_dir: Directory to save outputs
        
        Returns:
            The matplotlib figure object
        """
        begin_section("Visualizing Sculpture")
        
        try:
            # Create output directory if needed
            if save_path or save_array:
                os.makedirs(save_dir, exist_ok=True)
            
            # Plot the sculpture
            fig = self.visualizer.plot_sculpture(
                self.volumes_void,
                self.materials_void,
                title=title,
                hide_axis=hide_axis,
                save_path=save_path,
                save_array=save_array,
                save_dir=save_dir,
                show=True
            )
            
            log_success("Sculpture visualization completed")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error visualizing sculpture: {str(e)}")
            end_section("Sculpture visualization failed")
            raise
    
    def visualize_pointcloud(
        self,
        subdivision: int = 2,
        title: str = "Sculpture Point Cloud",
        save_path: Optional[str] = None
    ):
        """
        Visualize the sculpture as a point cloud.
        
        Args:
            subdivision: Number of points per voxel dimension
            title: Title for the visualization
            save_path: Path to save the visualization
        
        Returns:
            The plotly figure object
        """
        begin_section("Visualizing Sculpture as Point Cloud")
        
        try:
            # Convert to point cloud
            points = self.visualizer.voxel_to_pointcloud(self.volumes_void, subdivision)
            
            # Create colors based on the materials_void
            point_colors = []
            for x, y, z in points.astype(int):
                try:
                    color_str = self.materials_void[x, y, z]
                    if color_str == "red":
                        point_colors.append([255, 0, 0])
                    elif color_str == "green":
                        point_colors.append([0, 255, 0])
                    elif color_str == "blue":
                        point_colors.append([0, 0, 255])
                    elif color_str == "cyan":
                        point_colors.append([0, 255, 255])
                    elif color_str == "magenta":
                        point_colors.append([255, 0, 255])
                    elif color_str == "yellow":
                        point_colors.append([255, 255, 0])
                    else:
                        point_colors.append([100, 100, 100])  # Default gray
                except (IndexError, TypeError):
                    point_colors.append([100, 100, 100])  # Default gray
            
            # Plot the point cloud
            fig = self.visualizer.plot_pointcloud(
                points,
                colors=np.array(point_colors),
                size=3.0,
                alpha=0.7,
                title=title,
                save_path=save_path,
                show=True
            )
            
            log_success("Point cloud visualization completed")
            end_section()
            
            return fig
            
        except Exception as e:
            log_error(f"Error visualizing point cloud: {str(e)}")
            end_section("Point cloud visualization failed")
            raise
    
    def create_animated_rotation(
        self,
        n_frames: int = 36,
        fps: int = 10,
        title: str = "Rotating Sculpture",
        hide_axis: bool = True,
        save_path: Optional[str] = None
    ):
        """
        Create an animated rotation of the sculpture.
        
        Args:
            n_frames: Number of frames in the animation
            fps: Frames per second
            title: Title for the animation
            hide_axis: Whether to hide the axes
            save_path: Path to save the animation
        
        Returns:
            The matplotlib animation object
        """
        begin_section("Creating Animated Rotation")
        
        try:
            # Create the animation
            anim = self.visualizer.plot_animated_rotation(
                self.volumes_void,
                self.materials_void,
                n_frames=n_frames,
                fps=fps,
                title=title,
                hide_axis=hide_axis,
                save_path=save_path
            )
            
            log_success("Animation created successfully")
            end_section()
            
            return anim
            
        except Exception as e:
            log_error(f"Error creating animation: {str(e)}")
            end_section("Animation creation failed")
            raise
    
    def add_shape(
        self,
        shape_type: ShapeType,
        min_ratio: float = 0.1,
        max_ratio: float = 0.9
    ):
        """
        Add a single shape to the sculpture.
        
        Args:
            shape_type: Type of shape to add
            min_ratio: Minimum size ratio
            max_ratio: Maximum size ratio
        
        Returns:
            Self for method chaining
        """
        begin_section(f"Adding {shape_type.name} to Sculpture")
        
        try:
            # Add the shape
            self.volumes_void, self.materials_void = attach_shape(
                self.volumes_void,
                self.materials_void,
                shape_type,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                step=self.step,
                colors=self.colors,
                verbose=self.verbose
            )
            
            log_success(f"{shape_type.name} added successfully")
            end_section()
            
            return self
            
        except Exception as e:
            log_error(f"Error adding shape: {str(e)}")
            end_section("Shape addition failed")
            raise
    
    def reset(self):
        """
        Reset the sculpture to an empty void.
        
        Returns:
            Self for method chaining
        """
        begin_section("Resetting Sculpture")
        
        try:
            # Reset the void and materials
            self.volumes_void = np.zeros((self.void_dim, self.void_dim, self.void_dim))
            self.materials_void = np.empty(self.volumes_void.shape, dtype=object)
            
            log_success("Sculpture reset successfully")
            end_section()
            
            return self
            
        except Exception as e:
            log_error(f"Error resetting sculpture: {str(e)}")
            end_section("Sculpture reset failed")
            raise
    
    def save(
        self,
        directory: str = "output",
        save_volumes: bool = True,
        save_materials: bool = True
    ):
        """
        Save the sculpture arrays to files.
        
        Args:
            directory: Directory to save files
            save_volumes: Whether to save the volumes array
            save_materials: Whether to save the materials array
        
        Returns:
            Dictionary with paths to saved files
        """
        begin_section("Saving Sculpture")
        
        try:
            # Create output directory
            os.makedirs(directory, exist_ok=True)
            
            saved_files = {}
            
            # Generate timestamp for filenames
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save volumes if requested
            if save_volumes:
                volumes_dir = os.path.join(directory, "volume_array")
                os.makedirs(volumes_dir, exist_ok=True)
                
                volumes_path = os.path.join(volumes_dir, f"volume_array_{timestamp}.npy")
                np.save(volumes_path, self.volumes_void)
                
                saved_files["volumes"] = volumes_path
                log_success(f"Saved volumes to {volumes_path}")
            
            # Save materials if requested
            if save_materials:
                materials_dir = os.path.join(directory, "material_array")
                os.makedirs(materials_dir, exist_ok=True)
                
                materials_path = os.path.join(materials_dir, f"material_array_{timestamp}.npy")
                np.save(materials_path, self.materials_void)
                
                saved_files["materials"] = materials_path
                log_success(f"Saved materials to {materials_path}")
            
            log_success("Sculpture saved successfully")
            end_section()
            
            return saved_files
            
        except Exception as e:
            log_error(f"Error saving sculpture: {str(e)}")
            end_section("Sculpture save failed")
            raise
    
    @classmethod
    def load(
        cls,
        volumes_path: str,
        materials_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Load a sculpture from files.
        
        Args:
            volumes_path: Path to the volumes array file
            materials_path: Path to the materials array file
            verbose: Whether to print detailed information
        
        Returns:
            A new Sculptor instance with the loaded arrays
        """
        begin_section("Loading Sculpture")
        
        try:
            # Load volumes
            volumes = np.load(volumes_path)
            
            # Load materials if provided
            if materials_path:
                materials = np.load(materials_path, allow_pickle=True)
            else:
                materials = np.empty(volumes.shape, dtype=object)
            
            # Create a new Sculptor instance
            void_dim = volumes.shape[0]
            sculptor = cls(void_dim=void_dim, verbose=verbose)
            
            # Set the arrays
            sculptor.volumes_void = volumes
            sculptor.materials_void = materials
            
            log_success(f"Loaded volumes from {volumes_path}")
            if materials_path:
                log_success(f"Loaded materials from {materials_path}")
            
            log_success("Sculpture loaded successfully")
            end_section()
            
            return sculptor
            
        except Exception as e:
            log_error(f"Error loading sculpture: {str(e)}")
            end_section("Sculpture load failed")
            raise

# Main function for demonstrating the Sculptor
if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Start a sculpture generation demo
    begin_section("DeepSculpt Demonstration")
    log_info("Creating a complex 3D sculpture with multiple components")
    
    # Create a sculptor with customized parameters
    sculptor = Sculptor(
        void_dim=20,                # Size of the 3D grid
        edges=(3, 0.2, 0.6),        # 3 edges with sizes between 20-60% of void_dim
        planes=(2, 0.3, 0.7),       # 2 planes with sizes between 30-70% of void_dim
        pipes=(2, 0.4, 0.7),        # 2 pipes with sizes between 40-70% of void_dim
        grid=(1, 4),                # Enable grid with step size 4
        verbose=True                # Print detailed information
    )
    
    # Generate the sculpture
    log_action("Generating the sculpture", is_last=False)
    volumes, materials = sculptor.generate_sculpture()
    
    # Calculate statistics
    filled_voxels = np.sum(volumes > 0)
    total_voxels = volumes.size
    log_info(f"Created sculpture with {filled_voxels} filled voxels out of {total_voxels} ({filled_voxels/total_voxels*100:.2f}%)")
    
    # Visualize the sculpture from multiple angles
    log_action("Visualizing the sculpture", is_last=False)
    
    # Full multi-view visualization
    fig_path = os.path.join(output_dir, f"sculpture_multiview_{timestamp}.png")
    sculptor.visualize(
        title="DeepSculpt 3D Sculpture",
        hide_axis=True,
        save_path=fig_path,
        save_array=True,
        save_dir=output_dir
    )
    log_success(f"Saved multi-view visualization to {fig_path}")
    
    # Visualize different sections of the sculpture
    log_action("Visualizing sculpture sections", is_last=False)
    sections_path = os.path.join(output_dir, f"sculpture_sections_{timestamp}.png")
    section_fig = sculptor.visualizer.plot_sections(
        volumes,
        title="Sculpture Cross-Sections",
        show=False,
        save_path=sections_path
    )
    log_success(f"Saved sections visualization to {sections_path}")
    
    # Create a point cloud visualization
    log_action("Creating point cloud visualization", is_last=False)
    cloud_path = os.path.join(output_dir, f"sculpture_pointcloud_{timestamp}.html")
    sculptor.visualize_pointcloud(
        title="Sculpture Point Cloud",
        save_path=cloud_path
    )
    log_success(f"Saved point cloud visualization to {cloud_path}")
    
    # Create an animated rotation
    log_action("Creating animated rotation", is_last=False)
    animation_path = os.path.join(output_dir, f"sculpture_rotation_{timestamp}.gif")
    sculptor.create_animated_rotation(
        title="Rotating Sculpture",
        n_frames=36,  # One frame every 10 degrees
        fps=12,       # 12 frames per second (3 second animation)
        hide_axis=True,
        save_path=animation_path
    )
    log_success(f"Saved animation to {animation_path}")
    
    # Save the sculpture arrays
    log_action("Saving sculpture data", is_last=True)
    saved_files = sculptor.save(directory=output_dir)
    
    # Show all the figures
    plt.show()
    
    log_success("Demonstration completed successfully")
    end_section()