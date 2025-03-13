"""
Dataset Generation System for DeepSculpt
This module handles the generation of 3D sculptures for creating machine learning
datasets. It manages the creation, storage, and sampling of large collections of 
sculptures with consistent parametric variation for training purposes.

Key features:
- Individual sample generation: Creates and saves individual sculpture samples
- Organized storage: Manages samples in a structured date-based folder system
- Parameter variation: Controls diversity across the dataset
- Storage management: Handles file naming, directory creation, and persistence
- Sample visualization: Displays and saves visualizations of generated samples
- Progress tracking: Monitors generation progress with status updates

Dependencies:
- logger.py: For process tracking and status reporting
- sculptor.py: For individual sculpture generation
- visualization.py: For displaying samples from the dataset
- numpy: For array operations
- datetime: For timestamped file naming
- random: For sample selection
- tqdm: For progress bar visualization

Used by:
- curator.py: For loading and processing generated datasets
- training scripts: For generating training and validation data

Terminology:
- structure: 3D numpy array representing the sculpture shape (formerly "void")
- colors: 3D numpy array with color information (formerly "color_void")
"""

import os
import time
import random
import numpy as np
from datetime import date, datetime
from typing import List, Tuple, Dict, Any, Optional, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

from logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)
from sculptor import Sculptor
from visualization import Visualizer


class Collector:
    """
    Class for generating and collecting sculptures for dataset creation.
    """

    def __init__(
        self,
        void_dim: int = 32,
        edges: Tuple[int, float, float] = (0, 0.3, 0.5),
        planes: Tuple[int, float, float] = (0, 0.3, 0.5),
        pipes: Tuple[int, float, float] = (2, 0.3, 0.5),
        grid: Tuple[int, int] = (1, 4),
        step: Optional[int] = None,
        base_dir: Optional[str] = "data",
        total_samples: int = 100,
        colors: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Initialize the Collector instance.

        Args:
            void_dim: Size of the 3D grid in each dimension
            edges: Tuple of (count, min_ratio, max_ratio) for edge elements
            planes: Tuple of (count, min_ratio, max_ratio) for plane elements
            pipes: Tuple of (count, min_ratio, max_ratio) for pipe/volume elements
            grid: Tuple of (enable, step) for grid generation
            step: Step size for shape dimensions (if None, calculated as void_dim/6)
            base_dir: Base directory to save generated data
            total_samples: Total number of samples to generate
            colors: Dictionary of colors for different shape types
            seed: Random seed for reproducibility
            verbose: Whether to print detailed information
        """
        self.void_dim = void_dim
        self.edges = edges
        self.planes = planes
        self.pipes = pipes
        self.grid = grid

        # Calculate step if not provided
        self.step = int(void_dim / 6) if step is None else step

        # Set directory
        self.base_dir = base_dir

        # Set sample count
        self.total_samples = total_samples

        # Default colors if not provided
        if colors is None:
            self.colors = {
                "edges": "red",
                "planes": "green",
                "pipes": ["blue", "cyan", "magenta"],
                "volumes": ["purple", "brown", "orange"],
            }
        else:
            self.colors = colors

        self.verbose = verbose

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Create a visualizer
        self.visualizer = Visualizer(figsize=15, dpi=100)

        # Date string for folder structure
        self.date_str = date.today().isoformat()

        # Setup directory structure
        self._setup_directory_structure()

    def _setup_directory_structure(self):
        """Create the directory structure for samples and visualizations"""
        # Create date folder
        self.date_dir = os.path.join(self.base_dir, self.date_str)

        # Create samples directory
        self.samples_dir = os.path.join(self.date_dir, "samples")
        os.makedirs(self.samples_dir, exist_ok=True)

        # Create structures and colors subdirectories
        self.structures_dir = os.path.join(self.samples_dir, "structures")
        self.colors_dir = os.path.join(self.samples_dir, "colors")
        os.makedirs(self.structures_dir, exist_ok=True)
        os.makedirs(self.colors_dir, exist_ok=True)

        # Create visualizations directory
        self.visualizations_dir = os.path.join(self.date_dir, "visualizations")
        os.makedirs(self.visualizations_dir, exist_ok=True)

    def create_collection(self) -> List[Tuple[str, str]]:
        """
        Generate a collection of 3D sculpted shapes as individual samples.

        Returns:
            List of tuples containing (volume_path, material_path) for each sample
        """
        begin_section(f"Creating Collection of {self.total_samples} samples")

        try:
            # List to store paths to all generated samples
            samples_paths = []

            # Generate samples
            iterator = range(self.total_samples)
            if not self.verbose:
                iterator = tqdm(
                    iterator,
                    desc=f"Generating samples",
                    unit="sample",
                )

            for i in iterator:
                # Format sample number with leading zeros (5 digits)
                sample_num = f"{i+1:05d}"

                # Start timer for this sample
                start_time = time.time()

                if self.verbose:
                    log_action(f"Generating sample {sample_num}/{self.total_samples}")

                # Create a sculptor
                sculptor = Sculptor(
                    void_dim=self.void_dim,
                    edges=self.edges,
                    planes=self.planes,
                    pipes=self.pipes,
                    grid=self.grid,
                    colors=self.colors,
                    step=self.step,
                    verbose=False,  # Reduce verbosity for individual samples
                )

                # Generate a sculpture
                structure, colors = sculptor.generate_sculpture()

                # Save the sample
                structure_path = os.path.join(
                    self.structures_dir, f"structure_{sample_num}.npy"
                )
                colors_path = os.path.join(self.colors_dir, f"colors_{sample_num}.npy")

                np.save(file=structure_path, arr=structure)
                np.save(file=colors_path, arr=colors)

                # Add to list of samples
                samples_paths.append((structure_path, colors_path))

                # Create visualization (for every 10th sample or last sample)
                if (i + 1) % 10 == 0 or i == self.total_samples - 1:
                    viz_path = os.path.join(
                        self.visualizations_dir, f"sample_{sample_num}.png"
                    )
                    self.visualizer.plot_sculpture(
                        structure=structure,
                        colors=colors,
                        title=f"Sample {sample_num}",
                        hide_axis=True,
                        save_path=viz_path,
                    )

                    if self.verbose:
                        log_info(f"Saved visualization to {viz_path}")

                # Log progress for verbose mode
                if self.verbose:
                    log_success(
                        f"Generated sample {sample_num} in {time.time() - start_time:.2f} seconds"
                    )

            # Generate a metadata file with information about the collection
            self._save_metadata()

            log_success(f"Successfully generated {self.total_samples} samples")
            log_info(f"Samples saved in: {self.samples_dir}")
            log_info(f"Visualizations saved in: {self.visualizations_dir}")
            end_section()

            return samples_paths

        except Exception as e:
            log_error(f"Error creating collection: {str(e)}")
            end_section("Collection creation failed")
            raise

    def _save_metadata(self):
        """Save metadata about the collection"""
        metadata = {
            "date": self.date_str,
            "void_dim": self.void_dim,
            "edges": self.edges,
            "planes": self.planes,
            "pipes": self.pipes,
            "grid": self.grid,
            "step": self.step,
            "total_samples": self.total_samples,
            "timestamp": datetime.now().isoformat(),
        }

        # Convert to pretty JSON string
        import json

        metadata_str = json.dumps(metadata, indent=4)

        # Save to file
        metadata_path = os.path.join(self.date_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            f.write(metadata_str)

        log_success(f"Saved collection metadata to {metadata_path}")

    @staticmethod
    def list_available_collections(base_dir: str = "data") -> List[str]:
        """
        List all available collections by date.

        Args:
            base_dir: Base directory where collections are stored

        Returns:
            List of date strings representing available collections
        """
        if not os.path.exists(base_dir):
            return []

        # Get all subdirectories that could be date collections
        collections = []
        for item in os.listdir(base_dir):
            full_path = os.path.join(base_dir, item)
            if os.path.isdir(full_path):
                # Check if it has the expected structure
                if os.path.exists(
                    os.path.join(full_path, "samples")
                ) and os.path.exists(os.path.join(full_path, "visualizations")):
                    collections.append(item)

        # Sort by date (assuming YYYY-MM-DD format)
        collections.sort()
        return collections

    @staticmethod
    def load_sample(
        structure_path: str, colors_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a previously saved sample.

        Args:
            structure_path: Path to the structure .npy file
            colors_path: Path to the colors .npy file

        Returns:
            Tuple of (structure, colors) arrays
        """
        begin_section(f"Loading sample from {structure_path} and {colors_path}")

        try:
            # Load arrays
            structure = np.load(file=structure_path, allow_pickle=True)
            colors = np.load(file=colors_path, allow_pickle=True)

            log_success(
                f"Loaded structure with shape {structure.shape} and colors with shape {colors.shape}"
            )
            end_section()

            return structure, colors

        except Exception as e:
            log_error(f"Error loading sample: {str(e)}")
            end_section("Sample loading failed")
            raise


# Example usage
if __name__ == "__main__":
    # Set parameters for a small test
    output_dir = "data"

    # Create a collector
    collector = Collector(
        void_dim=20,  # Size of the 3D grid
        edges=(2, 0.2, 0.6),  # 2 edges with sizes between 20-60% of void_dim
        planes=(1, 0.3, 0.7),  # 1 plane with sizes between 30-70% of void_dim
        pipes=(1, 0.4, 0.7),  # 1 pipe with sizes between 40-70% of void_dim
        grid=(1, 4),  # Enable grid with step size 4
        step=None,  # Auto-calculate step size
        base_dir=output_dir,  # Output directory
        total_samples=10,  # Generate 10 samples
        verbose=True,  # Print detailed information
    )

    # Generate the collection
    sample_paths = collector.create_collection()

    # List available collections
    collections = Collector.list_available_collections(output_dir)
    print(f"Available collections: {collections}")

    # Load a sample (the first one generated)
    if sample_paths:
        structure_path, colors_path = sample_paths[0]
        structure, colors = Collector.load_sample(structure_path, colors_path)
        print(f"Loaded sample with shape {structure.shape}")

    # Visualize random samples from the collection
    if collections:
        latest_collection = collections[-1]
        collection_path = os.path.join(output_dir, latest_collection)
        samples_dir = os.path.join(collection_path, "samples")
        viz_dir = os.path.join(collection_path, "visualizations", "random_samples")

        # Create a visualizer
        visualizer = Visualizer(figsize=10, dpi=100)

        # Visualize random samples
        visualizer.visualize_samples_from_directory(
            directory=samples_dir,
            n_samples=3,  # Visualize 3 random samples
            output_dir=viz_dir,
            angles=[0, 1],  # Show two angles
        )
        print(f"Visualized random samples from {samples_dir} to {viz_dir}")
