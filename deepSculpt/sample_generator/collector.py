"""
Dataset Generation System for DeepSculpt
This module handles the batch generation of 3D sculptures for creating machine learning
datasets. It manages the creation, storage, and sampling of large collections of 
sculptures with consistent parametric variation for training purposes.

Key features:
- Batch generation: Creates multiple sculptures in parallel
- Chunked output: Manages data in manageable chunks for efficient processing
- Parameter variation: Controls diversity across the dataset
- Storage management: Handles file naming, directory creation, and persistence
- Sample visualization: Displays random samples from generated chunks
- Progress tracking: Monitors generation progress with status updates

Dependencies:
- logger.py: For process tracking and status reporting
- sculptor.py: For individual sculpture generation
- visualization.py: For displaying samples from the dataset
- numpy: For array operations and batch processing
- datetime: For timestamped file naming
- random: For sample selection
- tqdm: For progress bar visualization

Used by:
- curator.py: For loading and processing generated datasets
- training scripts: For generating training and validation data

TODO:
- Add parallel processing for faster generation
- Implement parameter space exploration for more diverse datasets
- Add metadata tracking for generated sculptures
- Support for cloud storage integration (S3, GCS)
- Add dataset validation and quality metrics
- Implement continuation of interrupted collections
- Add support for targeted dataset characteristics
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
    Class for generating and collecting batches of sculptures for dataset creation.
    """

    def __init__(
        self,
        void_dim: int = 32,
        edges: Tuple[int, float, float] = (0, 0.3, 0.5),
        planes: Tuple[int, float, float] = (0, 0.3, 0.5),
        pipes: Tuple[int, float, float] = (2, 0.3, 0.5),
        grid: Tuple[int, int] = (1, 4),
        step: Optional[int] = None,
        directory: Optional[str] = "data",
        chunk_size: int = 32,
        n_chunks: int = 10,
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
            directory: Directory to save generated data
            chunk_size: Number of sculptures per chunk
            n_chunks: Number of chunks to generate
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
        self.directory = directory

        # Set batch parameters
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks

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

    def create_collection(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a collection of 3D sculpted shapes in chunks.

        Returns:
            Tuple of the last generated chunk's volumes and materials arrays
        """
        begin_section(
            f"Creating Collection of {self.n_chunks} chunks with {self.chunk_size} sculptures each"
        )

        try:
            # Ensure the output directory exists
            os.makedirs(self.directory, exist_ok=True)

            # Keep track of the last chunk for return value
            last_volumes = None
            last_materials = None

            # Generate chunks
            for chunk_idx in range(self.n_chunks):
                begin_section(f"Generating Chunk {chunk_idx+1}/{self.n_chunks}")

                try:
                    # Start timer
                    start_time = time.time()

                    # Initialize arrays for this chunk
                    volumes_raw_data: List[np.ndarray] = []
                    materials_raw_data: List[np.ndarray] = []

                    # Generate sculptures for this chunk
                    log_action(
                        f"Generating {self.chunk_size} sculptures for chunk {chunk_idx+1}"
                    )

                    # Use tqdm for progress tracking if not verbose
                    iterator = range(self.chunk_size)
                    if not self.verbose:
                        iterator = tqdm(
                            iterator,
                            desc=f"Chunk {chunk_idx+1}/{self.n_chunks}",
                            unit="sculpture",
                        )

                    for i in iterator:
                        # Create a sculptor
                        sculptor = Sculptor(
                            void_dim=self.void_dim,
                            edges=self.edges,
                            planes=self.planes,
                            pipes=self.pipes,
                            grid=self.grid,
                            colors=self.colors,
                            step=self.step,
                            verbose=self.verbose,
                        )

                        # Generate a sculpture
                        (
                            sculpture_volumes,
                            sculpture_materials,
                        ) = sculptor.generate_sculpture()

                        # Add to our arrays
                        volumes_raw_data.append(sculpture_volumes.astype("int8"))
                        materials_raw_data.append(sculpture_materials)

                        # Log progress if verbose
                        if self.verbose and (i + 1) % 5 == 0:
                            log_info(
                                f"Generated {i+1}/{self.chunk_size} sculptures",
                                is_last=(i + 1 == self.chunk_size),
                            )

                    # Convert to numpy arrays
                    volumes_array = (
                        np.asarray(volumes_raw_data)
                        .reshape(
                            (
                                self.chunk_size,
                                self.void_dim,
                                self.void_dim,
                                self.void_dim,
                            )
                        )
                        .astype("int8")
                    )

                    materials_array = (
                        np.asarray(materials_raw_data)
                        .reshape(
                            (
                                self.chunk_size,
                                self.void_dim,
                                self.void_dim,
                                self.void_dim,
                            )
                        )
                        .astype("object")
                    )

                    log_success(
                        f"Generated chunk in {time.time() - start_time:.2f} seconds"
                    )

                    # Keep track of the last chunk
                    last_volumes = volumes_array
                    last_materials = materials_array

                    # Save this chunk
                    timestamp = date.today().isoformat()

                    # Paths for saving
                    volumes_path = os.path.join(
                        self.directory,
                        f"volume_data[{timestamp}]chunk[{chunk_idx+1}].npy",
                    )
                    materials_path = os.path.join(
                        self.directory,
                        f"material_data[{timestamp}]chunk[{chunk_idx+1}].npy",
                    )

                    # Save arrays
                    np.save(volumes_path, volumes_array, allow_pickle=True)
                    np.save(materials_path, materials_array, allow_pickle=True)

                    log_success(
                        f"Saved chunk {chunk_idx+1} to {volumes_path} and {materials_path}"
                    )

                    # Plot some random samples
                    self._plot_samples(
                        volumes_array,
                        materials_array,
                        n_samples=3,
                        chunk_idx=chunk_idx + 1,
                    )

                    end_section(f"Finished chunk {chunk_idx+1}/{self.n_chunks}")

                except Exception as e:
                    log_error(f"Error generating chunk {chunk_idx+1}: {str(e)}")
                    end_section(f"Failed to generate chunk {chunk_idx+1}")
                    raise

            log_success(
                f"Successfully generated {self.n_chunks} chunks with {self.chunk_size} sculptures each"
            )
            end_section()

            return last_volumes, last_materials

        except Exception as e:
            log_error(f"Error creating collection: {str(e)}")
            end_section("Collection creation failed")
            raise

    def _plot_samples(
        self,
        volumes: np.ndarray,
        materials: np.ndarray,
        n_samples: int = 3,
        chunk_idx: int = 0,
    ):
        """
        Plot random samples from the generated batch.

        Args:
            volumes: 4D array of volumes (batch_size, dim, dim, dim)
            materials: 4D array of materials (batch_size, dim, dim, dim)
            n_samples: Number of samples to plot
            chunk_idx: Current chunk index for file naming
        """
        begin_section(f"Plotting {n_samples} sample sculptures from chunk {chunk_idx}")

        try:
            # Create a visualizer
            visualizer = Visualizer(figsize=15, dpi=100)

            # Create samples directory
            samples_dir = os.path.join(self.directory, "samples")
            os.makedirs(samples_dir, exist_ok=True)

            # Get indices of random samples
            indices = random.sample(
                range(volumes.shape[0]), min(n_samples, volumes.shape[0])
            )

            # Plot each sample
            for i, idx in enumerate(indices):
                log_action(
                    f"Plotting sample {i+1}/{n_samples} (index {idx})",
                    is_last=(i == len(indices) - 1),
                )

                # Generate timestamp for unique filenames
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

                # Plot the sculpture
                sample_path = os.path.join(
                    samples_dir, f"sample_chunk{chunk_idx}_idx{idx}_{timestamp}.png"
                )

                visualizer.plot_sculpture(
                    volumes[idx],
                    materials[idx],
                    title=f"Sample {idx} from Chunk {chunk_idx}",
                    hide_axis=True,
                    save_path=sample_path,
                    show=False,
                )

                log_success(f"Saved sample plot to {sample_path}")

            log_success(f"Plotted {n_samples} samples from chunk {chunk_idx}")
            end_section()

        except Exception as e:
            log_error(f"Error plotting samples: {str(e)}")
            end_section("Sample plotting failed")
            raise

    @classmethod
    def load_chunk(
        cls, volumes_path: str, materials_path: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a previously saved chunk.

        Args:
            volumes_path: Path to the volumes .npy file
            materials_path: Path to the materials .npy file

        Returns:
            Tuple of (volumes, materials) arrays
        """
        begin_section(f"Loading chunk from {volumes_path} and {materials_path}")

        try:
            # Load arrays
            volumes = np.load(volumes_path, allow_pickle=True)
            materials = np.load(materials_path, allow_pickle=True)

            log_success(
                f"Loaded volumes with shape {volumes.shape} and materials with shape {materials.shape}"
            )
            end_section()

            return volumes, materials

        except Exception as e:
            log_error(f"Error loading chunk: {str(e)}")
            end_section("Chunk loading failed")
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
        directory=output_dir,  # Output directory
        chunk_size=5,  # 5 sculptures per chunk
        n_chunks=2,  # Generate 2 chunks
        verbose=True,  # Print detailed information
    )

    # Generate the collection
    volumes, materials = collector.create_collection()

    print(
        f"Generated {volumes.shape[0]} sculptures with dimensions {volumes.shape[1:]}."
    )
