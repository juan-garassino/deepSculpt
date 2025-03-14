"""
Dataset Generation System for DeepSculpt

This module manages the generation of 3D sculptures for machine learning datasets:
1. Creates individual and batches of 3D sculptures with controllable parameters
2. Manages efficient storage in a structured date-based folder system
3. Provides visualization and inspection tools for generated samples
4. Supports distributed generation and parallel processing
5. Includes dataset statistics and metadata management

Usage:
    from collector import Collector
    
    # Create a collector instance
    collector = Collector(void_dim=32, samples=100)
    
    # Generate a collection
    samples = collector.create_collection()
"""

import os
import time
import random
import numpy as np
import pandas as pd
import multiprocessing
import logging
import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union, Set, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepSculpt.Collector")

# Try to import DeepSculpt modules
try:
    from sculptor import Sculptor
    from visualization import Visualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    logger.error(f"Failed to import DeepSculpt modules: {e}")
    logger.error("Some functionality may be limited.")
    MODULES_AVAILABLE = False


class CollectionConfig:
    """Configuration holder for Collector."""
    
    def __init__(
        self,
        void_dim: int = 32,
        edges: Tuple[int, float, float] = (2, 0.2, 0.5),
        planes: Tuple[int, float, float] = (1, 0.3, 0.6),
        pipes: Tuple[int, float, float] = (2, 0.3, 0.6),
        grid: Tuple[int, int] = (1, 4),
        step: Optional[int] = None,
        total_samples: int = 100,
        colors: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the collection configuration.
        
        Args:
            void_dim: Size of the 3D grid in each dimension
            edges: Tuple of (count, min_ratio, max_ratio) for edge elements
            planes: Tuple of (count, min_ratio, max_ratio) for plane elements
            pipes: Tuple of (count, min_ratio, max_ratio) for pipe/volume elements
            grid: Tuple of (enable, step) for grid generation
            step: Step size for shape dimensions (if None, calculated as void_dim/6)
            total_samples: Total number of samples to generate
            colors: Dictionary of colors for different shape types
            seed: Random seed for reproducibility
            **kwargs: Additional parameters (for future extension)
        """
        self.void_dim = void_dim
        self.edges = edges
        self.planes = planes
        self.pipes = pipes
        self.grid = grid
        self.step = int(void_dim / 6) if step is None else step
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
            
        self.seed = seed
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "void_dim": self.void_dim,
            "edges": self.edges,
            "planes": self.planes,
            "pipes": self.pipes,
            "grid": self.grid,
            "step": self.step,
            "total_samples": self.total_samples,
            "colors": self.colors,
            "seed": self.seed
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CollectionConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def from_file(cls, file_path: str) -> 'CollectionConfig':
        """Load configuration from a JSON file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """Save configuration to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


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
        num_workers: int = 1,
        batch_size: int = 20,
        config: Optional[CollectionConfig] = None
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
            num_workers: Number of parallel worker processes (if > 1)
            batch_size: Batch size for parallel processing
            config: Optional configuration object (overrides other parameters)
        """
        # Use config if provided, otherwise create one from parameters
        if config:
            self.config = config
        else:
            self.config = CollectionConfig(
                void_dim=void_dim,
                edges=edges,
                planes=planes,
                pipes=pipes,
                grid=grid,
                step=step,
                total_samples=total_samples,
                colors=colors,
                seed=seed
            )
        
        # Store remaining parameters
        self.base_dir = base_dir
        self.verbose = verbose
        self.num_workers = max(1, num_workers)  # Ensure at least 1 worker
        self.batch_size = min(batch_size, total_samples)  # Ensure batch_size <= total_samples
        
        # Set random seed if provided
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
            logger.info(f"Random seed set to {self.config.seed}")
        
        # Create a visualizer if module is available
        self.visualizer = None
        if MODULES_AVAILABLE and 'Visualizer' in globals():
            self.visualizer = Visualizer(figsize=10, dpi=100)
            logger.info("Visualizer initialized")
        
        # Date string for folder structure
        self.date_str = date.today().isoformat()
        
        # Setup directory structure
        self._setup_directory_structure()
        
        # Statistics tracking
        self.statistics = {
            "start_time": None,
            "end_time": None,
            "elapsed_time": None,
            "samples_generated": 0,
            "errors": 0,
            "samples_per_second": 0
        }
        
        # Log initialization
        logger.info(f"Collector initialized with void_dim={self.config.void_dim}, samples={self.config.total_samples}")
        if self.num_workers > 1:
            logger.info(f"Using {self.num_workers} parallel workers with batch_size={self.batch_size}")

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
        
        # Create metadata directory
        self.metadata_dir = os.path.join(self.date_dir, "metadata")
        os.makedirs(self.metadata_dir, exist_ok=True)
        
        # Save configuration
        self.config.save_to_file(os.path.join(self.metadata_dir, "config.json"))
        
        logger.info(f"Directory structure created in {self.date_dir}")

    def _generate_sample(self, sample_idx: int) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a single sculpture sample.
        
        Args:
            sample_idx: Sample index
        
        Returns:
            Tuple of (structure_path, colors_path) or (None, None) on error
        """
        if not MODULES_AVAILABLE or 'Sculptor' not in globals():
            logger.error("Sculptor module not available. Cannot generate sample.")
            return None, None
        
        # Format sample number with leading zeros (5 digits)
        sample_num = f"{sample_idx:05d}"
        
        try:
            # Create a sculptor
            sculptor = Sculptor(
                void_dim=self.config.void_dim,
                edges=self.config.edges,
                planes=self.config.planes,
                pipes=self.config.pipes,
                grid=self.config.grid,
                colors=self.config.colors,
                step=self.config.step,
                verbose=False  # Reduce verbosity for individual samples
            )
            
            # Generate a sculpture
            structure, colors = sculptor.generate_sculpture()
            
            # Save the sample
            structure_path = os.path.join(self.structures_dir, f"structure_{sample_num}.npy")
            colors_path = os.path.join(self.colors_dir, f"colors_{sample_num}.npy")
            
            np.save(file=structure_path, arr=structure)
            np.save(file=colors_path, arr=colors)
            
            # Create visualization (if index is multiple of 10 or visualizer explicitly requested)
            if self.visualizer and (sample_idx % 10 == 0 or self.verbose):
                viz_path = os.path.join(self.visualizations_dir, f"sample_{sample_num}.png")
                self.visualizer.plot_sculpture(
                    structure=structure,
                    colors=colors,
                    title=f"Sample {sample_num}",
                    hide_axis=True,
                    save_path=viz_path,
                )
            
            return structure_path, colors_path
            
        except Exception as e:
            logger.error(f"Error generating sample {sample_num}: {e}")
            return None, None

    def _generate_batch(self, start_idx: int, batch_size: int) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Generate a batch of samples.
        
        Args:
            start_idx: Starting sample index
            batch_size: Number of samples to generate
            
        Returns:
            List of (structure_path, colors_path) tuples
        """
        results = []
        
        for i in range(batch_size):
            sample_idx = start_idx + i
            
            # Skip if beyond the total sample count
            if sample_idx >= self.config.total_samples:
                break
                
            result = self._generate_sample(sample_idx + 1)  # +1 for 1-based indexing
            results.append(result)
        
        return results

    def create_collection(self) -> List[Tuple[str, str]]:
        """
        Generate a collection of 3D sculpted shapes as individual samples.

        Returns:
            List of tuples containing (structure_path, colors_path) for each sample
        """
        logger.info(f"Creating collection of {self.config.total_samples} samples")
        
        # Reset and initialize statistics
        self.statistics["start_time"] = time.time()
        self.statistics["samples_generated"] = 0
        self.statistics["errors"] = 0
        
        # List to store paths to all generated samples
        samples_paths = []
        
        # Single-threaded or multi-threaded generation
        if self.num_workers <= 1:
            # Single-threaded generation
            iterator = range(self.config.total_samples)
            if not self.verbose:
                iterator = tqdm(
                    iterator,
                    desc=f"Generating samples",
                    unit="sample",
                )
            
            for i in iterator:
                sample_idx = i + 1  # Start from 1
                structure_path, colors_path = self._generate_sample(sample_idx)
                
                if structure_path and colors_path:
                    samples_paths.append((structure_path, colors_path))
                    self.statistics["samples_generated"] += 1
                else:
                    self.statistics["errors"] += 1
        
        else:
            # Multi-threaded generation using multiprocessing
            logger.info(f"Using {self.num_workers} workers for parallel generation")
            
            # Create a pool of workers
            pool = multiprocessing.Pool(processes=self.num_workers)
            
            # Prepare batch arguments
            batch_args = []
            for i in range(0, self.config.total_samples, self.batch_size):
                batch_size = min(self.batch_size, self.config.total_samples - i)
                batch_args.append((i, batch_size))
            
            # Execute batch generation in parallel
            try:
                results = []
                with tqdm(total=self.config.total_samples, desc="Generating samples", unit="sample") as pbar:
                    for batch_idx, (start_idx, batch_size) in enumerate(batch_args):
                        # Start this batch
                        batch_future = pool.apply_async(
                            self._generate_batch, 
                            args=(start_idx, batch_size),
                            callback=lambda x: pbar.update(len(x))
                        )
                        results.append(batch_future)
                    
                    # Wait for all futures to complete
                    for result in results:
                        batch_paths = result.get()
                        for structure_path, colors_path in batch_paths:
                            if structure_path and colors_path:
                                samples_paths.append((structure_path, colors_path))
                                self.statistics["samples_generated"] += 1
                            else:
                                self.statistics["errors"] += 1
            
            finally:
                # Clean up the pool
                pool.close()
                pool.join()
        
        # Record end time and calculate statistics
        self.statistics["end_time"] = time.time()
        self.statistics["elapsed_time"] = self.statistics["end_time"] - self.statistics["start_time"]
        
        if self.statistics["elapsed_time"] > 0:
            self.statistics["samples_per_second"] = self.statistics["samples_generated"] / self.statistics["elapsed_time"]
        
        # Save a metadata file with information about the collection
        self._save_metadata()
        
        # Log results
        logger.info(f"Collection created: {self.statistics['samples_generated']} samples "
                   f"in {self.statistics['elapsed_time']:.2f} seconds "
                   f"({self.statistics['samples_per_second']:.2f} samples/sec)")
        
        return samples_paths

    def _save_metadata(self):
        """Save metadata about the collection"""
        metadata = {
            "date": self.date_str,
            "config": self.config.to_dict(),
            "statistics": self.statistics,
            "timestamp": datetime.now().isoformat(),
            "directory_structure": {
                "base_dir": self.base_dir,
                "date_dir": self.date_dir,
                "samples_dir": self.samples_dir,
                "structures_dir": self.structures_dir,
                "colors_dir": self.colors_dir,
                "visualizations_dir": self.visualizations_dir,
                "metadata_dir": self.metadata_dir
            }
        }

        # Save to file
        metadata_path = os.path.join(self.metadata_dir, "collection_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        
        # Also save a copy to the date directory for easier access
        summary_path = os.path.join(self.date_dir, "collection_summary.json")
        
        # Create a simpler summary
        summary = {
            "date": self.date_str,
            "void_dim": self.config.void_dim,
            "total_samples": self.config.total_samples,
            "samples_generated": self.statistics["samples_generated"],
            "elapsed_time": self.statistics["elapsed_time"],
            "samples_per_second": self.statistics["samples_per_second"],
            "errors": self.statistics["errors"],
            "timestamp": datetime.now().isoformat()
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(f"Saved collection metadata to {metadata_path}")
        logger.info(f"Saved collection summary to {summary_path}")

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
                if os.path.exists(os.path.join(full_path, "samples")) and \
                   os.path.exists(os.path.join(full_path, "visualizations")):
                    collections.append(item)

        # Sort by date (assuming YYYY-MM-DD format)
        collections.sort()
        return collections

    @staticmethod
    def load_sample(structure_path: str, colors_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a previously saved sample.

        Args:
            structure_path: Path to the structure .npy file
            colors_path: Path to the colors .npy file

        Returns:
            Tuple of (structure, colors) arrays
        """
        logger.info(f"Loading sample from {structure_path} and {colors_path}")

        try:
            # Load arrays
            structure = np.load(file=structure_path, allow_pickle=True)
            colors = np.load(file=colors_path, allow_pickle=True)

            logger.info(f"Loaded structure with shape {structure.shape} and colors with shape {colors.shape}")
            return structure, colors

        except Exception as e:
            logger.error(f"Error loading sample: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection information
        """
        # Check if metadata file exists
        metadata_path = os.path.join(self.metadata_dir, "collection_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                return json.load(f)
        
        # If no metadata file, return basic info
        return {
            "date": self.date_str,
            "config": self.config.to_dict(),
            "statistics": self.statistics,
            "directory_structure": {
                "date_dir": self.date_dir,
                "samples_dir": self.samples_dir
            }
        }

    def visualize_random_samples(self, num_samples: int = 5, output_dir: Optional[str] = None) -> List[str]:
        """
        Visualize random samples from the collection.
        
        Args:
            num_samples: Number of samples to visualize
            output_dir: Output directory (defaults to visualizations_dir)
            
        Returns:
            List of paths to visualization files
        """
        if not self.visualizer:
            logger.error("Visualizer not available.")
            return []
        
        # Set default output directory
        output_dir = output_dir or os.path.join(self.visualizations_dir, "random_samples")
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all structure files
        structure_files = [f for f in os.listdir(self.structures_dir) if f.endswith(".npy")]
        
        if not structure_files:
            logger.error("No samples found.")
            return []
        
        # Select random samples
        sample_count = min(num_samples, len(structure_files))
        random_samples = random.sample(structure_files, sample_count)
        
        # Visualize each sample
        viz_paths = []
        for i, sample_file in enumerate(random_samples):
            try:
                # Get the corresponding colors file
                sample_id = sample_file.replace("structure_", "").replace(".npy", "")
                colors_file = f"colors_{sample_id}.npy"
                
                # Load the sample
                structure_path = os.path.join(self.structures_dir, sample_file)
                colors_path = os.path.join(self.colors_dir, colors_file)
                
                structure, colors = self.load_sample(structure_path, colors_path)
                
                # Create visualization
                viz_path = os.path.join(output_dir, f"random_sample_{i+1}_{sample_id}.png")
                
                self.visualizer.plot_sculpture(
                    structure=structure,
                    colors=colors,
                    title=f"Sample {sample_id}",
                    hide_axis=True,
                    save_path=viz_path
                )
                
                viz_paths.append(viz_path)
                
            except Exception as e:
                logger.error(f"Error visualizing sample {sample_file}: {e}")
        
        logger.info(f"Visualized {len(viz_paths)} random samples in {output_dir}")
        return viz_paths

    def create_dataset_summary(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a comprehensive summary of the dataset.
        
        Args:
            output_path: Path to save the summary (default: metadata_dir/dataset_summary.json)
            
        Returns:
            Dictionary with dataset summary
        """
        logger.info("Creating dataset summary")
        
        # Set default output path
        output_path = output_path or os.path.join(self.metadata_dir, "dataset_summary.json")
        
        # Get all structure files
        structure_files = [f for f in os.listdir(self.structures_dir) if f.endswith(".npy")]
        colors_files = [f for f in os.listdir(self.colors_dir) if f.endswith(".npy")]
        
        # Calculate statistics
        summary = {
            "date": self.date_str,
            "collection_path": self.date_dir,
            "samples_count": len(structure_files),
            "config": self.config.to_dict(),
            "statistics": self.statistics,
            "file_sizes": {
                "total_structure_files_size": sum(os.path.getsize(os.path.join(self.structures_dir, f)) for f in structure_files),
                "total_colors_files_size": sum(os.path.getsize(os.path.join(self.colors_dir, f)) for f in colors_files),
                "avg_structure_file_size": sum(os.path.getsize(os.path.join(self.structures_dir, f)) for f in structure_files) / max(1, len(structure_files)),
                "avg_colors_file_size": sum(os.path.getsize(os.path.join(self.colors_dir, f)) for f in colors_files) / max(1, len(colors_files))
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Calculate detailed statistics if there are samples
        if structure_files and len(structure_files) > 0:
            # Analyze a sample of structures for detailed statistics
            sample_size = min(10, len(structure_files))
            sample_files = random.sample(structure_files, sample_size)
            
            filled_voxel_counts = []
            
            for sample_file in sample_files:
                try:
                    structure_path = os.path.join(self.structures_dir, sample_file)
                    structure = np.load(structure_path, allow_pickle=True)
                    
                    # Count filled voxels
                    filled_voxels = np.sum(structure > 0)
                    filled_voxel_counts.append(filled_voxels)
                    
                except Exception as e:
                    logger.error(f"Error analyzing sample {sample_file}: {e}")
            
            if filled_voxel_counts:
                avg_filled_voxels = sum(filled_voxel_counts) / len(filled_voxel_counts)
                total_voxels = self.config.void_dim ** 3
                avg_fill_percentage = (avg_filled_voxels / total_voxels) * 100
                
                summary["structure_statistics"] = {
                    "avg_filled_voxels": avg_filled_voxels,
                    "avg_fill_percentage": avg_fill_percentage,
                    "total_voxels_per_sample": total_voxels,
                    "analyzed_sample_count": len(filled_voxel_counts)
                }
        
        # Save summary
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Dataset summary saved to {output_path}")
        return summary

    def clean_up(self, keep_samples: bool = True, keep_visualizations: bool = True, 
               keep_metadata: bool = True) -> bool:
        """
        Clean up the collection directory.
        
        Args:
            keep_samples: Whether to keep sample files
            keep_visualizations: Whether to keep visualization files
            keep_metadata: Whether to keep metadata files
            
        Returns:
            True if clean-up was successful
        """
        logger.info(f"Cleaning up collection directory: {self.date_dir}")
        
        try:
            # Remove samples if requested
            if not keep_samples and os.path.exists(self.samples_dir):
                logger.info(f"Removing samples directory: {self.samples_dir}")
                shutil.rmtree(self.samples_dir)
            
            # Remove visualizations if requested
            if not keep_visualizations and os.path.exists(self.visualizations_dir):
                logger.info(f"Removing visualizations directory: {self.visualizations_dir}")
                shutil.rmtree(self.visualizations_dir)
            
            # Remove metadata if requested
            if not keep_metadata and os.path.exists(self.metadata_dir):
                logger.info(f"Removing metadata directory: {self.metadata_dir}")
                shutil.rmtree(self.metadata_dir)
            
            # Check if the date directory is empty, remove if empty
            if not os.listdir(self.date_dir):
                logger.info(f"Removing empty date directory: {self.date_dir}")
                os.rmdir(self.date_dir)
            
            logger.info("Clean-up completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during clean-up: {e}")
            return False

    @classmethod
    def merge_collections(cls, collections: List[str], output_dir: str, 
                        copy_samples: bool = True, create_merged_visualizations: bool = True) -> Dict[str, Any]:
        """
        Merge multiple collections into a single directory.
        
        Args:
            collections: List of collection directories to merge
            output_dir: Output directory for the merged collection
            copy_samples: Whether to copy sample files (otherwise just create links)
            create_merged_visualizations: Whether to create visualizations for the merged collection
            
        Returns:
            Dictionary with merge information
        """
        logger.info(f"Merging {len(collections)} collections into {output_dir}")
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        merged_samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(merged_samples_dir, exist_ok=True)
        merged_structures_dir = os.path.join(merged_samples_dir, "structures")
        merged_colors_dir = os.path.join(merged_samples_dir, "colors")
        os.makedirs(merged_structures_dir, exist_ok=True)
        os.makedirs(merged_colors_dir, exist_ok=True)
        
        # Create metadata directory
        merged_metadata_dir = os.path.join(output_dir, "metadata")
        os.makedirs(merged_metadata_dir, exist_ok=True)
        
        # Create visualizations directory if needed
        merged_viz_dir = None
        if create_merged_visualizations:
            merged_viz_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(merged_viz_dir, exist_ok=True)
        
        # Initialize merge statistics
        merge_stats = {
            "collections_merged": len(collections),
            "collections": [],
            "total_samples": 0,
            "start_time": time.time(),
            "end_time": None,
            "elapsed_time": None
        }
        
        # Process each collection
        sample_count = 0
        
        for i, collection_dir in enumerate(collections):
            logger.info(f"Processing collection {i+1}/{len(collections)}: {collection_dir}")
            
            try:
                # Get collection metadata if available
                metadata_file = os.path.join(collection_dir, "metadata", "collection_metadata.json")
                collection_metadata = {}
                
                if os.path.exists(metadata_file):
                    with open(metadata_file, "r") as f:
                        collection_metadata = json.load(f)
                
                # Find samples directory
                samples_dir = os.path.join(collection_dir, "samples")
                if not os.path.exists(samples_dir):
                    logger.warning(f"Samples directory not found in {collection_dir}, skipping")
                    continue
                
                # Find structures and colors directories
                structures_dir = os.path.join(samples_dir, "structures")
                colors_dir = os.path.join(samples_dir, "colors")
                
                if not os.path.exists(structures_dir) or not os.path.exists(colors_dir):
                    # Try legacy directory structure
                    structures_dir = samples_dir
                    colors_dir = samples_dir
                
                # Get all structure files
                structure_files = []
                if os.path.exists(structures_dir):
                    structure_files = [f for f in os.listdir(structures_dir) 
                                    if f.endswith(".npy") and 
                                    ("structure_" in f or "volume_" in f)]
                
                # Process each sample
                collection_samples = 0
                
                for struct_file in structure_files:
                    # Determine sample ID and corresponding color file
                    sample_id = None
                    color_file = None
                    
                    if "structure_" in struct_file:
                        sample_id = struct_file.replace("structure_", "").replace(".npy", "")
                        color_file = f"colors_{sample_id}.npy"
                    elif "volume_" in struct_file:
                        sample_id = struct_file.replace("volume_", "").replace(".npy", "")
                        color_file = f"material_{sample_id}.npy"
                    
                    if not sample_id or not os.path.exists(os.path.join(colors_dir, color_file)):
                        logger.warning(f"No matching color file for {struct_file} in {colors_dir}")
                        continue
                    
                    # Create new sample ID for the merged collection
                    new_sample_id = f"{i+1:02d}_{sample_id}"
                    new_struct_file = f"structure_{new_sample_id}.npy"
                    new_color_file = f"colors_{new_sample_id}.npy"
                    
                    # Copy or link files
                    src_struct_path = os.path.join(structures_dir, struct_file)
                    src_color_path = os.path.join(colors_dir, color_file)
                    
                    dst_struct_path = os.path.join(merged_structures_dir, new_struct_file)
                    dst_color_path = os.path.join(merged_colors_dir, new_color_file)
                    
                    if copy_samples:
                        shutil.copy2(src_struct_path, dst_struct_path)
                        shutil.copy2(src_color_path, dst_color_path)
                    else:
                        # Create symbolic links
                        if os.path.exists(dst_struct_path):
                            os.remove(dst_struct_path)
                        if os.path.exists(dst_color_path):
                            os.remove(dst_color_path)
                        
                        os.symlink(os.path.abspath(src_struct_path), dst_struct_path)
                        os.symlink(os.path.abspath(src_color_path), dst_color_path)
                    
                    collection_samples += 1
                    sample_count += 1
                
                # Add collection statistics
                merge_stats["collections"].append({
                    "path": collection_dir,
                    "samples": collection_samples,
                    "metadata": collection_metadata
                })
                
                logger.info(f"Added {collection_samples} samples from {collection_dir}")
                
            except Exception as e:
                logger.error(f"Error processing collection {collection_dir}: {e}")
        
        # Update merge statistics
        merge_stats["total_samples"] = sample_count
        merge_stats["end_time"] = time.time()
        merge_stats["elapsed_time"] = merge_stats["end_time"] - merge_stats["start_time"]
        
        # Save merge metadata
        merge_metadata_path = os.path.join(merged_metadata_dir, "merge_metadata.json")
        with open(merge_metadata_path, "w") as f:
            json.dump(merge_stats, f, indent=4)
        
        # Create visualizations if requested
        if create_merged_visualizations and MODULES_AVAILABLE and 'Visualizer' in globals():
            logger.info("Creating visualizations for merged collection")
            
            # Create a visualizer
            visualizer = Visualizer(figsize=15, dpi=100)
            
            # Visualize random samples
            try:
                visualizer.visualize_samples_from_directory(
                    directory=merged_samples_dir,
                    n_samples=min(20, sample_count),
                    output_dir=merged_viz_dir,
                    angles=[0, 1, 2, 3]
                )
                logger.info(f"Created visualizations in {merged_viz_dir}")
            except Exception as e:
                logger.error(f"Error creating visualizations: {e}")
        
        logger.info(f"Merged {sample_count} samples from {len(collections)} collections")
        logger.info(f"Merged collection saved to {output_dir}")
        
        return merge_stats


# Command-line interface
def main():
    """Command-line interface for the Collector."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DeepSculpt Data Collector")
    
    # Main commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create collection command
    create_parser = subparsers.add_parser("create", help="Create a new collection")
    create_parser.add_argument("--void-dim", type=int, default=32, help="Dimension of void space")
    create_parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    create_parser.add_argument("--edges", type=int, default=2, help="Number of edge elements")
    create_parser.add_argument("--planes", type=int, default=1, help="Number of plane elements")
    create_parser.add_argument("--pipes", type=int, default=2, help="Number of pipe elements")
    create_parser.add_argument("--grid", type=int, default=1, help="Whether to include grid (1=yes, 0=no)")
    create_parser.add_argument("--grid-step", type=int, default=4, help="Grid step size")
    create_parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    create_parser.add_argument("--workers", type=int, default=1, help="Number of worker processes")
    create_parser.add_argument("--batch-size", type=int, default=20, help="Batch size for parallel processing")
    create_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    create_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    # List collections command
    list_parser = subparsers.add_parser("list", help="List available collections")
    list_parser.add_argument("--base-dir", type=str, default="./data", help="Base directory")
    
    # Visualize collection command
    viz_parser = subparsers.add_parser("visualize", help="Visualize samples from a collection")
    viz_parser.add_argument("--collection", type=str, default="latest", help="Collection to visualize (path or 'latest')")
    viz_parser.add_argument("--samples", type=int, default=5, help="Number of samples to visualize")
    viz_parser.add_argument("--base-dir", type=str, default="./data", help="Base directory")
    viz_parser.add_argument("--output-dir", type=str, default=None, help="Output directory for visualizations")
    
    # Merge collections command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple collections")
    merge_parser.add_argument("collections", nargs="+", help="Collections to merge (paths or date strings)")
    merge_parser.add_argument("--output-dir", type=str, required=True, help="Output directory for merged collection")
    merge_parser.add_argument("--base-dir", type=str, default="./data", help="Base directory for collections")
    merge_parser.add_argument("--link", action="store_true", help="Create links instead of copying files")
    merge_parser.add_argument("--no-visualizations", action="store_true", help="Skip creating visualizations")
    
    # Summary command
    summary_parser = subparsers.add_parser("summary", help="Create summary of a collection")
    summary_parser.add_argument("--collection", type=str, default="latest", help="Collection to summarize (path or 'latest')")
    summary_parser.add_argument("--base-dir", type=str, default="./data", help="Base directory")
    summary_parser.add_argument("--output-file", type=str, default=None, help="Output file for summary")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute command
    if args.command == "create":
        # Create a collection
        collector = Collector(
            void_dim=args.void_dim,
            edges=(args.edges, 0.2, 0.5),  # Default min/max ratios
            planes=(args.planes, 0.3, 0.6),
            pipes=(args.pipes, 0.3, 0.6),
            grid=(args.grid, args.grid_step),
            base_dir=args.output_dir,
            total_samples=args.samples,
            seed=args.seed,
            verbose=args.verbose,
            num_workers=args.workers,
            batch_size=args.batch_size
        )
        
        # Generate the collection
        sample_paths = collector.create_collection()
        
        # Print summary
        print(f"\nGenerated {len(sample_paths)} samples")
        print(f"Collection saved to: {collector.date_dir}")
        
    elif args.command == "list":
        # List available collections
        collections = Collector.list_available_collections(args.base_dir)
        
        if collections:
            print(f"Found {len(collections)} collections:")
            for i, collection in enumerate(collections):
                # Try to get collection size
                samples_dir = os.path.join(args.base_dir, collection, "samples", "structures")
                if os.path.exists(samples_dir):
                    sample_count = len([f for f in os.listdir(samples_dir) if f.endswith(".npy")])
                    print(f"  {i+1}. {collection} - {sample_count} samples")
                else:
                    print(f"  {i+1}. {collection}")
        else:
            print("No collections found.")
    
    elif args.command == "visualize":
        # Determine collection path
        collection_path = args.collection
        
        if collection_path == "latest":
            collections = Collector.list_available_collections(args.base_dir)
            if not collections:
                print("No collections found.")
                return
            collection_path = os.path.join(args.base_dir, collections[-1])
            print(f"Using latest collection: {collections[-1]}")
        
        # Create a collector for this existing collection
        collector = Collector(
            base_dir=args.base_dir,
            verbose=True
        )
        
        # Set the correct directories
        collector.date_dir = collection_path if os.path.isabs(collection_path) else os.path.join(args.base_dir, collection_path)
        collector.samples_dir = os.path.join(collector.date_dir, "samples")
        collector.structures_dir = os.path.join(collector.samples_dir, "structures")
        collector.colors_dir = os.path.join(collector.samples_dir, "colors")
        collector.visualizations_dir = os.path.join(collector.date_dir, "visualizations")
        
        # Create output directory if specified
        output_dir = args.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Visualize random samples
        viz_paths = collector.visualize_random_samples(
            num_samples=args.samples,
            output_dir=output_dir
        )
        
        if viz_paths:
            print(f"Created {len(viz_paths)} visualizations")
            print(f"Visualizations saved to: {output_dir or collector.visualizations_dir}")
        else:
            print("No visualizations created.")
    
    elif args.command == "merge":
        # Get full paths for collections
        collection_paths = []
        
        for collection in args.collections:
            if os.path.isabs(collection):
                collection_paths.append(collection)
            elif os.path.exists(os.path.join(args.base_dir, collection)):
                collection_paths.append(os.path.join(args.base_dir, collection))
            else:
                print(f"Warning: Collection not found: {collection}")
        
        if not collection_paths:
            print("No valid collections found to merge.")
            return
        
        # Merge collections
        merge_stats = Collector.merge_collections(
            collections=collection_paths,
            output_dir=args.output_dir,
            copy_samples=not args.link,
            create_merged_visualizations=not args.no_visualizations
        )
        
        # Print summary
        print(f"\nMerged {merge_stats['total_samples']} samples from {len(collection_paths)} collections")
        print(f"Merged collection saved to: {args.output_dir}")
    
    elif args.command == "summary":
        # Determine collection path
        collection_path = args.collection
        
        if collection_path == "latest":
            collections = Collector.list_available_collections(args.base_dir)
            if not collections:
                print("No collections found.")
                return
            collection_path = os.path.join(args.base_dir, collections[-1])
            print(f"Using latest collection: {collections[-1]}")
        
        # Create a collector for this existing collection
        collector = Collector(
            base_dir=args.base_dir,
            verbose=True
        )
        
        # Set the correct directories
        collector.date_dir = collection_path if os.path.isabs(collection_path) else os.path.join(args.base_dir, collection_path)
        collector.samples_dir = os.path.join(collector.date_dir, "samples")
        collector.structures_dir = os.path.join(collector.samples_dir, "structures")
        collector.colors_dir = os.path.join(collector.samples_dir, "colors")
        collector.visualizations_dir = os.path.join(collector.date_dir, "visualizations")
        collector.metadata_dir = os.path.join(collector.date_dir, "metadata")
        os.makedirs(collector.metadata_dir, exist_ok=True)
        
        # Create summary
        summary = collector.create_dataset_summary(args.output_file)
        
        # Print brief summary
        print(f"\nCollection Summary for {collector.date_dir}")
        print(f"  Samples: {summary['samples_count']}")
        if "structure_statistics" in summary:
            print(f"  Average fill: {summary['structure_statistics']['avg_fill_percentage']:.2f}%")
        print(f"  Total structure files size: {summary['file_sizes']['total_structure_files_size'] / 1024 / 1024:.2f} MB")
        print(f"  Total colors files size: {summary['file_sizes']['total_colors_files_size'] / 1024 / 1024:.2f} MB")
        print(f"Summary saved to: {args.output_file or os.path.join(collector.metadata_dir, 'dataset_summary.json')}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()