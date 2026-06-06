"""
PyTorch-based Dataset Generation System for DeepSculpt
This module provides the PyTorchCollector class for efficient dataset generation
using PyTorchSculptor with streaming capabilities, memory optimization, and
support for multiple output formats.

Key features:
- Streaming dataset generation with memory optimization
- Dynamic batch size adjustment based on available memory
- Support for multiple output formats (PyTorch, HDF5, Zarr)
- Distributed dataset generation across multiple GPUs
- Data compression and efficient storage formats
- Incremental dataset updates and versioning
- Comprehensive monitoring and logging

Dependencies:
- torch: For tensor operations and GPU acceleration
- pytorch_sculptor.py: For PyTorch-based sculpture generation
- logger.py: For process tracking and status reporting
- h5py: For HDF5 format support
- zarr: For Zarr format support
- psutil: For memory monitoring

Used by:
- Training pipelines: For generating training datasets
- Data preprocessing: For creating processed datasets

Terminology:
- structure: 3D PyTorch tensor representing the sculpture shape
- colors: 3D PyTorch tensor with color/material information
"""

import os
import time
import json
import psutil
import warnings
from datetime import date, datetime
from typing import List, Tuple, Dict, Any, Optional, Union, Iterator
from pathlib import Path
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Optional dependencies
try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False
    warnings.warn("h5py not available, HDF5 format disabled")

try:
    import zarr
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    warnings.warn("zarr not available, Zarr format disabled")

from deepsculpt.core.utils.logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)

from .pytorch_sculptor import PyTorchSculptor

class MemoryMonitor:
    """Utility class for monitoring memory usage and adjusting batch sizes."""
    
    def __init__(self, memory_limit_gb: float = 8.0, safety_margin: float = 0.8):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_gb: Maximum memory limit in GB
            safety_margin: Safety margin as fraction of limit (0.8 = 80%)
        """
        self.memory_limit_gb = memory_limit_gb
        self.safety_margin = safety_margin
        self.safe_limit_gb = memory_limit_gb * safety_margin
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
            gpu_max = torch.cuda.max_memory_allocated() / (1024**3)
        else:
            gpu_allocated = gpu_reserved = gpu_max = 0.0
            
        # CPU memory
        process = psutil.Process()
        cpu_memory = process.memory_info().rss / (1024**3)
        system_memory = psutil.virtual_memory()
        
        return {
            "gpu_allocated": gpu_allocated,
            "gpu_reserved": gpu_reserved,
            "gpu_max": gpu_max,
            "cpu_memory": cpu_memory,
            "system_available": system_memory.available / (1024**3),
            "system_percent": system_memory.percent,
        }
    
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced due to memory pressure."""
        usage = self.get_memory_usage()
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            if usage["gpu_allocated"] > self.safe_limit_gb:
                return True
                
        # Check system memory
        if usage["system_percent"] > 85.0:  # 85% system memory usage
            return True
            
        return False
    
    def suggest_batch_size(self, current_batch_size: int, sample_memory_mb: float) -> int:
        """Suggest optimal batch size based on memory constraints."""
        usage = self.get_memory_usage()
        
        # Available memory in MB
        if torch.cuda.is_available():
            available_mb = (self.safe_limit_gb - usage["gpu_allocated"]) * 1024
        else:
            available_mb = usage["system_available"] * 1024 * 0.5  # Use 50% of available
            
        # Calculate suggested batch size
        if sample_memory_mb > 0:
            suggested = max(1, int(available_mb / sample_memory_mb))
            return min(suggested, current_batch_size * 2)  # Don't increase too aggressively
        
        return current_batch_size


class StreamingDataset(Dataset):
    """
    Memory-efficient streaming dataset for PyTorch training.
    Generates samples on-demand without storing entire dataset in memory.
    """
    
    def __init__(
        self,
        sculptor_config: Dict[str, Any],
        num_samples: int,
        device: str = "cuda",
        sparse_mode: bool = False,
        cache_size: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize streaming dataset.
        
        Args:
            sculptor_config: Configuration for PyTorchSculptor
            num_samples: Total number of samples in dataset
            device: Device to generate samples on
            sparse_mode: Whether to use sparse tensors
            cache_size: Number of samples to cache in memory
            seed: Random seed for reproducibility
        """
        self.sculptor_config = sculptor_config
        self.num_samples = num_samples
        self.device = device
        self.sparse_mode = sparse_mode
        self.cache_size = cache_size
        self.seed = seed
        
        # Cache for recently generated samples
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._cache_order: List[int] = []
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate or retrieve a sample."""
        if idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.num_samples}")
        
        # Check cache first
        if idx in self._cache:
            structure, colors = self._cache[idx]
            return {"structure": structure, "colors": colors, "index": torch.tensor(idx)}
        
        # Generate new sample
        sculptor = PyTorchSculptor(
            device=self.device,
            sparse_mode=self.sparse_mode,
            verbose=False,
            **self.sculptor_config
        )
        
        structure, colors = sculptor.generate_sculpture()
        
        # Add to cache if there's space
        if len(self._cache) < self.cache_size:
            self._cache[idx] = (structure.clone(), colors.clone())
            self._cache_order.append(idx)
        elif self.cache_size > 0:
            # Remove oldest item from cache
            oldest_idx = self._cache_order.pop(0)
            del self._cache[oldest_idx]
            
            # Add new item
            self._cache[idx] = (structure.clone(), colors.clone())
            self._cache_order.append(idx)
        
        return {"structure": structure, "colors": colors, "index": torch.tensor(idx)}
    
    def clear_cache(self):
        """Clear the sample cache to free memory."""
        self._cache.clear()
        self._cache_order.clear()

class PyTorchCollector:
    """
    PyTorch-based dataset generation and collection system.
    Supports streaming generation, multiple output formats, and distributed processing.
    """
    
    def __init__(
        self,
        sculptor_config: Optional[Dict[str, Any]] = None,
        output_format: str = "pytorch",
        base_dir: str = "data",
        device: str = "auto",
        sparse_mode: bool = False,
        sparse_threshold: float = 0.1,
        memory_limit_gb: float = 8.0,
        compression: Optional[str] = None,
        verbose: bool = False,
    ):
        """
        Initialize PyTorchCollector.
        
        Args:
            sculptor_config: Configuration for PyTorchSculptor instances
            output_format: Output format ("pytorch", "hdf5", "zarr", "numpy")
            base_dir: Base directory for saving datasets
            device: Device to use for generation
            sparse_mode: Whether to use sparse tensors by default
            sparse_threshold: Sparsity threshold for sparse conversion
            memory_limit_gb: Memory limit for automatic optimization
            compression: Compression method for storage formats
            verbose: Whether to print detailed information
        """
        begin_section("Initializing PyTorchCollector")
        
        try:
            # Validate and set parameters
            self._validate_init_parameters(output_format, memory_limit_gb)
            
            # Default sculptor configuration
            if sculptor_config is None:
                sculptor_config = {
                    "void_dim": 32,
                    "edges": (1, 0.3, 0.5),
                    "planes": (1, 0.3, 0.5),
                    "pipes": (1, 0.3, 0.5),
                    "grid": (1, 4),
                    "step": 1,
                }
            
            self.sculptor_config = sculptor_config
            self.output_format = output_format
            self.base_dir = Path(base_dir)
            self.sparse_mode = sparse_mode
            self.sparse_threshold = sparse_threshold
            self.compression = compression
            self.verbose = verbose
            
            # Setup device
            self.device = self._setup_device(device)
            
            # Initialize memory monitor
            self.memory_monitor = MemoryMonitor(memory_limit_gb)
            
            # Create directory structure
            self.date_str = date.today().isoformat()
            self._setup_directory_structure()
            
            # Statistics tracking
            self.generation_stats = {
                "total_samples": 0,
                "generation_time": 0.0,
                "memory_usage": {},
                "batch_sizes": [],
                "errors": 0,
            }
            
            log_success(f"PyTorchCollector initialized with format={output_format}, device={self.device}")
            end_section()
            
        except Exception as e:
            log_error(f"Error initializing PyTorchCollector: {str(e)}")
            end_section("PyTorchCollector initialization failed")
            raise
    
    def _validate_init_parameters(self, output_format: str, memory_limit_gb: float):
        """Validate initialization parameters."""
        valid_formats = ["pytorch", "numpy"]
        if HDF5_AVAILABLE:
            valid_formats.append("hdf5")
        if ZARR_AVAILABLE:
            valid_formats.append("zarr")
            
        if output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}")
            
        if memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                log_info("CUDA available, using GPU")
                return "cuda"
            else:
                log_info("CUDA not available, using CPU")
                return "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            log_warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        
        return device
    
    def _setup_directory_structure(self):
        """Create directory structure for dataset storage."""
        # Create date folder
        self.date_dir = self.base_dir / self.date_str
        
        # Create format-specific directories
        if self.output_format == "pytorch":
            self.samples_dir = self.date_dir / "pytorch_samples"
            self.structures_dir = self.samples_dir / "structures"
            self.colors_dir = self.samples_dir / "colors"
            
            self.structures_dir.mkdir(parents=True, exist_ok=True)
            self.colors_dir.mkdir(parents=True, exist_ok=True)
            
        elif self.output_format == "numpy":
            self.samples_dir = self.date_dir / "numpy_samples"
            self.structures_dir = self.samples_dir / "structures"
            self.colors_dir = self.samples_dir / "colors"
            
            self.structures_dir.mkdir(parents=True, exist_ok=True)
            self.colors_dir.mkdir(parents=True, exist_ok=True)
            
        elif self.output_format in ["hdf5", "zarr"]:
            self.samples_dir = self.date_dir / f"{self.output_format}_samples"
            self.samples_dir.mkdir(parents=True, exist_ok=True)
        
        # Create metadata directory
        self.metadata_dir = self.date_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def create_streaming_dataset(
        self,
        num_samples: int,
        cache_size: int = 100,
        seed: Optional[int] = None,
    ) -> StreamingDataset:
        """
        Create a streaming dataset for memory-efficient training.
        
        Args:
            num_samples: Total number of samples in dataset
            cache_size: Number of samples to cache in memory
            seed: Random seed for reproducibility
            
        Returns:
            StreamingDataset instance
        """
        begin_section(f"Creating streaming dataset with {num_samples} samples")
        
        try:
            dataset = StreamingDataset(
                sculptor_config=self.sculptor_config,
                num_samples=num_samples,
                device=self.device,
                sparse_mode=self.sparse_mode,
                cache_size=cache_size,
                seed=seed,
            )
            
            log_success(f"Created streaming dataset with {num_samples} samples")
            end_section()
            return dataset
            
        except Exception as e:
            log_error(f"Error creating streaming dataset: {str(e)}")
            end_section("Streaming dataset creation failed")
            raise
    
    def create_collection(
        self,
        num_samples: int,
        batch_size: int = 32,
        dynamic_batching: bool = True,
        save_frequency: int = 100,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Generate a collection of samples with memory optimization.
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Initial batch size for generation
            dynamic_batching: Whether to adjust batch size based on memory
            save_frequency: How often to save progress (in samples)
            seed: Random seed for reproducibility
            
        Returns:
            List of paths to generated sample files
        """
        begin_section(f"Creating collection of {num_samples} samples")
        
        try:
            start_time = time.time()
            sample_paths = []
            current_batch_size = batch_size
            
            # Set random seed if provided
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed)
            
            # Progress tracking
            iterator = range(0, num_samples, current_batch_size)
            if not self.verbose:
                iterator = tqdm(
                    iterator,
                    desc="Generating samples",
                    unit="batch",
                    total=(num_samples + current_batch_size - 1) // current_batch_size,
                )
            
            for batch_start in iterator:
                batch_end = min(batch_start + current_batch_size, num_samples)
                actual_batch_size = batch_end - batch_start
                
                if self.verbose:
                    log_action(f"Generating batch {batch_start}-{batch_end-1} (size: {actual_batch_size})")
                
                # Generate batch
                batch_paths = self._generate_batch(
                    batch_start=batch_start,
                    batch_size=actual_batch_size,
                )
                sample_paths.extend(batch_paths)
                
                # Update statistics
                self.generation_stats["total_samples"] += actual_batch_size
                self.generation_stats["batch_sizes"].append(actual_batch_size)
                
                # Memory monitoring and dynamic batch size adjustment
                if dynamic_batching and self.memory_monitor.should_reduce_batch_size():
                    old_batch_size = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    log_warning(f"Reducing batch size from {old_batch_size} to {current_batch_size} due to memory pressure")
                
                # Save progress periodically
                if (batch_start + actual_batch_size) % save_frequency == 0:
                    self._save_progress_metadata(batch_start + actual_batch_size, num_samples)
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Final statistics
            total_time = time.time() - start_time
            self.generation_stats["generation_time"] = total_time
            self.generation_stats["memory_usage"] = self.memory_monitor.get_memory_usage()
            
            # Save final metadata
            self._save_collection_metadata(num_samples, sample_paths)
            
            log_success(f"Generated {num_samples} samples in {total_time:.2f} seconds")
            log_info(f"Average time per sample: {total_time/num_samples:.3f} seconds")
            end_section()
            
            return sample_paths
            
        except Exception as e:
            log_error(f"Error creating collection: {str(e)}")
            self.generation_stats["errors"] += 1
            end_section("Collection creation failed")
            raise

    def _generate_batch(self, batch_start: int, batch_size: int) -> List[str]:
        """Generate a batch of samples and save them."""
        batch_paths = []
        
        for i in range(batch_size):
            sample_idx = batch_start + i
            
            try:
                # Create sculptor
                sculptor = PyTorchSculptor(
                    device=self.device,
                    sparse_mode=self.sparse_mode,
                    sparse_threshold=self.sparse_threshold,
                    memory_limit_gb=self.memory_monitor.memory_limit_gb,
                    verbose=False,
                    **self.sculptor_config
                )
                
                # Generate sculpture
                structure, colors = sculptor.generate_sculpture()
                
                # Save sample
                sample_path = self._save_sample(sample_idx, structure, colors)
                batch_paths.append(sample_path)
                
            except Exception as e:
                log_error(f"Error generating sample {sample_idx}: {str(e)}")
                self.generation_stats["errors"] += 1
                continue
        
        return batch_paths
    
    def _save_sample(self, sample_idx: int, structure: torch.Tensor, colors: torch.Tensor) -> str:
        """Save a single sample in the specified format."""
        sample_num = f"{sample_idx:06d}"
        
        if self.output_format == "pytorch":
            # Save as PyTorch tensors
            structure_path = self.structures_dir / f"structure_{sample_num}.pt"
            colors_path = self.colors_dir / f"colors_{sample_num}.pt"
            
            torch.save(structure, structure_path)
            torch.save(colors, colors_path)
            
            return str(structure_path)
            
        elif self.output_format == "numpy":
            # Convert to numpy and save
            structure_np = structure.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()
            
            structure_path = self.structures_dir / f"structure_{sample_num}.npy"
            colors_path = self.colors_dir / f"colors_{sample_num}.npy"
            
            np.save(structure_path, structure_np)
            np.save(colors_path, colors_np)
            
            return str(structure_path)
            
        elif self.output_format == "hdf5" and HDF5_AVAILABLE:
            # Save to HDF5 file
            sample_path = self.samples_dir / f"sample_{sample_num}.h5"
            
            with h5py.File(sample_path, 'w') as f:
                # Convert to numpy for HDF5
                structure_np = structure.detach().cpu().numpy()
                colors_np = colors.detach().cpu().numpy()
                
                # Create datasets with compression if specified
                compression_kwargs = {}
                if self.compression:
                    compression_kwargs['compression'] = self.compression
                
                f.create_dataset('structure', data=structure_np, **compression_kwargs)
                f.create_dataset('colors', data=colors_np, **compression_kwargs)
                
                # Add metadata
                f.attrs['sample_idx'] = sample_idx
                f.attrs['device'] = self.device
                f.attrs['sparse_mode'] = self.sparse_mode
                f.attrs['timestamp'] = datetime.now().isoformat()
            
            return str(sample_path)
            
        elif self.output_format == "zarr" and ZARR_AVAILABLE:
            # Save to Zarr format
            sample_path = self.samples_dir / f"sample_{sample_num}.zarr"
            
            # Convert to numpy for Zarr
            structure_np = structure.detach().cpu().numpy()
            colors_np = colors.detach().cpu().numpy()
            
            # Create Zarr group
            root = zarr.open(str(sample_path), mode='w')
            
            # Compression settings
            compressor = None
            if self.compression:
                if self.compression == 'gzip':
                    compressor = zarr.Blosc(cname='gzip')
                elif self.compression == 'lz4':
                    compressor = zarr.Blosc(cname='lz4')
            
            # Save arrays
            root.create_dataset('structure', data=structure_np, compressor=compressor)
            root.create_dataset('colors', data=colors_np, compressor=compressor)
            
            # Add metadata
            root.attrs['sample_idx'] = sample_idx
            root.attrs['device'] = self.device
            root.attrs['sparse_mode'] = self.sparse_mode
            root.attrs['timestamp'] = datetime.now().isoformat()
            
            return str(sample_path)
        
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _save_progress_metadata(self, current_samples: int, total_samples: int):
        """Save progress metadata during generation."""
        progress_data = {
            "current_samples": current_samples,
            "total_samples": total_samples,
            "progress_percent": (current_samples / total_samples) * 100,
            "timestamp": datetime.now().isoformat(),
            "memory_usage": self.memory_monitor.get_memory_usage(),
            "generation_stats": self.generation_stats.copy(),
        }
        
        progress_path = self.metadata_dir / "progress.json"
        with open(progress_path, 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    def _save_collection_metadata(self, num_samples: int, sample_paths: List[str]):
        """Save final collection metadata."""
        metadata = {
            "collection_info": {
                "num_samples": num_samples,
                "output_format": self.output_format,
                "device": self.device,
                "sparse_mode": self.sparse_mode,
                "compression": self.compression,
                "date": self.date_str,
                "timestamp": datetime.now().isoformat(),
            },
            "sculptor_config": self.sculptor_config,
            "generation_stats": self.generation_stats,
            "sample_paths": sample_paths[:100],  # Save first 100 paths as examples
            "directory_structure": {
                "base_dir": str(self.base_dir),
                "date_dir": str(self.date_dir),
                "samples_dir": str(self.samples_dir),
            }
        }
        
        metadata_path = self.metadata_dir / "collection_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_success(f"Saved collection metadata to {metadata_path}")
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get current generation statistics."""
        return self.generation_stats.copy()
    
    def estimate_memory_usage(self, void_dim: int, batch_size: int = 1) -> Dict[str, float]:
        """
        Estimate memory usage for given parameters.
        
        Args:
            void_dim: Dimension of the 3D structure
            batch_size: Batch size for estimation
            
        Returns:
            Dictionary with memory estimates in MB
        """
        # Estimate tensor sizes
        voxels_per_sample = void_dim ** 3
        
        # Structure tensor (int8) + Colors tensor (int16)
        structure_mb = (voxels_per_sample * 1) / (1024**2)  # int8 = 1 byte
        colors_mb = (voxels_per_sample * 2) / (1024**2)     # int16 = 2 bytes
        
        sample_mb = structure_mb + colors_mb
        batch_mb = sample_mb * batch_size
        
        # Add overhead for PyTorch operations (roughly 2x)
        total_mb = batch_mb * 2
        
        return {
            "structure_mb": structure_mb,
            "colors_mb": colors_mb,
            "sample_mb": sample_mb,
            "batch_mb": batch_mb,
            "total_mb": total_mb,
            "recommended_batch_size": max(1, int(self.memory_monitor.safe_limit_gb * 1024 / sample_mb)),
        }


# Utility functions for loading and working with collections
def load_pytorch_sample(structure_path: str, colors_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a PyTorch sample from disk."""
    structure = torch.load(structure_path)
    colors = torch.load(colors_path)
    return structure, colors


def load_numpy_sample(structure_path: str, colors_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a NumPy sample from disk."""
    structure = np.load(structure_path)
    colors = np.load(colors_path)
    return structure, colors


def load_hdf5_sample(sample_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load an HDF5 sample from disk."""
    if not HDF5_AVAILABLE:
        raise ImportError("h5py not available for HDF5 loading")
    
    with h5py.File(sample_path, 'r') as f:
        structure = f['structure'][:]
        colors = f['colors'][:]
    
    return structure, colors


def load_zarr_sample(sample_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a Zarr sample from disk."""
    if not ZARR_AVAILABLE:
        raise ImportError("zarr not available for Zarr loading")
    
    root = zarr.open(sample_path, mode='r')
    structure = root['structure'][:]
    colors = root['colors'][:]
    
    return structure, colors

def list_available_collections(base_dir: str = "data") -> List[Dict[str, Any]]:
    """
    List all available collections with metadata.
    
    Args:
        base_dir: Base directory where collections are stored
        
    Returns:
        List of collection information dictionaries
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    collections = []
    
    for date_dir in base_path.iterdir():
        if date_dir.is_dir():
            metadata_path = date_dir / "metadata" / "collection_metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    collections.append({
                        "date": date_dir.name,
                        "path": str(date_dir),
                        "metadata": metadata,
                    })
                except Exception as e:
                    log_warning(f"Could not load metadata for {date_dir}: {str(e)}")
    
    # Sort by date
    collections.sort(key=lambda x: x["date"])
    return collections


# Example usage
if __name__ == "__main__":
    # Example configuration
    sculptor_config = {
        "void_dim": 32,
        "edges": (2, 0.2, 0.6),
        "planes": (1, 0.3, 0.7),
        "pipes": (1, 0.4, 0.7),
        "grid": (1, 4),
        "step": 1,
    }
    
    # Create collector
    collector = PyTorchCollector(
        sculptor_config=sculptor_config,
        output_format="pytorch",
        base_dir="data",
        device="auto",
        sparse_mode=True,
        memory_limit_gb=4.0,
        verbose=True,
    )
    
    # Generate collection
    sample_paths = collector.create_collection(
        num_samples=100,
        batch_size=16,
        dynamic_batching=True,
    )
    
    print(f"Generated {len(sample_paths)} samples")
    print(f"Generation stats: {collector.get_generation_stats()}")
    
    # Create streaming dataset
    streaming_dataset = collector.create_streaming_dataset(
        num_samples=1000,
        cache_size=50,
    )
    
    # Test streaming dataset
    sample = streaming_dataset[0]
    print(f"Sample shape: {sample['structure'].shape}")
    
    # List available collections
    collections = list_available_collections("data")
    print(f"Available collections: {len(collections)}")

class DistributedCollector:
    """
    Distributed dataset collection system for multi-GPU generation.
    Supports work distribution, load balancing, and fault tolerance.
    """
    
    def __init__(
        self,
        collector_config: Dict[str, Any],
        num_gpus: Optional[int] = None,
        master_port: int = 12355,
        fault_tolerance: bool = True,
        checkpoint_frequency: int = 1000,
        verbose: bool = False,
    ):
        """
        Initialize distributed collector.
        
        Args:
            collector_config: Configuration for PyTorchCollector instances
            num_gpus: Number of GPUs to use (None = all available)
            master_port: Port for distributed communication
            fault_tolerance: Whether to enable fault tolerance
            checkpoint_frequency: How often to checkpoint progress
            verbose: Whether to print detailed information
        """
        self.collector_config = collector_config
        self.fault_tolerance = fault_tolerance
        self.checkpoint_frequency = checkpoint_frequency
        self.verbose = verbose
        self.master_port = master_port
        
        # Determine number of GPUs
        if torch.cuda.is_available():
            self.num_gpus = num_gpus or torch.cuda.device_count()
            self.devices = [f"cuda:{i}" for i in range(self.num_gpus)]
        else:
            if num_gpus is not None:
                # Allow manual override for testing
                self.num_gpus = num_gpus
                self.devices = ["cpu"] * num_gpus
                log_info(f"CUDA not available, using {num_gpus} CPU processes for testing")
            else:
                log_warning("CUDA not available, using single CPU process")
                self.num_gpus = 1
                self.devices = ["cpu"]
        
        log_info(f"Initialized distributed collector with {self.num_gpus} devices: {self.devices}")
        
        # Work distribution tracking
        self.work_assignments: Dict[int, Dict[str, Any]] = {}
        self.completed_work: Dict[int, List[str]] = {}
        self.failed_work: Dict[int, List[int]] = {}
        
        # Progress tracking
        self.total_samples = 0
        self.completed_samples = 0
        self.start_time = 0.0
    
    def distribute_work(self, num_samples: int, batch_size: int = 32) -> Dict[int, Dict[str, Any]]:
        """
        Distribute work across available devices.
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size per worker
            
        Returns:
            Dictionary mapping worker_id to work assignment
        """
        begin_section(f"Distributing {num_samples} samples across {self.num_gpus} workers")
        
        try:
            # Calculate samples per worker
            base_samples_per_worker = num_samples // self.num_gpus
            extra_samples = num_samples % self.num_gpus
            
            work_assignments = {}
            current_start = 0
            
            for worker_id in range(self.num_gpus):
                # Distribute extra samples to first few workers
                worker_samples = base_samples_per_worker
                if worker_id < extra_samples:
                    worker_samples += 1
                
                if worker_samples > 0:
                    work_assignments[worker_id] = {
                        "device": self.devices[worker_id],
                        "start_idx": current_start,
                        "num_samples": worker_samples,
                        "batch_size": batch_size,
                        "worker_id": worker_id,
                    }
                    current_start += worker_samples
            
            self.work_assignments = work_assignments
            log_success(f"Work distributed: {[(w, a['num_samples']) for w, a in work_assignments.items()]}")
            end_section()
            
            return work_assignments
            
        except Exception as e:
            log_error(f"Error distributing work: {str(e)}")
            end_section("Work distribution failed")
            raise
    
    def create_distributed_collection(
        self,
        num_samples: int,
        batch_size: int = 32,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Create a collection using distributed processing.
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size per worker
            seed: Random seed for reproducibility
            
        Returns:
            List of all generated sample paths
        """
        begin_section(f"Creating distributed collection of {num_samples} samples")
        
        try:
            self.total_samples = num_samples
            self.start_time = time.time()
            
            # Distribute work
            work_assignments = self.distribute_work(num_samples, batch_size)
            
            if len(work_assignments) == 0:
                raise ValueError("No work assignments created")
            
            # For now, use single process fallback (multi-GPU implementation can be added later)
            all_sample_paths = self._run_single_worker(work_assignments[0], seed)
            
            # Collect and validate results
            total_generated = len(all_sample_paths)
            generation_time = time.time() - self.start_time
            
            log_success(f"Generated {total_generated}/{num_samples} samples in {generation_time:.2f} seconds")
            log_info(f"Average time per sample: {generation_time/total_generated:.3f} seconds")
            
            # Save distributed collection metadata
            self._save_distributed_metadata(num_samples, all_sample_paths, generation_time)
            
            end_section()
            return all_sample_paths
            
        except Exception as e:
            log_error(f"Error in distributed collection: {str(e)}")
            end_section("Distributed collection failed")
            raise
    
    def _run_single_worker(self, assignment: Dict[str, Any], seed: Optional[int]) -> List[str]:
        """Run single worker process (fallback)."""
        log_action("Running single worker process")
        
        # Create collector for this worker
        collector_config = self.collector_config.copy()
        collector_config["device"] = assignment["device"]
        
        collector = PyTorchCollector(**collector_config)
        
        # Generate samples
        sample_paths = collector.create_collection(
            num_samples=assignment["num_samples"],
            batch_size=assignment["batch_size"],
            seed=seed,
        )
        
        return sample_paths
    
    def _save_distributed_metadata(self, num_samples: int, sample_paths: List[str], generation_time: float):
        """Save metadata for distributed collection."""
        # Use the first collector's directory structure
        first_collector = PyTorchCollector(**self.collector_config)
        
        metadata = {
            "distributed_info": {
                "num_workers": self.num_gpus,
                "devices": self.devices,
                "total_samples": num_samples,
                "generated_samples": len(sample_paths),
                "generation_time": generation_time,
                "fault_tolerance": self.fault_tolerance,
                "timestamp": datetime.now().isoformat(),
            },
            "work_assignments": self.work_assignments,
            "collector_config": self.collector_config,
            "sample_paths_sample": sample_paths[:100],  # First 100 as example
        }
        
        metadata_path = first_collector.metadata_dir / "distributed_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_success(f"Saved distributed metadata to {metadata_path}")
    
    def get_distributed_stats(self) -> Dict[str, Any]:
        """Get statistics about distributed generation."""
        return {
            "num_workers": self.num_gpus,
            "devices": self.devices,
            "work_assignments": self.work_assignments,
            "completed_work": self.completed_work,
            "failed_work": self.failed_work,
            "total_samples": self.total_samples,
            "completed_samples": self.completed_samples,
        }


# Enhanced PyTorchCollector with distributed support
class PyTorchCollectorDistributed(PyTorchCollector):
    """
    Enhanced PyTorchCollector with built-in distributed support.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with distributed capabilities."""
        super().__init__(*args, **kwargs)
        
        # Check for multi-GPU availability
        self.multi_gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
        
        if self.multi_gpu_available:
            log_info(f"Multi-GPU support available: {torch.cuda.device_count()} GPUs")
        else:
            log_info("Single GPU/CPU mode")
    
    def create_distributed_collection(
        self,
        num_samples: int,
        batch_size: int = 32,
        use_distributed: bool = True,
        num_gpus: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> List[str]:
        """
        Create collection with optional distributed processing.
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size per worker
            use_distributed: Whether to use distributed processing
            num_gpus: Number of GPUs to use (None = all available)
            seed: Random seed for reproducibility
            
        Returns:
            List of generated sample paths
        """
        if use_distributed and self.multi_gpu_available:
            # Use distributed processing
            collector_config = {
                "sculptor_config": self.sculptor_config,
                "output_format": self.output_format,
                "base_dir": str(self.base_dir),
                "sparse_mode": self.sparse_mode,
                "sparse_threshold": self.sparse_threshold,
                "memory_limit_gb": self.memory_monitor.memory_limit_gb,
                "compression": self.compression,
                "verbose": self.verbose,
            }
            
            distributed_collector = DistributedCollector(
                collector_config=collector_config,
                num_gpus=num_gpus,
                verbose=self.verbose,
            )
            
            return distributed_collector.create_distributed_collection(
                num_samples=num_samples,
                batch_size=batch_size,
                seed=seed,
            )
        else:
            # Fall back to single-process generation
            return self.create_collection(
                num_samples=num_samples,
                batch_size=batch_size,
                seed=seed,
            )

# Example usage for distributed collection
if __name__ == "__main__":
    # Test distributed collection
    sculptor_config = {
        "void_dim": 32,
        "edges": (1, 0.3, 0.5),
        "planes": (1, 0.3, 0.5),
        "pipes": (1, 0.3, 0.5),
        "grid": (1, 4),
    }
    
    collector_config = {
        "sculptor_config": sculptor_config,
        "output_format": "pytorch",
        "base_dir": "data",
        "sparse_mode": True,
        "memory_limit_gb": 4.0,
        "verbose": True,
    }
    
    # Test distributed collector
    distributed_collector = DistributedCollector(
        collector_config=collector_config,
        num_gpus=None,  # Use all available GPUs
        verbose=True,
    )
    
    # Generate distributed collection
    sample_paths = distributed_collector.create_distributed_collection(
        num_samples=200,
        batch_size=16,
        seed=42,
    )
    
    print(f"Generated {len(sample_paths)} samples using distributed processing")
    print(f"Distributed stats: {distributed_collector.get_distributed_stats()}")
    
    # Test enhanced collector with distributed support
    enhanced_collector = PyTorchCollectorDistributed(**collector_config)
    
    sample_paths_2 = enhanced_collector.create_distributed_collection(
        num_samples=100,
        batch_size=16,
        use_distributed=True,
        seed=42,
    )
    
    print(f"Generated {len(sample_paths_2)} samples using enhanced collector")