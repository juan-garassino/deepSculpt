"""
DeepSculpt Workflow Manager

This module provides a consolidated workflow system for the DeepSculpt project, integrating:
1. Complete pipeline from data generation to model serving
2. Prefect workflows for orchestrating the entire process
3. MLflow tracking for comprehensive experiment monitoring
4. Notification system for alerting on workflow progress
5. Utilities for data, model, and visualization management

Usage:
    python workflow.py [--mode development|production] [--complete]
"""

import os
import numpy as np
import tensorflow as tf
import pandas as pd
import datetime
import requests
import errno
import json
import glob
import re
import time
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DeepSculpt")

# Try to import cloud storage
try:
    from google.cloud import storage
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False
    logger.warning("Google Cloud Storage not available. Cloud storage features will be disabled.")

# Try to import colorama for enhanced console output
try:
    from colorama import Fore, Style
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False
    logger.warning("Colorama not available. Colored output will be disabled.")
    
    # Create dummy Fore and Style classes
    class DummyColorClass:
        def __getattr__(self, name):
            return ""
    
    Fore = DummyColorClass()
    Style = DummyColorClass()

# Try to import workflow management libraries
try:
    from prefect import task, Flow, Parameter
    from prefect.schedules import IntervalSchedule
    from prefect.executors import LocalDaskExecutor
    from prefect.run_configs import LocalRun
    PREFECT_AVAILABLE = True
except ImportError:
    PREFECT_AVAILABLE = False
    logger.warning("Prefect not available. Workflow automation will be disabled.")
    
    # Create dummy decorators and classes
    def task(func=None, **kwargs):
        def decorator(f):
            return f
        
        if func:
            return decorator(func)
        return decorator
    
    class Parameter:
        def __init__(self, name, default=None):
            self.name = name
            self.default = default
    
    class Flow:
        def __init__(self, name, schedule=None):
            self.name = name
            self.schedule = schedule
        
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        
        def run(self, parameters=None):
            logger.warning("Prefect not available. Flow.run() is a no-op.")
            return None
        
        def register(self, project_name=None):
            logger.warning("Prefect not available. Flow.register() is a no-op.")
            return None

# Try to import ML experiment tracking
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Experiment tracking will be disabled.")

# Try to import project modules
try:
    # Import DeepSculpt modules
    from models import ModelFactory
    from trainer import DeepSculptTrainer, DataFrameDataLoader, EncodedDataLoader
    from collector import Collector
    from curator import Curator
    from visualization import Visualizer
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    logger.error(f"DeepSculpt modules not available: {e}")
    logger.error("Some functionality may be limited.")


class Manager:
    """
    General utility manager for the DeepSculpt project.
    Handles data loading, model management, visualization, and MLflow integration.
    """
    
    def __init__(self, model_name="deepSculpt", data_name="data", verbose=False):
        """
        Initialize the manager.
        
        Args:
            model_name: Name of the model
            data_name: Name of the dataset
            verbose: Whether to print detailed information
        """
        self.model_name = model_name
        self.data_name = data_name
        self.comment = f"{model_name}_{data_name}"
        self.data_subdir = f"{model_name}/{data_name}"
        self.verbose = verbose
        
        # Initialize visualizer if available
        self.visualizer = None
        if 'Visualizer' in globals():
            self.visualizer = Visualizer(figsize=15, dpi=100)

        # Print initialization message
        self._print_colored(
            f"\n🚀 Initialized DeepSculpt Manager: {model_name} with {data_name}",
            Fore.GREEN
        )
    
    def _print_colored(self, message, color=Fore.WHITE, end="\n"):
        """Print colored message if colorama is available."""
        print(f"{color}{message}{Style.RESET_ALL}", end=end)
    
    def load_locally(self, path_volumes_array, path_materials_array):
        """
        Load volume and material data from local files.
        
        Args:
            path_volumes_array: Path to volume data file
            path_materials_array: Path to material data file
            
        Returns:
            Tuple of (volumes_array, materials_array)
        """
        self._print_colored(f"Loading data from local files...", Fore.CYAN)
        
        try:
            raw_volumes_array = np.load(path_volumes_array, allow_pickle=True)
            raw_materials_array = np.load(path_materials_array, allow_pickle=True)
            
            self._print_colored(
                f"Loaded 'volume_data' shaped {raw_volumes_array.shape} and "
                f"'material_data' shaped {raw_materials_array.shape}",
                Fore.BLUE
            )
            
            return (raw_volumes_array, raw_materials_array)
        except Exception as e:
            self._print_colored(f"Error loading local data: {e}", Fore.RED)
            raise
    
    def load_from_gcp(self, path_volumes=None, path_materials=None):
        """
        Load volume and material data from GCP storage.
        
        Args:
            path_volumes: Path to volume data in GCP (optional)
            path_materials: Path to material data in GCP (optional)
            
        Returns:
            Tuple of (volumes_array, materials_array)
        """
        if not GCP_AVAILABLE:
            self._print_colored(
                "Google Cloud Storage not available. Please install google-cloud-storage.",
                Fore.RED
            )
            raise ImportError("Google Cloud Storage not available")
        
        self._print_colored(f"Loading data from GCP...", Fore.CYAN)
        
        self.path_volumes = path_volumes or "volume_data.npy"
        self.path_materials = path_materials or "material_data.npy"
        
        files = [self.path_volumes, self.path_materials]
        
        try:
            client = storage.Client().bucket(os.environ.get("BUCKET_NAME"))
            
            for file in files:
                blob = client.blob(os.environ.get("BUCKET_TRAIN_DATA_PATH") + "/" + file)
                blob.download_to_filename(file)
            
            train_size = int(os.environ.get("TRAIN_SIZE", "1000"))
            raw_volumes = np.load(self.path_volumes, allow_pickle=True)[:train_size]
            raw_materials = np.load(self.path_materials, allow_pickle=True)[:train_size]
            
            self._print_colored(
                f"Loaded 'volume_data' shaped {raw_volumes.shape} and "
                f"'material_data' shaped {raw_materials.shape}",
                Fore.BLUE
            )
            
            return (raw_volumes, raw_materials)
        except Exception as e:
            self._print_colored(f"Error loading data from GCP: {e}", Fore.RED)
            raise
    
    @staticmethod
    def upload_snapshot_to_gcp(snapshot_name, bucket_name=None):
        """
        Upload a snapshot image to GCP storage.
        
        Args:
            snapshot_name: Name of the snapshot file to upload
            bucket_name: Name of the GCP bucket (defaults to env var)
        """
        if not GCP_AVAILABLE:
            logger.warning("Google Cloud Storage not available. Cannot upload snapshot.")
            return
        
        try:
            STORAGE_FILENAME = snapshot_name
            storage_location = f"results/{STORAGE_FILENAME}"
            
            bucket_name = bucket_name or os.environ.get("BUCKET_NAME")
            bucket = storage.Client().bucket(bucket_name)
            blob = bucket.blob(storage_location)
            blob.upload_from_filename(STORAGE_FILENAME)
            
            logger.info(f"Uploaded snapshot to GCP: {STORAGE_FILENAME} @ {storage_location}")
        except Exception as e:
            logger.error(f"Error uploading snapshot to GCP: {e}")
    
    @staticmethod
    def save_mlflow_model(metrics=None, params=None, model=None, model_name=None, artifact_path=None):
        """
        Save model, parameters, and metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            params: Dictionary of parameters to log
            model: Keras model to save
            model_name: Name to register the model under (defaults to env var)
            artifact_path: Path in artifact store (defaults to 'model')
            
        Returns:
            Boolean indicating success
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Cannot save model.")
            return False
        
        try:
            # Retrieve MLflow env params
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
            mlflow_model_name = model_name or os.environ.get("MLFLOW_MODEL_NAME")
            artifact_path = artifact_path or "model"
            
            # Configure MLflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=mlflow_experiment)
            
            with mlflow.start_run():
                # STEP 1: Push parameters to MLflow
                if params is not None:
                    mlflow.log_params(params)
                
                # STEP 2: Push metrics to MLflow
                if metrics is not None:
                    mlflow.log_metrics(metrics)
                
                # STEP 3: Push model to MLflow
                if model is not None:
                    mlflow.keras.log_model(
                        keras_model=model,
                        artifact_path=artifact_path,
                        keras_module="tensorflow.keras",
                        registered_model_name=mlflow_model_name,
                    )
            
            logger.info(f"Data saved in MLflow: {mlflow_model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving to MLflow: {e}")
            return False
    
    @staticmethod
    def load_mlflow_model(stage="Production", model_name=None):
        """
        Load a model from MLflow.
        
        Args:
            stage: Stage of the model to load (e.g., "Production", "Staging")
            model_name: Name of the registered model (defaults to env var)
            
        Returns:
            Loaded Keras model or None if not found
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Cannot load model.")
            return None
        
        logger.info(f"Loading model {stage} stage from MLflow...")
        
        try:
            # Load model from MLflow
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
            mlflow_model_name = model_name or os.environ.get("MLFLOW_MODEL_NAME")
            
            model_uri = f"models:/{mlflow_model_name}/{stage}"
            logger.info(f"Model URI: {model_uri}")
            
            model = mlflow.keras.load_model(model_uri=model_uri)
            logger.info(f"Model loaded from MLflow: {model_uri}")
            
            return model
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            logger.warning(f"No model in stage {stage} on MLflow")
            return None
    
    @staticmethod
    def get_model_version(stage="Production", model_name=None):
        """
        Retrieve the version number of the latest model in the given stage.
        
        Args:
            stage: Stage of the model to check (e.g., "Production", "Staging")
            model_name: Name of the registered model (defaults to env var)
            
        Returns:
            Version number or None if not found
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available. Cannot get model version.")
            return None
        
        if os.environ.get("MODEL_TARGET") == "mlflow":
            try:
                mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
                mlflow_model_name = model_name or os.environ.get("MLFLOW_MODEL_NAME")
                
                client = MlflowClient()
                
                version = client.get_latest_versions(
                    name=mlflow_model_name, stages=[stage]
                )
                
                # Check whether a version of the model exists in the given stage
                if not version:
                    return None
                
                return int(version[0].version)
            except Exception as e:
                logger.error(f"Error getting model version: {e}")
                return None
        
        # Model version not handled
        return None
    
    @staticmethod
    def make_directory(directory):
        """Create a directory if it doesn't exist."""
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Directory created or verified: {directory}")
            return True
        except OSError as e:
            if e.errno != errno.EEXIST:
                logger.error(f"Error creating directory {directory}: {e}")
                raise
            return False
    
    @staticmethod
    def return_axis(void: np.ndarray, color_void: np.ndarray):
        """
        Selects a random plane from a 3D numpy array along a random axis.
        
        Args:
            void: The 3D numpy array to select a plane from
            color_void: The 3D numpy array that holds the color information
            
        Returns:
            Tuple of (working_plane, color_parameters, section)
        """
        section = np.random.randint(low=0, high=void.shape[0])
        axis_selection = np.random.randint(low=0, high=3)
        
        if axis_selection == 0:
            working_plane = void[section, :, :]
            color_parameters = color_void[section, :, :]
        elif axis_selection == 1:
            working_plane = void[:, section, :]
            color_parameters = color_void[:, section, :]
        elif axis_selection == 2:
            working_plane = void[:, :, section]
            color_parameters = color_void[:, :, section]
        else:
            logger.error("Axis selection value out of range.")
            raise ValueError("Axis selection value out of range.")
        
        return working_plane, color_parameters, section
    
    @staticmethod
    def create_animation(frames_path, output_name="animation", fps=30):
        """
        Create an animation from a sequence of images.
        
        Args:
            frames_path: Path to directory containing image frames
            output_name: Name of the output animation file
            fps: Frames per second for the animation
        """
        # Get a list of all image files in the directory
        image_files = sorted(
            [
                f
                for f in os.listdir(frames_path)
                if f.endswith(".png") or f.endswith(".jpg")
            ]
        )
        
        # Load the image files into an array of image arrays
        images = [imageio.imread(os.path.join(frames_path, f)) for f in image_files]
        
        # Create the animation and save it as a GIF file
        animation_path = os.path.join(frames_path, f"{output_name}.gif")
        
        imageio.mimsave(animation_path, images, fps=fps)
        logger.info(f"Animation saved to {animation_path}")
    
    @staticmethod
    def get_rgb_from_color_array(color_array):
        """Convert color array to RGB values."""
        rgb_array = np.zeros((*color_array.shape[:3], 3))
        for i in range(color_array.shape[0]):
            for j in range(color_array.shape[1]):
                for k in range(color_array.shape[2]):
                    color = color_array[i, j, k]
                    if color is not None:
                        rgb_array[i, j, k] = mcolors.to_rgb(color)
        return rgb_array * 255
    
    @staticmethod
    def convert_to_matplotlib_colors(arr):
        """
        Convert a 4D numpy array to matplotlib color strings.
        
        Args:
            arr: A 4D numpy array of shape (size, size, size, 3)
            
        Returns:
            A 3D numpy array with matplotlib color strings
        """
        size = arr.shape[0]
        # Initialize an empty array of the same shape as the input array
        result = np.empty((size, size, size), dtype=object)
        
        # Iterate over each pixel in the input array
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    # Get the RGB values of the pixel
                    r, g, b = arr[i, j, k, :]
                    
                    # Convert the RGB values to a matplotlib color string
                    color = mcolors.rgb2hex((r / 255, g / 255, b / 255))
                    
                    # Store the color string in the output array
                    result[i, j, k] = color
        
        return result
    
    @staticmethod
    def create_data_dataframe(data_folder, pattern=None):
        """
        Create a DataFrame with paths to volume and material data files.
        
        Args:
            data_folder: Folder containing data files
            pattern: Optional pattern to filter files (e.g., date string)
            
        Returns:
            Pandas DataFrame with columns for volume and material paths
        """
        # Find all npy files in the folder
        npy_files = glob.glob(os.path.join(data_folder, '**/*.npy'), recursive=True)
        
        # Filter by pattern if provided
        if pattern:
            npy_files = [f for f in npy_files if pattern in f]
        
        # Separate volume and material files
        volume_files = []
        material_files = []
        
        # Check for different naming patterns
        for file_path in npy_files:
            file_name = os.path.basename(file_path)
            if 'volume_data' in file_name or 'structure_' in file_name:
                volume_files.append(file_path)
            elif 'material_data' in file_name or 'colors_' in file_name:
                material_files.append(file_path)
        
        logger.info(f"Found {len(volume_files)} volume files and {len(material_files)} material files")
        
        # Extract chunk indices
        chunk_pattern = re.compile(r'(chunk\[(\d+)\]|_(\d+)\.npy)')
        
        # Create pairs of volume and material files
        data_pairs = []
        
        for volume_file in volume_files:
            # Extract sample identifier
            sample_id = None
            chunk_match = chunk_pattern.search(volume_file)
            
            if chunk_match:
                # Try to get the chunk index
                if chunk_match.group(2):
                    sample_id = chunk_match.group(2)
                elif chunk_match.group(3):
                    sample_id = chunk_match.group(3)
            
            if sample_id is None:
                # Try alternative matching for files like structure_00001.npy
                file_name = os.path.basename(volume_file)
                if 'structure_' in file_name:
                    sample_id = file_name.replace('structure_', '').replace('.npy', '')
                elif 'volume_' in file_name:
                    sample_id = file_name.replace('volume_', '').replace('.npy', '')
            
            # Find corresponding material file
            matching_material = None
            
            if sample_id:
                # Try several patterns for material files
                for material_file in material_files:
                    material_name = os.path.basename(material_file)
                    
                    # Check if the sample_id is in the material file name
                    if sample_id in material_name:
                        matching_material = material_file
                        break
            
            # If no match by ID, try matching by name pattern
            if matching_material is None:
                expected_material_file = None
                
                if 'volume_data' in volume_file:
                    expected_material_file = volume_file.replace('volume_data', 'material_data')
                elif 'structure_' in volume_file:
                    expected_material_file = volume_file.replace('structure_', 'colors_')
                
                if expected_material_file and expected_material_file in material_files:
                    matching_material = expected_material_file
            
            # If we found a matching material file, add the pair
            if matching_material:
                data_pairs.append({
                    'chunk_idx': sample_id if sample_id else -1,
                    'volume_path': volume_file,
                    'material_path': matching_material
                })
        
        # Create DataFrame
        df = pd.DataFrame(data_pairs)
        
        # Sort by chunk index if possible
        if not df.empty and 'chunk_idx' in df.columns:
            try:
                df['chunk_idx'] = pd.to_numeric(df['chunk_idx'])
                df = df.sort_values('chunk_idx')
            except:
                # If conversion fails, just keep the order
                pass
        
        return df


class WorkflowManager:
    """
    Manager for coordinating the complete DeepSculpt workflow:
    1. Data generation and collection
    2. Data preprocessing and curation
    3. Model training and evaluation
    4. Model deployment and serving
    """
    
    def __init__(self, base_dir="./data", results_dir="./results", verbose=False):
        """
        Initialize the workflow manager.
        
        Args:
            base_dir: Base directory for data
            results_dir: Directory for results
            verbose: Whether to print detailed information
        """
        self.base_dir = base_dir
        self.results_dir = results_dir
        self.verbose = verbose
        
        # Create directories
        os.makedirs(base_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize utilities
        self.manager = Manager(verbose=verbose)
        
        # Check module availability
        if not MODULES_AVAILABLE:
            logger.warning("Some DeepSculpt modules are not available.")
            logger.warning("Workflow functionality may be limited.")
        
        logger.info(f"Initialized WorkflowManager with base_dir={base_dir}, results_dir={results_dir}")
    
    def generate_data(self, void_dim=32, samples=100, edges=(2, 0.2, 0.5), 
                    planes=(1, 0.3, 0.6), pipes=(2, 0.3, 0.6), grid=(1, 4), 
                    output_dir=None):
        """
        Generate a dataset using the Collector.
        
        Args:
            void_dim: Size of the 3D grid in each dimension
            samples: Number of samples to generate
            edges: Tuple of (count, min_ratio, max_ratio) for edge elements
            planes: Tuple of (count, min_ratio, max_ratio) for plane elements
            pipes: Tuple of (count, min_ratio, max_ratio) for pipe elements
            grid: Tuple of (enable, step) for grid generation
            output_dir: Output directory (defaults to base_dir)
        
        Returns:
            Dictionary with information about the generated collection
        """
        if not MODULES_AVAILABLE or 'Collector' not in globals():
            logger.error("Collector module not available. Cannot generate data.")
            return None
        
        output_dir = output_dir or self.base_dir
        
        try:
            # Create a collector
            collector = Collector(
                void_dim=void_dim,
                edges=edges,
                planes=planes,
                pipes=pipes,
                grid=grid,
                base_dir=output_dir,
                total_samples=samples,
                verbose=self.verbose
            )
            
            # Generate the collection
            logger.info(f"Generating {samples} samples with void_dim={void_dim}...")
            start_time = time.time()
            sample_paths = collector.create_collection()
            elapsed = time.time() - start_time
            
            # Return collection information
            result = {
                'date_dir': collector.date_dir,
                'samples_dir': collector.samples_dir,
                'sample_count': len(sample_paths),
                'void_dim': void_dim,
                'elapsed_time': elapsed,
                'sample_paths': sample_paths
            }
            
            logger.info(f"Generated {len(sample_paths)} samples in {elapsed:.2f} seconds")
            logger.info(f"Collection saved to: {collector.date_dir}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error generating data: {e}")
            return None
    
    def curate_data(self, collection_dir=None, encoder="OHE", batch_size=32, 
                  output_dir=None, plot_samples=3):
        """
        Curate a collection for machine learning.
        
        Args:
            collection_dir: Path to collection directory (defaults to latest)
            encoder: Encoding method ('OHE', 'BINARY', or 'RGB')
            batch_size: Batch size for the dataset
            output_dir: Output directory (defaults to results_dir/processed)
            plot_samples: Number of samples to visualize
        
        Returns:
            Dictionary with information about the curated dataset
        """
        if not MODULES_AVAILABLE or 'Curator' not in globals():
            logger.error("Curator module not available. Cannot curate data.")
            return None
        
        # Determine collection directory
        if collection_dir is None:
            # Find the latest collection
            collections = Collector.list_available_collections(self.base_dir)
            
            if not collections:
                logger.error("No collections found. Generate data first.")
                return None
            
            collection_dir = os.path.join(self.base_dir, collections[-1])
            logger.info(f"Using latest collection: {collections[-1]}")
        
        # Determine output directory
        output_dir = output_dir or os.path.join(self.results_dir, "processed")
        
        try:
            # Create a curator
            curator = Curator(
                processing_method=encoder,
                output_dir=output_dir,
                verbose=self.verbose
            )
            
            # Process the collection
            logger.info(f"Curating collection with {encoder} encoding...")
            start_time = time.time()
            
            result = curator.preprocess_collection(
                collection_dir=collection_dir,
                batch_size=batch_size,
                buffer_size=1000,
                train_size=None,
                validation_split=0.2,
                plot_samples=plot_samples
            )
            
            elapsed = time.time() - start_time
            
            # Add elapsed time to result
            result['elapsed_time'] = elapsed
            
            logger.info(f"Curated {result['train_size']} training and {result['val_size']} validation samples")
            logger.info(f"Encoded shape: {result['encoded_shape']}")
            logger.info(f"Processed data saved to: {result['output_dir']}")
            logger.info(f"Time elapsed: {elapsed:.2f} seconds")
            
            return result
        
        except Exception as e:
            logger.error(f"Error curating data: {e}")
            return None
    
    def train_model(self, model_type="skip", epochs=100, batch_size=32, 
                  learning_rate=0.0002, curated_data=None, data_folder=None,
                  checkpoint_dir=None, snapshot_dir=None, snapshot_freq=5,
                  save_to_mlflow=True):
        """
        Train a DeepSculpt model.
        
        Args:
            model_type: Type of model to train
            epochs: Number of epochs to train for
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            curated_data: Path to curated data directory (optional)
            data_folder: Path to raw data folder (if curated_data not provided)
            checkpoint_dir: Directory for checkpoints (defaults to results_dir/checkpoints)
            snapshot_dir: Directory for snapshots (defaults to results_dir/snapshots)
            snapshot_freq: Frequency of saving snapshots (in epochs)
            save_to_mlflow: Whether to save the model to MLflow
        
        Returns:
            Dictionary with training results and metrics
        """
        if not MODULES_AVAILABLE:
            logger.error("DeepSculpt modules not available. Cannot train model.")
            return None
        
        # Create results directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set default directories if not provided
        checkpoint_dir = checkpoint_dir or os.path.join(self.results_dir, "checkpoints", model_type, timestamp)
        snapshot_dir = snapshot_dir or os.path.join(self.results_dir, "snapshots", model_type, timestamp)
        results_dir = os.path.join(self.results_dir, f"{model_type}_{timestamp}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(snapshot_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        logger.info(f"Training {model_type} model for {epochs} epochs")
        logger.info(f"Results will be saved to {results_dir}")
        
        try:
            # Determine whether to use curated data or raw data
            if curated_data:
                logger.info(f"Using curated data from {curated_data}")
                data_loader = EncodedDataLoader(
                    data_dir=curated_data,
                    batch_size=batch_size,
                    shuffle=True
                )
            elif data_folder:
                logger.info(f"Using raw data from {data_folder}")
                # Create DataFrame from raw data
                data_df = self.manager.create_data_dataframe(data_folder)
                
                if data_df.empty:
                    logger.error("No data files found! Please check your data folder.")
                    return None
                
                logger.info(f"Found {len(data_df)} data pairs")
                
                # Save DataFrame for future use
                data_file_path = os.path.join(results_dir, "data_paths.csv")
                data_df.to_csv(data_file_path, index=False)
                logger.info(f"Saved data paths to: {data_file_path}")
                
                # Create data loader
                data_loader = DataFrameDataLoader(
                    df=data_df,
                    batch_size=batch_size,
                    shuffle=True
                )
            else:
                logger.error("Either curated_data or data_folder must be provided")
                return None
            
            # Create models
            logger.info(f"Creating {model_type} models")
            
            if model_type == "autoencoder":
                # For autoencoder, create encoder and decoder
                from models import create_encoder, ModelFactory
                
                # Set environment variables for backwards compatibility
                os.environ["VOID_DIM"] = "64"
                os.environ["NOISE_DIM"] = "100"
                os.environ["COLOR"] = "1"
                
                latent_dim = 100
                void_dim = 64
                
                encoder = create_encoder(latent_dim=latent_dim, 
                                       input_shape=(void_dim, void_dim, void_dim, 3))
                decoder = ModelFactory.create_generator(model_type="autoencoder", 
                                                      void_dim=void_dim, 
                                                      noise_dim=latent_dim)
                
                # Create autoencoder trainer
                from trainer import AutoencoderTrainer
                
                trainer = AutoencoderTrainer(
                    encoder=encoder,
                    decoder=decoder,
                    learning_rate=learning_rate
                )
                
                # Train the model
                logger.info(f"Starting autoencoder training for {epochs} epochs")
                metrics = trainer.train(
                    data_loader=data_loader,
                    epochs=epochs,
                    checkpoint_dir=checkpoint_dir,
                    snapshot_dir=snapshot_dir,
                    snapshot_freq=snapshot_freq
                )
                
                # Save the final models
                encoder.save(os.path.join(results_dir, "encoder_final"))
                decoder.save(os.path.join(results_dir, "decoder_final"))
                
                # Save to MLflow if requested
                if save_to_mlflow and MLFLOW_AVAILABLE:
                    params = {
                        "model_type": model_type,
                        "latent_dim": latent_dim,
                        "void_dim": void_dim,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate
                    }
                    
                    # Prepare metrics for MLflow
                    mlflow_metrics = {}
                    if "total_loss" in metrics and metrics["total_loss"]:
                        mlflow_metrics["final_loss"] = float(metrics["total_loss"][-1])
                    if "epoch_times" in metrics:
                        mlflow_metrics["avg_epoch_time"] = float(np.mean(metrics["epoch_times"]))
                        mlflow_metrics["total_train_time"] = float(np.sum(metrics["epoch_times"]))
                    
                    # Save encoder and decoder to MLflow
                    self.manager.save_mlflow_model(
                        metrics=mlflow_metrics,
                        params=params,
                        model=encoder,
                        model_name=f"{os.environ.get('MLFLOW_MODEL_NAME', 'deepSculpt')}_encoder",
                        artifact_path="encoder"
                    )
                    
                    self.manager.save_mlflow_model(
                        metrics=mlflow_metrics,
                        params=params,
                        model=decoder,
                        model_name=f"{os.environ.get('MLFLOW_MODEL_NAME', 'deepSculpt')}_decoder",
                        artifact_path="decoder"
                    )
                
                # Plot and save metrics
                metrics_path = os.path.join(results_dir, "training_metrics.png")
                trainer.plot_metrics(save_path=metrics_path)
                
                return {
                    'model_type': model_type,
                    'results_dir': results_dir,
                    'checkpoint_dir': checkpoint_dir,
                    'snapshot_dir': snapshot_dir,
                    'metrics': metrics,
                    'encoder_model': encoder,
                    'decoder_model': decoder
                }
            
            else:
                # For GANs, create generator and discriminator
                generator = ModelFactory.create_generator(model_type=model_type)
                discriminator = ModelFactory.create_discriminator(model_type=model_type)
                
                # Create trainer
                trainer = DeepSculptTrainer(
                    generator=generator,
                    discriminator=discriminator,
                    learning_rate=learning_rate,
                    beta1=0.5,
                    beta2=0.999
                )
                
                # Train the model
                logger.info(f"Starting GAN training for {epochs} epochs")
                metrics = trainer.train(
                    data_loader=data_loader,
                    epochs=epochs,
                    checkpoint_dir=checkpoint_dir,
                    snapshot_dir=snapshot_dir,
                    snapshot_freq=snapshot_freq
                )
                
                # Save the final models
                generator.save(os.path.join(results_dir, "generator_final"))
                discriminator.save(os.path.join(results_dir, "discriminator_final"))
                
                # Save to MLflow if requested
                if save_to_mlflow and MLFLOW_AVAILABLE:
                    params = {
                        "model_type": model_type,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate
                    }
                    
                    # Prepare metrics for MLflow
                    mlflow_metrics = {}
                    if "gen_loss" in metrics and metrics["gen_loss"]:
                        mlflow_metrics["final_gen_loss"] = float(metrics["gen_loss"][-1])
                    if "disc_loss" in metrics and metrics["disc_loss"]:
                        mlflow_metrics["final_disc_loss"] = float(metrics["disc_loss"][-1])
                    if "epoch_times" in metrics:
                        mlflow_metrics["avg_epoch_time"] = float(np.mean(metrics["epoch_times"]))
                        mlflow_metrics["total_train_time"] = float(np.sum(metrics["epoch_times"]))
                    
                    # Save generator to MLflow
                    self.manager.save_mlflow_model(
                        metrics=mlflow_metrics,
                        params=params,
                        model=generator
                    )
                
                # Plot and save metrics
                metrics_path = os.path.join(results_dir, "training_metrics.png")
                trainer.plot_metrics(save_path=metrics_path)
                
                return {
                    'model_type': model_type,
                    'results_dir': results_dir,
                    'checkpoint_dir': checkpoint_dir,
                    'snapshot_dir': snapshot_dir,
                    'metrics': metrics,
                    'generator_model': generator,
                    'discriminator_model': discriminator
                }
        
        except Exception as e:
            logger.error(f"Error training model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def evaluate_model(self, model=None, model_path=None, stage="Production", 
                     num_samples=10, data_loader=None, data_folder=None):
        """
        Evaluate a model by generating samples and computing metrics.
        
        Args:
            model: Model to evaluate (optional)
            model_path: Path to saved model (optional)
            stage: MLflow stage to load from (if model and model_path not provided)
            num_samples: Number of samples to generate
            data_loader: Data loader for evaluation (optional)
            data_folder: Folder with evaluation data (if data_loader not provided)
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get the model
            if model is None:
                if model_path:
                    logger.info(f"Loading model from {model_path}")
                    model = tf.keras.models.load_model(model_path)
                else:
                    logger.info(f"Loading {stage} model from MLflow")
                    model = self.manager.load_mlflow_model(stage=stage)
                    
                    if model is None:
                        logger.error(f"No model found in MLflow stage: {stage}")
                        return None
            
            # Generate samples
            logger.info(f"Generating {num_samples} samples for evaluation")
            noise = tf.random.normal([num_samples, 100])  # Assuming noise_dim=100
            generated_samples = model(noise, training=False)
            
            # Calculate basic metrics
            avg_value = float(tf.reduce_mean(generated_samples).numpy())
            std_value = float(tf.math.reduce_std(generated_samples).numpy())
            
            # If a data loader or data folder is provided, we can calculate more advanced metrics
            if data_loader is not None or data_folder is not None:
                logger.info("Computing advanced metrics with real data")
                
                if data_loader is None and data_folder is not None:
                    # Create data loader from folder
                    data_df = self.manager.create_data_dataframe(data_folder)
                    
                    if data_df.empty:
                        logger.error("No data files found in evaluation folder")
                    else:
                        data_loader = DataFrameDataLoader(
                            df=data_df,
                            batch_size=32,
                            shuffle=False
                        )
                
                if data_loader is not None:
                    # Create TensorFlow dataset
                    dataset = data_loader.create_tf_dataset()
                    
                    # Get a batch of real samples
                    real_samples = next(iter(dataset.take(1)))
                    
                    # TODO: Implement FID score, Inception Score, or other GAN metrics
                    # For now, we'll use a simple MSE between statistics of real and generated
                    
                    real_mean = float(tf.reduce_mean(real_samples).numpy())
                    real_std = float(tf.math.reduce_std(real_samples).numpy())
                    
                    mean_diff = abs(real_mean - avg_value)
                    std_diff = abs(real_std - std_value)
                    
                    logger.info(f"Real samples statistics - Mean: {real_mean:.4f}, Std: {real_std:.4f}")
                    logger.info(f"Generated samples statistics - Mean: {avg_value:.4f}, Std: {std_value:.4f}")
                    logger.info(f"Differences - Mean: {mean_diff:.4f}, Std: {std_diff:.4f}")
                    
                    # Additional metrics
                    metrics = {
                        "real_mean": real_mean,
                        "real_std": real_std,
                        "generated_mean": avg_value,
                        "generated_std": std_value,
                        "mean_difference": mean_diff,
                        "std_difference": std_diff
                    }
                else:
                    metrics = {
                        "generated_mean": avg_value,
                        "generated_std": std_value
                    }
            else:
                metrics = {
                    "generated_mean": avg_value,
                    "generated_std": std_value
                }
            
            # Save generated samples
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            samples_dir = os.path.join(self.results_dir, "evaluation", timestamp)
            os.makedirs(samples_dir, exist_ok=True)
            
            # Save samples as numpy arrays
            np.save(os.path.join(samples_dir, "generated_samples.npy"), generated_samples.numpy())
            
            # Try to visualize samples if visualizer is available
            viz_paths = []
            if self.manager.visualizer:
                logger.info("Visualizing generated samples")
                
                for i in range(min(10, num_samples)):
                    # Convert to appropriate format
                    sample = generated_samples[i].numpy()
                    structure = (sample > 0.5).astype(np.int8)
                    
                    # Create visualization
                    viz_path = os.path.join(samples_dir, f"sample_{i:03d}.png")
                    
                    self.manager.visualizer.plot_sculpture(
                        structure=structure,
                        title=f"Generated Sample {i}",
                        hide_axis=True,
                        save_path=viz_path
                    )
                    
                    viz_paths.append(viz_path)
            
            # Return evaluation results
            return {
                "metrics": metrics,
                "samples_dir": samples_dir,
                "viz_paths": viz_paths,
                "timestamp": timestamp
            }
        
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_complete_workflow(self, config=None):
        """
        Run the complete DeepSculpt workflow from data generation to model evaluation.
        
        Args:
            config: Dictionary with workflow configuration
        
        Returns:
            Dictionary with workflow results
        """
        # Define default config
        default_config = {
            # Data generation
            "generate_data": True,
            "void_dim": 32,
            "samples": 100,
            "edges": (2, 0.2, 0.5),
            "planes": (1, 0.3, 0.6),
            "pipes": (2, 0.3, 0.6),
            "grid": (1, 4),
            
            # Data curation
            "curate_data": True,
            "encoder": "OHE",
            "batch_size": 32,
            "plot_samples": 3,
            
            # Model training
            "train_model": True,
            "model_type": "skip",
            "epochs": 100,
            "learning_rate": 0.0002,
            "snapshot_freq": 5,
            "save_to_mlflow": True,
            
            # Model evaluation
            "evaluate_model": True,
            "num_eval_samples": 10,
            
            # Notification
            "send_notification": True,
            "notification_url": None
        }
        
        # Update with provided config
        if config:
            for key, value in config.items():
                default_config[key] = value
        
        config = default_config
        
        # Track workflow results
        results = {
            "workflow": {
                "start_time": datetime.datetime.now().isoformat(),
                "config": config,
                "steps_completed": []
            }
        }
        
        # Step 1: Generate data if requested
        if config["generate_data"]:
            logger.info("Step 1: Generating data")
            data_result = self.generate_data(
                void_dim=config["void_dim"],
                samples=config["samples"],
                edges=config["edges"],
                planes=config["planes"],
                pipes=config["pipes"],
                grid=config["grid"]
            )
            
            if data_result:
                results["data_generation"] = data_result
                results["workflow"]["steps_completed"].append("data_generation")
                logger.info("Data generation completed successfully")
            else:
                logger.error("Data generation failed")
                if config["send_notification"]:
                    self._send_notification(
                        "DeepSculpt Workflow Failed at Data Generation",
                        config["notification_url"]
                    )
                return results
        
        # Step 2: Curate data if requested
        if config["curate_data"]:
            logger.info("Step 2: Curating data")
            collection_dir = None
            if "data_generation" in results:
                collection_dir = results["data_generation"]["date_dir"]
            
            curation_result = self.curate_data(
                collection_dir=collection_dir,
                encoder=config["encoder"],
                batch_size=config["batch_size"],
                plot_samples=config["plot_samples"]
            )
            
            if curation_result:
                results["data_curation"] = curation_result
                results["workflow"]["steps_completed"].append("data_curation")
                logger.info("Data curation completed successfully")
            else:
                logger.error("Data curation failed")
                if config["send_notification"]:
                    self._send_notification(
                        "DeepSculpt Workflow Failed at Data Curation",
                        config["notification_url"]
                    )
                return results
        
        # Step 3: Train model if requested
        if config["train_model"]:
            logger.info("Step 3: Training model")
            curated_data = None
            data_folder = None
            
            if "data_curation" in results:
                curated_data = results["data_curation"]["output_dir"]
            elif "data_generation" in results:
                data_folder = results["data_generation"]["date_dir"]
            
            training_result = self.train_model(
                model_type=config["model_type"],
                epochs=config["epochs"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                curated_data=curated_data,
                data_folder=data_folder,
                snapshot_freq=config["snapshot_freq"],
                save_to_mlflow=config["save_to_mlflow"]
            )
            
            if training_result:
                results["model_training"] = training_result
                results["workflow"]["steps_completed"].append("model_training")
                logger.info("Model training completed successfully")
            else:
                logger.error("Model training failed")
                if config["send_notification"]:
                    self._send_notification(
                        "DeepSculpt Workflow Failed at Model Training",
                        config["notification_url"]
                    )
                return results
        
        # Step 4: Evaluate model if requested
        if config["evaluate_model"]:
            logger.info("Step 4: Evaluating model")
            model = None
            data_folder = None
            
            if "model_training" in results:
                if "generator_model" in results["model_training"]:
                    model = results["model_training"]["generator_model"]
                elif "decoder_model" in results["model_training"]:
                    model = results["model_training"]["decoder_model"]
            
            if "data_generation" in results:
                data_folder = results["data_generation"]["date_dir"]
            
            evaluation_result = self.evaluate_model(
                model=model,
                num_samples=config["num_eval_samples"],
                data_folder=data_folder
            )
            
            if evaluation_result:
                results["model_evaluation"] = evaluation_result
                results["workflow"]["steps_completed"].append("model_evaluation")
                logger.info("Model evaluation completed successfully")
            else:
                logger.error("Model evaluation failed")
                if config["send_notification"]:
                    self._send_notification(
                        "DeepSculpt Workflow Failed at Model Evaluation",
                        config["notification_url"]
                    )
                # Continue since evaluation is optional
        
        # Record completion time
        results["workflow"]["end_time"] = datetime.datetime.now().isoformat()
        
        # Calculate elapsed time
        start_time = datetime.datetime.fromisoformat(results["workflow"]["start_time"])
        end_time = datetime.datetime.fromisoformat(results["workflow"]["end_time"])
        elapsed_seconds = (end_time - start_time).total_seconds()
        results["workflow"]["elapsed_seconds"] = elapsed_seconds
        
        # Log completion
        logger.info(f"Workflow completed in {elapsed_seconds:.2f} seconds")
        logger.info(f"Steps completed: {results['workflow']['steps_completed']}")
        
        # Send notification if requested
        if config["send_notification"]:
            self._send_notification(
                f"DeepSculpt Workflow Completed: {', '.join(results['workflow']['steps_completed'])}",
                config["notification_url"],
                results
            )
        
        return results
    
    def _send_notification(self, message, webhook_url=None, data=None):
        """
        Send a notification about workflow status.
        
        Args:
            message: Notification message
            webhook_url: URL for webhook (optional)
            data: Additional data to include (optional)
        
        Returns:
            Boolean indicating success
        """
        try:
            # Log the notification message
            logger.info(f"Notification: {message}")
            
            # If no webhook URL, just log the message
            if not webhook_url:
                return True
            
            # Format notification data
            payload = {
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Add data if provided
            if data:
                # Include summary information only
                summary = {}
                if "workflow" in data:
                    summary["workflow"] = {
                        "start_time": data["workflow"]["start_time"],
                        "end_time": data["workflow"].get("end_time"),
                        "elapsed_seconds": data["workflow"].get("elapsed_seconds"),
                        "steps_completed": data["workflow"]["steps_completed"]
                    }
                
                steps_data = {}
                for step in ["data_generation", "data_curation", "model_training", "model_evaluation"]:
                    if step in data:
                        # Include only key metrics for each step
                        if step == "data_generation":
                            steps_data[step] = {
                                "sample_count": data[step]["sample_count"],
                                "elapsed_time": data[step]["elapsed_time"]
                            }
                        elif step == "data_curation":
                            steps_data[step] = {
                                "train_size": data[step]["train_size"],
                                "val_size": data[step]["val_size"],
                                "encoder": data[step]["encoder"] if "encoder" in data[step] else None,
                                "elapsed_time": data[step]["elapsed_time"]
                            }
                        elif step == "model_training":
                            steps_data[step] = {
                                "model_type": data[step]["model_type"],
                                "results_dir": data[step]["results_dir"]
                            }
                            # Add final metrics if available
                            if "metrics" in data[step]:
                                metrics = data[step]["metrics"]
                                if "gen_loss" in metrics and metrics["gen_loss"]:
                                    steps_data[step]["final_gen_loss"] = float(metrics["gen_loss"][-1])
                                if "disc_loss" in metrics and metrics["disc_loss"]:
                                    steps_data[step]["final_disc_loss"] = float(metrics["disc_loss"][-1])
                                if "total_loss" in metrics and metrics["total_loss"]:
                                    steps_data[step]["final_loss"] = float(metrics["total_loss"][-1])
                        elif step == "model_evaluation":
                            if "metrics" in data[step]:
                                steps_data[step] = data[step]["metrics"]
                
                payload["data"] = summary
                payload["steps"] = steps_data
            
            # Send the webhook request
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info(f"Notification sent successfully to {webhook_url}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
            return False


# Prefect task definitions
@task
def preprocess_data(experiment, data_folder):
    """
    Preprocess DeepSculpt data and create a DataFrame for training.
    
    Args:
        experiment: MLflow experiment name
        data_folder: Folder containing the data
        
    Returns:
        Path to the saved DataFrame
    """
    logger.info("Preprocessing data...")
    
    # Create Manager instance
    manager = Manager()
    
    # Create DataFrame for data
    data_df = manager.create_data_dataframe(data_folder)
    
    if data_df.empty:
        logger.error("No data files found!")
        return None
    
    # Save DataFrame to disk
    output_path = os.path.join(data_folder, "processed", "data_paths.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_df.to_csv(output_path, index=False)
    
    logger.info(f"Processed {len(data_df)} data pairs")
    return output_path


@task
def evaluate_model(data_path, model_type="skip", stage="Production"):
    """
    Evaluate the current production model on new data.
    
    Args:
        data_path: Path to the preprocessed data DataFrame
        model_type: Type of model to evaluate
        stage: MLflow model stage to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info(f"Evaluating {stage} model...")
    
    # Create Manager instance
    manager = Manager()
    
    # Load the model from MLflow
    model = manager.load_mlflow_model(stage=stage)
    
    if model is None:
        logger.error(f"No model found in {stage} stage")
        return {"gen_loss": float("inf"), "disc_loss": float("inf")}
    
    # Load data DataFrame
    data_df = pd.read_csv(data_path)
    
    # Create data loader
    data_loader = DataFrameDataLoader(
        df=data_df,
        batch_size=32,
        shuffle=False
    )
    
    # Create TensorFlow dataset
    dataset = data_loader.create_tf_dataset()
    
    # Evaluate the model
    logger.info("Running evaluation...")
    
    # For GANs, we might need custom evaluation metrics
    # Here we'll use a simple approach - generate samples and calculate statistics
    
    # Generate some samples using fixed noise
    noise = tf.random.normal([16, 100])  # Assuming noise_dim=100
    generated_samples = model(noise, training=False)
    
    # Calculate some basic metrics (this is a placeholder - adapt to your needs)
    # In a real GAN evaluation, you might use FID, Inception Score, etc.
    avg_value = tf.reduce_mean(generated_samples).numpy()
    std_value = tf.math.reduce_std(generated_samples).numpy()
    
    metrics = {
        "avg_value": float(avg_value),
        "std_value": float(std_value),
        # Add more metrics as needed
    }
    
    logger.info(f"Evaluation complete: {metrics}")
    return metrics


@task
def train_model(data_path, model_type="skip", epochs=10):
    """
    Train a new DeepSculpt model.
    
    Args:
        data_path: Path to the preprocessed data DataFrame
        model_type: Type of model to train
        epochs: Number of epochs to train for
        
    Returns:
        Dictionary of training metrics
    """
    logger.info(f"Training new {model_type} model for {epochs} epochs...")
    
    # Load data DataFrame
    data_df = pd.read_csv(data_path)
    
    # Create data loader
    data_loader = DataFrameDataLoader(
        df=data_df,
        batch_size=32,
        shuffle=True
    )
    
    # Set environment variables for backwards compatibility
    os.environ["VOID_DIM"] = "64"
    os.environ["NOISE_DIM"] = "100"
    os.environ["COLOR"] = "1"
    
    # Create models
    generator = ModelFactory.create_generator(model_type=model_type)
    discriminator = ModelFactory.create_discriminator(model_type=model_type)
    
    # Print model summaries
    print("\nGenerator Summary:")
    generator.summary()
    
    print("\nDiscriminator Summary:")
    discriminator.summary()
    
    # Create results directories
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"./results/{model_type}_{timestamp}"
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    snapshot_dir = os.path.join(results_dir, "snapshots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    # Create trainer
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999
    )
    
    # Train the model
    logger.info("Starting training...")
    metrics = trainer.train(
        data_loader=data_loader,
        epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        snapshot_dir=snapshot_dir,
        snapshot_freq=5
    )
    
    # Save the final models
    generator.save(os.path.join(results_dir, "generator_final"))
    discriminator.save(os.path.join(results_dir, "discriminator_final"))
    
    # Get final metrics
    final_metrics = {
        "gen_loss": float(metrics["gen_loss"][-1]) if metrics["gen_loss"] else float("inf"),
        "disc_loss": float(metrics["disc_loss"][-1]) if metrics["disc_loss"] else float("inf"),
        "training_time": sum(metrics["epoch_times"]) if "epoch_times" in metrics else 0
    }
    
    # Save to MLflow
    Manager.save_mlflow_model(
        metrics=final_metrics,
        params={"model_type": model_type, "epochs": epochs},
        model=generator
    )
    
    logger.info(f"Training complete. Results saved to {results_dir}")
    return final_metrics


@task
def compare_and_promote(eval_metrics, train_metrics, threshold=0.1):
    """
    Compare evaluation and training metrics to decide whether to promote the new model.
    
    Args:
        eval_metrics: Metrics from the evaluation task
        train_metrics: Metrics from the training task
        threshold: Threshold for improvement
        
    Returns:
        Boolean indicating whether the new model should be promoted
    """
    logger.info("Comparing models...")
    
    # For GANs, the comparison is more complex than just comparing a single metric
    # This is a simplified comparison - adapt to your specific needs
    
    # Check if training improved over evaluation
    if "gen_loss" in train_metrics and "gen_loss" in eval_metrics:
        improvement = eval_metrics["gen_loss"] - train_metrics["gen_loss"]
        relative_improvement = improvement / eval_metrics["gen_loss"] if eval_metrics["gen_loss"] > 0 else float("inf")
        
        logger.info(f"Generator loss: {eval_metrics['gen_loss']} -> {train_metrics['gen_loss']} (Improvement: {improvement:.4f})")
        
        if relative_improvement > threshold:
            logger.info(f"New model is better by {relative_improvement:.2%}")
            return True
        else:
            logger.info(f"New model is not significantly better ({relative_improvement:.2%})")
            return False
    
    # Default to promoting if we can't compare (first run)
    logger.info("Cannot compare models, defaulting to promote")
    return True


@task
def promote_model(should_promote, model_path=None):
    """
    Promote the new model to production if indicated.
    
    Args:
        should_promote: Boolean indicating whether to promote
        model_path: Path to the model to promote (optional)
    """
    if not should_promote:
        logger.info("Model promotion skipped")
        return
    
    logger.info("Promoting model to production...")
    
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Cannot promote model.")
        return
    
    try:
        # Get MLflow client
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        client = MlflowClient()
        
        # Get the latest model version
        model_name = os.environ.get("MLFLOW_MODEL_NAME")
        latest_version = Manager.get_model_version(stage="None")
        
        if latest_version is None:
            logger.error("No model version found to promote")
            return
        
        # Transition the model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version,
            stage="Production"
        )
        
        logger.info(f"Model version {latest_version} promoted to Production")
    
    except Exception as e:
        logger.error(f"Error promoting model: {e}")


@task
def notify(eval_metrics, train_metrics, promoted):
    """
    Send a notification about the workflow results.
    
    Args:
        eval_metrics: Metrics from evaluation
        train_metrics: Metrics from training
        promoted: Whether the model was promoted
    """
    # This is a simple Slack-style notification - replace with your preferred method
    logger.info("Sending notification...")
    
    # Prepare message
    message = "DeepSculpt Workflow Completed\n"
    message += f"Evaluation Metrics: {json.dumps(eval_metrics, indent=2)}\n"
    message += f"Training Metrics: {json.dumps(train_metrics, indent=2)}\n"
    message += f"Model Promoted: {'Yes' if promoted else 'No'}"
    
    # Example: Send to a webhook (replace with your actual notification method)
    try:
        webhook_url = os.environ.get("NOTIFICATION_WEBHOOK")
        
        if webhook_url:
            payload = {
                "text": message,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
            logger.info("Notification sent")
        
        logger.info(f"\nNotification Message:\n{message}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send notification: {e}")
        return False


def build_flow(schedule=None):
    """
    Build the Prefect workflow for DeepSculpt.
    
    Args:
        schedule: Optional schedule for the workflow
        
    Returns:
        Prefect Flow object
    """
    if not PREFECT_AVAILABLE:
        logger.warning("Prefect not available. Workflow creation is limited.")
        return None
    
    flow_name = os.environ.get("PREFECT_FLOW_NAME", "deepSculpt_workflow")
    
    with Flow(name=flow_name, schedule=schedule) as flow:
        # Parameters
        mlflow_experiment = Parameter("experiment", default=os.environ.get("MLFLOW_EXPERIMENT", "deepSculpt"))
        data_folder = Parameter("data_folder", default="./data")
        model_type = Parameter("model_type", default="skip")
        epochs = Parameter("epochs", default=10)
        
        # 1. Preprocess data
        data_path = preprocess_data(mlflow_experiment, data_folder)
        
        # 2. Evaluate current production model
        eval_metrics = evaluate_model(data_path, model_type)
        
        # 3. Train new model
        train_metrics = train_model(data_path, model_type, epochs)
        
        # 4. Compare models and decide whether to promote
        should_promote = compare_and_promote(eval_metrics, train_metrics)
        
        # 5. Promote if indicated
        promotion_result = promote_model(should_promote)
        
        # 6. Send notification
        notify(eval_metrics, train_metrics, should_promote)
    
    return flow


def run_prefect_flow(mode="development", data_folder="./data", model_type="skip", 
                   epochs=10, schedule=False):
    """
    Run the Prefect workflow.
    
    Args:
        mode: Execution mode ('development' or 'production')
        data_folder: Path to data folder
        model_type: Type of model to train
        epochs: Number of epochs for training
        schedule: Whether to run with schedule
        
    Returns:
        Flow execution result or None
    """
    if not PREFECT_AVAILABLE:
        logger.warning("Prefect not available. Cannot run workflow.")
        return None
    
    # Set environment variables
    os.environ["PREFECT_BACKEND"] = mode
    
    # Set up schedule if requested
    workflow_schedule = None
    if schedule:
        workflow_schedule = IntervalSchedule(
            interval=datetime.timedelta(days=1),
            end_date=datetime.datetime.now() + datetime.timedelta(days=30)
        )
    
    # Build the flow
    flow = build_flow(workflow_schedule)
    if flow is None:
        return None
    
    # Configure executor
    flow.executor = LocalDaskExecutor()
    
    # Run or register flow based on mode
    if mode == "development":
        # In development mode, run the flow locally
        logger.info("Running workflow in development mode...")
        
        # Run the flow with parameters
        result = flow.run(parameters={
            "experiment": os.environ.get("MLFLOW_EXPERIMENT", "deepSculpt"),
            "data_folder": data_folder,
            "model_type": model_type,
            "epochs": epochs
        })
        
        return result
        
    elif mode == "production":
        # In production mode, register the flow with Prefect
        logger.info("Registering workflow for production...")
        
        # Get environment variables
        try:
            env_dict = None
            try:
                from dotenv import dotenv_values
                env_dict = dotenv_values(".env")
            except ImportError:
                logger.warning("dotenv not installed, using current environment")
                env_dict = dict(os.environ)
            
            flow.run_config = LocalRun(env=env_dict)
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
        
        # Register the flow
        flow.register(os.environ.get("PREFECT_FLOW_NAME", "deepSculpt_project"))
        logger.info("Workflow registered successfully")
        
        return True
    
    else:
        logger.error(f"Invalid mode: {mode}")
        return None


def main():
    """Main entry point for the DeepSculpt workflow."""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DeepSculpt Workflow Manager")
    parser.add_argument("--mode", type=str, choices=["development", "production"], 
                        default="development", help="Execution mode")
    parser.add_argument("--data-folder", type=str, default="./data",
                        help="Path to data folder")
    parser.add_argument("--model-type", type=str, default="skip",
                        choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--schedule", action="store_true",
                        help="Run with schedule")
    parser.add_argument("--complete", action="store_true",
                        help="Run complete workflow (generate data, curate, train, evaluate)")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate new data collection")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--void-dim", type=int, default=32,
                        help="Dimension of void space")
    parser.add_argument("--curate", action="store_true",
                        help="Curate data collection")
    parser.add_argument("--encoder", type=str, default="OHE",
                        choices=["OHE", "BINARY", "RGB"],
                        help="Encoding method for curation")
    parser.add_argument("--evaluate", action="store_true",
                        help="Evaluate model")
    parser.add_argument("--mlflow", action="store_true",
                        help="Save models to MLflow")
    parser.add_argument("--notify", action="store_true",
                        help="Send notification on completion")
    parser.add_argument("--webhook-url", type=str, default=None,
                        help="URL for notification webhook")
    
    args = parser.parse_args()
    
    # Set up environment variables if not already set
    if "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    if "MLFLOW_EXPERIMENT" not in os.environ:
        os.environ["MLFLOW_EXPERIMENT"] = "deepSculpt"
    
    if "MLFLOW_MODEL_NAME" not in os.environ:
        os.environ["MLFLOW_MODEL_NAME"] = "deepSculpt_generator"
    
    if "PREFECT_FLOW_NAME" not in os.environ:
        os.environ["PREFECT_FLOW_NAME"] = "deepSculpt_workflow"
    
    if args.webhook_url:
        os.environ["NOTIFICATION_WEBHOOK"] = args.webhook_url
    
    if args.complete:
        # Run complete workflow
        logger.info("Running complete DeepSculpt workflow")
        
        # Create workflow manager
        workflow_manager = WorkflowManager(
            base_dir=args.data_folder,
            results_dir="./results",
            verbose=True
        )
        
        # Define workflow configuration
        config = {
            # Data generation
            "generate_data": True,
            "void_dim": args.void_dim,
            "samples": args.samples,
            "edges": (2, 0.2, 0.5),
            "planes": (1, 0.3, 0.6),
            "pipes": (2, 0.3, 0.6),
            "grid": (1, 4),
            
            # Data curation
            "curate_data": True,
            "encoder": args.encoder,
            "batch_size": 32,
            "plot_samples": 3,
            
            # Model training
            "train_model": True,
            "model_type": args.model_type,
            "epochs": args.epochs,
            "learning_rate": 0.0002,
            "snapshot_freq": 5,
            "save_to_mlflow": args.mlflow,
            
            # Model evaluation
            "evaluate_model": True,
            "num_eval_samples": 10,
            
            # Notification
            "send_notification": args.notify,
            "notification_url": args.webhook_url
        }
        
        # Run the workflow
        results = workflow_manager.run_complete_workflow(config)
        
        # Print workflow summary
        print("\n" + "=" * 80)
        print("DeepSculpt Workflow Summary")
        print("=" * 80)
        print(f"Steps completed: {results['workflow']['steps_completed']}")
        print(f"Total time: {results['workflow']['elapsed_seconds']:.2f} seconds")
        print("=" * 80)
        
    elif args.generate_data:
        # Generate data only
        logger.info("Generating data collection")
        
        # Create workflow manager
        workflow_manager = WorkflowManager(
            base_dir=args.data_folder,
            results_dir="./results",
            verbose=True
        )
        
        # Generate data
        result = workflow_manager.generate_data(
            void_dim=args.void_dim,
            samples=args.samples
        )
        
        if result:
            print(f"Generated {result['sample_count']} samples in {result['elapsed_time']:.2f} seconds")
            print(f"Collection saved to: {result['date_dir']}")
        
    elif args.curate:
        # Curate data only
        logger.info("Curating data collection")
        
        # Create workflow manager
        workflow_manager = WorkflowManager(
            base_dir=args.data_folder,
            results_dir="./results",
            verbose=True
        )
        
        # Curate data
        result = workflow_manager.curate_data(
            encoder=args.encoder
        )
        
        if result:
            print(f"Curated {result['train_size']} training and {result['val_size']} validation samples")
            print(f"Processed data saved to: {result['output_dir']}")
    
    elif args.evaluate:
        # Evaluate model only
        logger.info("Evaluating model")
        
        # Create workflow manager
        workflow_manager = WorkflowManager(
            base_dir=args.data_folder,
            results_dir="./results",
            verbose=True
        )
        
        # Evaluate model
        result = workflow_manager.evaluate_model(
            stage="Production",
            num_samples=10,
            data_folder=args.data_folder
        )
        
        if result:
            print("\nEvaluation Metrics:")
            for k, v in result["metrics"].items():
                print(f"  {k}: {v:.6f}")
            print(f"Samples saved to: {result['samples_dir']}")
    
    else:
        # Run Prefect workflow
        logger.info(f"Running Prefect workflow in {args.mode} mode")
        result = run_prefect_flow(
            mode=args.mode,
            data_folder=args.data_folder,
            model_type=args.model_type,
            epochs=args.epochs,
            schedule=args.schedule
        )
        
        if result:
            logger.info("Workflow execution completed successfully")
        else:
            logger.error("Workflow execution failed")


if __name__ == "__main__":
    main()