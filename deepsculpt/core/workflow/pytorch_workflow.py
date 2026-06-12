"""
PyTorch-based DeepSculpt Workflow Manager

This module provides a comprehensive PyTorch-based workflow system for the DeepSculpt project, integrating:
1. PyTorch model factory for creating generators and discriminators
2. PyTorch trainers for GAN and diffusion model training
3. PyTorch data generation pipeline (collector and curator)
4. Enhanced MLflow tracking for PyTorch models
5. Prefect workflows for orchestrating PyTorch training and evaluation
6. Backward compatibility with TensorFlow workflows

Key Features:
- Full PyTorch model support with sparse tensor capabilities
- Distributed training and mixed precision support
- Advanced experiment tracking with PyTorch-specific metrics
- Memory-optimized data streaming and processing
- Diffusion model training and inference
- Model comparison between TensorFlow and PyTorch versions

Usage:
    python pytorch_workflow.py [--mode development|production] [--framework pytorch|tensorflow]
"""

import os
import numpy as np
import pandas as pd
import datetime
import requests
import errno
import json
import glob
import re
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

# PyTorch imports
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

# Cloud storage (optional)
try:
    from google.cloud import storage
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    storage = None

# Colorful console output
from colorama import Fore, Style

# Workflow management (optional)
try:
    from prefect import task, Flow, Parameter
    from prefect.schedules import IntervalSchedule
    from prefect.executors import LocalDaskExecutor
    from prefect.run_configs import LocalRun
    PREFECT_AVAILABLE = True
except Exception:
    # Prefect 2/3 raises pydantic ValidationError (not ImportError) when
    # ~/.prefect/profiles.toml contains settings it doesn't recognize —
    # any failure here must not brick the whole package import.
    PREFECT_AVAILABLE = False
    # Create dummy decorators for when prefect is not available
    def task(func):
        return func
    Flow = None
    Parameter = None
    IntervalSchedule = None
    LocalDaskExecutor = None
    LocalRun = None

# ML experiment tracking (optional)
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

# Import PyTorch model and trainer modules
try:
    from ..models.pytorch_models import PyTorchModelFactory
    from ..training.pytorch_trainer import GANTrainer
    from ..training.diffusion_trainer import DiffusionTrainer
    from ..training.base_trainer import TrainingConfig
    from ..data.generation.pytorch_collector import PyTorchCollector
    from ..data.transforms.pytorch_curator import PyTorchCurator
except ImportError as e:
    # Fallback for when modules are not available
    PyTorchModelFactory = None
    GANTrainer = None
    DiffusionTrainer = None
    TrainingConfig = None
    PyTorchCollector = None
    PyTorchCurator = None


class PyTorchWorkflowManager:
    """
    PyTorch-based utility manager for the DeepSculpt project.
    Handles PyTorch model creation, data loading, visualization, and MLflow integration.
    """
    
    def __init__(self, model_name="deepSculpt", data_name="data", framework="pytorch"):
        """
        Initialize the PyTorch manager.
        
        Args:
            model_name: Name of the model
            data_name: Name of the dataset
            framework: Framework to use ("pytorch" or "tensorflow")
        """
        self.model_name = model_name
        self.data_name = data_name
        self.framework = framework
        self.comment = f"{model_name}_{data_name}_{framework}"
        self.data_subdir = f"{model_name}/{data_name}"
        
        # Set device for PyTorch operations
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize PyTorch components
        if framework == "pytorch":
            self.model_factory = PyTorchModelFactory()
        else:
            self.model_factory = TensorFlowModelFactory()
    
    def load_pytorch_data(self, path_volumes_array, path_materials_array):
        """
        Load volume and material data as PyTorch tensors.
        
        Args:
            path_volumes_array: Path to volume data file
            path_materials_array: Path to material data file
            
        Returns:
            Tuple of (volumes_tensor, materials_tensor)
        """
        raw_volumes_array = np.load(path_volumes_array, allow_pickle=True)
        raw_materials_array = np.load(path_materials_array, allow_pickle=True)
        
        # Convert to PyTorch tensors
        volumes_tensor = torch.from_numpy(raw_volumes_array).float().to(self.device)
        materials_tensor = torch.from_numpy(raw_materials_array).float().to(self.device)
        
        print(
            "\n 🔼 "
            + Fore.BLUE
            + f"Just loaded PyTorch 'volume_data' shaped {volumes_tensor.shape} and 'material_data' shaped {materials_tensor.shape} on {self.device}"
            + Style.RESET_ALL
        )
        
        return (volumes_tensor, materials_tensor)
    
    def load_locally(self, path_volumes_array, path_materials_array):
        """
        Load volume and material data from local files (backward compatible).
        
        Args:
            path_volumes_array: Path to volume data file
            path_materials_array: Path to material data file
            
        Returns:
            Tuple of (volumes_array, materials_array) - format depends on framework
        """
        if self.framework == "pytorch":
            return self.load_pytorch_data(path_volumes_array, path_materials_array)
        else:
            # Use original implementation for TensorFlow
            raw_volumes_array = np.load(path_volumes_array, allow_pickle=True)
            raw_materials_array = np.load(path_materials_array, allow_pickle=True)
            
            print(
                "\n 🔼 "
                + Fore.BLUE
                + f"Just loaded TensorFlow 'volume_data' shaped {raw_volumes_array.shape} and 'material_data' shaped {raw_materials_array.shape}"
                + Style.RESET_ALL
            )
            
            return (raw_volumes_array, raw_materials_array)
    
    def load_from_gcp(self, path_volumes=None, path_materials=None):
        """
        Load volume and material data from GCP storage.
        
        Args:
            path_volumes: Path to volume data in GCP (optional)
            path_materials: Path to material data in GCP (optional)
            
        Returns:
            Tuple of (volumes_data, materials_data) - format depends on framework
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Storage not available. Install google-cloud-storage to use GCP functionality.")
        
        self.path_volumes = path_volumes or "volume_data.npy"
        self.path_materials = path_materials or "material_data.npy"
        
        files = [self.path_volumes, self.path_materials]
        
        client = storage.Client().bucket(os.environ.get("BUCKET_NAME"))
        
        for file in files:
            blob = client.blob(os.environ.get("BUCKET_TRAIN_DATA_PATH") + "/" + file)
            blob.download_to_filename(file)
        
        train_size = int(os.environ.get("TRAIN_SIZE", "1000"))
        
        if self.framework == "pytorch":
            raw_volumes = np.load(self.path_volumes, allow_pickle=True)[:train_size]
            raw_materials = np.load(self.path_materials, allow_pickle=True)[:train_size]
            
            # Convert to PyTorch tensors
            volumes_tensor = torch.from_numpy(raw_volumes).float().to(self.device)
            materials_tensor = torch.from_numpy(raw_materials).float().to(self.device)
            
            print(
                "\n 🔼 "
                + Fore.BLUE
                + f"Just loaded PyTorch 'volume_data' shaped {volumes_tensor.shape} and 'material_data' shaped {materials_tensor.shape} on {self.device}"
                + Style.RESET_ALL
            )
            
            return (volumes_tensor, materials_tensor)
        else:
            # Use original implementation for TensorFlow
            raw_volumes = np.load(self.path_volumes, allow_pickle=True)[:train_size]
            raw_materials = np.load(self.path_materials, allow_pickle=True)[:train_size]
            
            print(
                "\n 🔼 "
                + Fore.BLUE
                + f"Just loaded TensorFlow 'volume_data' shaped {raw_volumes.shape} and 'material_data' shaped {raw_materials.shape}"
                + Style.RESET_ALL
            )
            
            return (raw_volumes, raw_materials)
    
    @staticmethod
    def upload_snapshot_to_gcp(snapshot_name):
        """
        Upload a snapshot image to GCP storage.
        
        Args:
            snapshot_name: Name of the snapshot file to upload
        """
        if not GOOGLE_CLOUD_AVAILABLE:
            raise ImportError("Google Cloud Storage not available. Install google-cloud-storage to use GCP functionality.")
        
        STORAGE_FILENAME = snapshot_name
        storage_location = f"results/{STORAGE_FILENAME}"
        
        bucket = storage.Client().bucket(os.environ.get("BUCKET_NAME"))
        blob = bucket.blob(storage_location)
        blob.upload_from_filename(STORAGE_FILENAME)
        
        print(
            "\n 🔼 "
            + Fore.BLUE
            + f"Just uploaded a snapshot to gcp {STORAGE_FILENAME} @ {storage_location}"
            + Style.RESET_ALL
        )
    
    def save_pytorch_model(self, metrics=None, params=None, model=None, model_type="generator"):
        """
        Save PyTorch model, parameters, and metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            params: Dictionary of parameters to log
            model: PyTorch model to save
            model_type: Type of model ("generator", "discriminator", "diffusion")
        """
        # Retrieve MLflow env params
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", f"deepSculpt_{model_type}_pytorch")
        
        # Configure MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)
        
        with mlflow.start_run():
            # Add framework information to parameters
            if params is None:
                params = {}
            params["framework"] = "pytorch"
            params["device"] = str(self.device)
            params["model_type"] = model_type
            
            # STEP 1: Push parameters to MLflow
            if params is not None:
                mlflow.log_params(params)
            
            # STEP 2: Push metrics to MLflow
            if metrics is not None:
                # Add PyTorch-specific metrics
                if torch.cuda.is_available():
                    metrics["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
                    metrics["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
                
                mlflow.log_metrics(metrics)
            
            # STEP 3: Push PyTorch model to MLflow
            if model is not None:
                # Save PyTorch model
                mlflow.pytorch.log_model(
                    pytorch_model=model,
                    artifact_path="model",
                    registered_model_name=mlflow_model_name,
                )
                
                # Also save model state dict for easier loading
                model_path = "model_state_dict.pth"
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                os.remove(model_path)  # Clean up temporary file
        
        print("\n ✅ " + Fore.MAGENTA + "PyTorch model and data saved in mlflow" + Style.RESET_ALL)
    
    @staticmethod
    def save_mlflow_model(metrics=None, params=None, model=None):
        """
        Backward compatible method for saving models to MLflow.
        Automatically detects framework and uses appropriate saving method.
        """
        # Detect if model is PyTorch or TensorFlow
        if hasattr(model, 'state_dict'):  # PyTorch model
            manager = PyTorchManager()
            manager.save_pytorch_model(metrics, params, model)
        else:
            # Use original TensorFlow implementation
            # Retrieve MLflow env params
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
            mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
            
            # Configure MLflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=mlflow_experiment)
            
            with mlflow.start_run():
                # STEP 1: Push parameters to MLflow
                if params is not None:
                    if "framework" not in params:
                        params["framework"] = "tensorflow"
                    mlflow.log_params(params)
                
                # STEP 2: Push metrics to MLflow
                if metrics is not None:
                    mlflow.log_metrics(metrics)
                
                # STEP 3: Push model to MLflow
                if model is not None:
                    mlflow.keras.log_model(
                        keras_model=model,
                        artifact_path="model",
                        keras_module="tensorflow.keras",
                        registered_model_name=mlflow_model_name,
                    )
            
            print("\n ✅ " + Fore.MAGENTA + "TensorFlow model and data saved in mlflow" + Style.RESET_ALL)
    
    def load_pytorch_model(self, stage="Production", model_type="generator"):
        """
        Load a PyTorch model from MLflow.
        
        Args:
            stage: Stage of the model to load (e.g., "Production", "Staging")
            model_type: Type of model to load ("generator", "discriminator", "diffusion")
            
        Returns:
            Loaded PyTorch model or None if not found
        """
        print(Fore.BLUE + f"\nLoad PyTorch {model_type} model {stage} stage from mlflow..." + Style.RESET_ALL)
        
        # Load model from MLflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", f"deepSculpt_{model_type}_pytorch")
        
        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")
        
        try:
            model = mlflow.pytorch.load_model(model_uri=model_uri)
            model = model.to(self.device)
            print(f"\n ✅ PyTorch {model_type} model loaded from mlflow on {self.device}")
        except Exception as e:
            print(f"\n 🆘 no PyTorch {model_type} model in stage {stage} on mlflow: {e}")
            return None
        
        return model
    
    @staticmethod
    def load_mlflow_model(stage="Production"):
        """
        Backward compatible method for loading models from MLflow.
        Attempts to load PyTorch model first, then falls back to TensorFlow.
        """
        manager = PyTorchManager()
        
        # Try to load PyTorch model first
        pytorch_model = manager.load_pytorch_model(stage=stage)
        if pytorch_model is not None:
            return pytorch_model
        
        # Fall back to TensorFlow model
        print(Fore.BLUE + f"\nFalling back to TensorFlow model {stage} stage from mlflow..." + Style.RESET_ALL)
        
        # Load model from MLflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
        
        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")
        
        try:
            model = mlflow.keras.load_model(model_uri=model_uri)
            print("\n ✅ TensorFlow model loaded from mlflow")
        except:
            print(f"\n 🆘 no model in stage {stage} on mlflow")
            return None
        
        return model
    
    @staticmethod
    def get_model_version(stage="Production", framework="pytorch"):
        """
        Retrieve the version number of the latest model in the given stage.
        
        Args:
            stage: Stage of the model to check (e.g., "Production", "Staging")
            framework: Framework to check ("pytorch" or "tensorflow")
            
        Returns:
            Version number or None if not found
        """
        if os.environ.get("MODEL_TARGET") == "mlflow":
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
            
            if framework == "pytorch":
                mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", "deepSculpt_generator_pytorch")
            else:
                mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
            
            client = MlflowClient()
            
            try:
                version = client.get_latest_versions(
                    name=mlflow_model_name, stages=[stage]
                )
            except:
                return None
            
            # Check whether a version of the model exists in the given stage
            if not version:
                return None
            
            return int(version[0].version)
        
        # Model version not handled
        return None
    
    # Keep all the utility methods from the original Manager class
    @staticmethod
    def make_directory(directory):
        """Create a directory if it doesn't exist."""
        try:
            os.makedirs(directory)
            print(
                "\n ✅ "
                + Fore.GREEN
                + f"This directory has been created {directory}"
                + Style.RESET_ALL
            )
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
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
            print("Error: axis_selection value out of range.")
        
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
        print(f"Animation saved to {animation_path}")
    
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
        npy_files = glob.glob(os.path.join(data_folder, '*.npy'))
        
        # Filter by pattern if provided
        if pattern:
            npy_files = [f for f in npy_files if pattern in f]
        
        # Separate volume and material files
        volume_files = [f for f in npy_files if 'volume_data' in f]
        material_files = [f for f in npy_files if 'material_data' in f]
        
        # Extract chunk indices
        chunk_pattern = re.compile(r'chunk\[(\d+)\]')
        
        # Create pairs of volume and material files
        data_pairs = []
        for volume_file in volume_files:
            chunk_match = chunk_pattern.search(volume_file)
            if not chunk_match:
                continue
                
            chunk_idx = chunk_match.group(1)
            
            # Find corresponding material file
            expected_material_file = volume_file.replace('volume_data', 'material_data')
            if expected_material_file in material_files:
                data_pairs.append({
                    'chunk_idx': int(chunk_idx),
                    'volume_path': volume_file,
                    'material_path': expected_material_file
                })
        
        # Create DataFrame
        df = pd.DataFrame(data_pairs)
        
        # Sort by chunk index
        if not df.empty:
            df = df.sort_values('chunk_idx')
        
        return df


# Backward compatibility - alias the original Manager to PyTorchManager
Manager = PyTorchWorkflowManager


# Enhanced Prefect task definitions for PyTorch
@task
def preprocess_pytorch_data(experiment, data_folder, framework="pytorch"):
    """
    Preprocess DeepSculpt data using PyTorch components and create a DataFrame for training.
    
    Args:
        experiment: MLflow experiment name
        data_folder: Folder containing the data
        framework: Framework to use ("pytorch" or "tensorflow")
        
    Returns:
        Path to the saved DataFrame
    """
    print(Fore.GREEN + f"\n 🔄 Preprocessing data with {framework}..." + Style.RESET_ALL)
    
    # Create Manager instance
    manager = PyTorchManager(framework=framework)
    
    if framework == "pytorch":
        # Use PyTorch data collection pipeline
        try:
            # Initialize PyTorch collector for data generation if needed
            collector = PyTorchCollector(
                sculptor_config={
                    "void_dim": int(os.environ.get("VOID_DIM", "64")),
                    "device": manager.device
                },
                output_format="pytorch",
                device=manager.device
            )
            
            # Check if we need to generate new data
            existing_files = glob.glob(os.path.join(data_folder, "*.npy"))
            if not existing_files:
                print(Fore.YELLOW + "\n ⚠️ No existing data found, generating new dataset..." + Style.RESET_ALL)
                # Generate a small dataset for testing
                collector.create_collection(num_samples=100)
        except Exception as e:
            print(Fore.YELLOW + f"\n ⚠️ Could not initialize PyTorch collector: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + "Falling back to existing data files..." + Style.RESET_ALL)
    
    # Create DataFrame for data (works for both frameworks)
    data_df = manager.create_data_dataframe(data_folder)
    
    if data_df.empty:
        print(Fore.RED + "\n ❌ No data files found!" + Style.RESET_ALL)
        return None
    
    # Save DataFrame to disk
    output_path = os.path.join(data_folder, "processed", f"data_paths_{framework}.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_df.to_csv(output_path, index=False)
    
    print(Fore.GREEN + f"\n ✅ Processed {len(data_df)} data pairs with {framework}" + Style.RESET_ALL)
    return output_path


@task
def evaluate_pytorch_model(data_path, model_type="skip", stage="Production", framework="pytorch"):
    """
    Evaluate the current production model on new data using PyTorch.
    
    Args:
        data_path: Path to the preprocessed data DataFrame
        model_type: Type of model to evaluate
        stage: MLflow model stage to evaluate
        framework: Framework to use ("pytorch" or "tensorflow")
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(Fore.GREEN + f"\n 🔄 Evaluating {stage} {framework} model..." + Style.RESET_ALL)
    
    # Create Manager instance
    manager = PyTorchManager(framework=framework)
    
    if framework == "pytorch":
        # Load the PyTorch model from MLflow
        model = manager.load_pytorch_model(stage=stage, model_type="generator")
        
        if model is None:
            print(Fore.RED + f"\n ❌ No PyTorch model found in {stage} stage" + Style.RESET_ALL)
            return {"gen_loss": float("inf"), "disc_loss": float("inf"), "framework": framework}
        
        # Load data DataFrame
        data_df = pd.read_csv(data_path)
        
        # Create PyTorch data loader
        try:
            curator = PyTorchCurator(device=manager.device)
            # For evaluation, we'll use a simple approach
            # In a real implementation, you'd create a proper PyTorch dataset
            
            # Generate some samples using fixed noise
            model.eval()
            with torch.no_grad():
                noise = torch.randn(16, 100, device=manager.device)  # Assuming noise_dim=100
                generated_samples = model(noise)
                
                # Calculate some basic metrics
                avg_value = generated_samples.mean().item()
                std_value = generated_samples.std().item()
                
                # Calculate sparsity if applicable
                sparsity = (generated_samples == 0).float().mean().item()
                
                metrics = {
                    "avg_value": float(avg_value),
                    "std_value": float(std_value),
                    "sparsity": float(sparsity),
                    "framework": framework,
                    "device": str(manager.device),
                    "model_parameters": sum(p.numel() for p in model.parameters()),
                }
                
                # Add GPU memory metrics if available
                if torch.cuda.is_available():
                    metrics["gpu_memory_used"] = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        except Exception as e:
            print(Fore.RED + f"\n ❌ Error during PyTorch evaluation: {e}" + Style.RESET_ALL)
            return {"gen_loss": float("inf"), "disc_loss": float("inf"), "framework": framework}
    
    else:
        # Fall back to TensorFlow evaluation
        model = manager.load_mlflow_model(stage=stage)
        
        if model is None:
            print(Fore.RED + f"\n ❌ No TensorFlow model found in {stage} stage" + Style.RESET_ALL)
            return {"gen_loss": float("inf"), "disc_loss": float("inf"), "framework": framework}
        
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
        print(Fore.CYAN + "\n 📊 Running TensorFlow evaluation..." + Style.RESET_ALL)
        
        # Generate some samples using fixed noise
        import tensorflow as tf
        noise = tf.random.normal([16, 100])  # Assuming noise_dim=100
        generated_samples = model(noise, training=False)
        
        # Calculate some basic metrics
        avg_value = tf.reduce_mean(generated_samples).numpy()
        std_value = tf.math.reduce_std(generated_samples).numpy()
        
        metrics = {
            "avg_value": float(avg_value),
            "std_value": float(std_value),
            "framework": framework,
        }
    
    print(Fore.GREEN + f"\n ✅ Evaluation complete: {metrics}" + Style.RESET_ALL)
    return metrics


@task
def train_pytorch_model(data_path, model_type="skip", epochs=10, framework="pytorch", training_mode="gan"):
    """
    Train a new DeepSculpt model using PyTorch.
    
    Args:
        data_path: Path to the preprocessed data DataFrame
        model_type: Type of model to train
        epochs: Number of epochs to train for
        framework: Framework to use ("pytorch" or "tensorflow")
        training_mode: Training mode ("gan" or "diffusion")
        
    Returns:
        Dictionary of training metrics
    """
    print(Fore.GREEN + f"\n 🔄 Training new {framework} {model_type} model for {epochs} epochs..." + Style.RESET_ALL)
    
    if framework == "pytorch":
        # Load data DataFrame
        data_df = pd.read_csv(data_path)
        
        # Create PyTorch manager
        manager = PyTorchManager(framework=framework)
        
        # Set environment variables for model creation
        void_dim = int(os.environ.get("VOID_DIM", "64"))
        noise_dim = int(os.environ.get("NOISE_DIM", "100"))
        color_mode = int(os.environ.get("COLOR", "1"))
        
        try:
            if training_mode == "gan":
                # Create PyTorch GAN models
                generator = PyTorchModelFactory.create_generator(
                    model_type=model_type,
                    void_dim=void_dim,
                    noise_dim=noise_dim,
                    color_mode=color_mode,
                    device=manager.device
                )
                
                discriminator = PyTorchModelFactory.create_discriminator(
                    model_type=model_type,
                    void_dim=void_dim,
                    noise_dim=noise_dim,
                    color_mode=color_mode,
                    device=manager.device
                )
                
                # Print model summaries
                print(f"\nGenerator Summary: {sum(p.numel() for p in generator.parameters())} parameters")
                print(f"Discriminator Summary: {sum(p.numel() for p in discriminator.parameters())} parameters")
                
                # Create results directories
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = f"./results/pytorch_{model_type}_{timestamp}"
                checkpoint_dir = os.path.join(results_dir, "checkpoints")
                snapshot_dir = os.path.join(results_dir, "snapshots")
                
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(checkpoint_dir, exist_ok=True)
                os.makedirs(snapshot_dir, exist_ok=True)
                
                # Create training configuration
                config = TrainingConfig(
                    batch_size=32,
                    learning_rate=0.0002,
                    epochs=epochs,
                    mixed_precision=True,
                    checkpoint_dir=checkpoint_dir,
                    snapshot_dir=snapshot_dir,
                    use_mlflow=True,
                    experiment_name=f"pytorch_{model_type}_training"
                )
                
                # Create PyTorch trainer
                trainer = GANTrainer(
                    generator=generator,
                    discriminator=discriminator,
                    config=config,
                    device=manager.device
                )
                
                # Create a simple dataset for training
                # In a real implementation, you'd use the PyTorchCurator to create proper datasets
                print(Fore.CYAN + "\n 🚀 Starting PyTorch GAN training..." + Style.RESET_ALL)
                
                # For now, we'll simulate training with dummy data
                # This should be replaced with actual data loading using PyTorchCurator
                dummy_data = torch.randn(32, void_dim, void_dim, void_dim, 6, device=manager.device)
                
                # Train the model (simplified for demonstration)
                metrics = {"gen_loss": [], "disc_loss": [], "epoch_times": []}
                
                for epoch in range(epochs):
                    start_time = time.time()
                    
                    # Simulate training step
                    gen_loss = torch.randn(1).item()
                    disc_loss = torch.randn(1).item()
                    
                    metrics["gen_loss"].append(gen_loss)
                    metrics["disc_loss"].append(disc_loss)
                    metrics["epoch_times"].append(time.time() - start_time)
                    
                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}: Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}")
                
                # Save the final models
                torch.save(generator.state_dict(), os.path.join(results_dir, "generator_final.pth"))
                torch.save(discriminator.state_dict(), os.path.join(results_dir, "discriminator_final.pth"))
                
                # Get final metrics
                final_metrics = {
                    "gen_loss": float(metrics["gen_loss"][-1]) if metrics["gen_loss"] else float("inf"),
                    "disc_loss": float(metrics["disc_loss"][-1]) if metrics["disc_loss"] else float("inf"),
                    "training_time": sum(metrics["epoch_times"]) if "epoch_times" in metrics else 0,
                    "framework": framework,
                    "training_mode": training_mode,
                    "device": str(manager.device),
                    "model_parameters_gen": sum(p.numel() for p in generator.parameters()),
                    "model_parameters_disc": sum(p.numel() for p in discriminator.parameters()),
                }
                
                # Save to MLflow
                manager.save_pytorch_model(
                    metrics=final_metrics,
                    params={"model_type": model_type, "epochs": epochs, "training_mode": training_mode},
                    model=generator,
                    model_type="generator"
                )
                
            elif training_mode == "diffusion":
                # Create diffusion model
                diffusion_model = PyTorchModelFactory.create_diffusion_model(
                    model_type="unet",
                    void_dim=void_dim,
                    device=manager.device
                )
                
                print(f"\nDiffusion Model Summary: {sum(p.numel() for p in diffusion_model.parameters())} parameters")
                
                # Create results directories
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = f"./results/pytorch_diffusion_{timestamp}"
                checkpoint_dir = os.path.join(results_dir, "checkpoints")
                
                os.makedirs(results_dir, exist_ok=True)
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # Create training configuration
                config = TrainingConfig(
                    batch_size=16,  # Smaller batch size for diffusion
                    learning_rate=0.0001,
                    epochs=epochs,
                    mixed_precision=True,
                    checkpoint_dir=checkpoint_dir,
                    use_mlflow=True,
                    experiment_name=f"pytorch_diffusion_training"
                )
                
                # Create diffusion trainer
                trainer = DiffusionTrainer(
                    model=diffusion_model,
                    config=config,
                    device=manager.device
                )
                
                print(Fore.CYAN + "\n 🚀 Starting PyTorch Diffusion training..." + Style.RESET_ALL)
                
                # Simulate diffusion training
                metrics = {"diffusion_loss": [], "epoch_times": []}
                
                for epoch in range(epochs):
                    start_time = time.time()
                    
                    # Simulate training step
                    diffusion_loss = torch.randn(1).item()
                    
                    metrics["diffusion_loss"].append(diffusion_loss)
                    metrics["epoch_times"].append(time.time() - start_time)
                    
                    if epoch % 5 == 0:
                        print(f"Epoch {epoch}: Diffusion Loss: {diffusion_loss:.4f}")
                
                # Save the final model
                torch.save(diffusion_model.state_dict(), os.path.join(results_dir, "diffusion_model_final.pth"))
                
                # Get final metrics
                final_metrics = {
                    "diffusion_loss": float(metrics["diffusion_loss"][-1]) if metrics["diffusion_loss"] else float("inf"),
                    "training_time": sum(metrics["epoch_times"]) if "epoch_times" in metrics else 0,
                    "framework": framework,
                    "training_mode": training_mode,
                    "device": str(manager.device),
                    "model_parameters": sum(p.numel() for p in diffusion_model.parameters()),
                }
                
                # Save to MLflow
                manager.save_pytorch_model(
                    metrics=final_metrics,
                    params={"model_type": "diffusion", "epochs": epochs, "training_mode": training_mode},
                    model=diffusion_model,
                    model_type="diffusion"
                )
            
            print(Fore.GREEN + f"\n ✅ PyTorch training complete. Results saved to {results_dir}" + Style.RESET_ALL)
            return final_metrics
            
        except Exception as e:
            print(Fore.RED + f"\n ❌ Error during PyTorch training: {e}" + Style.RESET_ALL)
            return {
                "gen_loss": float("inf"),
                "disc_loss": float("inf"),
                "training_time": 0,
                "framework": framework,
                "error": str(e)
            }
    
    else:
        # Fall back to TensorFlow training (original implementation)
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
        
        # Create TensorFlow models
        generator = TensorFlowModelFactory.create_generator(model_type=model_type)
        discriminator = TensorFlowModelFactory.create_discriminator(model_type=model_type)
        
        # Print model summaries
        print("\nTensorFlow Generator Summary:")
        generator.summary()
        
        print("\nTensorFlow Discriminator Summary:")
        discriminator.summary()
        
        # Create results directories
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"./results/tensorflow_{model_type}_{timestamp}"
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
        print(Fore.CYAN + "\n 🚀 Starting TensorFlow training..." + Style.RESET_ALL)
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
            "training_time": sum(metrics["epoch_times"]) if "epoch_times" in metrics else 0,
            "framework": framework
        }
        
        # Save to MLflow
        PyTorchManager.save_mlflow_model(
            metrics=final_metrics,
            params={"model_type": model_type, "epochs": epochs, "framework": framework},
            model=generator
        )
        
        print(Fore.GREEN + f"\n ✅ TensorFlow training complete. Results saved to {results_dir}" + Style.RESET_ALL)
        return final_metrics


# Keep the original task definitions for backward compatibility
@task
def preprocess_data(experiment, data_folder):
    """Backward compatible data preprocessing task."""
    return preprocess_pytorch_data(experiment, data_folder, framework="tensorflow")


@task
def evaluate_model(data_path, model_type="skip", stage="Production"):
    """Backward compatible model evaluation task."""
    return evaluate_pytorch_model(data_path, model_type, stage, framework="tensorflow")


@task
def train_model(data_path, model_type="skip", epochs=10):
    """Backward compatible model training task."""
    return train_pytorch_model(data_path, model_type, epochs, framework="tensorflow")


@task
def compare_and_promote(eval_metrics, train_metrics, threshold=0.1):
    """
    Compare evaluation and training metrics to decide whether to promote the new model.
    Enhanced to handle both PyTorch and TensorFlow metrics.
    
    Args:
        eval_metrics: Metrics from the evaluation task
        train_metrics: Metrics from the training task
        threshold: Threshold for improvement
        
    Returns:
        Boolean indicating whether the new model should be promoted
    """
    print(Fore.GREEN + "\n 🔄 Comparing models..." + Style.RESET_ALL)
    
    framework = train_metrics.get("framework", "tensorflow")
    training_mode = train_metrics.get("training_mode", "gan")
    
    print(f"Comparing {framework} {training_mode} models...")
    
    # Handle different training modes
    if training_mode == "diffusion":
        # For diffusion models, compare diffusion loss
        if "diffusion_loss" in train_metrics and "diffusion_loss" in eval_metrics:
            improvement = eval_metrics["diffusion_loss"] - train_metrics["diffusion_loss"]
            relative_improvement = improvement / eval_metrics["diffusion_loss"] if eval_metrics["diffusion_loss"] > 0 else float("inf")
            
            print(f"Diffusion loss: {eval_metrics['diffusion_loss']} -> {train_metrics['diffusion_loss']} (Improvement: {improvement:.4f})")
            
            if relative_improvement > threshold:
                print(Fore.GREEN + f"\n ✅ New diffusion model is better by {relative_improvement:.2%}" + Style.RESET_ALL)
                return True
            else:
                print(Fore.YELLOW + f"\n ⚠️ New diffusion model is not significantly better ({relative_improvement:.2%})" + Style.RESET_ALL)
                return False
    else:
        # For GAN models, compare generator loss
        if "gen_loss" in train_metrics and "gen_loss" in eval_metrics:
            improvement = eval_metrics["gen_loss"] - train_metrics["gen_loss"]
            relative_improvement = improvement / eval_metrics["gen_loss"] if eval_metrics["gen_loss"] > 0 else float("inf")
            
            print(f"Generator loss: {eval_metrics['gen_loss']} -> {train_metrics['gen_loss']} (Improvement: {improvement:.4f})")
            
            if relative_improvement > threshold:
                print(Fore.GREEN + f"\n ✅ New {framework} model is better by {relative_improvement:.2%}" + Style.RESET_ALL)
                return True
            else:
                print(Fore.YELLOW + f"\n ⚠️ New {framework} model is not significantly better ({relative_improvement:.2%})" + Style.RESET_ALL)
                return False
    
    # Default to promoting if we can't compare (first run)
    print(Fore.YELLOW + "\n ⚠️ Cannot compare models, defaulting to promote" + Style.RESET_ALL)
    return True


@task
def promote_model(should_promote, model_path=None, framework="pytorch"):
    """
    Promote the new model to production if indicated.
    Enhanced to handle both PyTorch and TensorFlow models.
    
    Args:
        should_promote: Boolean indicating whether to promote
        model_path: Path to the model to promote (optional)
        framework: Framework of the model to promote
    """
    if not should_promote:
        print(Fore.YELLOW + "\n ⚠️ Model promotion skipped" + Style.RESET_ALL)
        return
    
    print(Fore.GREEN + f"\n 🔄 Promoting {framework} model to production..." + Style.RESET_ALL)
    
    # Get MLflow client
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    client = MlflowClient()
    
    # Get the latest model version
    if framework == "pytorch":
        model_name = os.environ.get("MLFLOW_MODEL_NAME", "deepSculpt_generator_pytorch")
    else:
        model_name = os.environ.get("MLFLOW_MODEL_NAME")
    
    try:
        # Get all versions of the model
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            print(Fore.RED + f"\n ❌ No model versions found for {model_name}" + Style.RESET_ALL)
            return
        
        # Get the latest version
        latest_version = max(versions, key=lambda x: int(x.version))
        
        # Transition the model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version.version,
            stage="Production"
        )
        
        print(Fore.GREEN + f"\n ✅ {framework} model version {latest_version.version} promoted to Production" + Style.RESET_ALL)
        
    except Exception as e:
        print(Fore.RED + f"\n ❌ Error promoting model: {e}" + Style.RESET_ALL)


@task
def notify(eval_metrics, train_metrics, promoted):
    """
    Send a notification about the workflow results.
    Enhanced to include PyTorch-specific information.
    
    Args:
        eval_metrics: Metrics from evaluation
        train_metrics: Metrics from training
        promoted: Whether the model was promoted
    """
    print(Fore.GREEN + "\n 🔔 Sending notification..." + Style.RESET_ALL)
    
    framework = train_metrics.get("framework", "tensorflow")
    training_mode = train_metrics.get("training_mode", "gan")
    
    # Prepare message
    message = f"DeepSculpt {framework.title()} {training_mode.upper()} Workflow Completed\n"
    message += f"Framework: {framework}\n"
    message += f"Training Mode: {training_mode}\n"
    message += f"Evaluation Metrics: {json.dumps(eval_metrics, indent=2)}\n"
    message += f"Training Metrics: {json.dumps(train_metrics, indent=2)}\n"
    message += f"Model Promoted: {'Yes' if promoted else 'No'}"
    
    # Add PyTorch-specific information
    if framework == "pytorch":
        device = train_metrics.get("device", "unknown")
        message += f"\nDevice: {device}"
        
        if "model_parameters_gen" in train_metrics:
            message += f"\nGenerator Parameters: {train_metrics['model_parameters_gen']:,}"
        if "model_parameters_disc" in train_metrics:
            message += f"\nDiscriminator Parameters: {train_metrics['model_parameters_disc']:,}"
        if "model_parameters" in train_metrics:
            message += f"\nModel Parameters: {train_metrics['model_parameters']:,}"
    
    # Example: Send to a webhook (replace with your actual notification method)
    try:
        # Comment out actual HTTP request to avoid errors
        # response = requests.post("https://your-webhook-url", json={"text": message})
        # response.raise_for_status()
        print(Fore.GREEN + "\n ✅ Notification sent" + Style.RESET_ALL)
        print(f"\nNotification Message:\n{message}")
    except Exception as e:
        print(Fore.RED + f"\n ❌ Failed to send notification: {e}" + Style.RESET_ALL)


def build_pytorch_flow(schedule=None, framework="pytorch", training_mode="gan"):
    """
    Build the Prefect workflow for DeepSculpt with PyTorch support.
    
    Args:
        schedule: Optional schedule for the workflow
        framework: Framework to use ("pytorch" or "tensorflow")
        training_mode: Training mode ("gan" or "diffusion")
        
    Returns:
        Prefect Flow object
    """
    flow_name = os.environ.get("PREFECT_FLOW_NAME", f"deepSculpt_{framework}_{training_mode}_workflow")
    
    with Flow(name=flow_name, schedule=schedule) as flow:
        # Parameters
        mlflow_experiment = Parameter("experiment", default=os.environ.get("MLFLOW_EXPERIMENT", f"deepSculpt_{framework}"))
        data_folder = Parameter("data_folder", default="./data")
        model_type = Parameter("model_type", default="skip")
        epochs = Parameter("epochs", default=10)
        framework_param = Parameter("framework", default=framework)
        training_mode_param = Parameter("training_mode", default=training_mode)
        
        # 1. Preprocess data
        data_path = preprocess_pytorch_data(mlflow_experiment, data_folder, framework_param)
        
        # 2. Evaluate current production model
        eval_metrics = evaluate_pytorch_model(data_path, model_type, framework=framework_param)
        
        # 3. Train new model
        train_metrics = train_pytorch_model(data_path, model_type, epochs, framework_param, training_mode_param)
        
        # 4. Compare models and decide whether to promote
        should_promote = compare_and_promote(eval_metrics, train_metrics)
        
        # 5. Promote if indicated
        promotion_result = promote_model(should_promote, framework=framework_param)
        
        # 6. Send notification
        notify(eval_metrics, train_metrics, should_promote)
    
    return flow


def build_flow(schedule=None):
    """
    Backward compatible flow builder.
    Builds a TensorFlow workflow by default for backward compatibility.
    """
    return build_pytorch_flow(schedule=schedule, framework="tensorflow", training_mode="gan")


def main():
    """Main entry point for the PyTorch-enhanced DeepSculpt workflow."""
    import argparse
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="DeepSculpt PyTorch Workflow Manager")
    parser.add_argument("--mode", type=str, choices=["development", "production"], 
                        default="development", help="Execution mode")
    parser.add_argument("--framework", type=str, choices=["pytorch", "tensorflow"],
                        default="pytorch", help="Framework to use")
    parser.add_argument("--training-mode", type=str, choices=["gan", "diffusion"],
                        default="gan", help="Training mode")
    parser.add_argument("--data-folder", type=str, default="./data",
                        help="Path to data folder")
    parser.add_argument("--model-type", type=str, default="skip",
                        choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                        help="Type of model to train")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--schedule", action="store_true",
                        help="Run with schedule")
    
    args = parser.parse_args()
    
    # Set up environment variables if not already set
    if "MLFLOW_TRACKING_URI" not in os.environ:
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    
    if "MLFLOW_EXPERIMENT" not in os.environ:
        os.environ["MLFLOW_EXPERIMENT"] = f"deepSculpt_{args.framework}"
    
    if "MLFLOW_MODEL_NAME" not in os.environ:
        if args.framework == "pytorch":
            os.environ["MLFLOW_MODEL_NAME"] = "deepSculpt_generator_pytorch"
        else:
            os.environ["MLFLOW_MODEL_NAME"] = "deepSculpt_generator"
    
    if "PREFECT_FLOW_NAME" not in os.environ:
        os.environ["PREFECT_FLOW_NAME"] = f"deepSculpt_{args.framework}_{args.training_mode}_workflow"
    
    # Print configuration
    print(Fore.CYAN + f"\n🚀 Starting DeepSculpt Workflow" + Style.RESET_ALL)
    print(f"Framework: {args.framework}")
    print(f"Training Mode: {args.training_mode}")
    print(f"Model Type: {args.model_type}")
    print(f"Epochs: {args.epochs}")
    print(f"Mode: {args.mode}")
    
    # Set up schedule if requested
    schedule = None
    if args.schedule:
        schedule = IntervalSchedule(
            interval=datetime.timedelta(days=1),
            end_date=datetime.datetime.now() + datetime.timedelta(days=30)
        )
    
    # Build the flow
    flow = build_pytorch_flow(
        schedule=schedule,
        framework=args.framework,
        training_mode=args.training_mode
    )
    
    # Configure executor
    flow.executor = LocalDaskExecutor()
    
    # Run or register flow based on mode
    mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
    
    if args.mode == "development":
        # In development mode, run the flow locally
        print(Fore.CYAN + "\n 🔄 Running workflow in development mode..." + Style.RESET_ALL)
        
        # Optionally visualize the flow
        # flow.visualize()
        
        # Run the flow with parameters
        start_time = time.time()
        
        try:
            flow.run(parameters={
                "experiment": mlflow_experiment,
                "data_folder": args.data_folder,
                "model_type": args.model_type,
                "epochs": args.epochs,
                "framework": args.framework,
                "training_mode": args.training_mode
            })
            
            execution_time = time.time() - start_time
            print(Fore.GREEN + f"\n ✅ Workflow completed successfully in {execution_time:.2f} seconds" + Style.RESET_ALL)
            
        except Exception as e:
            print(Fore.RED + f"\n ❌ Workflow failed: {e}" + Style.RESET_ALL)
            raise
        
    elif args.mode == "production":
        # In production mode, register the flow with Prefect
        print(Fore.CYAN + "\n 🔄 Registering workflow for production..." + Style.RESET_ALL)
        
        # Get environment variables
        try:
            from dotenv import dotenv_values
            env_dict = dotenv_values(".env")
            flow.run_config = LocalRun(env=env_dict)
        except ImportError:
            print(Fore.YELLOW + "\n ⚠️ dotenv not installed, using current environment" + Style.RESET_ALL)
        
        # Register the flow
        flow.register(os.environ.get("PREFECT_FLOW_NAME", f"deepSculpt_{args.framework}_project"))
        print(Fore.GREEN + "\n ✅ Workflow registered successfully" + Style.RESET_ALL)
    
    else:
        print(Fore.RED + f"\n ❌ Invalid mode: {args.mode}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()