"""
DeepSculpt Workflow Manager

This module provides a consolidated workflow system for the DeepSculpt project, integrating:
1. General Manager utilities for data handling and visualization
2. Prefect workflows for orchestrating training and evaluation
3. MLflow tracking for experiment monitoring

Usage:
    python workflow.py [--mode development|production]
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
from typing import Dict, List, Tuple, Optional, Union, Any

# Visualization imports
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

# Cloud storage
from google.cloud import storage

# Colorful console output
from colorama import Fore, Style

# Workflow management
from prefect import task, Flow, Parameter
from prefect.schedules import IntervalSchedule
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import LocalRun

# ML experiment tracking
import mlflow
from mlflow.tracking import MlflowClient

# Import our model and trainer modules
from models import ModelFactory
from trainer import DeepSculptTrainer, DataFrameDataLoader


class Manager:
    """
    General utility manager for the DeepSculpt project.
    Handles data loading, visualization, and MLflow integration.
    """
    
    def __init__(self, model_name="deepSculpt", data_name="data"):
        """
        Initialize the manager.
        
        Args:
            model_name: Name of the model
            data_name: Name of the dataset
        """
        self.model_name = model_name
        self.data_name = data_name
        self.comment = f"{model_name}_{data_name}"
        self.data_subdir = f"{model_name}/{data_name}"
    
    def load_locally(self, path_volumes_array, path_materials_array):
        """
        Load volume and material data from local files.
        
        Args:
            path_volumes_array: Path to volume data file
            path_materials_array: Path to material data file
            
        Returns:
            Tuple of (volumes_array, materials_array)
        """
        raw_volumes_array = np.load(path_volumes_array, allow_pickle=True)
        raw_materials_array = np.load(path_materials_array, allow_pickle=True)
        
        print(
            "\n 🔼 "
            + Fore.BLUE
            + f"Just loaded 'volume_data' shaped {raw_volumes_array.shape} and 'material_data' shaped {raw_materials_array.shape}"
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
            Tuple of (volumes_array, materials_array)
        """
        self.path_volumes = path_volumes or "volume_data.npy"
        self.path_materials = path_materials or "material_data.npy"
        
        files = [self.path_volumes, self.path_materials]
        
        client = storage.Client().bucket(os.environ.get("BUCKET_NAME"))
        
        for file in files:
            blob = client.blob(os.environ.get("BUCKET_TRAIN_DATA_PATH") + "/" + file)
            blob.download_to_filename(file)
        
        train_size = int(os.environ.get("TRAIN_SIZE", "1000"))
        raw_volumes = np.load(self.path_volumes, allow_pickle=True)[:train_size]
        raw_materials = np.load(self.path_materials, allow_pickle=True)[:train_size]
        
        print(
            "\n 🔼 "
            + Fore.BLUE
            + f"Just loaded 'volume_data' shaped {raw_volumes.shape} and 'material_data' shaped {raw_materials.shape}"
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
    
    @staticmethod
    def save_mlflow_model(metrics=None, params=None, model=None):
        """
        Save model, parameters, and metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            params: Dictionary of parameters to log
            model: Keras model to save
        """
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
        
        print("\n ✅ " + Fore.MAGENTA + "Data saved in mlflow" + Style.RESET_ALL)
    
    @staticmethod
    def load_mlflow_model(stage="Production"):
        """
        Load a model from MLflow.
        
        Args:
            stage: Stage of the model to load (e.g., "Production", "Staging")
            
        Returns:
            Loaded Keras model or None if not found
        """
        print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)
        
        # Load model from MLflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
        
        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")
        
        try:
            model = mlflow.keras.load_model(model_uri=model_uri)
            print("\n ✅ model loaded from mlflow")
        except:
            print(f"\n 🆘 no model in stage {stage} on mlflow")
            return None
        
        return model
    
    @staticmethod
    def get_model_version(stage="Production"):
        """
        Retrieve the version number of the latest model in the given stage.
        
        Args:
            stage: Stage of the model to check (e.g., "Production", "Staging")
            
        Returns:
            Version number or None if not found
        """
        if os.environ.get("MODEL_TARGET") == "mlflow":
            mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
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
    print(Fore.GREEN + "\n 🔄 Preprocessing data..." + Style.RESET_ALL)
    
    # Create Manager instance
    manager = Manager()
    
    # Create DataFrame for data
    data_df = manager.create_data_dataframe(data_folder)
    
    if data_df.empty:
        print(Fore.RED + "\n ❌ No data files found!" + Style.RESET_ALL)
        return None
    
    # Save DataFrame to disk
    output_path = os.path.join(data_folder, "processed", "data_paths.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_df.to_csv(output_path, index=False)
    
    print(Fore.GREEN + f"\n ✅ Processed {len(data_df)} data pairs" + Style.RESET_ALL)
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
    print(Fore.GREEN + f"\n 🔄 Evaluating {stage} model..." + Style.RESET_ALL)
    
    # Create Manager instance
    manager = Manager()
    
    # Load the model from MLflow
    model = manager.load_mlflow_model(stage=stage)
    
    if model is None:
        print(Fore.RED + f"\n ❌ No model found in {stage} stage" + Style.RESET_ALL)
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
    print(Fore.CYAN + "\n 📊 Running evaluation..." + Style.RESET_ALL)
    
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
    
    print(Fore.GREEN + f"\n ✅ Evaluation complete: {metrics}" + Style.RESET_ALL)
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
    print(Fore.GREEN + f"\n 🔄 Training new {model_type} model for {epochs} epochs..." + Style.RESET_ALL)
    
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
    print(Fore.CYAN + "\n 🚀 Starting training..." + Style.RESET_ALL)
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
    
    print(Fore.GREEN + f"\n ✅ Training complete. Results saved to {results_dir}" + Style.RESET_ALL)
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
    print(Fore.GREEN + "\n 🔄 Comparing models..." + Style.RESET_ALL)
    
    # For GANs, the comparison is more complex than just comparing a single metric
    # This is a simplified comparison - adapt to your specific needs
    
    # Check if training improved over evaluation
    if "gen_loss" in train_metrics and "gen_loss" in eval_metrics:
        improvement = eval_metrics["gen_loss"] - train_metrics["gen_loss"]
        relative_improvement = improvement / eval_metrics["gen_loss"] if eval_metrics["gen_loss"] > 0 else float("inf")
        
        print(f"Generator loss: {eval_metrics['gen_loss']} -> {train_metrics['gen_loss']} (Improvement: {improvement:.4f})")
        
        if relative_improvement > threshold:
            print(Fore.GREEN + f"\n ✅ New model is better by {relative_improvement:.2%}" + Style.RESET_ALL)
            return True
        else:
            print(Fore.YELLOW + f"\n ⚠️ New model is not significantly better ({relative_improvement:.2%})" + Style.RESET_ALL)
            return False
    
    # Default to promoting if we can't compare (first run)
    print(Fore.YELLOW + "\n ⚠️ Cannot compare models, defaulting to promote" + Style.RESET_ALL)
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
        print(Fore.YELLOW + "\n ⚠️ Model promotion skipped" + Style.RESET_ALL)
        return
    
    print(Fore.GREEN + "\n 🔄 Promoting model to production..." + Style.RESET_ALL)
    
    # Get MLflow client
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    client = MlflowClient()
    
    # Get the latest model version
    model_name = os.environ.get("MLFLOW_MODEL_NAME")
    latest_version = Manager.get_model_version(stage="None")
    
    if latest_version is None:
        print(Fore.RED + "\n ❌ No model version found to promote" + Style.RESET_ALL)
        return
    
    # Transition the model to Production
    client.transition_model_version_stage(
        name=model_name,
        version=latest_version,
        stage="Production"
    )
    
    print(Fore.GREEN + f"\n ✅ Model version {latest_version} promoted to Production" + Style.RESET_ALL)


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
    print(Fore.GREEN + "\n 🔔 Sending notification..." + Style.RESET_ALL)
    
    # Prepare message
    message = "DeepSculpt Workflow Completed\n"
    message += f"Evaluation Metrics: {json.dumps(eval_metrics, indent=2)}\n"
    message += f"Training Metrics: {json.dumps(train_metrics, indent=2)}\n"
    message += f"Model Promoted: {'Yes' if promoted else 'No'}"
    
    # Example: Send to a webhook (replace with your actual notification method)
    try:
        # Comment out actual HTTP request to avoid errors
        # response = requests.post("https://your-webhook-url", json={"text": message})
        # response.raise_for_status()
        print(Fore.GREEN + "\n ✅ Notification sent" + Style.RESET_ALL)
        print(f"\nNotification Message:\n{message}")
    except Exception as e:
        print(Fore.RED + f"\n ❌ Failed to send notification: {e}" + Style.RESET_ALL)


def build_flow(schedule=None):
    """
    Build the Prefect workflow for DeepSculpt.
    
    Args:
        schedule: Optional schedule for the workflow
        
    Returns:
        Prefect Flow object
    """
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
    
    # Set up schedule if requested
    schedule = None
    if args.schedule:
        schedule = IntervalSchedule(
            interval=datetime.timedelta(days=1),
            end_date=datetime.datetime.now() + datetime.timedelta(days=30)
        )
    
    # Build the flow
    flow = build_flow(schedule)
    
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
        flow.run(parameters={
            "experiment": mlflow_experiment,
            "data_folder": args.data_folder,
            "model_type": args.model_type,
            "epochs": args.epochs
        })
        
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
        flow.register(os.environ.get("PREFECT_FLOW_NAME", "deepSculpt_project"))
        print(Fore.GREEN + "\n ✅ Workflow registered successfully" + Style.RESET_ALL)
    
    else:
        print(Fore.RED + f"\n ❌ Invalid mode: {args.mode}" + Style.RESET_ALL)


if __name__ == "__main__":
    main()