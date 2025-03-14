"""
Training pipeline for DeepSculpt models with support for preprocessed data.

This module provides:
1. Data loading from various sources (raw files, pandas DataFrame, preprocessed datasets)
2. Training loops for different model architectures (GANs, Autoencoders)
3. Visualization and monitoring during training
4. Integration with MLflow for experiment tracking
5. Checkpoint management and model persistence
"""

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.train import Checkpoint, CheckpointManager
import matplotlib.pyplot as plt
from datetime import datetime
import json
import glob
from typing import Dict, List, Optional, Union, Tuple, Any, Callable

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# Import visualization module if available
try:
    from visualization import Visualizer
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False


# Loss functions
cross_entropy = BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """Calculate discriminator loss."""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    """Calculate generator loss."""
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def autoencoder_loss(real_data, reconstructed_data, alpha=1.0, beta=0.1):
    """
    Calculate loss for autoencoder models.
    
    Args:
        real_data: Original input data
        reconstructed_data: Reconstructed data from the autoencoder
        alpha: Weight for reconstruction loss
        beta: Weight for regularization loss
        
    Returns:
        Combined loss value
    """
    # Reconstruction loss (mean squared error)
    reconstruction_loss = tf.reduce_mean(tf.square(real_data - reconstructed_data))
    
    # Regularization loss
    reg_loss = tf.reduce_mean(tf.square(reconstructed_data))
    
    # Total loss
    total_loss = alpha * reconstruction_loss + beta * reg_loss
    
    return total_loss, reconstruction_loss, reg_loss


class BaseDataLoader:
    """Base class for data loaders."""
    
    def __init__(self, batch_size=32, shuffle=True):
        """
        Initialize the data loader.
        
        Args:
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data between epochs
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = 0
    
    def __len__(self):
        """Get number of batches per epoch."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def create_tf_dataset(self):
        """Create a TensorFlow dataset."""
        raise NotImplementedError("Subclasses must implement create_tf_dataset()")


class DataFrameDataLoader(BaseDataLoader):
    """Load training data from paths stored in a pandas DataFrame."""
    
    def __init__(self, df, batch_size=32, shuffle=True, volume_col='volume_path', 
                 material_col='material_path', processor=None):
        """
        Initialize the data loader.
        
        Args:
            df: Pandas DataFrame with paths to volume and material files
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data between epochs
            volume_col: Column name for volume file paths
            material_col: Column name for material file paths
            processor: Optional preprocessing function to apply to loaded data
        """
        super().__init__(batch_size, shuffle)
        self.df = df
        self.volume_col = volume_col
        self.material_col = material_col
        self.processor = processor
        self.num_samples = len(df)
        self.indices = np.arange(self.num_samples)
        
        # Check if the specified columns exist
        if volume_col not in df.columns:
            raise ValueError(f"Column {volume_col} not found in DataFrame")
        if material_col not in df.columns:
            raise ValueError(f"Column {material_col} not found in DataFrame")
        
        # Shuffle if necessary
        if shuffle:
            np.random.shuffle(self.indices)
    
    def _load_sample(self, idx):
        """Load a single sample from disk."""
        volume_path = self.df.iloc[idx][self.volume_col]
        material_path = self.df.iloc[idx][self.material_col]
        
        try:
            volume_data = np.load(volume_path, allow_pickle=True)
            material_data = np.load(material_path, allow_pickle=True)
            
            # Apply preprocessing if provided
            if self.processor:
                return self.processor(volume_data, material_data)
            else:
                return volume_data
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return None or a placeholder
            return None
    
    def _process_batch(self, batch_indices):
        """Load and process a batch of samples."""
        batch_data = []
        
        for idx in batch_indices:
            sample = self._load_sample(idx)
            if sample is not None:
                batch_data.append(sample)
        
        # If no valid samples were loaded, return None
        if not batch_data:
            return None
        
        # Convert to array
        return np.array(batch_data)
    
    def create_tf_dataset(self):
        """Create a TensorFlow dataset from this loader."""
        def generator():
            # Shuffle indices if requested
            if self.shuffle:
                np.random.shuffle(self.indices)
            
            # Yield batches
            for i in range(0, self.num_samples, self.batch_size):
                batch_indices = self.indices[i:min(i + self.batch_size, self.num_samples)]
                batch = self._process_batch(batch_indices)
                if batch is not None:
                    yield batch
        
        # Create dataset with unknown shape since samples may have different dimensions
        output_shapes = None
        if self.num_samples > 0:
            sample_data = self._load_sample(0)
            if sample_data is not None:
                output_shapes = tf.TensorShape([None] + list(sample_data.shape))
        
        if output_shapes:
            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=tf.float32,
                output_shapes=output_shapes
            )
        else:
            dataset = tf.data.Dataset.from_generator(
                generator,
                output_types=tf.float32
            )
        
        return dataset.prefetch(tf.data.AUTOTUNE)
    
    def iterate_batches(self):
        """Manually iterate through batches without TF dataset."""
        # Shuffle indices if requested
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Yield batches
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i:min(i + self.batch_size, self.num_samples)]
            batch = self._process_batch(batch_indices)
            if batch is not None:
                yield batch


class EncodedDataLoader(BaseDataLoader):
    """
    Data loader for preprocessed/encoded data from curator.py.
    
    This loader can work with data encoded by the various encoders in curator.py:
    - OneHotEncoderDecoder
    - BinaryEncoderDecoder
    - RGBEncoderDecoder
    """
    
    def __init__(self, data_dir, batch_size=32, shuffle=True, split="train"):
        """
        Initialize the encoded data loader.
        
        Args:
            data_dir: Directory containing processed data (from curator.py)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle data
            split: Which split to use ("train" or "val")
        """
        super().__init__(batch_size, shuffle)
        self.data_dir = data_dir
        self.split = split
        
        # Load metadata
        metadata_path = os.path.join(data_dir, "preprocessing_metadata.json")
        try:
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
                print(f"Loaded metadata: {self.metadata['processing_method']} encoding")
        except Exception as e:
            print(f"Warning: Failed to load metadata: {e}")
            self.metadata = {}
        
        # Determine dataset format based on metadata and available files
        # For this implementation, we'll scan for TensorFlow dataset files
        if split == "train":
            dataset_path = os.path.join(data_dir, "train_dataset")
        else:
            dataset_path = os.path.join(data_dir, "val_dataset")
        
        if os.path.exists(dataset_path):
            self.tf_dataset = tf.data.Dataset.load(dataset_path)
            # Get sample count
            if split == "train" and "train_size" in self.metadata:
                self.num_samples = self.metadata["train_size"]
            elif split == "val" and "val_size" in self.metadata:
                self.num_samples = self.metadata["val_size"]
            else:
                self.num_samples = sum(1 for _ in self.tf_dataset)
            
            print(f"Loaded {self.num_samples} samples from {dataset_path}")
        else:
            # Try loading from numpy arrays
            data_file = os.path.join(data_dir, f"{split}_data.npy")
            if os.path.exists(data_file):
                self.data = np.load(data_file)
                self.num_samples = len(self.data)
                print(f"Loaded {self.num_samples} samples from {data_file}")
            else:
                raise ValueError(f"No dataset found in {data_dir} for split '{split}'")
    
    def create_tf_dataset(self):
        """
        Create a TensorFlow dataset from the encoded data.
        
        Returns:
            TensorFlow dataset
        """
        if hasattr(self, "tf_dataset"):
            # Dataset already loaded
            if self.shuffle:
                buffer_size = min(10000, self.num_samples)
                return self.tf_dataset.shuffle(buffer_size).batch(self.batch_size)
            else:
                return self.tf_dataset.batch(self.batch_size)
        elif hasattr(self, "data"):
            # Create dataset from numpy array
            dataset = tf.data.Dataset.from_tensor_slices(self.data)
            if self.shuffle:
                buffer_size = min(10000, self.num_samples)
                dataset = dataset.shuffle(buffer_size)
            return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            raise ValueError("No data available to create TensorFlow dataset")


class BaseTrainer:
    """Base class for trainers."""
    
    def __init__(self, learning_rate=0.0002):
        """
        Initialize the trainer.
        
        Args:
            learning_rate: Learning rate for optimizers
        """
        self.learning_rate = learning_rate
        self.metrics = {}
        
        # Initialize visualizer if available
        self.visualizer = None
        if VISUALIZER_AVAILABLE:
            self.visualizer = Visualizer(figsize=10, dpi=100)
    
    def create_checkpoint(self, checkpoint_dir):
        """
        Create checkpoint objects.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Tuple of (checkpoint, manager)
        """
        raise NotImplementedError("Subclasses must implement create_checkpoint()")
    
    def train(self, data_loader, epochs, checkpoint_dir=None, snapshot_dir=None):
        """
        Train the model.
        
        Args:
            data_loader: Data loader instance
            epochs: Number of epochs to train for
            checkpoint_dir: Directory for saving checkpoints (optional)
            snapshot_dir: Directory for saving snapshots (optional)
            
        Returns:
            Training metrics
        """
        raise NotImplementedError("Subclasses must implement train()")
    
    def save_to_mlflow(self, mlflow_model_name=None, params=None):
        """
        Save the model to MLflow.
        
        Args:
            mlflow_model_name: Name of the model in MLflow
            params: Additional parameters to log
            
        Returns:
            Boolean indicating success
        """
        raise NotImplementedError("Subclasses must implement save_to_mlflow()")


class DeepSculptTrainer(BaseTrainer):
    """Trainer for DeepSculpt 3D generation models."""
    
    def __init__(self, generator, discriminator, learning_rate=0.0002, beta1=0.5, beta2=0.999):
        """
        Initialize the trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            learning_rate: Learning rate for optimizers
            beta1: Beta1 parameter for Adam optimizer
            beta2: Beta2 parameter for Adam optimizer
        """
        super().__init__(learning_rate)
        self.generator = generator
        self.discriminator = discriminator
        
        # Create optimizers
        self.generator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
        self.discriminator_optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
        
        # Initialize metrics tracking
        self.metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'epoch_times': []
        }
        
        # Create seed noise for consistent evaluation
        self.seed = tf.random.normal([16, 100])  # Assuming noise_dim=100
    
    def create_checkpoint(self, checkpoint_dir):
        """
        Create checkpoint objects.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Tuple of (checkpoint, manager)
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = Checkpoint(
            step=tf.Variable(1),
            generator_optimizer=self.generator_optimizer,
            discriminator_optimizer=self.discriminator_optimizer,
            generator=self.generator,
            discriminator=self.discriminator
        )
        
        manager = CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3,
            checkpoint_name="checkpoint"
        )
        
        return checkpoint, manager
    
    def restore_from_checkpoint(self, manager):
        """
        Restore from checkpoint if available.
        
        Args:
            manager: CheckpointManager object
            
        Returns:
            Boolean indicating whether restoration was successful
        """
        if manager.latest_checkpoint:
            status = manager.checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from checkpoint: {manager.latest_checkpoint}")
            return True
        else:
            print("No checkpoint found, starting from scratch")
            return False
    
    @tf.function
    def train_step(self, real_images):
        """
        Execute a single training step.
        
        Args:
            real_images: Batch of real images
            
        Returns:
            Tuple of (gen_loss, disc_loss)
        """
        # Generate random noise
        batch_size = tf.shape(real_images)[0]
        noise = tf.random.normal([batch_size, 100])  # Assuming noise_dim=100
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images
            generated_images = self.generator(noise, training=True)
            
            # Get discriminator predictions
            real_output = self.discriminator(real_images, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            
            # Calculate losses
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.generator_optimizer.apply_gradients(
            zip(gen_gradients, self.generator.trainable_variables)
        )
        self.discriminator_optimizer.apply_gradients(
            zip(disc_gradients, self.discriminator.trainable_variables)
        )
        
        return gen_loss, disc_loss
    
    def train(self, data_loader, epochs, checkpoint_dir=None, snapshot_dir=None, snapshot_freq=1,
              callbacks=None, log_freq=1, save_best_only=False):
        """
        Train the model.
        
        Args:
            data_loader: DataLoader object with create_tf_dataset() method
            epochs: Number of epochs to train for
            checkpoint_dir: Directory for saving checkpoints (optional)
            snapshot_dir: Directory for saving snapshots (optional)
            snapshot_freq: Frequency of saving snapshots (in epochs)
            callbacks: List of callback functions to call after each epoch
            log_freq: Frequency of logging progress (in batches)
            save_best_only: Whether to save only the best checkpoint
            
        Returns:
            Training metrics
        """
        # Create TensorFlow dataset
        print("Creating TensorFlow dataset...")
        dataset = data_loader.create_tf_dataset()
        
        # Set up checkpointing if requested
        manager = None
        if checkpoint_dir:
            checkpoint, manager = self.create_checkpoint(checkpoint_dir)
            self.restore_from_checkpoint(manager)
        
        # Set up snapshot directory if requested
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
        
        # Initialize best metrics for save_best_only
        best_gen_loss = float('inf')
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            start_time = time.time()
            
            # Initialize metrics for this epoch
            epoch_gen_losses = []
            epoch_disc_losses = []
            batches = 0
            
            # Create progress bar if tqdm is available
            if TQDM_AVAILABLE:
                # We need to determine the number of batches for tqdm
                # Since we don't know the exact number, we'll use an estimate
                estimated_batches = len(data_loader)
                batch_iterator = tqdm(
                    dataset, total=estimated_batches, 
                    desc=f"Epoch {epoch+1}/{epochs}",
                    unit="batch"
                )
            else:
                batch_iterator = dataset
            
            # Train on batches
            for batch in batch_iterator:
                gen_loss, disc_loss = self.train_step(batch)
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(float(disc_loss))
                batches += 1
                
                # Print progress for long epochs
                if not TQDM_AVAILABLE and batches % log_freq == 0:
                    print(f"  Processed {batches} batches")
            
            # Calculate average losses
            avg_gen_loss = np.mean(epoch_gen_losses) if epoch_gen_losses else 0
            avg_disc_loss = np.mean(epoch_disc_losses) if epoch_disc_losses else 0
            
            # Update metrics
            self.metrics['gen_loss'].append(avg_gen_loss)
            self.metrics['disc_loss'].append(avg_disc_loss)
            
            # Record epoch time
            epoch_time = time.time() - start_time
            self.metrics['epoch_times'].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Save checkpoint if requested
            if manager:
                if save_best_only:
                    if avg_gen_loss < best_gen_loss:
                        save_path = manager.save()
                        best_gen_loss = avg_gen_loss
                        print(f"Saved best checkpoint (gen_loss: {avg_gen_loss:.4f}): {save_path}")
                elif (epoch + 1) % 5 == 0:
                    save_path = manager.save()
                    print(f"Saved checkpoint: {save_path}")
            
            # Generate and save snapshot if requested
            if snapshot_dir and (epoch + 1) % snapshot_freq == 0:
                self._save_snapshot(snapshot_dir, epoch)
            
            # Call callbacks if provided
            if callbacks:
                epoch_info = {
                    'epoch': epoch + 1,
                    'gen_loss': avg_gen_loss,
                    'disc_loss': avg_disc_loss,
                    'time': epoch_time,
                    'generator': self.generator,
                    'discriminator': self.discriminator
                }
                for callback in callbacks:
                    callback(epoch_info)
        
        print("Training complete!")
        return self.metrics
    
    def _save_snapshot(self, snapshot_dir, epoch):
        """Generate and save a snapshot of the current model state."""
        print(f"Generating snapshot for epoch {epoch+1}...")
        
        # Generate images from seed
        predictions = self.generator(self.seed, training=False)
        
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        
        # We need to determine the visualization approach based on the data
        # For 3D data, we can show slices or projections
        if predictions.shape[-1] > 1:  # Multi-channel data
            # Show middle slice of first 16 samples with RGB channels
            for i in range(min(16, predictions.shape[0])):
                plt.subplot(4, 4, i+1)
                
                # Get middle slice
                middle_slice = predictions[i, predictions.shape[1]//2, :, :, :3]
                
                # Normalize to [0, 1] for display
                middle_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice) + 1e-6)
                
                plt.imshow(middle_slice)
                plt.axis('off')
        else:
            # Show middle slice of first 16 samples (grayscale)
            for i in range(min(16, predictions.shape[0])):
                plt.subplot(4, 4, i+1)
                
                # Get middle slice
                middle_slice = predictions[i, predictions.shape[1]//2, :, :, 0]
                
                plt.imshow(middle_slice, cmap='gray')
                plt.axis('off')
        
        plt.tight_layout()
        
        # Save figure
        snapshot_path = os.path.join(snapshot_dir, f'snapshot_epoch_{epoch+1:04d}.png')
        plt.savefig(snapshot_path)
        plt.close(fig)
        
        print(f"Snapshot saved to {snapshot_path}")
        
        # If visualizer is available, try to use it for more advanced visualization
        if self.visualizer is not None:
            try:
                # Convert predictions to appropriate format for the visualizer
                # This is a simplified example and may need adjustment based on your specific needs
                for i in range(min(2, predictions.shape[0])):
                    structure = (predictions[i].numpy() > 0.5).astype(np.int8)
                    viz_path = os.path.join(snapshot_dir, f'viz_sample_{i}_epoch_{epoch+1:04d}.png')
                    
                    # Use the visualizer to create a 3D visualization
                    self.visualizer.plot_sculpture(
                        structure=structure,
                        title=f"Sample {i} - Epoch {epoch+1}",
                        hide_axis=True,
                        save_path=viz_path,
                    )
            except Exception as e:
                print(f"Warning: Failed to create advanced visualization: {e}")
    
    def plot_metrics(self, save_path=None, show=False):
        """Plot training metrics and optionally save to file."""
        if not self.metrics['gen_loss']:
            print("No metrics to plot yet")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.metrics['gen_loss']) + 1)
        plt.plot(epochs, self.metrics['gen_loss'], 'b-', label='Generator Loss')
        plt.plot(epochs, self.metrics['disc_loss'], 'r-', label='Discriminator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot epoch times
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.metrics['epoch_times'], 'g-')
        plt.title('Epoch Times')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved metrics plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_to_mlflow(self, mlflow_model_name=None, params=None):
        """
        Save the model to MLflow.
        
        Args:
            mlflow_model_name: Name of the model in MLflow
            params: Additional parameters to log
            
        Returns:
            Boolean indicating success
        """
        try:
            # Import MLflow
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Retrieve MLflow env params
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
            if mlflow_model_name is None:
                mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
            
            # Configure MLflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=mlflow_experiment)
            
            # Prepare metrics
            metrics = {}
            if self.metrics:
                if self.metrics.get('gen_loss'):
                    metrics["final_gen_loss"] = float(self.metrics['gen_loss'][-1])
                if self.metrics.get('disc_loss'):
                    metrics["final_disc_loss"] = float(self.metrics['disc_loss'][-1])
                if self.metrics.get('epoch_times'):
                    metrics["avg_epoch_time"] = float(np.mean(self.metrics['epoch_times']))
                    metrics["total_train_time"] = float(np.sum(self.metrics['epoch_times']))
            
            # Start MLflow run
            with mlflow.start_run():
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log the generator model
                mlflow.keras.log_model(
                    keras_model=self.generator,
                    artifact_path="model",
                    keras_module="tensorflow.keras",
                    registered_model_name=mlflow_model_name,
                )
            
            print("\n✓ Data saved in MLflow")
            return True
        
        except Exception as e:
            print(f"Error saving to MLflow: {e}")
            return False
    
    def generate_samples(self, num_samples=1, noise_dim=100, seed=None, slice_axis=0, slice_position=0.5):
        """
        Generate samples from the trained generator.
        
        Args:
            num_samples: Number of samples to generate
            noise_dim: Dimension of the noise vector
            seed: Random seed for reproducible generation
            slice_axis: Axis to use for 2D slice visualization
            slice_position: Position along the axis for slice (0.0-1.0)
            
        Returns:
            Dictionary with generated samples and visualizations
        """
        # Set seed for reproducibility if provided
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
        
        # Generate noise
        noise = tf.random.normal([num_samples, noise_dim])
        
        # Generate samples
        predictions = self.generator(noise, training=False)
        
        # Create visualizations
        if self.visualizer:
            # Convert predictions to appropriate format for the visualizer
            # This is a simplified example and may need adjustment
            structures = []
            for i in range(num_samples):
                structure = (predictions[i].numpy() > 0.5).astype(np.int8)
                structures.append(structure)
            
            # Create a directory for visualizations
            viz_dir = os.path.join("results", "generated", datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(viz_dir, exist_ok=True)
            
            # Generate visualizations
            viz_paths = []
            for i, structure in enumerate(structures):
                viz_path = os.path.join(viz_dir, f'sample_{i:03d}.png')
                self.visualizer.plot_sculpture(
                    structure=structure,
                    title=f"Generated Sample {i}",
                    hide_axis=True,
                    save_path=viz_path,
                )
                viz_paths.append(viz_path)
            
            # Also create 2D slice visualizations
            slice_paths = []
            for i, pred in enumerate(predictions):
                # Get slice based on axis and position
                slice_idx = int(slice_position * pred.shape[slice_axis])
                
                if slice_axis == 0:
                    slice_data = pred[slice_idx, :, :, :3].numpy()  # Use first 3 channels for RGB
                elif slice_axis == 1:
                    slice_data = pred[:, slice_idx, :, :3].numpy()
                else:
                    slice_data = pred[:, :, slice_idx, :3].numpy()
                
                # Normalize for display
                slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data) + 1e-6)
                
                # Create figure
                fig = plt.figure(figsize=(6, 6))
                plt.imshow(slice_data)
                plt.axis('off')
                plt.title(f"Sample {i} - Slice")
                
                # Save figure
                slice_path = os.path.join(viz_dir, f'slice_{i:03d}.png')
                plt.savefig(slice_path)
                plt.close(fig)
                slice_paths.append(slice_path)
            
            return {
                'samples': predictions.numpy(),
                'viz_dir': viz_dir,
                'viz_paths': viz_paths,
                'slice_paths': slice_paths
            }
        else:
            # Just return the raw predictions
            return {
                'samples': predictions.numpy()
            }


class AutoencoderTrainer(BaseTrainer):
    """Trainer for DeepSculpt 3D autoencoder models."""
    
    def __init__(self, encoder, decoder, learning_rate=0.0001, beta1=0.9, beta2=0.999):
        """
        Initialize the trainer.
        
        Args:
            encoder: Encoder model
            decoder: Decoder model
            learning_rate: Learning rate for optimizer
            beta1: Beta1 parameter for Adam optimizer
            beta2: Beta2 parameter for Adam optimizer
        """
        super().__init__(learning_rate)
        self.encoder = encoder
        self.decoder = decoder
        
        # Create optimizer
        self.optimizer = Adam(learning_rate=learning_rate, beta_1=beta1, beta_2=beta2)
        
        # Initialize metrics tracking
        self.metrics = {
            'total_loss': [],
            'reconstruction_loss': [],
            'reg_loss': [],
            'epoch_times': []
        }
    
    def create_checkpoint(self, checkpoint_dir):
        """
        Create checkpoint objects.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            
        Returns:
            Tuple of (checkpoint, manager)
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = Checkpoint(
            step=tf.Variable(1),
            optimizer=self.optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        
        manager = CheckpointManager(
            checkpoint=checkpoint,
            directory=checkpoint_dir,
            max_to_keep=3,
            checkpoint_name="autoencoder_checkpoint"
        )
        
        return checkpoint, manager
    
    def restore_from_checkpoint(self, manager):
        """
        Restore from checkpoint if available.
        
        Args:
            manager: CheckpointManager object
            
        Returns:
            Boolean indicating whether restoration was successful
        """
        if manager.latest_checkpoint:
            status = manager.checkpoint.restore(manager.latest_checkpoint)
            print(f"Restored from checkpoint: {manager.latest_checkpoint}")
            return True
        else:
            print("No checkpoint found, starting from scratch")
            return False
    
    @tf.function
    def train_step(self, batch):
        """
        Execute a single training step.
        
        Args:
            batch: Batch of input data
            
        Returns:
            Tuple of (total_loss, reconstruction_loss, reg_loss)
        """
        with tf.GradientTape() as tape:
            # Forward pass through encoder and decoder
            latent = self.encoder(batch, training=True)
            reconstructed = self.decoder(latent, training=True)
            
            # Calculate losses
            total_loss, reconstruction_loss, reg_loss = autoencoder_loss(
                batch, reconstructed, alpha=1.0, beta=0.1
            )
        
        # Calculate gradients
        trainable_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return total_loss, reconstruction_loss, reg_loss
    
    def train(self, data_loader, epochs, checkpoint_dir=None, snapshot_dir=None, snapshot_freq=1,
              callbacks=None, log_freq=1, save_best_only=False):
        """
        Train the autoencoder model.
        
        Args:
            data_loader: DataLoader object with create_tf_dataset() method
            epochs: Number of epochs to train for
            checkpoint_dir: Directory for saving checkpoints (optional)
            snapshot_dir: Directory for saving snapshots (optional)
            snapshot_freq: Frequency of saving snapshots (in epochs)
            callbacks: List of callback functions to call after each epoch
            log_freq: Frequency of logging progress (in batches)
            save_best_only: Whether to save only the best checkpoint
            
        Returns:
            Training metrics
        """
        # Create TensorFlow dataset
        print("Creating TensorFlow dataset...")
        dataset = data_loader.create_tf_dataset()
        
        # Set up checkpointing if requested
        manager = None
        if checkpoint_dir:
            checkpoint, manager = self.create_checkpoint(checkpoint_dir)
            self.restore_from_checkpoint(manager)
        
        # Set up snapshot directory if requested
        if snapshot_dir:
            os.makedirs(snapshot_dir, exist_ok=True)
        
        # Initialize best metrics for save_best_only
        best_total_loss = float('inf')
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            start_time = time.time()
            
            # Initialize metrics for this epoch
            epoch_total_losses = []
            epoch_reconstruction_losses = []
            epoch_reg_losses = []
            batches = 0
            
            # Create progress bar if tqdm is available
            if TQDM_AVAILABLE:
                estimated_batches = len(data_loader)
                batch_iterator = tqdm(
                    dataset, total=estimated_batches, 
                    desc=f"Epoch {epoch+1}/{epochs}",
                    unit="batch"
                )
            else:
                batch_iterator = dataset
            
            # Train on batches
            for batch in batch_iterator:
                total_loss, reconstruction_loss, reg_loss = self.train_step(batch)
                epoch_total_losses.append(float(total_loss))
                epoch_reconstruction_losses.append(float(reconstruction_loss))
                epoch_reg_losses.append(float(reg_loss))
                batches += 1
                
                # Print progress for long epochs
                if not TQDM_AVAILABLE and batches % log_freq == 0:
                    print(f"  Processed {batches} batches")
            
            # Calculate average losses
            avg_total_loss = np.mean(epoch_total_losses) if epoch_total_losses else 0
            avg_reconstruction_loss = np.mean(epoch_reconstruction_losses) if epoch_reconstruction_losses else 0
            avg_reg_loss = np.mean(epoch_reg_losses) if epoch_reg_losses else 0
            
            # Update metrics
            self.metrics['total_loss'].append(avg_total_loss)
            self.metrics['reconstruction_loss'].append(avg_reconstruction_loss)
            self.metrics['reg_loss'].append(avg_reg_loss)
            
            # Record epoch time
            epoch_time = time.time() - start_time
            self.metrics['epoch_times'].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - Total Loss: {avg_total_loss:.4f}, "
                  f"Reconstruction Loss: {avg_reconstruction_loss:.4f}, "
                  f"Reg Loss: {avg_reg_loss:.4f}, "
                  f"Time: {epoch_time:.2f}s")
            
            # Save checkpoint if requested
            if manager:
                if save_best_only:
                    if avg_total_loss < best_total_loss:
                        save_path = manager.save()
                        best_total_loss = avg_total_loss
                        print(f"Saved best checkpoint (loss: {avg_total_loss:.4f}): {save_path}")
                elif (epoch + 1) % 5 == 0:
                    save_path = manager.save()
                    print(f"Saved checkpoint: {save_path}")
            
            # Generate and save snapshot if requested
            if snapshot_dir and (epoch + 1) % snapshot_freq == 0:
                self._save_snapshot(dataset, snapshot_dir, epoch)
            
            # Call callbacks if provided
            if callbacks:
                epoch_info = {
                    'epoch': epoch + 1,
                    'total_loss': avg_total_loss,
                    'reconstruction_loss': avg_reconstruction_loss,
                    'reg_loss': avg_reg_loss,
                    'time': epoch_time,
                    'encoder': self.encoder,
                    'decoder': self.decoder
                }
                for callback in callbacks:
                    callback(epoch_info)
        
        print("Training complete!")
        return self.metrics
    
    def _save_snapshot(self, dataset, snapshot_dir, epoch):
        """Generate and save a snapshot of the current model state."""
        print(f"Generating snapshot for epoch {epoch+1}...")
        
        # Take a few samples from the dataset
        try:
            samples = next(iter(dataset.take(1)))
            
            # For small batch sizes, we might need to select just a few samples
            if len(samples) > 8:
                samples = samples[:8]
            
            # Encode and decode the samples
            latent = self.encoder(samples, training=False)
            reconstructed = self.decoder(latent, training=False)
            
            # Create a figure
            fig = plt.figure(figsize=(12, 8))
            
            # We need to determine the visualization approach based on the data
            # For 3D data, we can show slices or projections
            for i in range(min(samples.shape[0], 4)):
                # Original sample - middle slice
                plt.subplot(4, 4, i*4 + 1)
                if samples.shape[-1] > 1:  # Multi-channel data
                    middle_slice = samples[i, samples.shape[1]//2, :, :, :3]
                    middle_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice) + 1e-6)
                    plt.imshow(middle_slice)
                else:
                    middle_slice = samples[i, samples.shape[1]//2, :, :, 0]
                    plt.imshow(middle_slice, cmap='gray')
                plt.title(f"Original {i}")
                plt.axis('off')
                
                # Reconstructed sample - middle slice
                plt.subplot(4, 4, i*4 + 2)
                if reconstructed.shape[-1] > 1:  # Multi-channel data
                    middle_slice = reconstructed[i, reconstructed.shape[1]//2, :, :, :3]
                    middle_slice = (middle_slice - np.min(middle_slice)) / (np.max(middle_slice) - np.min(middle_slice) + 1e-6)
                    plt.imshow(middle_slice)
                else:
                    middle_slice = reconstructed[i, reconstructed.shape[1]//2, :, :, 0]
                    plt.imshow(middle_slice, cmap='gray')
                plt.title(f"Reconstructed {i}")
                plt.axis('off')
                
                # Latent representation
                plt.subplot(4, 4, i*4 + 3)
                latent_sample = latent[i].numpy()
                latent_2d = latent_sample.reshape(int(np.sqrt(latent_sample.shape[0])), -1)
                plt.imshow(latent_2d, cmap='viridis')
                plt.title(f"Latent {i}")
                plt.axis('off')
                
                # Difference (error) visualization
                plt.subplot(4, 4, i*4 + 4)
                if samples.shape[-1] > 1:  # Multi-channel data
                    diff = np.abs(
                        samples[i, samples.shape[1]//2, :, :, :3] - 
                        reconstructed[i, reconstructed.shape[1]//2, :, :, :3]
                    )
                    # Normalize for better visualization
                    diff = diff / np.max(diff) if np.max(diff) > 0 else diff
                    plt.imshow(diff, cmap='hot')
                else:
                    diff = np.abs(
                        samples[i, samples.shape[1]//2, :, :, 0] - 
                        reconstructed[i, reconstructed.shape[1]//2, :, :, 0]
                    )
                    # Normalize for better visualization
                    diff = diff / np.max(diff) if np.max(diff) > 0 else diff
                    plt.imshow(diff, cmap='hot')
                plt.title(f"Error {i}")
                plt.axis('off')
            
            plt.tight_layout()
            
            # Save figure
            snapshot_path = os.path.join(snapshot_dir, f'ae_snapshot_epoch_{epoch+1:04d}.png')
            plt.savefig(snapshot_path)
            plt.close(fig)
            
            print(f"Snapshot saved to {snapshot_path}")
            
        except Exception as e:
            print(f"Warning: Failed to create snapshot: {e}")
    
    def plot_metrics(self, save_path=None, show=False):
        """Plot training metrics and optionally save to file."""
        if not self.metrics['total_loss']:
            print("No metrics to plot yet")
            return
        
        plt.figure(figsize=(15, 6))
        
        # Plot losses
        plt.subplot(1, 3, 1)
        epochs = range(1, len(self.metrics['total_loss']) + 1)
        plt.plot(epochs, self.metrics['total_loss'], 'b-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.metrics['reconstruction_loss'], 'g-', label='Reconstruction Loss')
        plt.plot(epochs, self.metrics['reg_loss'], 'r-', label='Regularization Loss')
        plt.title('Component Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot epoch times
        plt.subplot(1, 3, 3)
        plt.plot(epochs, self.metrics['epoch_times'], 'g-')
        plt.title('Epoch Times')
        plt.xlabel('Epoch')
        plt.ylabel('Time (seconds)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved metrics plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_to_mlflow(self, mlflow_model_name=None, params=None):
        """
        Save the model to MLflow.
        
        Args:
            mlflow_model_name: Name of the model in MLflow
            params: Additional parameters to log
            
        Returns:
            Boolean indicating success
        """
        try:
            # Import MLflow
            import mlflow
            from mlflow.tracking import MlflowClient
            
            # Retrieve MLflow env params
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
            mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
            if mlflow_model_name is None:
                mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME", "autoencoder")
            
            # Configure MLflow
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name=mlflow_experiment)
            
            # Prepare metrics
            metrics = {}
            if self.metrics:
                if self.metrics.get('total_loss'):
                    metrics["final_total_loss"] = float(self.metrics['total_loss'][-1])
                if self.metrics.get('reconstruction_loss'):
                    metrics["final_reconstruction_loss"] = float(self.metrics['reconstruction_loss'][-1])
                if self.metrics.get('reg_loss'):
                    metrics["final_reg_loss"] = float(self.metrics['reg_loss'][-1])
                if self.metrics.get('epoch_times'):
                    metrics["avg_epoch_time"] = float(np.mean(self.metrics['epoch_times']))
                    metrics["total_train_time"] = float(np.sum(self.metrics['epoch_times']))
            
            # Start MLflow run
            with mlflow.start_run():
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                if metrics:
                    mlflow.log_metrics(metrics)
                
                # Log the encoder and decoder models
                mlflow.keras.log_model(
                    keras_model=self.encoder,
                    artifact_path="encoder",
                    keras_module="tensorflow.keras",
                    registered_model_name=f"{mlflow_model_name}_encoder",
                )
                
                mlflow.keras.log_model(
                    keras_model=self.decoder,
                    artifact_path="decoder",
                    keras_module="tensorflow.keras",
                    registered_model_name=f"{mlflow_model_name}_decoder",
                )
            
            print("\n✓ Data saved in MLflow")
            return True
        
        except Exception as e:
            print(f"Error saving to MLflow: {e}")
            return False
    
    def encode_samples(self, samples):
        """
        Encode samples to latent space.
        
        Args:
            samples: Input samples to encode
            
        Returns:
            Latent vectors
        """
        return self.encoder(samples, training=False)
    
    def decode_samples(self, latent_vectors):
        """
        Decode latent vectors to samples.
        
        Args:
            latent_vectors: Latent vectors to decode
            
        Returns:
            Reconstructed samples
        """
        return self.decoder(latent_vectors, training=False)
    
    def reconstruct_samples(self, samples):
        """
        Reconstruct samples through the autoencoder.
        
        Args:
            samples: Input samples to reconstruct
            
        Returns:
            Reconstructed samples
        """
        latent = self.encoder(samples, training=False)
        return self.decoder(latent, training=False)
    
    def interpolate_samples(self, sample1, sample2, num_steps=10):
        """
        Interpolate between two samples in latent space.
        
        Args:
            sample1: First sample
            sample2: Second sample
            num_steps: Number of interpolation steps
            
        Returns:
            Interpolated samples
        """
        # Ensure samples have batch dimension
        if len(sample1.shape) < len(self.encoder.input.shape):
            sample1 = tf.expand_dims(sample1, 0)
        if len(sample2.shape) < len(self.encoder.input.shape):
            sample2 = tf.expand_dims(sample2, 0)
        
        # Encode samples to latent space
        latent1 = self.encoder(sample1, training=False)
        latent2 = self.encoder(sample2, training=False)
        
        # Generate interpolated latent vectors
        alphas = np.linspace(0, 1, num_steps)
        interp_latents = []
        
        for alpha in alphas:
            interp_latent = latent1 * (1 - alpha) + latent2 * alpha
            interp_latents.append(interp_latent)
        
        # Stack interpolated latent vectors
        interp_latents = tf.concat(interp_latents, axis=0)
        
        # Decode interpolated latent vectors
        interp_samples = self.decoder(interp_latents, training=False)
        
        return interp_samples.numpy()


def create_data_dataframe(data_folder, pattern=None):
    """
    Create a DataFrame with paths to volume and material data files.
    
    Args:
        data_folder: Folder containing data files
        pattern: Optional pattern to filter files (e.g., date string)
        
    Returns:
        Pandas DataFrame with columns for volume and material paths
    """
    # Import necessary modules
    import glob
    import re
    
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
    
    print(f"Found {len(volume_files)} volume files and {len(material_files)} material files")
    
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


def preprocess_volume_material(volume_data, material_data, color_mode=1):
    """
    Preprocess volume and material data.
    
    Args:
        volume_data: Volume data array
        material_data: Material data array
        color_mode: 0 for monochrome, 1 for color
        
    Returns:
        Processed data ready for model input
    """
    # Apply one-hot encoding or other preprocessing here
    # This is a placeholder - adjust based on your specific needs
    
    # For now, we'll just return the volume data as a float32 array
    return volume_data.astype(np.float32)


# Example usage
if __name__ == "__main__":
    import argparse
    try:
        from models import ModelFactory
    except ImportError:
        print("Warning: models.py not found. Some functionality may be limited.")
        ModelFactory = None
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DeepSculpt model")
    parser.add_argument("--data-folder", type=str, required=True, help="Folder containing data files")
    parser.add_argument("--model-type", type=str, default="skip", choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                        help="Type of model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory for saving results")
    parser.add_argument("--mlflow", action="store_true", help="Save models to MLflow")
    args = parser.parse_args()
    
    # Create results directories
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    snapshot_dir = os.path.join(args.results_dir, "snapshots")
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Create data DataFrame
    print(f"Creating data DataFrame from {args.data_folder}...")
    data_df = create_data_dataframe(args.data_folder)
    print(f"Found {len(data_df)} data pairs")
    
    # Create data loader
    data_loader = DataFrameDataLoader(
        data_df,
        batch_size=args.batch_size,
        processor=preprocess_volume_material
    )
    
    # Create models
    if ModelFactory is not None:
        print(f"Creating {args.model_type} models...")
        
        if args.model_type == "autoencoder":
            # Create encoder and decoder for autoencoder
            from models import create_encoder
            
            latent_dim = 100
            void_dim = 64
            
            encoder = create_encoder(latent_dim=latent_dim, input_shape=(void_dim, void_dim, void_dim, 1))
            decoder = ModelFactory.create_generator(model_type="autoencoder", void_dim=void_dim, noise_dim=latent_dim)
            
            # Create autoencoder trainer
            trainer = AutoencoderTrainer(encoder=encoder, decoder=decoder)
        else:
            # Create generator and discriminator for GAN
            generator = ModelFactory.create_generator(model_type=args.model_type)
            discriminator = ModelFactory.create_discriminator(model_type=args.model_type)
            
            # Create GAN trainer
            trainer = DeepSculptTrainer(generator=generator, discriminator=discriminator)
        
        # Train the model
        print("Starting training...")
        metrics = trainer.train(
            data_loader=data_loader,
            epochs=args.epochs,
            checkpoint_dir=checkpoint_dir,
            snapshot_dir=snapshot_dir
        )
        
        # Plot and save metrics
        metrics_path = os.path.join(args.results_dir, "training_metrics.png")
        trainer.plot_metrics(save_path=metrics_path, show=False)
        
        # Save to MLflow if requested
        if args.mlflow:
            params = {
                "model_type": args.model_type,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "data_folder": args.data_folder
            }
            trainer.save_to_mlflow(params=params)
        
        print(f"Training complete. Results saved to {args.results_dir}")
    
    else:
        print("ModelFactory not available. Unable to create models.")