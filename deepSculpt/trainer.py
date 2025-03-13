"""
Training pipeline for DeepSculpt models using pandas DataFrame for data management.
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


class DataFrameDataLoader:
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
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
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
    
    def __len__(self):
        """Get number of batches per epoch."""
        return (self.num_samples + self.batch_size - 1) // self.batch_size
    
    def _load_sample(self, idx):
        """Load a single sample from disk."""
        volume_path = self.df.iloc[idx][self.volume_col]
        material_path = self.df.iloc[idx][self.material_col]
        
        try:
            volume_data = np.load(volume_path)
            material_data = np.load(material_path)
            
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


class DeepSculptTrainer:
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
    
    def train(self, data_loader, epochs, checkpoint_dir=None, snapshot_dir=None, snapshot_freq=1):
        """
        Train the model.
        
        Args:
            data_loader: DataFrameDataLoader or any object with create_tf_dataset() method
            epochs: Number of epochs to train for
            checkpoint_dir: Directory for saving checkpoints (optional)
            snapshot_dir: Directory for saving snapshots (optional)
            snapshot_freq: Frequency of saving snapshots (in epochs)
            
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
        
        # Training loop
        print(f"Starting training for {epochs} epochs...")
        for epoch in range(epochs):
            start_time = time.time()
            
            # Initialize metrics for this epoch
            epoch_gen_losses = []
            epoch_disc_losses = []
            batches = 0
            
            # Train on batches
            for batch in dataset:
                gen_loss, disc_loss = self.train_step(batch)
                epoch_gen_losses.append(float(gen_loss))
                epoch_disc_losses.append(float(disc_loss))
                batches += 1
                
                # Print progress for long epochs
                if batches % 10 == 0:
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
            if manager and (epoch + 1) % 5 == 0:
                save_path = manager.save()
                print(f"Saved checkpoint: {save_path}")
            
            # Generate and save snapshot if requested
            if snapshot_dir and (epoch + 1) % snapshot_freq == 0:
                self._save_snapshot(snapshot_dir, epoch)
        
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
    
    def plot_metrics(self, save_path=None):
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
        
        plt.show()


def create_data_dataframe(data_folder, pattern=None):
    """
    Create a DataFrame with paths to volume and material data files.
    
    Args:
        data_folder: Folder containing data files
        pattern: Optional pattern to filter files (e.g., date string)
        
    Returns:
        Pandas DataFrame with columns for volume and material paths
    """
    import glob
    import re
    
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
    from models import ModelFactory
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train DeepSculpt model")
    parser.add_argument("--data-folder", type=str, required=True, help="Folder containing data files")
    parser.add_argument("--model-type", type=str, default="skip", choices=["simple", "complex", "skip", "monochrome"],
                        help="Type of model architecture")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory for saving results")
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
    print(f"Creating {args.model_type} models...")
    generator = ModelFactory.create_generator(model_type=args.model_type)
    discriminator = ModelFactory.create_discriminator(model_type=args.model_type)
    
    # Create trainer
    trainer = DeepSculptTrainer(generator, discriminator)
    
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
    trainer.plot_metrics(save_path=metrics_path)
    
    print(f"Training complete. Results saved to {args.results_dir}")