"""
End-to-End Test Script for DeepSculpt

This module runs a complete end-to-end test of the DeepSculpt pipeline:
1. Data Generation: Creates synthetic 3D volumes and materials
2. Data Preprocessing: Prepares data for model consumption
3. Model Training: Trains a 3D generation model
4. Inference: Generates new 3D models
5. Visualization: Creates visualizations of the process and results

Tests the entire pipeline from data generation to model output.
"""

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd

# Optional: import logging if available
try:
    from logger import begin_section, end_section, log_action, log_success, log_info, log_error
    logging_available = True
except ImportError:
    # Create simplified versions if logger module is not available
    logging_available = False
    def begin_section(msg): print(f"\n== {msg} ==")
    def end_section(msg=""): print(f"== {msg if msg else 'Complete'} ==")
    def log_action(msg): print(f"→ {msg}")
    def log_success(msg): print(f"✓ {msg}")
    def log_info(msg): print(f"i {msg}")
    def log_error(msg): print(f"✗ {msg}")

# Import DeepSculpt modules
try:
    from models import ModelFactory
    from trainer import DeepSculptTrainer, DataFrameDataLoader, create_data_dataframe
    from workflow import Manager
except ImportError as e:
    log_error(f"Cannot import DeepSculpt modules: {e}")
    log_error("Make sure the modules are in your PYTHONPATH")
    sys.exit(1)


def create_directories():
    """Create the necessary directories for test results"""
    begin_section("Creating Test Directories")
    
    # Create main results directory
    results_dir = "e2e_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories for different stages
    data_dir = os.path.join(results_dir, "data")
    models_dir = os.path.join(results_dir, "models")
    train_viz_dir = os.path.join(results_dir, "training_visualizations")
    gen_viz_dir = os.path.join(results_dir, "generation_visualizations")
    checkpoints_dir = os.path.join(results_dir, "checkpoints")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(train_viz_dir, exist_ok=True)
    os.makedirs(gen_viz_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    log_success(f"Created test directories in {results_dir}")
    end_section()
    
    return results_dir, data_dir, models_dir, train_viz_dir, gen_viz_dir, checkpoints_dir


def generate_test_data(data_dir, num_samples=10, void_dim=32, num_chunks=2):
    """Generate synthetic data for testing"""
    begin_section("Generating Test Data")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    volume_paths = []
    material_paths = []
    
    for chunk_idx in range(1, num_chunks+1):
        log_action(f"Generating chunk {chunk_idx} with {num_samples} samples")
        
        # Create volume data with 3D shapes
        volumes = np.zeros((num_samples, void_dim, void_dim, void_dim), dtype=np.int32)
        
        # Create material data
        materials = np.zeros_like(volumes)
        
        # Generate different 3D shapes for each sample
        for i in range(num_samples):
            # Add a sphere or cube to each sample
            shape_type = np.random.choice(['sphere', 'cube', 'cylinder'])
            center = void_dim // 2
            
            if shape_type == 'sphere':
                # Create a sphere
                radius = np.random.randint(void_dim // 6, void_dim // 3)
                for x in range(void_dim):
                    for y in range(void_dim):
                        for z in range(void_dim):
                            distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                            if distance < radius:
                                volumes[i, x, y, z] = 1
                                # Base material
                                materials[i, x, y, z] = 1
            
            elif shape_type == 'cube':
                # Create a cube
                size = np.random.randint(void_dim // 6, void_dim // 3)
                min_coord = center - size
                max_coord = center + size
                volumes[i, min_coord:max_coord, min_coord:max_coord, min_coord:max_coord] = 1
                materials[i, min_coord:max_coord, min_coord:max_coord, min_coord:max_coord] = 2
            
            else:  # cylinder
                # Create a cylinder
                radius = np.random.randint(void_dim // 6, void_dim // 3)
                height = np.random.randint(void_dim // 4, void_dim // 2)
                min_z = center - height // 2
                max_z = center + height // 2
                
                for x in range(void_dim):
                    for y in range(void_dim):
                        distance_2d = np.sqrt((x - center)**2 + (y - center)**2)
                        if distance_2d < radius:
                            volumes[i, x, y, min_z:max_z] = 1
                            materials[i, x, y, min_z:max_z] = 3
            
            # Add some smaller shapes with different materials
            num_extra_shapes = np.random.randint(1, 4)
            for j in range(num_extra_shapes):
                shape_center = np.random.randint(void_dim // 4, void_dim * 3 // 4, size=3)
                shape_size = np.random.randint(2, void_dim // 6)
                material_id = np.random.randint(1, 6)  # Random material ID
                
                # Randomly choose shape type
                extra_shape = np.random.choice(['sphere', 'cube'])
                
                if extra_shape == 'sphere':
                    for x in range(max(0, shape_center[0] - shape_size), min(void_dim, shape_center[0] + shape_size)):
                        for y in range(max(0, shape_center[1] - shape_size), min(void_dim, shape_center[1] + shape_size)):
                            for z in range(max(0, shape_center[2] - shape_size), min(void_dim, shape_center[2] + shape_size)):
                                distance = np.sqrt((x - shape_center[0])**2 + 
                                                  (y - shape_center[1])**2 + 
                                                  (z - shape_center[2])**2)
                                if distance < shape_size:
                                    volumes[i, x, y, z] = 1
                                    materials[i, x, y, z] = material_id
                
                else:  # cube
                    min_x = max(0, shape_center[0] - shape_size)
                    max_x = min(void_dim, shape_center[0] + shape_size)
                    min_y = max(0, shape_center[1] - shape_size)
                    max_y = min(void_dim, shape_center[1] + shape_size)
                    min_z = max(0, shape_center[2] - shape_size)
                    max_z = min(void_dim, shape_center[2] + shape_size)
                    
                    volumes[i, min_x:max_x, min_y:max_y, min_z:max_z] = 1
                    materials[i, min_x:max_x, min_y:max_y, min_z:max_z] = material_id
        
        # Save data to disk
        volume_path = os.path.join(data_dir, f"volume_data[{timestamp}]chunk[{chunk_idx}].npy")
        material_path = os.path.join(data_dir, f"material_data[{timestamp}]chunk[{chunk_idx}].npy")
        
        np.save(volume_path, volumes)
        np.save(material_path, materials)
        
        volume_paths.append(volume_path)
        material_paths.append(material_path)
        
        log_success(f"Generated and saved chunk {chunk_idx}")
        log_info(f"Volume data shape: {volumes.shape}")
    
    # Create a visualization of one sample
    if len(volume_paths) > 0:
        volumes = np.load(volume_paths[0])
        materials = np.load(material_paths[0])
        
        log_action("Creating sample visualization")
        
        fig = plt.figure(figsize=(15, 5))
        
        # Show cross-sections of first sample
        sample_idx = 0
        volume = volumes[sample_idx]
        material = materials[sample_idx]
        
        # Get middle slices
        mid_x = volume.shape[0] // 2
        mid_y = volume.shape[1] // 2
        mid_z = volume.shape[2] // 2
        
        # Volume visualization
        plt.subplot(2, 3, 1)
        plt.imshow(volume[mid_x, :, :], cmap='gray')
        plt.title("Volume: X middle slice")
        plt.colorbar()
        
        plt.subplot(2, 3, 2)
        plt.imshow(volume[:, mid_y, :], cmap='gray')
        plt.title("Volume: Y middle slice")
        plt.colorbar()
        
        plt.subplot(2, 3, 3)
        plt.imshow(volume[:, :, mid_z], cmap='gray')
        plt.title("Volume: Z middle slice")
        plt.colorbar()
        
        # Material visualization
        plt.subplot(2, 3, 4)
        plt.imshow(material[mid_x, :, :], cmap='viridis')
        plt.title("Material: X middle slice")
        plt.colorbar()
        
        plt.subplot(2, 3, 5)
        plt.imshow(material[:, mid_y, :], cmap='viridis')
        plt.title("Material: Y middle slice")
        plt.colorbar()
        
        plt.subplot(2, 3, 6)
        plt.imshow(material[:, :, mid_z], cmap='viridis')
        plt.title("Material: Z middle slice")
        plt.colorbar()
        
        plt.tight_layout()
        
        # Save the visualization
        viz_path = os.path.join(data_dir, "sample_visualization.png")
        plt.savefig(viz_path)
        plt.close(fig)
        
        log_success(f"Created and saved sample visualization to {viz_path}")
    
    # Create a DataFrame of file paths
    df = create_data_dataframe(data_dir)
    df_path = os.path.join(data_dir, "data_paths.csv")
    df.to_csv(df_path, index=False)
    
    log_success(f"Created dataset with {len(df)} samples")
    log_info(f"DataFrame saved to {df_path}")
    
    end_section()
    
    return df_path


def train_model(data_path, models_dir, train_viz_dir, checkpoints_dir):
    """Train a model on the generated data"""
    begin_section("Training Model")
    
    # Load data DataFrame
    df = pd.read_csv(data_path)
    
    # Set smaller dimensions for faster testing
    void_dim = 32  # Should match the data generation
    noise_dim = 64
    
    # Create data loader
    log_action("Creating data loader")
    data_loader = DataFrameDataLoader(
        df=df,
        batch_size=4,  # Small batch size for testing
        shuffle=True,
        processor=lambda v, m: v.astype(np.float32)  # Simple preprocessing
    )
    
    # Create models
    log_action("Creating models")
    generator = ModelFactory.create_generator(
        model_type="skip",  # Skip connections work well for 3D
        void_dim=void_dim,
        noise_dim=noise_dim,
        color_mode=1
    )
    
    discriminator = ModelFactory.create_discriminator(
        model_type="skip",
        void_dim=void_dim,
        noise_dim=noise_dim,
        color_mode=1
    )
    
    log_info(f"Generator: {generator.count_params():,} parameters")
    log_info(f"Discriminator: {discriminator.count_params():,} parameters")
    
    # Create trainer
    log_action("Creating trainer")
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=0.0002,
        beta1=0.5,
        beta2=0.999
    )
    
    # Train for a minimal number of epochs
    epochs = 3  # Just enough to see if training works
    log_action(f"Training for {epochs} epochs")
    
    metrics = trainer.train(
        data_loader=data_loader,
        epochs=epochs,
        checkpoint_dir=checkpoints_dir,
        snapshot_dir=train_viz_dir,
        snapshot_freq=1
    )
    
    # Plot training metrics
    log_action("Creating training metrics visualization")
    metrics_path = os.path.join(train_viz_dir, "training_metrics.png")
    trainer.plot_metrics(save_path=metrics_path)
    
    # Save models
    log_action("Saving trained models")
    generator_path = os.path.join(models_dir, "generator")
    discriminator_path = os.path.join(models_dir, "discriminator")
    
    generator.save(generator_path)
    discriminator.save(discriminator_path)
    
    log_success(f"Training completed - {epochs} epochs")
    log_info(f"Final generator loss: {metrics['gen_loss'][-1]:.4f}")
    log_info(f"Final discriminator loss: {metrics['disc_loss'][-1]:.4f}")
    log_info(f"Models saved to {models_dir}")
    
    end_section()
    
    return generator, discriminator


def generate_samples(generator, gen_viz_dir, num_samples=16, void_dim=32, noise_dim=64):
    """Generate samples from the trained model"""
    begin_section("Generating Samples")
    
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Generate samples
    log_action(f"Generating {num_samples} samples")
    
    noise = tf.random.normal([num_samples, noise_dim])
    generated_samples = generator(noise, training=False)
    
    log_info(f"Generated samples shape: {generated_samples.shape}")
    
    # Create a grid visualization
    log_action("Creating grid visualization")
    
    # Determine grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig = plt.figure(figsize=(grid_size * 3, grid_size * 3))
    
    for i in range(num_samples):
        if i >= grid_size * grid_size:
            break
            
        plt.subplot(grid_size, grid_size, i + 1)
        
        # Get middle slice on Z axis
        mid_z = void_dim // 2
        slice_data = generated_samples[i, :, :, mid_z, :3].numpy()
        
        # Normalize for better visualization
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max > slice_min:
            slice_data = (slice_data - slice_min) / (slice_max - slice_min)
        
        plt.imshow(slice_data)
        plt.title(f"Sample {i+1}")
        plt.axis('off')
    
    plt.suptitle("Generated 3D Samples (Z-middle slices)")
    plt.tight_layout()
    
    # Save the grid visualization
    grid_path = os.path.join(gen_viz_dir, "generated_samples_grid.png")
    plt.savefig(grid_path)
    plt.close(fig)
    
    log_success(f"Generated grid visualization saved to {grid_path}")
    
    # Create detailed visualizations of a few samples
    log_action("Creating detailed sample visualizations")
    
    for i in range(min(3, num_samples)):
        sample = generated_samples[i].numpy()
        
        # Create a figure with multiple slices
        fig = plt.figure(figsize=(15, 5))
        plt.suptitle(f"Generated Sample {i+1} - Detailed View")
        
        # Get slice indices
        slices = [void_dim // 4, void_dim // 2, void_dim * 3 // 4]
        
        # X slices
        for j, x in enumerate(slices):
            plt.subplot(3, 3, j + 1)
            plt.imshow(sample[x, :, :, :3])
            plt.title(f"X-Slice {x}")
            plt.axis('off')
        
        # Y slices
        for j, y in enumerate(slices):
            plt.subplot(3, 3, j + 4)
            plt.imshow(sample[:, y, :, :3])
            plt.title(f"Y-Slice {y}")
            plt.axis('off')
        
        # Z slices
        for j, z in enumerate(slices):
            plt.subplot(3, 3, j + 7)
            plt.imshow(sample[:, :, z, :3])
            plt.title(f"Z-Slice {z}")
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save the detailed visualization
        detail_path = os.path.join(gen_viz_dir, f"sample_{i+1}_detailed.png")
        plt.savefig(detail_path)
        plt.close(fig)
        
        log_success(f"Detailed visualization for sample {i+1} saved")
    
    end_section()
    
    return generated_samples


def validate_results(generated_samples, gen_viz_dir):
    """Validate the quality of the generated samples"""
    begin_section("Validating Results")
    
    # Basic validation tests
    num_samples = len(generated_samples)
    log_info(f"Validating {num_samples} generated samples")
    
    # Test 1: Check that samples have non-zero variance
    sample_means = []
    sample_stds = []
    
    for i in range(num_samples):
        sample = generated_samples[i].numpy()
        sample_means.append(np.mean(sample))
        sample_stds.append(np.std(sample))
    
    overall_mean = np.mean(sample_means)
    overall_std = np.mean(sample_stds)
    
    log_info(f"Mean value across samples: {overall_mean:.4f}")
    log_info(f"Mean standard deviation: {overall_std:.4f}")
    
    # Check if samples have sufficient variance
    has_variance = overall_std > 0.05
    log_info(f"Samples have sufficient variance: {has_variance}")
    
    # Test 2: Check that samples are different from each other
    sample_differences = []
    
    for i in range(num_samples):
        for j in range(i+1, num_samples):
            # Calculate mean absolute difference between samples
            diff = np.mean(np.abs(generated_samples[i] - generated_samples[j]))
            sample_differences.append(diff)
    
    avg_difference = np.mean(sample_differences)
    log_info(f"Average difference between samples: {avg_difference:.4f}")
    
    # Check if samples are sufficiently different
    has_diversity = avg_difference > 0.1
    log_info(f"Samples have sufficient diversity: {has_diversity}")
    
    # Create a visual report
    log_action("Creating validation report")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot sample means
    ax1.bar(range(num_samples), sample_means)
    ax1.axhline(y=overall_mean, color='r', linestyle='-', label=f"Mean: {overall_mean:.4f}")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Mean Value")
    ax1.set_title("Sample Mean Values")
    ax1.legend()
    
    # Plot sample standard deviations
    ax2.bar(range(num_samples), sample_stds)
    ax2.axhline(y=overall_std, color='r', linestyle='-', label=f"Mean: {overall_std:.4f}")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Standard Deviation")
    ax2.set_title("Sample Standard Deviations")
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the report
    report_path = os.path.join(gen_viz_dir, "validation_report.png")
    plt.savefig(report_path)
    plt.close(fig)
    
    log_success(f"Validation report saved to {report_path}")
    
    # Create a text report
    report_text = [
        "# DeepSculpt Validation Report",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Validation Metrics",
        f"- Number of samples: {num_samples}",
        f"- Mean value across samples: {overall_mean:.4f}",
        f"- Mean standard deviation: {overall_std:.4f}",
        f"- Average difference between samples: {avg_difference:.4f}",
        "",
        "## Validation Results",
        f"- Samples have sufficient variance: {'✓ PASS' if has_variance else '✗ FAIL'}",
        f"- Samples have sufficient diversity: {'✓ PASS' if has_diversity else '✗ FAIL'}",
        "",
        f"## Overall Result: {'✓ PASS' if (has_variance and has_diversity) else '✗ FAIL'}"
    ]
    
    # Save text report
    text_report_path = os.path.join(gen_viz_dir, "validation_report.md")
    with open(text_report_path, 'w') as f:
        f.write('\n'.join(report_text))
    
    log_success(f"Validation text report saved to {text_report_path}")
    log_success(f"Validation {'passed' if (has_variance and has_diversity) else 'failed'}")
    
    end_section()
    
    return has_variance and has_diversity


def run_e2e_test():
    """Run complete end-to-end test"""
    begin_section("DeepSculpt End-to-End Test")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Create directory structure
        results_dir, data_dir, models_dir, train_viz_dir, gen_viz_dir, checkpoints_dir = create_directories()
        
        # Generate test data
        data_path = generate_test_data(
            data_dir=data_dir,
            num_samples=8,  # Small number for faster testing
            void_dim=32,
            num_chunks=2
        )
        
        # Train model
        generator, discriminator = train_model(
            data_path=data_path,
            models_dir=models_dir,
            train_viz_dir=train_viz_dir,
            checkpoints_dir=checkpoints_dir
        )
        
        # Generate samples
        generated_samples = generate_samples(
            generator=generator,
            gen_viz_dir=gen_viz_dir,
            num_samples=16,
            void_dim=32,
            noise_dim=64
        )
        
        # Validate results
        validation_passed = validate_results(
            generated_samples=generated_samples,
            gen_viz_dir=gen_viz_dir
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Print final results
        log_success("End-to-end test completed")
        log_info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        log_info(f"Results saved in {os.path.abspath(results_dir)}")
        
        # Final status
        if validation_passed:
            log_success("END-TO-END TEST PASSED")
        else:
            log_error("END-TO-END TEST FAILED - See validation report for details")
        
    except Exception as e:
        log_error(f"Error during end-to-end test: {str(e)}")
        import traceback
        traceback.print_exc()
    
    end_section("DeepSculpt Testing")


if __name__ == "__main__":
    # Set matplotlib to non-interactive mode
    plt.ioff()
    
    # Close any existing figures
    plt.close("all")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Run end-to-end test
    run_e2e_test()
    
    # Ensure all plots are closed
    plt.close("all")