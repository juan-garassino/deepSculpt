"""
Dataset Generation Test Script for DeepSculpt

This module tests the data generation and preparation pipeline:
- Volume data generation
- Material data generation
- Dataset collection and chunking
- Data preprocessing for model consumption
- Visualization of dataset samples

Creates synthetic 3D data and prepares it for model training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd

# Optional: use logging from your logger module if available
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

# Import custom data loader from trainer.py
try:
    from trainer import create_data_dataframe, DataFrameDataLoader
except ImportError:
    log_error("Cannot import from trainer.py. Make sure it's in your path.")
    raise


def create_directories():
    """Create the necessary directories for test results"""
    begin_section("Creating Test Directories")
    
    # Create main results directory
    results_dir = "dataset_test_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create subdirectories
    data_dir = os.path.join(results_dir, "data")
    viz_dir = os.path.join(results_dir, "visualizations")
    processed_dir = os.path.join(results_dir, "processed")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    log_success(f"Created test directories in {results_dir}")
    end_section()
    
    return results_dir, data_dir, viz_dir, processed_dir


def generate_test_volumes(data_dir, num_samples=16, void_dim=32, num_chunks=2):
    """Generate synthetic volume data"""
    begin_section("Generating Volume Data")
    
    timestamp = datetime.now().strftime("%Y%m%d")
    volume_paths = []
    material_paths = []
    
    for chunk_idx in range(1, num_chunks+1):
        log_action(f"Generating chunk {chunk_idx} with {num_samples} samples")
        
        # Create volume data (binary voxels)
        volumes = np.random.randint(0, 2, size=(num_samples, void_dim, void_dim, void_dim))
        
        # Add some structure to make it more like real data:
        # 1. Create a sphere in the middle of each volume
        for i in range(num_samples):
            center = void_dim // 2
            radius = void_dim // 4
            
            # Create a simple sphere
            for x in range(void_dim):
                for y in range(void_dim):
                    for z in range(void_dim):
                        distance = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
                        if distance < radius:
                            volumes[i, x, y, z] = 1
            
            # Add some random shapes
            num_shapes = np.random.randint(1, 4)
            for _ in range(num_shapes):
                shape_center = np.random.randint(0, void_dim, size=3)
                shape_radius = np.random.randint(3, 8)
                
                # Add a smaller sphere somewhere in the volume
                x_min = max(0, shape_center[0] - shape_radius)
                x_max = min(void_dim, shape_center[0] + shape_radius)
                y_min = max(0, shape_center[1] - shape_radius)
                y_max = min(void_dim, shape_center[1] + shape_radius)
                z_min = max(0, shape_center[2] - shape_radius)
                z_max = min(void_dim, shape_center[2] + shape_radius)
                
                for x in range(x_min, x_max):
                    for y in range(y_min, y_max):
                        for z in range(z_min, z_max):
                            distance = np.sqrt((x - shape_center[0])**2 + 
                                              (y - shape_center[1])**2 + 
                                              (z - shape_center[2])**2)
                            if distance < shape_radius:
                                volumes[i, x, y, z] = 1
        
        # Create material data (categories 0-5 for different materials)
        materials = np.zeros_like(volumes)
        
        # Assign materials to volume voxels
        for i in range(num_samples):
            # Assign materials only where volume exists
            mask = volumes[i] > 0
            
            # Base material (1)
            materials[i][mask] = 1
            
            # Add some variety with other materials
            for material_id in range(2, 6):
                # Pick random center for this material
                material_center = np.random.randint(0, void_dim, size=3)
                material_radius = np.random.randint(3, 10)
                
                # Apply material in a sphere around the center
                for x in range(void_dim):
                    for y in range(void_dim):
                        for z in range(void_dim):
                            if not volumes[i, x, y, z]:
                                continue  # Skip if no volume here
                                
                            distance = np.sqrt((x - material_center[0])**2 + 
                                              (y - material_center[1])**2 + 
                                              (z - material_center[2])**2)
                            if distance < material_radius:
                                materials[i, x, y, z] = material_id
        
        # Save to disk
        volume_path = os.path.join(data_dir, f"volume_data[{timestamp}]chunk[{chunk_idx}].npy")
        material_path = os.path.join(data_dir, f"material_data[{timestamp}]chunk[{chunk_idx}].npy")
        
        np.save(volume_path, volumes)
        np.save(material_path, materials)
        
        volume_paths.append(volume_path)
        material_paths.append(material_path)
        
        log_success(f"Generated chunk {chunk_idx} and saved to disk")
        log_info(f"Volume data shape: {volumes.shape}")
        log_info(f"Material data shape: {materials.shape}")
    
    log_success(f"Generated {num_chunks} chunks with {num_samples} samples each")
    end_section()
    
    return volume_paths, material_paths


def visualize_data_samples(volume_paths, material_paths, viz_dir, num_samples=3):
    """Create visualizations of the generated data"""
    begin_section("Visualizing Data Samples")
    
    # Load a subset of data for visualization
    volume_path = volume_paths[0]
    material_path = material_paths[0]
    
    volumes = np.load(volume_path)
    materials = np.load(material_path)
    
    log_info(f"Loaded sample data from {os.path.basename(volume_path)}")
    log_info(f"Volume data shape: {volumes.shape}")
    
    # Visualize a few samples
    for sample_idx in range(min(num_samples, len(volumes))):
        log_action(f"Creating visualization for sample {sample_idx+1}")
        
        volume = volumes[sample_idx]
        material = materials[sample_idx]
        
        # Create a figure with multiple subplots for different slices and views
        fig = plt.figure(figsize=(15, 10))
        
        # Title for the entire figure
        plt.suptitle(f"Sample {sample_idx+1} Visualization", fontsize=16)
        
        # Volume data visualizations (top row)
        # X-slice
        ax1 = plt.subplot(2, 3, 1)
        mid_x = volume.shape[0] // 2
        plt.imshow(volume[mid_x, :, :], cmap='gray')
        plt.title(f"Volume: X-Slice (Middle)")
        plt.colorbar(label="Volume Value")
        
        # Y-slice
        ax2 = plt.subplot(2, 3, 2)
        mid_y = volume.shape[1] // 2
        plt.imshow(volume[:, mid_y, :], cmap='gray')
        plt.title(f"Volume: Y-Slice (Middle)")
        plt.colorbar(label="Volume Value")
        
        # Z-slice
        ax3 = plt.subplot(2, 3, 3)
        mid_z = volume.shape[2] // 2
        plt.imshow(volume[:, :, mid_z], cmap='gray')
        plt.title(f"Volume: Z-Slice (Middle)")
        plt.colorbar(label="Volume Value")
        
        # Material data visualizations (bottom row)
        # X-slice
        ax4 = plt.subplot(2, 3, 4)
        plt.imshow(material[mid_x, :, :], cmap='viridis')
        plt.title(f"Material: X-Slice (Middle)")
        plt.colorbar(label="Material ID")
        
        # Y-slice
        ax5 = plt.subplot(2, 3, 5)
        plt.imshow(material[:, mid_y, :], cmap='viridis')
        plt.title(f"Material: Y-Slice (Middle)")
        plt.colorbar(label="Material ID")
        
        # Z-slice
        ax6 = plt.subplot(2, 3, 6)
        plt.imshow(material[:, :, mid_z], cmap='viridis')
        plt.title(f"Material: Z-Slice (Middle)")
        plt.colorbar(label="Material ID")
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        
        # Save the visualization
        viz_path = os.path.join(viz_dir, f"sample_{sample_idx+1}_visualization.png")
        plt.savefig(viz_path)
        plt.close(fig)
        
        log_success(f"Saved visualization to {viz_path}")
    
    # Create a 3D visualization of one sample
    if volumes.shape[0] > 0:
        log_action("Creating 3D visualization")
        
        sample_volume = volumes[0]
        
        # Create a 3D plot of the volume data
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Get voxel coordinates
        x, y, z = np.where(sample_volume > 0)
        
        # Plot points
        ax.scatter(x, y, z, c='blue', marker='o', alpha=0.05, s=5)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Visualization of Sample Volume')
        
        # Set equal aspect ratio
        max_range = max([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()])
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range * 0.5, mid_x + max_range * 0.5)
        ax.set_ylim(mid_y - max_range * 0.5, mid_y + max_range * 0.5)
        ax.set_zlim(mid_z - max_range * 0.5, mid_z + max_range * 0.5)
        
        # Save the 3D visualization
        viz_3d_path = os.path.join(viz_dir, "sample_3d_visualization.png")
        plt.savefig(viz_3d_path)
        plt.close(fig)
        
        log_success(f"Saved 3D visualization to {viz_3d_path}")
    
    end_section()


def create_dataset_dataframe(volume_paths, material_paths, processed_dir):
    """Create a DataFrame of file paths for the dataset"""
    begin_section("Creating Dataset DataFrame")
    
    # Create a dictionary to build the DataFrame
    data_dict = {
        'chunk_idx': [],
        'volume_path': [],
        'material_path': []
    }
    
    # Extract chunk indices and build dict
    chunk_pattern = r"chunk\[(\d+)\]"
    import re
    
    for vol_path, mat_path in zip(volume_paths, material_paths):
        # Extract chunk index
        vol_match = re.search(chunk_pattern, vol_path)
        mat_match = re.search(chunk_pattern, mat_path)
        
        if vol_match and mat_match:
            chunk_idx = int(vol_match.group(1))
            data_dict['chunk_idx'].append(chunk_idx)
            data_dict['volume_path'].append(vol_path)
            data_dict['material_path'].append(mat_path)
    
    # Create DataFrame
    df = pd.DataFrame(data_dict)
    
    # Save to disk
    df_path = os.path.join(processed_dir, "dataset_paths.csv")
    df.to_csv(df_path, index=False)
    
    log_success(f"Created DataFrame with {len(df)} entries")
    log_info(f"DataFrame saved to {df_path}")
    
    # Also use create_data_dataframe to verify compatibility
    log_action("Verifying with create_data_dataframe function")
    data_dir = os.path.dirname(volume_paths[0])  # Get the directory
    
    try:
        alt_df = create_data_dataframe(data_dir)
        log_success(f"create_data_dataframe function produced {len(alt_df)} entries")
    except Exception as e:
        log_error(f"Failed to verify with create_data_dataframe: {str(e)}")
    
    end_section()
    
    return df, df_path


def test_dataframe_loader(df_path, processed_dir):
    """Test the DataFrameDataLoader with the dataset"""
    begin_section("Testing DataFrameDataLoader")
    
    try:
        # Load the DataFrame
        df = pd.read_csv(df_path)
        
        # Create a data loader
        batch_size = 4
        log_action(f"Creating DataFrameDataLoader with batch_size={batch_size}")
        
        data_loader = DataFrameDataLoader(
            df=df,
            batch_size=batch_size,
            shuffle=True,
            processor=lambda v, m: v.astype(np.float32)  # Simple preprocessing function
        )
        
        # Test TensorFlow dataset creation
        log_action("Creating TensorFlow dataset")
        tf_dataset = data_loader.create_tf_dataset()
        
        # Iterate through a few batches
        log_action("Iterating through dataset batches")
        
        batch_counts = 0
        sample_counts = 0
        
        for batch in data_loader.iterate_batches():
            batch_counts += 1
            sample_counts += len(batch)
            
            # Stop after a few batches
            if batch_counts >= 3:
                break
        
        log_success(f"Successfully iterated through {batch_counts} batches ({sample_counts} samples)")
        
        # Save a visualization of a random batch
        if batch_counts > 0:
            log_action("Creating batch visualization")
            
            # Create a sample batch
            batch = next(iter(data_loader.iterate_batches()))
            
            # Create visualization
            fig = plt.figure(figsize=(15, 10))
            
            # Display up to 4 samples from the batch
            for i in range(min(4, len(batch))):
                # Get middle slices
                sample = batch[i]
                
                mid_x = sample.shape[0] // 2
                mid_y = sample.shape[1] // 2
                mid_z = sample.shape[2] // 2
                
                # Plot the slices
                plt.subplot(4, 3, i*3 + 1)
                plt.imshow(sample[mid_x, :, :], cmap='gray')
                plt.title(f"Sample {i+1}: X-Slice")
                plt.axis('off')
                
                plt.subplot(4, 3, i*3 + 2)
                plt.imshow(sample[:, mid_y, :], cmap='gray')
                plt.title(f"Sample {i+1}: Y-Slice")
                plt.axis('off')
                
                plt.subplot(4, 3, i*3 + 3)
                plt.imshow(sample[:, :, mid_z], cmap='gray')
                plt.title(f"Sample {i+1}: Z-Slice")
                plt.axis('off')
            
            plt.tight_layout()
            
            # Save visualization
            batch_viz_path = os.path.join(processed_dir, "batch_visualization.png")
            plt.savefig(batch_viz_path)
            plt.close(fig)
            
            log_success(f"Saved batch visualization to {batch_viz_path}")
    
    except Exception as e:
        log_error(f"Error testing DataFrameDataLoader: {str(e)}")
        import traceback
        traceback.print_exc()
    
    end_section()


def run_dataset_tests():
    """Run all dataset generation and processing tests"""
    begin_section("DeepSculpt Dataset Generation Tests")
    
    # Record start time
    start_time = time.time()
    
    try:
        # Create test directories
        results_dir, data_dir, viz_dir, processed_dir = create_directories()
        
        # Generate test volumes
        volume_paths, material_paths = generate_test_volumes(
            data_dir=data_dir,
            num_samples=16,
            void_dim=32,
            num_chunks=2
        )
        
        # Visualize data samples
        visualize_data_samples(
            volume_paths=volume_paths,
            material_paths=material_paths,
            viz_dir=viz_dir,
            num_samples=3
        )
        
        # Create dataset DataFrame
        df, df_path = create_dataset_dataframe(
            volume_paths=volume_paths,
            material_paths=material_paths,
            processed_dir=processed_dir
        )
        
        # Test DataFrameDataLoader
        test_dataframe_loader(df_path=df_path, processed_dir=processed_dir)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        log_success("All dataset tests completed successfully")
        log_info(f"Total execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
        log_info(f"Results saved in {os.path.abspath(results_dir)}")
        
    except Exception as e:
        log_error(f"Error during dataset testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    end_section("DeepSculpt Dataset Testing")


if __name__ == "__main__":
    # Set matplotlib to non-interactive mode
    plt.ioff()
    
    # Close any existing figures
    plt.close("all")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run dataset tests
    run_dataset_tests()
    
    # Ensure all plots are closed
    plt.close("all")