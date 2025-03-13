"""
Comprehensive Test Script for DeepSculpt
This module tests all components of the DeepSculpt package, including:
- Shape generation
- Sculpture creation
- Batch collection
- Visualization
- Dataset preprocessing

Creates a dataset of 2 batches with 16 sculptures each, saves visualizations,
and stores all results (including numpy files) in a results folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

# Import DeepSculpt modules
from shapes import ShapeType
from sculptor import Sculptor
from collector import Collector
from curator import (
    Curator,
    OneHotEncoderDecoder,
    BinaryEncoderDecoder,
    RGBEncoderDecoder,
)
from visualization import Visualizer
from logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_info,
    log_error,
)


def create_directories():
    """Create the necessary directories for results"""
    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Subdirectories for organization
    plots_dir = os.path.join(results_dir, "plots")
    dataset_dir = os.path.join(results_dir, "dataset")
    samples_dir = os.path.join(results_dir, "samples")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)

    return results_dir, plots_dir, dataset_dir, samples_dir


def test_single_sculpture(results_dir, plots_dir):
    """Test creation and visualization of a single sculpture"""
    begin_section("Testing Single Sculpture Creation")

    # Create a sculptor with specific parameters
    sculptor = Sculptor(
        void_dim=16,  # Size of the 3D grid
        edges=(2, 0.2, 0.5),  # 2 edges with sizes between 20-50% of void_dim
        planes=(1, 0.3, 0.6),  # 1 plane with sizes between 30-60% of void_dim
        pipes=(1, 0.3, 0.6),  # 1 pipe with sizes between 30-60% of void_dim
        grid=(1, 4),  # Enable grid with step size 4
        verbose=True,  # Print detailed information
    )

    # Generate a sculpture
    log_action("Generating a single sculpture")
    volumes, materials = sculptor.generate_sculpture()

    # Save visualization
    log_action("Saving sculpture visualizations")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # 3D visualization - pass show=False to the underlying visualizer functions
    viz_path = os.path.join(plots_dir, f"single_sculpture_{timestamp}.png")

    # Modify the visualizer's functions to not show plots
    sculptor.visualizer.plot_sculpture = (
        lambda *args, **kwargs: sculptor.visualizer.__class__.plot_sculpture(
            sculptor.visualizer, *args, **{**kwargs, "show": False}
        )
    )

    fig = sculptor.visualize(
        title="Test Sculpture", hide_axis=True, save_path=viz_path, save_array=False
    )
    plt.close(fig)
    log_success(f"Saved 3D visualization to {viz_path}")

    # Sections visualization
    sections_path = os.path.join(plots_dir, f"sculpture_sections_{timestamp}.png")
    sections_fig = sculptor.visualizer.plot_sections(
        volumes=volumes,
        title="Sculpture Cross-Sections",
        show=False,
        save_path=sections_path,
        cmap="gray",
    )
    plt.close(sections_fig)
    log_success(f"Saved sections visualization to {sections_path}")

    # Save the raw data
    volumes_path = os.path.join(results_dir, f"single_volumes_{timestamp}.npy")
    materials_path = os.path.join(results_dir, f"single_materials_{timestamp}.npy")
    np.save(file=volumes_path, arr=volumes)
    np.save(file=materials_path, arr=materials)
    log_success(f"Saved raw data to {volumes_path} and {materials_path}")

    end_section("Single sculpture test completed successfully")
    return volumes, materials


def test_batch_generation(dataset_dir):
    """Test batch generation of sculptures"""
    begin_section("Testing Batch Generation")

    # Create a collector for dataset generation
    collector = Collector(
        void_dim=16,  # Size of the 3D grid
        edges=(2, 0.2, 0.5),  # 2 edges with sizes between 20-50% of void_dim
        planes=(1, 0.3, 0.6),  # 1 plane with sizes between 30-60% of void_dim
        pipes=(1, 0.3, 0.6),  # 1 pipe with sizes between 30-60% of void_dim
        grid=(1, 4),  # Enable grid with step size 4
        directory=dataset_dir,  # Output directory
        chunk_size=16,  # 16 sculptures per chunk
        n_chunks=2,  # Generate 2 chunks
        verbose=True,  # Print detailed information
    )

    # Patch the collector's visualizer to not show plots
    def patch_visualizer(visualizer):
        original_plot_sculpture = visualizer.plot_sculpture

        def patched_plot_sculpture(*args, **kwargs):
            return original_plot_sculpture(*args, **{**kwargs, "show": False})

        visualizer.plot_sculpture = patched_plot_sculpture
        return visualizer

    # Override the collector's _plot_samples method to not show plots
    original_plot_samples = collector._plot_samples

    def patched_plot_samples(*args, **kwargs):
        # Create a visualizer and patch it
        visualizer = Visualizer(figsize=15, dpi=100)
        collector.visualizer = patch_visualizer(visualizer)

        # Call the original method
        result = original_plot_samples(*args, **kwargs)

        # Close all plots
        plt.close("all")

        return result

    collector._plot_samples = patched_plot_samples

    # Generate the collection
    log_action("Generating a collection of sculptures")
    volumes, materials = collector.create_collection()

    log_success(f"Generated 2 chunks with 16 sculptures each")
    log_info(f"Total sculptures generated: {2 * 16}")

    # List the generated files
    log_info("Generated files:")
    for filename in os.listdir(dataset_dir):
        log_info(f"  - {filename}")

    end_section("Batch generation test completed successfully")
    return volumes, materials


def test_dataset_preprocessing(dataset_dir, plots_dir):
    """Test dataset preprocessing with different encoding methods"""
    begin_section("Testing Dataset Preprocessing")

    # Find the most recent volume and material files
    volume_files = [f for f in os.listdir(dataset_dir) if f.startswith("volume_data")]
    material_files = [
        f for f in os.listdir(dataset_dir) if f.startswith("material_data")
    ]

    if not volume_files or not material_files:
        log_error("No dataset files found. Please run test_batch_generation first.")
        end_section("Dataset preprocessing test skipped")
        return

    volume_file = sorted(volume_files)[-1]  # Get the most recent file
    material_file = sorted(material_files)[-1]  # Get the most recent file

    volumes_path = os.path.join(dataset_dir, volume_file)
    materials_path = os.path.join(dataset_dir, material_file)

    # Test preprocessing with different encoding methods
    encoding_methods = ["OHE", "BINARY", "RGB"]

    for method in encoding_methods:
        log_action(f"Testing {method} encoding")

        # Create a curator with the specified encoding method
        curator = Curator(processing_method=method, verbose=True)

        # Patch curator's visualizer to not show plots
        original_plot_sculpture = curator.visualizer.plot_sculpture

        def patched_plot_sculpture(*args, **kwargs):
            return original_plot_sculpture(*args, **{**kwargs, "show": False})

        curator.visualizer.plot_sculpture = patched_plot_sculpture

        # Preprocess the dataset
        dataset, encoder = curator.preprocess_collection(
            volumes_path=volumes_path,
            materials_path=materials_path,
            plot_samples=1,  # Plot 1 random sample
            buffer_size=100,
            batch_size=4,
            train_size=8,  # Use 8 samples to keep it quick
        )

        # Get original data for comparison
        volumes, materials = Collector.load_chunk(
            volumes_path=volumes_path, materials_path=materials_path
        )

        # Extract encoded data for visualization
        try:
            encoded_batch = next(iter(dataset))
            encoded_data = encoded_batch.numpy()

            # Visualize encoded sample
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            curator.visualize_encoded(
                encoded_data=encoded_data,
                encoder_decoder=encoder,
                sample_index=0,  # Visualize the first sample
                original_materials=materials,  # Compare with original
            )
        except Exception as e:
            log_error(f"Error visualizing encoded data: {str(e)}")

        # Close all figures to avoid displaying them during the test
        plt.close("all")

        log_success(f"Completed {method} encoding test")

    end_section("Dataset preprocessing test completed successfully")


def run_all_tests():
    """Run all tests"""
    begin_section("Running All DeepSculpt Tests")

    try:
        # Create directories
        results_dir, plots_dir, dataset_dir, samples_dir = create_directories()

        # Test single sculpture creation and visualization
        volumes, materials = test_single_sculpture(
            results_dir=results_dir, plots_dir=plots_dir
        )

        # Test batch generation
        batch_volumes, batch_materials = test_batch_generation(dataset_dir=dataset_dir)

        # Test dataset preprocessing
        test_dataset_preprocessing(dataset_dir=dataset_dir, plots_dir=plots_dir)

        log_success("All tests completed successfully")
        log_info(f"Results saved in {os.path.abspath(results_dir)}")

    except Exception as e:
        log_error(f"Error during testing: {str(e)}")
        import traceback

        traceback.print_exc()

    end_section("DeepSculpt Testing")


if __name__ == "__main__":
    # Set matplotlib to non-interactive mode to avoid displaying plots during testing
    plt.ioff()

    # Close any existing figures
    plt.close("all")

    # Record the start time
    start_time = time.time()

    # Run all tests
    run_all_tests()

    # Calculate and log the total execution time
    execution_time = time.time() - start_time
    print(
        f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
    )

    # Ensure all plots are closed
    plt.close("all")
