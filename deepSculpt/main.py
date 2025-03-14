#!/usr/bin/env python3
"""
DeepSculpt - Main Orchestrator

This is the main entry point for the DeepSculpt project, integrating all components:
1. Data generation and collection
2. Dataset curation and preprocessing
3. Model training and evaluation
4. Workflow automation
5. API server and Telegram bot interface

Usage:
    # Data Generation
    python main.py create-collection --void-dim=32 --samples=1000

    # Data Curation
    python main.py curate --collection=latest --encoder=OHE

    # Model Training
    python main.py train --model-type=skip --epochs=100 --data-folder=./data

    # Complete Pipeline
    python main.py pipeline --create-data --curate --train --model-type=skip

    # API and Bot
    python main.py serve-api --port=8000
    python main.py run-bot --token=YOUR_TELEGRAM_TOKEN
    python main.py all --mode=production
"""

import os
import sys
import argparse
import subprocess
import threading
import time
import pandas as pd
import tensorflow as tf
from datetime import datetime
import signal
import uvicorn
import multiprocessing
import glob
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# Import DeepSculpt modules
try:
    # Core components
    from models import ModelFactory
    from trainer import DeepSculptTrainer, DataFrameDataLoader, create_data_dataframe
    from workflow import Manager, build_flow
    import api
    import bot

    # Data generation and curation components
    from collector import Collector
    from curator import Curator
    from visualization import Visualizer
except ImportError as e:
    print(f"Error importing DeepSculpt modules: {e}")
    print("Make sure all required modules are in the same directory or in PYTHONPATH")
    sys.exit(1)


# Configure environment variables if not already set
def setup_environment():
    """Set up environment variables with default values if not already set."""
    env_defaults = {
        "VOID_DIM": "64",
        "NOISE_DIM": "100",
        "COLOR": "1",
        "INSTANCE": "0",
        "MINIBATCH_SIZE": "32",
        "EPOCHS": "100",
        "MODEL_CHECKPOINT": "5",
        "PICTURE_SNAPSHOT": "1",
        "MLFLOW_TRACKING_URI": "http://localhost:5000",
        "MLFLOW_EXPERIMENT": "deepSculpt",
        "MLFLOW_MODEL_NAME": "deepSculpt_generator",
        "PREFECT_FLOW_NAME": "deepSculpt_workflow",
        "PREFECT_BACKEND": "development",
        "DEEPSCULPT_API_URL": "http://localhost:8000"
    }
    
    for key, default in env_defaults.items():
        if key not in os.environ:
            os.environ[key] = default
            print(f"Setting default environment variable: {key}={default}")


def create_collection(args):
    """
    Create a collection of 3D sculpture samples using the Collector.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print(f"Starting data generation with void_dim={args.void_dim}")
    
    # Create a collector with provided parameters
    collector = Collector(
        void_dim=args.void_dim,
        edges=(args.edges_count, args.edges_min_ratio, args.edges_max_ratio),
        planes=(args.planes_count, args.planes_min_ratio, args.planes_max_ratio),
        pipes=(args.pipes_count, args.pipes_min_ratio, args.pipes_max_ratio),
        grid=(int(args.grid_enabled), args.grid_step),
        step=args.step,
        base_dir=args.output_dir,
        total_samples=args.samples,
        verbose=args.verbose
    )
    
    try:
        # Generate the collection
        print(f"Generating {args.samples} samples...")
        start_time = time.time()
        sample_paths = collector.create_collection()
        elapsed = time.time() - start_time
        
        # Print summary
        print(f"\n✅ Collection created successfully!")
        print(f"Generated {len(sample_paths)} samples in {elapsed:.2f} seconds")
        print(f"Collection saved to: {collector.date_dir}")
        
        # Save collection path for downstream tasks if requested
        if args.save_info:
            info_path = os.path.join(args.output_dir, "last_collection_info.json")
            info = {
                "date_dir": collector.date_dir,
                "samples_dir": collector.samples_dir,
                "sample_count": len(sample_paths),
                "void_dim": args.void_dim,
                "timestamp": datetime.now().isoformat()
            }
            
            with open(info_path, "w") as f:
                json.dump(info, f, indent=2)
            
            print(f"Collection info saved to: {info_path}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error creating collection: {e}")
        return 1


def curate_collection(args):
    """
    Curate and preprocess a collection for model training.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print(f"Starting data curation with encoder={args.encoder}")
    
    # Determine collection path
    collection_dir = args.collection
    base_dir = args.data_dir
    
    if collection_dir == "latest":
        # Find the most recent collection
        collections = Collector.list_available_collections(base_dir)
        
        if not collections:
            print("❌ No collections found. Please generate data first.")
            return 1
        
        collection_dir = os.path.join(base_dir, collections[-1])
        print(f"Using latest collection: {collections[-1]}")
    else:
        # Check if the provided path exists
        if not os.path.exists(collection_dir):
            # Try appending to base_dir
            full_path = os.path.join(base_dir, collection_dir)
            if os.path.exists(full_path):
                collection_dir = full_path
            else:
                print(f"❌ Collection not found: {collection_dir}")
                return 1
    
    try:
        # Create a curator with specified encoder
        curator = Curator(
            processing_method=args.encoder,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
        
        # Process the collection
        print(f"Preprocessing collection...")
        start_time = time.time()
        
        result = curator.preprocess_collection(
            collection_dir=collection_dir,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            train_size=args.train_size,
            validation_split=args.validation_split,
            plot_samples=args.plot_samples
        )
        
        elapsed = time.time() - start_time
        
        # Print summary
        print(f"\n✅ Collection curated successfully!")
        print(f"Processed {result['train_size']} training and {result['val_size']} validation samples")
        print(f"Encoded shape: {result['encoded_shape']}")
        print(f"Processed data saved to: {result['output_dir']}")
        print(f"Time elapsed: {elapsed:.2f} seconds")
        
        # Save curation info for downstream tasks
        info_path = os.path.join(args.output_dir, "last_curation_info.json")
        info = {
            "output_dir": result["output_dir"],
            "train_size": result["train_size"],
            "val_size": result["val_size"],
            "encoded_shape": [int(x) for x in result["encoded_shape"]],
            "encoder": args.encoder,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        
        print(f"Curation info saved to: {info_path}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Error curating collection: {e}")
        return 1


def train_model(args):
    """
    Train a DeepSculpt model.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print(f"Starting training with model type: {args.model_type}")
    
    # Set environment variables for compatibility
    os.environ["VOID_DIM"] = str(args.void_dim)
    os.environ["NOISE_DIM"] = str(args.noise_dim)
    os.environ["COLOR"] = "1" if args.color else "0"
    
    # Create results directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    checkpoint_dir = os.path.join(results_dir, "checkpoints")
    snapshot_dir = os.path.join(results_dir, "snapshots")
    
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)
    
    print(f"Results will be saved to {results_dir}")
    
    # Check for curated data if encoder is specified
    if args.encoder:
        # Try to find the curated data directory
        last_curation_path = os.path.join(args.output_dir, "last_curation_info.json")
        if os.path.exists(last_curation_path):
            with open(last_curation_path, "r") as f:
                curation_info = json.load(f)
            
            print(f"Found curated data from {curation_info['timestamp']}")
            print(f"Encoder: {curation_info['encoder']}")
            print(f"Training samples: {curation_info['train_size']}")
            
            # TODO: Load and use the curated dataset
            # This requires modification of the training process
            # to use the encoded datasets rather than raw data
            
            # For now, we'll proceed with the existing flow
            print(f"⚠️ Training with curated data not yet implemented")
            print(f"Proceeding with standard training...")
    
    # Load or create data DataFrame
    print(f"Processing data from folder: {args.data_folder}")
    
    if args.data_file and os.path.exists(args.data_file):
        # Load existing DataFrame if provided
        print(f"Loading data paths from: {args.data_file}")
        data_df = pd.read_csv(args.data_file)
    else:
        # Create new DataFrame from data folder
        print(f"Scanning data folder: {args.data_folder}")
        data_df = create_data_dataframe(args.data_folder)
    
    if data_df.empty:
        print("❌ Error: No data files found! Please check your data folder.")
        return 1
    
    print(f"Found {len(data_df)} data pairs")
    
    # Save DataFrame for future use
    data_file_path = os.path.join(results_dir, "data_paths.csv")
    data_df.to_csv(data_file_path, index=False)
    print(f"Saved data paths to: {data_file_path}")
    
    # Create models
    print(f"Creating {args.model_type} models")
    generator = ModelFactory.create_generator(
        model_type=args.model_type, 
        void_dim=args.void_dim, 
        noise_dim=args.noise_dim,
        color_mode=1 if args.color else 0
    )
    
    discriminator = ModelFactory.create_discriminator(
        model_type=args.model_type,
        void_dim=args.void_dim,
        noise_dim=args.noise_dim,
        color_mode=1 if args.color else 0
    )
    
    # Add regularization if requested
    if args.dropout > 0:
        from models import add_regularization
        generator = add_regularization(generator, dropout_rate=args.dropout)
        print(f"Added dropout regularization with rate: {args.dropout}")
    
    # Print model summaries
    if args.verbose:
        print("\nGenerator Summary:")
        generator.summary()
        
        print("\nDiscriminator Summary:")
        discriminator.summary()
    
    # Create data loader
    data_loader = DataFrameDataLoader(
        df=data_df,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Create trainer
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2
    )
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs")
    metrics = trainer.train(
        data_loader=data_loader,
        epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        snapshot_dir=snapshot_dir,
        snapshot_freq=args.snapshot_freq
    )
    
    # Save the final models
    generator.save(os.path.join(results_dir, "generator_final"))
    discriminator.save(os.path.join(results_dir, "discriminator_final"))
    
    # Plot and save metrics
    metrics_path = os.path.join(results_dir, "training_metrics.png")
    trainer.plot_metrics(save_path=metrics_path)
    
    # Save to MLflow if requested
    if args.mlflow:
        print("Saving model to MLflow")
        
        # Prepare parameters and metrics
        params = {
            "model_type": args.model_type,
            "void_dim": args.void_dim,
            "noise_dim": args.noise_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "beta1": args.beta1,
            "beta2": args.beta2,
            "color_mode": 1 if args.color else 0,
            "dropout": args.dropout
        }
        
        final_metrics = {}
        if metrics.get("gen_loss") and metrics.get("disc_loss"):
            final_metrics = {
                "final_gen_loss": float(metrics["gen_loss"][-1]),
                "final_disc_loss": float(metrics["disc_loss"][-1]),
                "training_time": sum(metrics.get("epoch_times", [0]))
            }
        
        # Save to MLflow
        Manager.save_mlflow_model(
            metrics=final_metrics,
            params=params,
            model=generator
        )
    
    print(f"✅ Training complete! Results saved to {results_dir}")
    return 0


def run_api_server(args):
    """
    Run the FastAPI server.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print(f"Starting API server on port {args.port}")
    
    # Run with uvicorn
    uvicorn.run(
        "api:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )
    
    return 0


def run_telegram_bot(args):
    """
    Run the Telegram bot.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print("Starting Telegram bot")
    
    # Set Telegram token
    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    elif "TELEGRAM_BOT_TOKEN" not in os.environ:
        print("❌ Error: Telegram bot token not provided. Use --token or set TELEGRAM_BOT_TOKEN environment variable")
        return 1
    
    # Set API URL if provided
    if args.api_url:
        os.environ["DEEPSCULPT_API_URL"] = args.api_url
    
    # Run the bot
    try:
        bot.main()
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"❌ Error running bot: {e}")
        return 1
    
    return 0


def run_workflow(args):
    """
    Run the DeepSculpt workflow.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print(f"Running workflow in {args.mode} mode")
    
    # Set mode
    os.environ["PREFECT_BACKEND"] = args.mode
    
    # Run the workflow
    from workflow import main as workflow_main
    
    # Create sys.argv for workflow
    sys_argv = ["workflow.py", f"--mode={args.mode}"]
    if args.data_folder:
        sys_argv.append(f"--data-folder={args.data_folder}")
    if args.model_type:
        sys_argv.append(f"--model-type={args.model_type}")
    if args.epochs:
        sys_argv.append(f"--epochs={args.epochs}")
    if args.schedule:
        sys_argv.append("--schedule")
    
    # Save original sys.argv
    original_argv = sys.argv
    
    try:
        # Replace sys.argv
        sys.argv = sys_argv
        
        # Run workflow
        workflow_main()
    except Exception as e:
        print(f"❌ Error running workflow: {e}")
        return 1
    finally:
        # Restore original sys.argv
        sys.argv = original_argv
    
    return 0


def run_all_services(args):
    """
    Run all services (API server, Telegram bot, and optionally workflow).
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print("Starting all DeepSculpt services")
    
    # Define process functions
    def run_api():
        api_args = argparse.Namespace(
            host=args.host,
            port=args.port,
            reload=False
        )
        run_api_server(api_args)
    
    def run_bot():
        bot_args = argparse.Namespace(
            token=args.token,
            api_url=f"http://{args.host}:{args.port}"
        )
        run_telegram_bot(bot_args)
    
    def run_work():
        workflow_args = argparse.Namespace(
            mode=args.mode,
            data_folder=args.data_folder,
            model_type=args.model_type,
            epochs=args.epochs,
            schedule=args.schedule
        )
        run_workflow(workflow_args)
    
    # Create and start processes
    processes = []
    
    # API process
    api_process = multiprocessing.Process(target=run_api)
    api_process.start()
    processes.append(api_process)
    print(f"API server started on {args.host}:{args.port}")
    
    # Give API time to start
    time.sleep(3)
    
    # Bot process
    bot_process = multiprocessing.Process(target=run_bot)
    bot_process.start()
    processes.append(bot_process)
    print("Telegram bot started")
    
    # Workflow process (if requested)
    if args.workflow:
        workflow_process = multiprocessing.Process(target=run_work)
        workflow_process.start()
        processes.append(workflow_process)
        print(f"Workflow started in {args.mode} mode")
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("Shutting down all services...")
        for process in processes:
            if process.is_alive():
                process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Wait for processes
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("Received keyboard interrupt, shutting down...")
        for process in processes:
            if process.is_alive():
                process.terminate()
    
    return 0


def run_pipeline(args):
    """
    Run a complete pipeline from data generation to model training.
    
    Args:
        args: Command line arguments
    
    Returns:
        Integer status code (0 for success)
    """
    print("Starting complete DeepSculpt pipeline")
    start_time = time.time()
    
    try:
        # Step 1: Create collection if requested
        if args.create_data:
            print("\n🔹 STEP 1: Creating data collection")
            create_args = argparse.Namespace(
                void_dim=args.void_dim,
                edges_count=args.edges_count,
                edges_min_ratio=args.edges_min_ratio,
                edges_max_ratio=args.edges_max_ratio,
                planes_count=args.planes_count,
                planes_min_ratio=args.planes_min_ratio,
                planes_max_ratio=args.planes_max_ratio,
                pipes_count=args.pipes_count,
                pipes_min_ratio=args.pipes_min_ratio,
                pipes_max_ratio=args.pipes_max_ratio,
                grid_enabled=args.grid_enabled,
                grid_step=args.grid_step,
                step=args.step,
                samples=args.samples,
                output_dir=args.data_dir,
                save_info=True,
                verbose=args.verbose
            )
            
            result = create_collection(create_args)
            if result != 0:
                print("❌ Pipeline failed at the data creation step")
                return result
        
        # Step 2: Curate collection if requested
        if args.curate:
            print("\n🔹 STEP 2: Curating data collection")
            curate_args = argparse.Namespace(
                collection="latest",
                encoder=args.encoder,
                batch_size=args.batch_size,
                buffer_size=1000,
                train_size=None,
                validation_split=0.2,
                plot_samples=5,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                verbose=args.verbose
            )
            
            result = curate_collection(curate_args)
            if result != 0:
                print("❌ Pipeline failed at the data curation step")
                return result
        
        # Step 3: Train model if requested
        if args.train:
            print("\n🔹 STEP 3: Training model")
            train_args = argparse.Namespace(
                model_type=args.model_type,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                beta1=0.5,
                beta2=0.999,
                void_dim=args.void_dim,
                noise_dim=100,
                color=True,
                snapshot_freq=args.snapshot_freq,
                data_folder=args.data_dir,
                data_file=None,
                output_dir=args.output_dir,
                dropout=args.dropout,
                mlflow=args.mlflow,
                verbose=args.verbose,
                encoder=args.encoder
            )
            
            result = train_model(train_args)
            if result != 0:
                print("❌ Pipeline failed at the training step")
                return result
        
        # Step 4: Start services if requested
        if args.serve:
            print("\n🔹 STEP 4: Starting services")
            services_args = argparse.Namespace(
                host=args.host,
                port=args.port,
                token=args.token,
                mode=args.mode,
                workflow=args.workflow,
                data_folder=args.data_dir,
                model_type=args.model_type,
                epochs=args.epochs,
                schedule=args.schedule
            )
            
            # Run services (this will block until interrupted)
            return run_all_services(services_args)
        
        # If we get here, all requested steps completed successfully
        elapsed = time.time() - start_time
        print(f"\n✅ Pipeline completed successfully in {elapsed:.2f} seconds!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return 1


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSculpt - Deep Learning for 3D Generation")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create Collection command
    collection_parser = subparsers.add_parser("create-collection", help="Generate a collection of 3D samples")
    collection_parser.add_argument("--void-dim", type=int, default=32,
                                 help="Size of the 3D grid in each dimension")
    collection_parser.add_argument("--edges-count", type=int, default=2,
                                 help="Number of edge elements per sculpture")
    collection_parser.add_argument("--edges-min-ratio", type=float, default=0.2,
                                 help="Minimum size ratio for edge elements")
    collection_parser.add_argument("--edges-max-ratio", type=float, default=0.5,
                                 help="Maximum size ratio for edge elements")
    collection_parser.add_argument("--planes-count", type=int, default=1,
                                 help="Number of plane elements per sculpture")
    collection_parser.add_argument("--planes-min-ratio", type=float, default=0.3,
                                 help="Minimum size ratio for plane elements")
    collection_parser.add_argument("--planes-max-ratio", type=float, default=0.6,
                                 help="Maximum size ratio for plane elements")
    collection_parser.add_argument("--pipes-count", type=int, default=2,
                                 help="Number of pipe elements per sculpture")
    collection_parser.add_argument("--pipes-min-ratio", type=float, default=0.3,
                                 help="Minimum size ratio for pipe elements")
    collection_parser.add_argument("--pipes-max-ratio", type=float, default=0.6,
                                 help="Maximum size ratio for pipe elements")
    collection_parser.add_argument("--grid-enabled", type=int, default=1,
                                 help="Whether to enable grid (0=disabled, 1=enabled)")
    collection_parser.add_argument("--grid-step", type=int, default=4,
                                 help="Step size for grid generation")
    collection_parser.add_argument("--step", type=int, default=None,
                                 help="Step size for shape dimensions (default: void_dim/6)")
    collection_parser.add_argument("--samples", type=int, default=100,
                                 help="Number of samples to generate")
    collection_parser.add_argument("--output-dir", type=str, default="./data",
                                 help="Directory to save generated data")
    collection_parser.add_argument("--save-info", action="store_true",
                                 help="Save collection info for downstream tasks")
    collection_parser.add_argument("--verbose", action="store_true",
                                 help="Print verbose output")
    
    # Curate Collection command
    curate_parser = subparsers.add_parser("curate", help="Preprocess a collection for training")
    curate_parser.add_argument("--collection", type=str, default="latest",
                             help="Collection to curate (path or 'latest')")
    curate_parser.add_argument("--encoder", type=str, default="OHE",
                             choices=["OHE", "BINARY", "RGB"],
                             help="Encoding method to use")
    curate_parser.add_argument("--batch-size", type=int, default=32,
                             help="Batch size for training")
    curate_parser.add_argument("--buffer-size", type=int, default=1000,
                             help="Buffer size for dataset shuffling")
    curate_parser.add_argument("--train-size", type=int, default=None,
                             help="Number of samples to use for training (None for all)")
    curate_parser.add_argument("--validation-split", type=float, default=0.2,
                             help="Fraction of data to use for validation")
    curate_parser.add_argument("--plot-samples", type=int, default=5,
                             help="Number of random samples to plot")
    curate_parser.add_argument("--data-dir", type=str, default="./data",
                             help="Base directory for collections")
    curate_parser.add_argument("--output-dir", type=str, default="./processed_data",
                             help="Directory to save processed data")
    curate_parser.add_argument("--verbose", action="store_true",
                             help="Print verbose output")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model-type", type=str, default="skip",
                            choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                            help="Type of model to train")
    train_parser.add_argument("--epochs", type=int, default=100,
                            help="Number of epochs to train for")
    train_parser.add_argument("--batch-size", type=int, default=32,
                            help="Batch size for training")
    train_parser.add_argument("--learning-rate", type=float, default=0.0002,
                            help="Learning rate for optimizers")
    train_parser.add_argument("--beta1", type=float, default=0.5,
                            help="Beta1 parameter for Adam optimizer")
    train_parser.add_argument("--beta2", type=float, default=0.999,
                            help="Beta2 parameter for Adam optimizer")
    train_parser.add_argument("--void-dim", type=int, default=64,
                            help="Dimension of void space")
    train_parser.add_argument("--noise-dim", type=int, default=100,
                            help="Dimension of noise vector")
    train_parser.add_argument("--color", action="store_true",
                            help="Use color mode (default: True)")
    train_parser.add_argument("--snapshot-freq", type=int, default=5,
                            help="Frequency of saving snapshots (epochs)")
    train_parser.add_argument("--data-folder", type=str, default="./data",
                            help="Path to data folder")
    train_parser.add_argument("--data-file", type=str, default=None,
                            help="Path to CSV file with data paths")
    train_parser.add_argument("--output-dir", type=str, default="./results",
                            help="Directory for output files")
    train_parser.add_argument("--dropout", type=float, default=0.0,
                            help="Dropout rate for regularization")
    train_parser.add_argument("--mlflow", action="store_true",
                            help="Save model to MLflow")
    train_parser.add_argument("--encoder", type=str, default=None,
                            choices=[None, "OHE", "BINARY", "RGB"],
                            help="Use curated data with this encoder")
    train_parser.add_argument("--verbose", action="store_true",
                            help="Print verbose output")
    
    # API server command
    api_parser = subparsers.add_parser("serve-api", help="Run the API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0",
                          help="Host to bind the server to")
    api_parser.add_argument("--port", type=int, default=8000,
                          help="Port to bind the server to")
    api_parser.add_argument("--reload", action="store_true",
                          help="Enable auto-reload on code changes")
    
    # Telegram bot command
    bot_parser = subparsers.add_parser("run-bot", help="Run the Telegram bot")
    bot_parser.add_argument("--token", type=str, default=None,
                          help="Telegram bot token")
    bot_parser.add_argument("--api-url", type=str, default=None,
                          help="URL of the DeepSculpt API server")
    
    # Workflow command
    workflow_parser = subparsers.add_parser("workflow", help="Run the workflow")
    workflow_parser.add_argument("--mode", type=str, choices=["development", "production"],
                               default="development", help="Execution mode")
    workflow_parser.add_argument("--data-folder", type=str, default="./data",
                               help="Path to data folder")
    workflow_parser.add_argument("--model-type", type=str, default="skip",
                               choices=["simple", "complex", "skip", "monochrome"],
                               help="Type of model to train")
    workflow_parser.add_argument("--epochs", type=int, default=10,
                               help="Number of epochs for training")
    workflow_parser.add_argument("--schedule", action="store_true",
                               help="Run with schedule")
    
    # Pipeline command (combines multiple steps)
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--create-data", action="store_true",
                               help="Create data collection")
    pipeline_parser.add_argument("--curate", action="store_true",
                               help="Curate data collection")
    pipeline_parser.add_argument("--train", action="store_true",
                               help="Train a model")
    pipeline_parser.add_argument("--serve", action="store_true",
                               help="Start API and bot services")
    
    # Data generation parameters
    pipeline_parser.add_argument("--void-dim", type=int, default=32,
                               help="Size of the 3D grid in each dimension")
    pipeline_parser.add_argument("--edges-count", type=int, default=2,
                               help="Number of edge elements per sculpture")
    pipeline_parser.add_argument("--edges-min-ratio", type=float, default=0.2,
                               help="Minimum size ratio for edge elements")
    pipeline_parser.add_argument("--edges-max-ratio", type=float, default=0.5,
                               help="Maximum size ratio for edge elements")
    pipeline_parser.add_argument("--planes-count", type=int, default=1,
                               help="Number of plane elements per sculpture")
    pipeline_parser.add_argument("--planes-min-ratio", type=float, default=0.3,
                               help="Minimum size ratio for plane elements")
    pipeline_parser.add_argument("--planes-max-ratio", type=float, default=0.6,
                               help="Maximum size ratio for plane elements")
    pipeline_parser.add_argument("--pipes-count", type=int, default=2,
                               help="Number of pipe elements per sculpture")
    pipeline_parser.add_argument("--pipes-min-ratio", type=float, default=0.3,
                               help="Minimum size ratio for pipe elements")
    pipeline_parser.add_argument("--pipes-max-ratio", type=float, default=0.6,
                               help="Maximum size ratio for pipe elements")
    pipeline_parser.add_argument("--grid-enabled", type=int, default=1,
                               help="Whether to enable grid (0=disabled, 1=enabled)")
    pipeline_parser.add_argument("--grid-step", type=int, default=4,
                               help="Step size for grid generation")
    pipeline_parser.add_argument("--step", type=int, default=None,
                               help="Step size for shape dimensions (default: void_dim/6)")
    pipeline_parser.add_argument("--samples", type=int, default=100,
                               help="Number of samples to generate")
    
    # Curation parameters
    pipeline_parser.add_argument("--encoder", type=str, default="OHE",
                               choices=["OHE", "BINARY", "RGB"],
                               help="Encoding method to use")
    
    # Training parameters
    pipeline_parser.add_argument("--model-type", type=str, default="skip",
                               choices=["simple", "complex", "skip", "monochrome", "autoencoder"],
                               help="Type of model to train")
    pipeline_parser.add_argument("--epochs", type=int, default=100,
                               help="Number of epochs to train for")
    pipeline_parser.add_argument("--batch-size", type=int, default=32,
                               help="Batch size for training")
    pipeline_parser.add_argument("--learning-rate", type=float, default=0.0002,
                               help="Learning rate for optimizers")
    pipeline_parser.add_argument("--dropout", type=float, default=0.0,
                               help="Dropout rate for regularization")
    pipeline_parser.add_argument("--snapshot-freq", type=int, default=5,
                               help="Frequency of saving snapshots (epochs)")
    pipeline_parser.add_argument("--mlflow", action="store_true",
                               help="Save model to MLflow")
    
    # Service parameters
    pipeline_parser.add_argument("--host", type=str, default="0.0.0.0",
                               help="Host for API server")
    pipeline_parser.add_argument("--port", type=int, default=8000,
                               help="Port for API server")
    pipeline_parser.add_argument("--token", type=str, default=None,
                               help="Telegram bot token")
    pipeline_parser.add_argument("--mode", type=str, choices=["development", "production"],
                               default="development", help="Execution mode for workflow")
    pipeline_parser.add_argument("--workflow", action="store_true",
                               help="Run workflow in addition to API and bot")
    pipeline_parser.add_argument("--schedule", action="store_true",
                               help="Run workflow with schedule")
    
    # Directory parameters
    pipeline_parser.add_argument("--data-dir", type=str, default="./data",
                               help="Directory for raw data")
    pipeline_parser.add_argument("--output-dir", type=str, default="./results",
                               help="Directory for outputs")
    
    # Other parameters
    pipeline_parser.add_argument("--verbose", action="store_true",
                               help="Print verbose output")
    
    # Run all services command
    all_parser = subparsers.add_parser("all", help="Run all services")
    all_parser.add_argument("--host", type=str, default="0.0.0.0",
                          help="Host for API server")
    all_parser.add_argument("--port", type=int, default=8000,
                          help="Port for API server")
    all_parser.add_argument("--token", type=str, default=None,
                          help="Telegram bot token")
    all_parser.add_argument("--mode", type=str, choices=["development", "production"],
                          default="development", help="Execution mode for workflow")
    all_parser.add_argument("--workflow", action="store_true",
                          help="Run workflow in addition to API and bot")
    all_parser.add_argument("--data-folder", type=str, default="./data",
                          help="Path to data folder for workflow")
    all_parser.add_argument("--model-type", type=str, default="skip",
                          choices=["simple", "complex", "skip", "monochrome"],
                          help="Type of model for workflow")
    all_parser.add_argument("--epochs", type=int, default=10,
                          help="Number of epochs for workflow training")
    all_parser.add_argument("--schedule", action="store_true",
                          help="Run workflow with schedule")
    
    return parser.parse_args()


def main():
    """Main entry point for DeepSculpt."""
    # Setup environment
    setup_environment()
    
    # Parse arguments
    args = parse_arguments()
    
    # Execute requested command
    if args.command == "create-collection":
        return create_collection(args)
    elif args.command == "curate":
        return curate_collection(args)
    elif args.command == "train":
        return train_model(args)
    elif args.command == "serve-api":
        return run_api_server(args)
    elif args.command == "run-bot":
        return run_telegram_bot(args)
    elif args.command == "workflow":
        return run_workflow(args)
    elif args.command == "pipeline":
        return run_pipeline(args)
    elif args.command == "all":
        return run_all_services(args)
    else:
        print("Error: No command specified")
        print("Use one of: create-collection, curate, train, serve-api, run-bot, workflow, pipeline, all")
        return 1


if __name__ == "__main__":
    sys.exit(main())