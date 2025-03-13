#!/usr/bin/env python3
"""
DeepSculpt - Main Orchestrator

This is the main entry point for the DeepSculpt project, integrating all components:
1. Model creation and management
2. Training pipeline 
3. Workflow automation
4. API server
5. Telegram bot interface

Usage:
    python main.py train --model-type=skip --epochs=100 --data-folder=./data
    python main.py serve-api --port=8000
    python main.py run-bot --token=YOUR_TELEGRAM_TOKEN
    python main.py workflow --mode=development
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

# Import DeepSculpt modules
# Use try/except to handle potential import errors gracefully
try:
    from models import ModelFactory
    from trainer import DeepSculptTrainer, DataFrameDataLoader, create_data_dataframe
    from workflow import Manager, build_flow
    import api
    import bot
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


def train_model(args):
    """
    Train a DeepSculpt model.
    
    Args:
        args: Command line arguments
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
    
    # Create data DataFrame
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
        print("Error: No data files found! Please check your data folder.")
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
    
    print(f"Training complete! Results saved to {results_dir}")
    return 0


def run_api_server(args):
    """
    Run the FastAPI server.
    
    Args:
        args: Command line arguments
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
    """
    print("Starting Telegram bot")
    
    # Set Telegram token
    if args.token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.token
    elif "TELEGRAM_BOT_TOKEN" not in os.environ:
        print("Error: Telegram bot token not provided. Use --token or set TELEGRAM_BOT_TOKEN environment variable")
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
        print(f"Error running bot: {e}")
        return 1
    
    return 0


def run_workflow(args):
    """
    Run the DeepSculpt workflow.
    
    Args:
        args: Command line arguments
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
        print(f"Error running workflow: {e}")
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


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepSculpt - Deep Learning for 3D Generation")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
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
    if args.command == "train":
        return train_model(args)
    elif args.command == "serve-api":
        return run_api_server(args)
    elif args.command == "run-bot":
        return run_telegram_bot(args)
    elif args.command == "workflow":
        return run_workflow(args)
    elif args.command == "all":
        return run_all_services(args)
    else:
        print("Error: No command specified")
        print("Use one of: train, serve-api, run-bot, workflow, all")
        return 1


if __name__ == "__main__":
    sys.exit(main())