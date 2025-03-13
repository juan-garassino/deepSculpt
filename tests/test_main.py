"""
Tests for the DeepSculpt main orchestrator.
"""

import pytest
import os
import sys
import tempfile
import argparse
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import main module
import main


@pytest.fixture
def setup_env():
    """Setup environment variables for tests."""
    # Save original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    os.environ["VOID_DIM"] = "32"
    os.environ["NOISE_DIM"] = "50"
    os.environ["COLOR"] = "1"
    os.environ["INSTANCE"] = "0"
    os.environ["MINIBATCH_SIZE"] = "32"
    os.environ["EPOCHS"] = "10"
    os.environ["MODEL_CHECKPOINT"] = "5"
    os.environ["PICTURE_SNAPSHOT"] = "1"
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_EXPERIMENT"] = "test_experiment"
    os.environ["MLFLOW_MODEL_NAME"] = "test_model"
    os.environ["PREFECT_FLOW_NAME"] = "test_workflow"
    os.environ["PREFECT_BACKEND"] = "development"
    os.environ["TELEGRAM_BOT_TOKEN"] = "test_token"
    os.environ["DEEPSCULPT_API_URL"] = "http://localhost:8000"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


def test_setup_environment():
    """Test the setup_environment function."""
    # Clear environment
    original_env = dict(os.environ)
    os.environ.clear()
    
    # Call function
    main.setup_environment()
    
    # Check that environment variables were set
    assert "VOID_DIM" in os.environ
    assert "NOISE_DIM" in os.environ
    assert "COLOR" in os.environ
    assert "INSTANCE" in os.environ
    assert "MINIBATCH_SIZE" in os.environ
    assert "EPOCHS" in os.environ
    assert "MODEL_CHECKPOINT" in os.environ
    assert "PICTURE_SNAPSHOT" in os.environ
    assert "MLFLOW_TRACKING_URI" in os.environ
    assert "MLFLOW_EXPERIMENT" in os.environ
    assert "MLFLOW_MODEL_NAME" in os.environ
    assert "PREFECT_FLOW_NAME" in os.environ
    assert "DEEPSCULPT_API_URL" in os.environ
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@patch("main.ModelFactory.create_generator")
@patch("main.ModelFactory.create_discriminator")
@patch("main.create_data_dataframe")
@patch("main.DataFrameDataLoader")
@patch("main.DeepSculptTrainer")
@patch("os.makedirs")
def test_train_model(mock_makedirs, mock_trainer, mock_loader, mock_create_df, 
                   mock_create_disc, mock_create_gen, setup_env):
    """Test the train_model function."""
    # Setup mocks
    mock_generator = MagicMock()
    mock_discriminator = MagicMock()
    mock_create_gen.return_value = mock_generator
    mock_create_disc.return_value = mock_discriminator
    
    mock_df = MagicMock()
    mock_create_df.return_value = mock_df
    
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = {
        "gen_loss": [1.0, 0.9, 0.8],
        "disc_loss": [0.5, 0.4, 0.3],
        "epoch_times": [1.0, 1.0, 1.0]
    }
    mock_trainer.return_value = mock_trainer_instance
    
    # Create mock args
    args = MagicMock()
    args.model_type = "skip"
    args.void_dim = 32
    args.noise_dim = 50
    args.color = True
    args.output_dir = tempfile.gettempdir()
    args.data_folder = tempfile.gettempdir()
    args.data_file = None
    args.batch_size = 32
    args.epochs = 3
    args.learning_rate = 0.0002
    args.beta1 = 0.5
    args.beta2 = 0.999
    args.dropout = 0.0
    args.mlflow = False
    args.verbose = False
    args.snapshot_freq = 1
    
    # Call function
    with patch("pandas.DataFrame.to_csv"):
        result = main.train_model(args)
    
    # Check return value
    assert result == 0
    
    # Check that models were created
    mock_create_gen.assert_called_once_with(
        model_type="skip", 
        void_dim=32, 
        noise_dim=50,
        color_mode=1
    )
    mock_create_disc.assert_called_once_with(
        model_type="skip",
        void_dim=32,
        noise_dim=50,
        color_mode=1
    )
    
    # Check that trainer was created and train was called
    mock_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once()


@patch("uvicorn.run")
def test_run_api_server(mock_uvicorn_run, setup_env):
    """Test the run_api_server function."""
    # Create mock args
    args = MagicMock()
    args.host = "0.0.0.0"
    args.port = 8000
    args.reload = False
    
    # Call function
    result = main.run_api_server(args)
    
    # Check return value
    assert result == 0
    
    # Check that uvicorn.run was called
    mock_uvicorn_run.assert_called_once_with(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )


@patch("main.bot.main")
def test_run_telegram_bot(mock_bot_main, setup_env):
    """Test the run_telegram_bot function."""
    # Create mock args
    args = MagicMock()
    args.token = "test_token"
    args.api_url = "http://localhost:8000"
    
    # Call function
    result = main.run_telegram_bot(args)
    
    # Check return value
    assert result == 0
    
    # Check that bot.main was called
    mock_bot_main.assert_called_once()


@patch("workflow.main")
def test_run_workflow(mock_workflow_main, setup_env):
    """Test the run_workflow function."""
    # Create mock args
    args = MagicMock()
    args.mode = "development"
    args.data_folder = "/tmp/data"
    args.model_type = "skip"
    args.epochs = 10
    args.schedule = False
    
    # Call function
    result = main.run_workflow(args)
    
    # Check return value
    assert result == 0
    
    # Check that workflow.main was called
    mock_workflow_main.assert_called_once()


@patch("multiprocessing.Process")
@patch("time.sleep")
def test_run_all_services(mock_sleep, mock_process, setup_env):
    """Test the run_all_services function."""
    # Setup mocks
    mock_process_instance = MagicMock()
    mock_process.return_value = mock_process_instance
    mock_sleep.return_value = None
    
    # Create mock args
    args = MagicMock()
    args.host = "0.0.0.0"
    args.port = 8000
    args.token = "test_token"
    args.mode = "development"
    args.workflow = True
    args.data_folder = "/tmp/data"
    args.model_type = "skip"
    args.epochs = 10
    args.schedule = False
    
    # Call function
    with patch("signal.signal"):
        result = main.run_all_services(args)
    
    # Check return value
    assert result == 0
    
    # Check that processes were created and started
    assert mock_process.call_count == 3  # API, bot, and workflow
    assert mock_process_instance.start.call_count == 3
    assert mock_process_instance.join.call_count == 3


@patch("argparse.ArgumentParser.parse_args")
def test_parse_arguments(mock_parse_args):
    """Test the parse_arguments function."""
    # Setup mock
    mock_args = MagicMock()
    mock_parse_args.return_value = mock_args
    
    # Call function
    args = main.parse_arguments()
    
    # Check return value
    assert args == mock_args
    
    # Check that parse_args was called
    mock_parse_args.assert_called_once()


@patch("main.setup_environment")
@patch("main.parse_arguments")
@patch("main.train_model")
@patch("main.run_api_server")
@patch("main.run_telegram_bot")
@patch("main.run_workflow")
@patch("main.run_all_services")
def test_main_function(mock_run_all, mock_run_workflow, mock_run_bot, 
                      mock_run_api, mock_train, mock_parse_args, 
                      mock_setup_env, setup_env):
    """Test the main function."""
    # Setup mocks
    mock_args = MagicMock()
    mock_parse_args.return_value = mock_args
    
    # Test train command
    mock_args.command = "train"
    mock_train.return_value = 0
    
    result = main.main()
    assert result == 0
    mock_train.assert_called_once_with(mock_args)
    
    # Reset mocks
    mock_train.reset_mock()
    
    # Test serve-api command
    mock_args.command = "serve-api"
    mock_run_api.return_value = 0
    
    result = main.main()
    assert result == 0
    mock_run_api.assert_called_once_with(mock_args)
    
    # Reset mocks
    mock_run_api.reset_mock()
    
    # Test run-bot command
    mock_args.command = "run-bot"
    mock_run_bot.return_value = 0
    
    result = main.main()
    assert result == 0
    mock_run_bot.assert_called_once_with(mock_args)
    
    # Reset mocks
    mock_run_bot.reset_mock()
    
    # Test workflow command
    mock_args.command = "workflow"
    mock_run_workflow.return_value = 0
    
    result = main.main()
    assert result == 0
    mock_run_workflow.assert_called_once_with(mock_args)
    
    # Reset mocks
    mock_run_workflow.reset_mock()
    
    # Test all command
    mock_args.command = "all"
    mock_run_all.return_value = 0
    
    result = main.main()
    assert result == 0
    mock_run_all.assert_called_once_with(mock_args)
    
    # Reset mocks
    mock_run_all.reset_mock()
    
    # Test no command
    mock_args.command = None
    
    result = main.main()
    assert result == 1


@patch("sys.exit")
def test_main_script(mock_exit, setup_env):
    """Test the main script execution."""
    # Mock sys.exit and main function
    with patch("main.main") as mock_main:
        mock_main.return_value = 0
        
        # Call the script
        import main as main_script
        
        # Check that main was called
        mock_main.assert_called_once()
        
        # Check that sys.exit was called with the return value
        mock_exit.assert_called_once_with(0)


def test_import_errors():
    """Test handling of import errors."""
    # Save original imports
    original_imports = {}
    for module in ["models", "trainer", "workflow", "api", "bot"]:
        if module in sys.modules:
            original_imports[module] = sys.modules[module]
            del sys.modules[module]
    
    # Mock imports to raise ImportError
    with patch("builtins.__import__", side_effect=ImportError("Test error")):
        try:
            # This should raise SystemExit
            with pytest.raises(SystemExit):
                import main
        except ImportError:
            # If it doesn't exit, fail the test
            pytest.fail("Import error not handled properly")
    
    # Restore original imports
    for module, value in original_imports.items():
        sys.modules[module] = value