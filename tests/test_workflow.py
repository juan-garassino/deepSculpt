"""
Tests for the DeepSculpt workflow automation.
"""

import pytest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the workflow module
from workflow import Manager, preprocess_data, evaluate_model, train_model, compare_and_promote, promote_model


@pytest.fixture
def setup_env():
    """Setup environment variables for tests."""
    # Save original environment
    original_env = dict(os.environ)
    
    # Set test environment variables
    os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
    os.environ["MLFLOW_EXPERIMENT"] = "test_experiment"
    os.environ["MLFLOW_MODEL_NAME"] = "test_model"
    os.environ["PREFECT_FLOW_NAME"] = "test_workflow"
    os.environ["VOID_DIM"] = "32"
    os.environ["NOISE_DIM"] = "50"
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_data_directory():
    """Create a temporary directory with sample data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data files
        vol_data = np.random.rand(5, 8, 8, 8).astype(np.float32)
        mat_data = np.random.randint(0, 6, (5, 8, 8, 8)).astype(np.int32)
        
        vol_path = os.path.join(tmpdir, "volume_data[2023-01-01]chunk[1].npy")
        mat_path = os.path.join(tmpdir, "material_data[2023-01-01]chunk[1].npy")
        
        np.save(vol_path, vol_data)
        np.save(mat_path, mat_data)
        
        yield tmpdir


def test_manager_initialization():
    """Test Manager initialization."""
    manager = Manager(model_name="test_model", data_name="test_data")
    
    assert manager.model_name == "test_model"
    assert manager.data_name == "test_data"
    assert manager.comment == "test_model_test_data"
    assert manager.data_subdir == "test_model/test_data"


def test_manager_create_data_dataframe(sample_data_directory):
    """Test Manager.create_data_dataframe method."""
    manager = Manager()
    
    # Create DataFrame
    df = manager.create_data_dataframe(sample_data_directory)
    
    # Check DataFrame
    assert len(df) == 1
    assert df.iloc[0]['chunk_idx'] == 1
    assert os.path.basename(df.iloc[0]['volume_path']) == "volume_data[2023-01-01]chunk[1].npy"
    assert os.path.basename(df.iloc[0]['material_path']) == "material_data[2023-01-01]chunk[1].npy"


def test_manager_make_directory(tmp_path):
    """Test Manager.make_directory static method."""
    test_dir = os.path.join(tmp_path, "test_dir")
    
    # Make sure directory doesn't exist yet
    assert not os.path.exists(test_dir)
    
    # Create directory
    Manager.make_directory(test_dir)
    
    # Check that directory was created
    assert os.path.exists(test_dir)
    assert os.path.isdir(test_dir)


@patch("mlflow.set_tracking_uri")
@patch("mlflow.set_experiment")
@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metrics")
@patch("mlflow.keras.log_model")
def test_manager_save_mlflow_model(mock_log_model, mock_log_metrics, mock_log_params,
                                   mock_start_run, mock_set_experiment, mock_set_tracking_uri,
                                   setup_env):
    """Test Manager.save_mlflow_model static method."""
    # Create mock model, metrics, and params
    mock_model = MagicMock()
    metrics = {"metric1": 0.5, "metric2": 0.7}
    params = {"param1": "value1", "param2": "value2"}
    
    # Call method
    Manager.save_mlflow_model(metrics=metrics, params=params, model=mock_model)
    
    # Check that MLflow functions were called
    mock_set_tracking_uri.assert_called_once_with(os.environ["MLFLOW_TRACKING_URI"])
    mock_set_experiment.assert_called_once_with(experiment_name=os.environ["MLFLOW_EXPERIMENT"])
    assert mock_start_run.called
    mock_log_params.assert_called_once_with(params)
    mock_log_metrics.assert_called_once_with(metrics)
    mock_log_model.assert_called_once()


@patch("mlflow.set_tracking_uri")
@patch("mlflow.keras.load_model")
def test_manager_load_mlflow_model(mock_load_model, mock_set_tracking_uri, setup_env):
    """Test Manager.load_mlflow_model static method."""
    # Setup mock
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    # Call method
    model = Manager.load_mlflow_model(stage="Production")
    
    # Check that MLflow functions were called
    mock_set_tracking_uri.assert_called_once_with(os.environ["MLFLOW_TRACKING_URI"])
    mock_load_model.assert_called_once()
    
    # Check that model was returned
    assert model == mock_model


@patch("workflow.Manager.create_data_dataframe")
def test_preprocess_data(mock_create_df, sample_data_directory, tmp_path):
    """Test preprocess_data task."""
    # Setup mock
    mock_df = pd.DataFrame({"test": [1, 2, 3]})
    mock_create_df.return_value = mock_df
    
    # Call task
    with patch("os.makedirs"):
        output_path = preprocess_data("test_experiment", sample_data_directory)
    
    # Check that create_data_dataframe was called
    mock_create_df.assert_called_once_with(sample_data_directory)
    
    # Check that output path was returned
    assert isinstance(output_path, str)
    assert output_path.endswith("data_paths.csv")


@patch("workflow.Manager.load_mlflow_model")
@patch("pandas.read_csv")
@patch("workflow.DataFrameDataLoader")
def test_evaluate_model(mock_loader, mock_read_csv, mock_load_model):
    """Test evaluate_model task."""
    # Setup mocks
    mock_model = MagicMock()
    mock_load_model.return_value = mock_model
    
    mock_df = pd.DataFrame({"test": [1, 2, 3]})
    mock_read_csv.return_value = mock_df
    
    mock_dataset = MagicMock()
    mock_loader_instance = MagicMock()
    mock_loader_instance.create_tf_dataset.return_value = mock_dataset
    mock_loader.return_value = mock_loader_instance
    
    # Mock model generation that returns a tensor
    mock_model.return_value = tf.ones([16, 32, 32, 32, 3])
    
    # Call task
    with patch("tensorflow.random.normal"):
        metrics = evaluate_model("test_data_path", model_type="skip", stage="Production")
    
    # Check that metrics were returned
    assert isinstance(metrics, dict)
    assert "avg_value" in metrics
    assert "std_value" in metrics


@patch("workflow.ModelFactory.create_generator")
@patch("workflow.ModelFactory.create_discriminator")
@patch("pandas.read_csv")
@patch("workflow.DataFrameDataLoader")
@patch("workflow.DeepSculptTrainer")
@patch("os.makedirs")
@patch("workflow.Manager.save_mlflow_model")
def test_train_model(mock_save_mlflow, mock_makedirs, mock_trainer, mock_loader, 
                    mock_read_csv, mock_create_disc, mock_create_gen):
    """Test train_model task."""
    # Setup mocks
    mock_generator = MagicMock()
    mock_discriminator = MagicMock()
    mock_create_gen.return_value = mock_generator
    mock_create_disc.return_value = mock_discriminator
    
    mock_df = pd.DataFrame({"test": [1, 2, 3]})
    mock_read_csv.return_value = mock_df
    
    mock_loader_instance = MagicMock()
    mock_loader.return_value = mock_loader_instance
    
    mock_trainer_instance = MagicMock()
    mock_trainer_instance.train.return_value = {
        "gen_loss": [1.0, 0.9, 0.8],
        "disc_loss": [0.5, 0.4, 0.3],
        "epoch_times": [1.0, 1.0, 1.0]
    }
    mock_trainer.return_value = mock_trainer_instance
    
    # Call task
    metrics = train_model("test_data_path", model_type="skip", epochs=3)
    
    # Check that metrics were returned
    assert isinstance(metrics, dict)
    assert "gen_loss" in metrics
    assert "disc_loss" in metrics
    assert "training_time" in metrics
    
    # Check that models were created
    mock_create_gen.assert_called_once_with(model_type="skip")
    mock_create_disc.assert_called_once_with(model_type="skip")
    
    # Check that trainer was created and train was called
    mock_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once()
    
    # Check that MLflow model was saved
    mock_save_mlflow.assert_called_once()


def test_compare_and_promote():
    """Test compare_and_promote task."""
    # Test case where new model is better
    eval_metrics = {"gen_loss": 1.0}
    train_metrics = {"gen_loss": 0.8}
    
    result = compare_and_promote(eval_metrics, train_metrics, threshold=0.1)
    assert result is True
    
    # Test case where new model is not significantly better
    eval_metrics = {"gen_loss": 1.0}
    train_metrics = {"gen_loss": 0.95}
    
    result = compare_and_promote(eval_metrics, train_metrics, threshold=0.1)
    assert result is False
    
    # Test case with missing metrics
    eval_metrics = {"other_metric": 1.0}
    train_metrics = {"other_metric": 0.8}
    
    result = compare_and_promote(eval_metrics, train_metrics, threshold=0.1)
    assert result is True  # Default to promote if metrics can't be compared


@patch("mlflow.set_tracking_uri")
@patch("mlflow.tracking.MlflowClient")
@patch("workflow.Manager.get_model_version")
def test_promote_model(mock_get_version, mock_client, mock_set_tracking_uri, setup_env):
    """Test promote_model task."""
    # Setup mocks
    mock_client_instance = MagicMock()
    mock_client.return_value = mock_client_instance
    
    mock_get_version.return_value = 1
    
    # Test with should_promote=True
    promote_model(True)
    
    # Check that model was promoted
    mock_client_instance.transition_model_version_stage.assert_called_once()
    
    # Reset mocks
    mock_client_instance.reset_mock()
    
    # Test with should_promote=False
    promote_model(False)
    
    # Check that model was not promoted
    mock_client_instance.transition_model_version_stage.assert_not_called()


@patch("workflow.preprocess_data")
@patch("workflow.evaluate_model")
@patch("workflow.train_model")
@patch("workflow.compare_and_promote")
@patch("workflow.promote_model")
@patch("workflow.notify")
def test_build_flow(mock_notify, mock_promote, mock_compare, mock_train, mock_evaluate, mock_preprocess):
    """Test build_flow function."""
    # Import build_flow
    from workflow import build_flow
    
    # Setup mocks
    mock_preprocess.return_value = "test_data_path"
    mock_evaluate.return_value = {"gen_loss": 1.0}
    mock_train.return_value = {"gen_loss": 0.8}
    mock_compare.return_value = True
    
    # Build flow
    flow = build_flow(schedule=None)
    
    # Check that flow was created
    assert flow is not None
    assert hasattr(flow, "run")