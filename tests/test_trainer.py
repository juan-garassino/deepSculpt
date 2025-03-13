"""
Tests for the DeepSculpt training pipeline.
"""

import pytest
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer import DeepSculptTrainer, DataFrameDataLoader, create_data_dataframe
from models import ModelFactory


@pytest.fixture
def sample_data_df():
    """Create a sample DataFrame for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data files
        vol_data1 = np.random.rand(5, 8, 8, 8).astype(np.float32)
        mat_data1 = np.random.randint(0, 6, (5, 8, 8, 8)).astype(np.int32)
        
        vol_data2 = np.random.rand(5, 8, 8, 8).astype(np.float32)
        mat_data2 = np.random.randint(0, 6, (5, 8, 8, 8)).astype(np.int32)
        
        vol_path1 = os.path.join(tmpdir, "volume_data[2023-01-01]chunk[1].npy")
        mat_path1 = os.path.join(tmpdir, "material_data[2023-01-01]chunk[1].npy")
        
        vol_path2 = os.path.join(tmpdir, "volume_data[2023-01-01]chunk[2].npy")
        mat_path2 = os.path.join(tmpdir, "material_data[2023-01-01]chunk[2].npy")
        
        np.save(vol_path1, vol_data1)
        np.save(mat_path1, mat_data1)
        np.save(vol_path2, vol_data2)
        np.save(mat_path2, mat_data2)
        
        # Create DataFrame
        df = pd.DataFrame({
            'chunk_idx': [1, 2],
            'volume_path': [vol_path1, vol_path2],
            'material_path': [mat_path1, mat_path2]
        })
        
        yield df


@pytest.fixture
def simple_models():
    """Create simple generator and discriminator models for testing."""
    # Create very small models for testing
    generator = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Reshape((4, 4, 4, 1))
    ])
    
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(4, 4, 4, 1)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return generator, discriminator


def test_create_data_dataframe():
    """Test creating a DataFrame from data folder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data files
        vol_data = np.random.rand(5, 8, 8, 8).astype(np.float32)
        mat_data = np.random.randint(0, 6, (5, 8, 8, 8)).astype(np.int32)
        
        vol_path = os.path.join(tmpdir, "volume_data[2023-01-01]chunk[1].npy")
        mat_path = os.path.join(tmpdir, "material_data[2023-01-01]chunk[1].npy")
        
        np.save(vol_path, vol_data)
        np.save(mat_path, mat_data)
        
        # Create DataFrame
        df = create_data_dataframe(tmpdir)
        
        # Check DataFrame
        assert len(df) == 1
        assert df.iloc[0]['chunk_idx'] == 1
        assert df.iloc[0]['volume_path'] == vol_path
        assert df.iloc[0]['material_path'] == mat_path


def test_dataframe_data_loader(sample_data_df):
    """Test DataFrameDataLoader initialization and iteration."""
    loader = DataFrameDataLoader(
        df=sample_data_df,
        batch_size=2,
        shuffle=True
    )
    
    # Test length
    assert len(loader) > 0
    
    # Test iteration
    for batch in loader.iterate_batches():
        assert isinstance(batch, np.ndarray)
        assert batch.shape[0] <= 2  # Batch size should be less than or equal to 2
        break


def test_dataframe_data_loader_tf_dataset(sample_data_df):
    """Test DataFrameDataLoader TensorFlow dataset creation."""
    loader = DataFrameDataLoader(
        df=sample_data_df,
        batch_size=2,
        shuffle=True
    )
    
    # Create TensorFlow dataset
    dataset = loader.create_tf_dataset()
    
    # Check that it's a TensorFlow dataset
    assert isinstance(dataset, tf.data.Dataset)
    
    # Check that we can iterate over it
    for batch in dataset.take(1):
        assert isinstance(batch, tf.Tensor)
        assert batch.shape[0] <= 2  # Batch size should be less than or equal to 2


def test_deepsculpt_trainer_initialization(simple_models):
    """Test DeepSculptTrainer initialization."""
    generator, discriminator = simple_models
    
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=0.001,
        beta1=0.5,
        beta2=0.999
    )
    
    # Check attributes
    assert trainer.generator == generator
    assert trainer.discriminator == discriminator
    assert trainer.generator_optimizer is not None
    assert trainer.discriminator_optimizer is not None
    assert trainer.metrics is not None


@patch("builtins.open")
@patch("matplotlib.pyplot.savefig")
def test_deepsculpt_trainer_save_snapshot(mock_savefig, mock_open, simple_models, tmp_path):
    """Test saving a snapshot of the model."""
    generator, discriminator = simple_models
    
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator
    )
    
    # Mock the savefig method
    mock_savefig.return_value = None
    
    # Call the save snapshot method
    trainer._save_snapshot(str(tmp_path), 0)
    
    # Check that savefig was called
    assert mock_savefig.called


@patch("matplotlib.pyplot.savefig")
def test_deepsculpt_trainer_plot_metrics(mock_savefig, simple_models):
    """Test plotting training metrics."""
    generator, discriminator = simple_models
    
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator
    )
    
    # Add some metrics
    trainer.metrics['gen_loss'] = [1.0, 0.9, 0.8]
    trainer.metrics['disc_loss'] = [0.5, 0.4, 0.3]
    trainer.metrics['epoch_times'] = [1.0, 1.0, 1.0]
    
    # Mock the savefig method
    mock_savefig.return_value = None
    
    # Call the plot metrics method
    trainer.plot_metrics(save_path="test.png")
    
    # Check that savefig was called
    assert mock_savefig.called


@patch("tensorflow.keras.models.save_model")
@patch.object(DeepSculptTrainer, "_save_snapshot")
@pytest.mark.parametrize("with_manager", [True, False])
def test_deepsculpt_trainer_train(mock_save_snapshot, mock_save_model, simple_models, with_manager):
    """Test the training loop."""
    generator, discriminator = simple_models
    
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=0.001
    )
    
    # Create a mock data loader
    mock_loader = MagicMock()
    mock_dataset = tf.data.Dataset.from_tensor_slices(
        tf.random.normal([10, 4, 4, 4, 1])
    ).batch(2)
    mock_loader.create_tf_dataset.return_value = mock_dataset
    
    # Create a mock checkpoint manager
    if with_manager:
        mock_checkpoint = MagicMock()
        mock_manager = MagicMock()
        mock_manager.save.return_value = "checkpoint_path"
    else:
        mock_checkpoint = None
        mock_manager = None
    
    # Train for a few epochs
    metrics = trainer.train(
        data_loader=mock_loader,
        epochs=2,
        checkpoint_dir="test" if with_manager else None,
        snapshot_dir="test",
        snapshot_freq=1
    )
    
    # Check that metrics were recorded
    assert 'gen_loss' in metrics
    assert 'disc_loss' in metrics
    assert len(metrics['gen_loss']) == 2
    assert len(metrics['disc_loss']) == 2
    
    # Check that save_snapshot was called
    assert mock_save_snapshot.called


def test_train_step(simple_models):
    """Test a single training step."""
    generator, discriminator = simple_models
    
    trainer = DeepSculptTrainer(
        generator=generator,
        discriminator=discriminator,
        learning_rate=0.001
    )
    
    # Create a sample batch
    batch = tf.random.normal([2, 4, 4, 4, 1])
    
    # Run a training step
    gen_loss, disc_loss = trainer.train_step(batch)
    
    # Check that losses were returned
    assert isinstance(gen_loss, tf.Tensor)
    assert isinstance(disc_loss, tf.Tensor)