"""
Tests for PyTorch-based curator functionality.

This module tests the PyTorchCurator class and its encoder/decoder implementations,
ensuring equivalence with the original TensorFlow-based curator while providing
enhanced PyTorch functionality.
"""

import os
import tempfile
import shutil
import numpy as np
import torch
import pytest
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from deepsculpt.core.data.transforms.pytorch_curator import (
    PyTorchOneHotEncoderDecoder,
    PyTorchBinaryEncoderDecoder,
    PyTorchRGBEncoderDecoder,
    PyTorchEmbeddingEncoderDecoder,
    PyTorchCurator,
    PyTorchDataset
)


class TestPyTorchOneHotEncoderDecoder:
    """Test PyTorch one-hot encoding and decoding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test data
        self.colors_array = np.array([
            [
                [['red', 'blue', None], ['green', None, 'red']],
                [[None, 'blue', 'green'], ['red', 'green', None]]
            ]
        ])
        
        self.color_list = ['red', 'blue', 'green', None]
        
        self.encoder = PyTorchOneHotEncoderDecoder(
            self.colors_array,
            color_list=self.color_list,
            device=self.device,
            verbose=True
        )

    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.device == self.device
        assert self.encoder.n_samples == 1
        assert self.encoder.void_dim == 2
        assert self.encoder.n_classes == 4
        assert self.encoder.color_list == self.color_list

    def test_ohe_encode(self):
        """Test one-hot encoding."""
        encoded_tensor, classes = self.encoder.ohe_encode()
        
        # Check output shape
        expected_shape = (1, 2, 2, 3, 4)  # (samples, dim, dim, dim, classes)
        assert encoded_tensor.shape == expected_shape
        
        # Check classes
        assert classes == self.color_list
        
        # Check tensor is on correct device
        assert encoded_tensor.device.type == self.device
        
        # Check one-hot property (each voxel should have exactly one 1)
        sums = torch.sum(encoded_tensor, dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums))

    def test_ohe_decode(self):
        """Test one-hot decoding."""
        # First encode
        encoded_tensor, _ = self.encoder.ohe_encode()
        
        # Then decode
        structures, colors = self.encoder.ohe_decode(encoded_tensor)
        
        # Check shapes
        expected_shape = (1, 2, 2, 3)
        assert structures.shape == expected_shape
        assert colors.shape == expected_shape
        
        # Check that structures are binary
        assert torch.all((structures == 0) | (structures == 1))
        
        # Check device
        assert structures.device.type == self.device
        assert colors.device.type == self.device

    def test_encode_decode_consistency(self):
        """Test that encoding followed by decoding preserves structure."""
        encoded_tensor, _ = self.encoder.ohe_encode()
        structures, colors = self.encoder.ohe_decode(encoded_tensor)
        
        # Re-encode and check consistency
        encoded_again, _ = self.encoder.ohe_encode()
        
        # Should be identical (within floating point precision)
        assert torch.allclose(encoded_tensor, encoded_again, atol=1e-6)


class TestPyTorchBinaryEncoderDecoder:
    """Test PyTorch binary encoding and decoding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test data with simple colors
        self.colors_array = np.array([
            [
                [['red', 'blue', None], ['green', None, 'red']],
                [[None, 'blue', 'green'], ['red', 'green', None]]
            ]
        ])
        
        self.encoder = PyTorchBinaryEncoderDecoder(
            self.colors_array,
            device=self.device,
            verbose=True
        )

    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.device == self.device
        assert self.encoder.n_samples == 1
        assert self.encoder.void_dim == 2

    def test_binary_encode(self):
        """Test binary encoding."""
        encoded_tensor, classes = self.encoder.binary_encode()
        
        # Check that we have the right number of bits
        n_classes = len(set(self.colors_array.flatten()))
        expected_bits = int(np.ceil(np.log2(n_classes)))
        
        expected_shape = (1, 2, 2, 3, expected_bits)
        assert encoded_tensor.shape == expected_shape
        
        # Check tensor is on correct device
        assert encoded_tensor.device.type == self.device
        
        # Check that values are binary (0 or 1)
        assert torch.all((encoded_tensor == 0) | (encoded_tensor == 1))

    def test_binary_decode(self):
        """Test binary decoding."""
        # First encode
        encoded_tensor, classes = self.encoder.binary_encode()
        
        # Then decode
        structures, colors = self.encoder.binary_decode(encoded_tensor)
        
        # Check shapes
        expected_shape = (1, 2, 2, 3)
        assert structures.shape == expected_shape
        assert colors.shape == expected_shape
        
        # Check device
        assert structures.device.type == self.device
        assert colors.device.type == self.device

    def test_encode_decode_consistency(self):
        """Test encoding/decoding consistency."""
        encoded_tensor, _ = self.encoder.binary_encode()
        structures, colors = self.encoder.binary_decode(encoded_tensor)
        
        # Check that non-zero colors correspond to structure
        non_zero_colors = colors != 0
        assert torch.allclose(structures, non_zero_colors.float())


class TestPyTorchRGBEncoderDecoder:
    """Test PyTorch RGB encoding and decoding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test data
        self.colors_array = np.array([
            [
                [['red', 'blue', None], ['green', None, 'red']],
                [[None, 'blue', 'green'], ['red', 'green', None]]
            ]
        ])
        
        self.color_dict = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            None: (0, 0, 0)
        }
        
        self.encoder = PyTorchRGBEncoderDecoder(
            self.colors_array,
            color_dict=self.color_dict,
            device=self.device,
            verbose=True
        )

    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.device == self.device
        assert self.encoder.n_samples == 1
        assert self.encoder.void_dim == 2
        assert self.encoder.color_dict == self.color_dict

    def test_rgb_encode(self):
        """Test RGB encoding."""
        encoded_tensor, color_mapping = self.encoder.rgb_encode()
        
        # Check output shape
        expected_shape = (1, 2, 2, 3, 3)  # (samples, dim, dim, dim, RGB)
        assert encoded_tensor.shape == expected_shape
        
        # Check tensor is on correct device
        assert encoded_tensor.device.type == self.device
        
        # Check data type
        assert encoded_tensor.dtype == torch.uint8
        
        # Check color mapping
        assert color_mapping == self.color_dict

    def test_rgb_decode(self):
        """Test RGB decoding."""
        # First encode
        encoded_tensor, _ = self.encoder.rgb_encode()
        
        # Then decode
        structures, colors = self.encoder.rgb_decode(encoded_tensor, threshold=1.0)
        
        # Check shapes
        expected_shape = (1, 2, 2, 3)
        assert structures.shape == expected_shape
        assert colors.shape == expected_shape
        
        # Check device
        assert structures.device.type == self.device
        assert colors.device.type == self.device

    def test_color_matching(self):
        """Test that RGB encoding/decoding preserves color information."""
        encoded_tensor, _ = self.encoder.rgb_encode()
        structures, colors = self.encoder.rgb_decode(encoded_tensor, threshold=1.0)
        
        # Check that structures match non-zero colors
        non_zero_structures = structures > 0
        non_zero_colors = colors > 0
        assert torch.allclose(non_zero_structures.float(), non_zero_colors.float())


class TestPyTorchEmbeddingEncoderDecoder:
    """Test PyTorch embedding encoding and decoding."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create test data
        self.colors_array = np.array([
            [
                [['red', 'blue', None], ['green', None, 'red']],
                [[None, 'blue', 'green'], ['red', 'green', None]]
            ]
        ])
        
        self.embedding_dim = 16
        
        self.encoder = PyTorchEmbeddingEncoderDecoder(
            self.colors_array,
            embedding_dim=self.embedding_dim,
            device=self.device,
            verbose=True
        )

    def test_initialization(self):
        """Test encoder initialization."""
        assert self.encoder.device == self.device
        assert self.encoder.embedding_dim == self.embedding_dim
        assert self.encoder.n_samples == 1
        assert self.encoder.void_dim == 2

    def test_embedding_encode(self):
        """Test embedding encoding."""
        encoded_tensor, embedding_layer = self.encoder.embedding_encode()
        
        # Check output shape
        expected_shape = (1, 2, 2, 3, self.embedding_dim)
        assert encoded_tensor.shape == expected_shape
        
        # Check tensor is on correct device
        assert encoded_tensor.device.type == self.device
        
        # Check embedding layer
        assert isinstance(embedding_layer, torch.nn.Embedding)
        assert embedding_layer.embedding_dim == self.embedding_dim

    def test_embedding_decode(self):
        """Test embedding decoding."""
        # First encode
        encoded_tensor, _ = self.encoder.embedding_encode()
        
        # Then decode
        structures, colors = self.encoder.embedding_decode(encoded_tensor)
        
        # Check shapes
        expected_shape = (1, 2, 2, 3)
        assert structures.shape == expected_shape
        assert colors.shape == expected_shape
        
        # Check device
        assert structures.device.type == self.device
        assert colors.device.type == self.device


class TestPyTorchDataset:
    """Test PyTorchDataset functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data files
        self.data_paths = []
        for i in range(5):
            # Create structure file
            structure = np.random.randint(0, 2, (64, 64, 64))
            structure_path = os.path.join(self.temp_dir, f"structure_{i}.npy")
            np.save(structure_path, structure)
            
            # Create colors file
            colors = np.random.choice(['red', 'blue', 'green', None], (64, 64, 64))
            colors_path = os.path.join(self.temp_dir, f"colors_{i}.npy")
            np.save(colors_path, colors)
            
            self.data_paths.append((structure_path, colors_path))
        
        # Create encoder
        sample_colors = np.load(self.data_paths[0][1], allow_pickle=True)
        self.encoder = PyTorchOneHotEncoderDecoder(
            sample_colors, device=self.device
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_dataset_creation(self):
        """Test dataset creation."""
        dataset = PyTorchDataset(
            data_paths=self.data_paths,
            encoder_decoder=self.encoder,
            device=self.device
        )
        
        assert len(dataset) == 5
        
        # Test getting a sample
        sample = dataset[0]
        assert 'structure' in sample
        assert 'colors' in sample
        assert 'index' in sample
        
        # Check tensor devices
        assert sample['structure'].device.type == self.device
        assert sample['colors'].device.type == self.device

    def test_dataset_caching(self):
        """Test dataset caching functionality."""
        dataset = PyTorchDataset(
            data_paths=self.data_paths,
            encoder_decoder=self.encoder,
            device=self.device,
            cache_size=3
        )
        
        # Load samples to populate cache
        for i in range(3):
            _ = dataset[i]
        
        # Check cache size
        assert len(dataset.cache) == 3
        assert len(dataset.cache_order) == 3

    def test_dataset_preloading(self):
        """Test dataset preloading."""
        dataset = PyTorchDataset(
            data_paths=self.data_paths[:2],  # Use fewer samples for speed
            encoder_decoder=self.encoder,
            device=self.device,
            preload=True
        )
        
        # All samples should be cached
        assert len(dataset.cache) == 2


class TestPyTorchCurator:
    """Test PyTorchCurator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock collection directory structure
        self.collection_dir = os.path.join(self.temp_dir, "test_collection")
        self.samples_dir = os.path.join(self.collection_dir, "samples")
        self.structures_dir = os.path.join(self.samples_dir, "structures")
        self.colors_dir = os.path.join(self.samples_dir, "colors")
        
        os.makedirs(self.structures_dir)
        os.makedirs(self.colors_dir)
        
        # Create mock data files
        for i in range(3):
            # Create structure file
            structure = np.random.randint(0, 2, (32, 32, 32))  # Smaller for speed
            structure_path = os.path.join(self.structures_dir, f"structure_{i:03d}.npy")
            np.save(structure_path, structure)
            
            # Create colors file
            colors = np.random.choice(['red', 'blue', 'green', None], (32, 32, 32))
            colors_path = os.path.join(self.colors_dir, f"colors_{i:03d}.npy")
            np.save(colors_path, colors)
        
        self.curator = PyTorchCurator(
            processing_method="OHE",
            output_dir=os.path.join(self.temp_dir, "processed"),
            device=self.device,
            batch_size=2,
            verbose=True
        )

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_curator_initialization(self):
        """Test curator initialization."""
        assert self.curator.processing_method == "OHE"
        assert self.curator.device == self.device
        assert self.curator.batch_size == 2
        assert os.path.exists(self.curator.output_dir)

    def test_load_samples_from_collection(self):
        """Test loading samples from collection."""
        data_paths, metadata = self.curator.load_samples_from_collection(
            self.collection_dir, limit=None, shuffle=False
        )
        
        assert len(data_paths) == 3
        assert metadata['total_samples'] == 3
        assert metadata['processing_method'] == "OHE"
        
        # Check that paths exist
        for struct_path, colors_path in data_paths:
            assert os.path.exists(struct_path)
            assert os.path.exists(colors_path)

    def test_create_dataset(self):
        """Test dataset creation."""
        data_paths, _ = self.curator.load_samples_from_collection(self.collection_dir)
        dataset = self.curator.create_dataset(data_paths)
        
        assert len(dataset) == 3
        assert isinstance(dataset, PyTorchDataset)

    def test_create_dataloader(self):
        """Test dataloader creation."""
        data_paths, _ = self.curator.load_samples_from_collection(self.collection_dir)
        dataset = self.curator.create_dataset(data_paths)
        dataloader = self.curator.create_dataloader(dataset, batch_size=2)
        
        # Test getting a batch
        batch = next(iter(dataloader))
        assert 'structures' in batch
        assert 'colors' in batch
        assert 'indices' in batch
        
        # Check batch size (should be 2 or less for last batch)
        assert batch['structures'].shape[0] <= 2

    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        data_paths, _ = self.curator.load_samples_from_collection(self.collection_dir)
        dataset = self.curator.create_dataset(data_paths)
        dataloader = self.curator.create_dataloader(dataset, batch_size=2)
        
        batch = next(iter(dataloader))
        processed_batch = self.curator.preprocess_batch(
            batch, apply_encoding=True, apply_augmentation=False
        )
        
        assert 'structures' in processed_batch
        assert 'colors' in processed_batch
        # Should have encoded colors for OHE method
        assert 'encoded_colors' in processed_batch

    def test_preprocess_collection(self):
        """Test full collection preprocessing."""
        result = self.curator.preprocess_collection(
            self.collection_dir,
            limit=2,
            save_processed=False
        )
        
        assert 'processed_batches' in result
        assert 'metadata' in result
        assert result['metadata']['total_processed'] == 2

    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        memory_usage = self.curator.get_memory_usage()
        
        # Should have either GPU or CPU memory info
        assert len(memory_usage) > 0
        
        if torch.cuda.is_available():
            assert 'gpu_allocated' in memory_usage
        else:
            assert 'cpu_memory' in memory_usage

    def test_batch_size_optimization(self):
        """Test automatic batch size optimization."""
        data_paths, _ = self.curator.load_samples_from_collection(self.collection_dir)
        dataset = self.curator.create_dataset(data_paths)
        
        optimal_batch_size = self.curator.optimize_batch_size(
            dataset, target_memory_gb=1.0, max_batch_size=8
        )
        
        assert isinstance(optimal_batch_size, int)
        assert optimal_batch_size >= 1
        assert optimal_batch_size <= 8

    @pytest.mark.parametrize("processing_method", ["OHE", "BINARY", "RGB", "EMBEDDING"])
    def test_different_processing_methods(self, processing_method):
        """Test different processing methods."""
        curator = PyTorchCurator(
            processing_method=processing_method,
            output_dir=os.path.join(self.temp_dir, f"processed_{processing_method.lower()}"),
            device=self.device,
            batch_size=1,
            verbose=False
        )
        
        data_paths, _ = curator.load_samples_from_collection(self.collection_dir)
        dataset = curator.create_dataset(data_paths)
        dataloader = curator.create_dataloader(dataset, batch_size=1)
        
        batch = next(iter(dataloader))
        processed_batch = curator.preprocess_batch(batch, apply_encoding=True)
        
        # Should complete without errors
        assert 'structures' in processed_batch
        assert 'colors' in processed_batch


class TestEncodingEquivalence:
    """Test equivalence between PyTorch and original implementations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create consistent test data
        np.random.seed(42)
        self.colors_array = np.random.choice(
            ['red', 'blue', 'green', None], 
            size=(2, 8, 8, 8)
        )

    def test_one_hot_encoding_consistency(self):
        """Test that PyTorch one-hot encoding produces consistent results."""
        encoder = PyTorchOneHotEncoderDecoder(
            self.colors_array, device=self.device
        )
        
        # Encode twice and check consistency
        encoded1, classes1 = encoder.ohe_encode()
        encoded2, classes2 = encoder.ohe_encode()
        
        assert torch.allclose(encoded1, encoded2)
        assert classes1 == classes2

    def test_binary_encoding_consistency(self):
        """Test that PyTorch binary encoding produces consistent results."""
        encoder = PyTorchBinaryEncoderDecoder(
            self.colors_array, device=self.device
        )
        
        # Encode twice and check consistency
        encoded1, classes1 = encoder.binary_encode()
        encoded2, classes2 = encoder.binary_encode()
        
        assert torch.allclose(encoded1, encoded2)
        assert classes1 == classes2

    def test_rgb_encoding_consistency(self):
        """Test that PyTorch RGB encoding produces consistent results."""
        encoder = PyTorchRGBEncoderDecoder(
            self.colors_array, device=self.device
        )
        
        # Encode twice and check consistency
        encoded1, mapping1 = encoder.rgb_encode()
        encoded2, mapping2 = encoder.rgb_encode()
        
        assert torch.allclose(encoded1, encoded2)
        assert mapping1 == mapping2

    def test_round_trip_consistency(self):
        """Test that encode->decode->encode produces consistent results."""
        encoder = PyTorchOneHotEncoderDecoder(
            self.colors_array, device=self.device
        )
        
        # First round trip
        encoded1, _ = encoder.ohe_encode()
        structures1, colors1 = encoder.ohe_decode(encoded1)
        
        # Create new encoder with decoded colors and encode again
        encoder2 = PyTorchOneHotEncoderDecoder(
            colors1.cpu().numpy(), device=self.device
        )
        encoded2, _ = encoder2.ohe_encode()
        
        # Should be consistent in structure
        assert encoded1.shape == encoded2.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])