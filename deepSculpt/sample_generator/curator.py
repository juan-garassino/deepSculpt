"""
Dataset Preprocessing System for DeepSculpt
This module handles the preprocessing of sculpture datasets for machine learning tasks.
It provides various encoding methods (one-hot, binary, RGB), creates TensorFlow datasets,
and includes visualization tools for encoded data. It bridges the gap between raw
sculpture data and training-ready formats.

Key features:
- Multiple encoding methods: One-hot, binary, and RGB encoding options
- TensorFlow integration: Creates properly formatted datasets for training
- Visualization: Tools for inspecting encoded and decoded data
- Decoding functionality: Converts encoded data back to sculptural form
- Data inspection: Analysis and validation of preprocessing results
- Batch processing: Efficient handling of large datasets

Dependencies:
- logger.py: For process tracking and status reporting
- collector.py: For loading generated sculpture data
- visualization.py: For displaying processed samples
- numpy: For array operations
- sklearn: For label encoding and one-hot encoding
- tensorflow: For dataset creation and management
- matplotlib.colors: For color mapping in RGB encoding
- random: For sample selection
- tqdm: For progress visualization

Used by:
- Training scripts: For preparing data for machine learning models
- Model evaluation: For processing inference results

TODO:
- Add support for sparse tensor encoding for memory efficiency
- Implement more sophisticated data augmentation techniques
- Add normalization options for different model architectures
- Support for multi-GPU data pipeline optimization
- Add dataset statistics and distribution analysis
- Implement feature engineering options for different ML approaches
- Add support for progressive loading of very large datasets
"""

import os
import time
import random
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Any, Optional, Union, Set
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.data import Dataset

from logger import begin_section, end_section, log_action, log_success, log_error, log_info, log_warning
from collector import Collector
from visualization import Visualizer

class EncoderDecoder:
    """Base class for encoding and decoding methods."""
    
    def __init__(
        self,
        materials: np.ndarray,
        verbose: bool = False
    ):
        """
        Initialize the EncoderDecoder.
        
        Args:
            materials: Array of materials to encode
            verbose: Whether to print detailed information
        """
        self.materials = materials
        self.verbose = verbose
        self.unique_colors = self._get_unique_colors()
        
    def _get_unique_colors(self) -> Set[Any]:
        """
        Extract all unique colors from the materials array.
        
        Returns:
            Set of unique color values
        """
        unique_colors = set()
        # Flatten the array and add each unique color to the set
        for color in np.ndarray.flatten(self.materials):
            unique_colors.add(color)
        
        if self.verbose:
            log_info(f"Found {len(unique_colors)} unique colors: {unique_colors}")
        
        return unique_colors

class OneHotEncoderDecoder(EncoderDecoder):
    """
    Class for one-hot encoding and decoding material colors using sklearn's OneHotEncoder.
    
    This class encodes color and material labels into one-hot encoded arrays and can
    decode the one-hot encoded arrays back into the original labels.
    """
    
    def __init__(
        self,
        materials_labels_array: np.ndarray,
        material_list: Optional[List[Any]] = None,
        verbose: bool = False
    ):
        """
        Initialize the OneHotEncoderDecoder.
        
        Args:
            materials_labels_array: Array of material labels to encode
            material_list: List of all possible material values (if None, extracted from data)
            verbose: Whether to print detailed information
        """
        super().__init__(materials_labels_array, verbose)
        self.materials_labels_array = materials_labels_array
        self.void_dim = self.materials_labels_array.shape[1]
        self.n_samples = self.materials_labels_array.shape[0]
        self.n_classes = None
        self.classes = None
        
        # Use provided material list or determine from data
        if material_list is not None:
            self.material_list = material_list
        else:
            self.material_list = sorted(list(self.unique_colors), key=lambda x: str(x))
        
        # Initialize the one-hot encoder
        self.one_hot_encoder = OneHotEncoder(
            categories=[self.material_list], 
            handle_unknown='ignore'
        )
    
    def ohe_encode(self) -> Tuple[np.ndarray, List[Any]]:
        """
        Encode materials using one-hot encoding.
        
        Returns:
            Tuple of (encoded_array, color_classes)
        """
        begin_section("One-Hot Encoding Materials")
        
        try:
            if not self.material_list:
                raise ValueError("The list of materials cannot be empty.")
            
            # Reshape the array for encoding
            flat_materials = self.materials_labels_array.reshape(-1, 1)
            
            # Fit and transform using sklearn's OneHotEncoder
            encoded_array = self.one_hot_encoder.fit_transform(flat_materials)
            
            # Get the classes from the encoder
            self.classes = self.one_hot_encoder.categories_[0]
            self.n_classes = len(self.classes)
            
            log_info(f"Encoded {self.n_samples} samples into {self.n_classes} classes")
            log_info(f"Classes: {self.classes}")
            
            # Reshape the encoded array back to the original dimensions plus one-hot dimension
            encoded_reshaped = encoded_array.toarray().reshape(
                (
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_classes
                )
            )
            
            log_success(f"One-hot encoded to shape {encoded_reshaped.shape}")
            end_section()
            
            return encoded_reshaped, self.classes
            
        except Exception as e:
            log_error(f"Error during one-hot encoding: {str(e)}")
            end_section("One-hot encoding failed")
            raise
    
    def ohe_decode(
        self,
        one_hot_encoded_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode one-hot encoded materials back to colors.
        
        Args:
            one_hot_encoded_array: One-hot encoded array
            
        Returns:
            Tuple of (volumes_array, materials_array) where:
              - volumes_array has 1s where material exists and 0s elsewhere
              - materials_array contains the original material names
        """
        begin_section("One-Hot Decoding Materials")
        
        try:
            self.n_samples = one_hot_encoded_array.shape[0]
            
            # Reshape the array for decoding
            flat_encoded = one_hot_encoded_array.reshape(
                (
                    self.n_samples * self.void_dim * self.void_dim * self.void_dim,
                    self.n_classes
                )
            )
            
            # Inverse transform using the encoder
            decoded_materials = self.one_hot_encoder.inverse_transform(flat_encoded)
            
            # Create a binary volume array (1 where material exists, 0 elsewhere)
            decoded_volumes = np.where(decoded_materials == None, 0, 1)
            
            # Reshape back to original dimensions
            decoded_volumes_reshaped = decoded_volumes.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )
            
            decoded_materials_reshaped = decoded_materials.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )
            
            log_success(f"Decoded to shapes: volumes {decoded_volumes_reshaped.shape}, materials {decoded_materials_reshaped.shape}")
            end_section()
            
            return decoded_volumes_reshaped, decoded_materials_reshaped
            
        except Exception as e:
            log_error(f"Error during one-hot decoding: {str(e)}")
            end_section("One-hot decoding failed")
            raise

class BinaryEncoderDecoder(EncoderDecoder):
    """
    Class for binary encoding and decoding material colors using sklearn's LabelEncoder.
    
    This class encodes material labels into binary format (using the minimum number of bits
    needed to represent all unique materials) and can decode the binary representation
    back to the original labels.
    """
    
    def __init__(
        self,
        materials_labels_array: np.ndarray,
        verbose: bool = False
    ):
        """
        Initialize the BinaryEncoderDecoder.
        
        Args:
            materials_labels_array: Array of material labels to encode
            verbose: Whether to print detailed information
        """
        super().__init__(materials_labels_array, verbose)
        self.materials_labels_array = materials_labels_array
        self.void_dim = self.materials_labels_array.shape[1]
        self.n_samples = self.materials_labels_array.shape[0]
        self.classes = None
        self.n_bit = None
        self.binarizer_encoder = LabelEncoder()
    
    def binary_encode(self) -> Tuple[np.ndarray, List[Any]]:
        """
        Encode materials using binary encoding.
        
        Returns:
            Tuple of (encoded_array, color_classes)
        """
        begin_section("Binary Encoding Materials")
        
        try:
            # Flatten the array for label encoding
            flat_materials = self.materials_labels_array.reshape(-1)
            
            # Transform using sklearn's LabelEncoder
            label_encoded = self.binarizer_encoder.fit_transform(flat_materials)
            
            # Get the classes from the encoder
            self.classes = self.binarizer_encoder.classes_
            
            # Calculate how many bits we need to represent all classes
            self.n_bit = int(np.ceil(np.log2(len(self.classes))))
            
            log_info(f"Encoding {len(self.classes)} classes using {self.n_bit} bits")
            
            # Convert each label to its binary representation
            binary_format = f"{{:0{self.n_bit}b}}"
            
            # Convert to binary and split into individual bits
            binary_encoded = np.array([
                [int(bit) for bit in binary_format.format(label)]
                for label in label_encoded
            ], dtype=float)
            
            # Reshape to original dimensions plus bit dimension
            binary_encoded_reshaped = binary_encoded.reshape(
                (
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_bit
                )
            )
            
            log_success(f"Binary encoded to shape {binary_encoded_reshaped.shape}")
            end_section()
            
            return binary_encoded_reshaped, list(self.classes)
            
        except Exception as e:
            log_error(f"Error during binary encoding: {str(e)}")
            end_section("Binary encoding failed")
            raise
    
    def binary_decode(
        self,
        binary_encoded_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode binary encoded materials back to colors.
        
        Args:
            binary_encoded_array: Binary encoded array
            
        Returns:
            Tuple of (volumes_array, materials_array) where:
              - volumes_array has 1s where material exists and 0s elsewhere
              - materials_array contains the original material names
        """
        begin_section("Binary Decoding Materials")
        
        try:
            self.n_samples = binary_encoded_array.shape[0]
            
            # Reshape the array for decoding
            flat_encoded = binary_encoded_array.reshape(
                (self.n_samples * self.void_dim**3, self.n_bit)
            )
            
            # Convert each binary vector back to an integer
            # First, convert to strings of 0s and 1s
            binary_strings = ["".join(str(int(bit)) for bit in vector) for vector in flat_encoded]
            
            # Convert binary strings to integers
            label_indices = [int(binary, 2) for binary in binary_strings]
            
            # Convert integer labels back to original classes
            decoded_materials = self.binarizer_encoder.inverse_transform(label_indices)
            
            # Create a binary volume array (1 where material exists, 0 elsewhere)
            decoded_volumes = np.where(decoded_materials == None, 0, 1)
            
            # Reshape back to original dimensions
            decoded_volumes_reshaped = decoded_volumes.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )
            
            decoded_materials_reshaped = decoded_materials.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )
            
            log_success(f"Decoded to shapes: volumes {decoded_volumes_reshaped.shape}, materials {decoded_materials_reshaped.shape}")
            end_section()
            
            return decoded_volumes_reshaped, decoded_materials_reshaped
            
        except Exception as e:
            log_error(f"Error during binary decoding: {str(e)}")
            end_section("Binary decoding failed")
            raise

class RGBEncoderDecoder(EncoderDecoder):
    """
    Class for RGB encoding and decoding material colors.
    
    This class converts between material names and RGB color values,
    supporting both built-in color mappings and custom color dictionaries.
    """
    
    def __init__(
        self,
        materials_labels_array: Optional[np.ndarray] = None,
        color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None,
        verbose: bool = False
    ):
        """
        Initialize the RGBEncoderDecoder.
        
        Args:
            materials_labels_array: Array of material labels to encode (optional)
            color_dict: Dictionary mapping material names to RGB tuples (optional)
            verbose: Whether to print detailed information
        """
        if materials_labels_array is not None:
            super().__init__(materials_labels_array, verbose)
            self.materials_labels_array = materials_labels_array
            self.void_dim = materials_labels_array.shape[1] if len(materials_labels_array.shape) > 1 else 0
            self.n_samples = materials_labels_array.shape[0] if len(materials_labels_array.shape) > 0 else 0
        else:
            self.verbose = verbose
            self.materials_labels_array = None
            self.void_dim = 0
            self.n_samples = 0
        
        # Initialize color dictionary
        self.color_dict = self._initialize_color_dict(color_dict)
    
    def _initialize_color_dict(
        self,
        color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None
    ) -> Dict[Any, Tuple[int, int, int]]:
        """
        Initialize the color dictionary with defaults if not provided.
        
        Args:
            color_dict: Dictionary mapping material names to RGB tuples
            
        Returns:
            Complete color dictionary
        """
        if color_dict is not None:
            return color_dict
        
        # Create default color dictionary using matplotlib colors
        result_dict = {}
        
        # Add TABLEAU colors (common visualization colors)
        for name, hex_color in mcolors.TABLEAU_COLORS.items():
            rgb = tuple(int(x * 255) for x in mcolors.to_rgb(hex_color))
            result_dict[name] = rgb
        
        # Add CSS4 colors (wide range of named colors)
        for name, hex_color in mcolors.CSS4_COLORS.items():
            if name not in result_dict:  # Don't overwrite TABLEAU colors
                rgb = tuple(int(x * 255) for x in mcolors.to_rgb(hex_color))
                result_dict[name] = rgb
        
        # Add special case for None (transparent/empty)
        result_dict[None] = (0, 0, 0)
        
        # Add common color names that might be used
        common_colors = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
            'yellow': (255, 255, 0),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
        }
        
        for name, rgb in common_colors.items():
            if name not in result_dict:
                result_dict[name] = rgb
        
        if self.verbose:
            log_info(f"Initialized color dictionary with {len(result_dict)} colors")
        
        return result_dict
    
    def rgb_encode(self) -> Tuple[np.ndarray, Dict[Any, Tuple[int, int, int]]]:
        """
        Encode materials array using RGB values.
        
        Returns:
            Tuple of (encoded_array, color_mapping)
        """
        begin_section("RGB Encoding Materials")
        
        try:
            if self.materials_labels_array is None:
                raise ValueError("Materials array not provided")
            
            # Initialize output array with RGB channels
            rgb_array = np.zeros(
                (
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    3
                ),
                dtype=np.uint8
            )
            
            # Convert each material to its RGB value
            for s in range(self.n_samples):
                for i in range(self.void_dim):
                    for j in range(self.void_dim):
                        for k in range(self.void_dim):
                            material = self.materials_labels_array[s, i, j, k]
                            if material in self.color_dict:
                                rgb_array[s, i, j, k] = self.color_dict[material]
                            elif material is not None:
                                # Assign a default gray for unknown materials
                                rgb_array[s, i, j, k] = (128, 128, 128)
            
            log_success(f"RGB encoded to shape {rgb_array.shape}")
            end_section()
            
            return rgb_array, self.color_dict
            
        except Exception as e:
            log_error(f"Error during RGB encoding: {str(e)}")
            end_section("RGB encoding failed")
            raise
    
    def rgb_decode(
        self,
        rgb_array: np.ndarray,
        threshold: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode RGB encoded array back to materials.
        
        Args:
            rgb_array: RGB encoded array
            threshold: Threshold for color matching (Euclidean distance)
            
        Returns:
            Tuple of (volumes_array, materials_array) where:
              - volumes_array has 1s where material exists and 0s elsewhere
              - materials_array contains the original material names
        """
        begin_section("RGB Decoding Materials")
        
        try:
            n_samples = rgb_array.shape[0]
            void_dim = rgb_array.shape[1]
            
            # Prepare output arrays
            materials_array = np.empty((n_samples, void_dim, void_dim, void_dim), dtype=object)
            volumes_array = np.zeros((n_samples, void_dim, void_dim, void_dim), dtype=np.uint8)
            
            # Invert the color dictionary for lookup
            rgb_to_material = {rgb: material for material, rgb in self.color_dict.items()}
            
            # Process each voxel
            for s in range(n_samples):
                for i in range(void_dim):
                    for j in range(void_dim):
                        for k in range(void_dim):
                            rgb = tuple(rgb_array[s, i, j, k])
                            
                            # Check for empty/black voxels
                            if rgb == (0, 0, 0):
                                materials_array[s, i, j, k] = None
                                continue
                            
                            # Try exact match first
                            if rgb in rgb_to_material:
                                materials_array[s, i, j, k] = rgb_to_material[rgb]
                                volumes_array[s, i, j, k] = 1
                            else:
                                # Find closest color within threshold
                                min_dist = float('inf')
                                closest_material = None
                                
                                for known_rgb, material in rgb_to_material.items():
                                    if known_rgb == (0, 0, 0):  # Skip None/black
                                        continue
                                        
                                    # Calculate Euclidean distance in RGB space
                                    dist = np.sqrt(sum((a - b)**2 for a, b in zip(rgb, known_rgb)))
                                    
                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_material = material
                                
                                # Assign if within threshold, otherwise None
                                if min_dist <= threshold and closest_material is not None:
                                    materials_array[s, i, j, k] = closest_material
                                    volumes_array[s, i, j, k] = 1
                                else:
                                    materials_array[s, i, j, k] = None
            
            log_success(f"RGB decoded to shapes: volumes {volumes_array.shape}, materials {materials_array.shape}")
            end_section()
            
            return volumes_array, materials_array
            
        except Exception as e:
            log_error(f"Error during RGB decoding: {str(e)}")
            end_section("RGB decoding failed")
            raise

class Curator:
    """Class for preprocessing sculpture data for machine learning."""
    
    def __init__(
        self,
        processing_method: str = "OHE",
        verbose: bool = False
    ):
        """
        Initialize the Curator instance.
        
        Args:
            processing_method: Type of encoding to use ('OHE', 'BINARY', or 'RGB')
            verbose: Whether to print detailed information
        """
        self.processing_method = processing_method
        self.verbose = verbose
        self.visualizer = Visualizer(figsize=15, dpi=100)
    
    def preprocess_collection(
        self,
        volumes_path: str,
        materials_path: str,
        plot_samples: int = 3,
        buffer_size: int = 1000,
        batch_size: int = 32,
        train_size: Optional[int] = None
    ) -> Tuple[tf.data.Dataset, Any]:
        """
        Preprocess a collection for machine learning.
        
        Args:
            volumes_path: Path to the volumes .npy file
            materials_path: Path to the materials .npy file
            plot_samples: Number of random samples to plot
            buffer_size: Buffer size for dataset shuffling
            batch_size: Batch size for the dataset
            train_size: Number of samples to take for training (None for all)
            
        Returns:
            Tuple of (tensorflow_dataset, encoder_decoder_instance)
        """
        begin_section(f"Preprocessing Collection with {self.processing_method}")
        
        try:
            # Load the data
            log_action("Loading sculpture data")
            volumes, materials = Collector.load_chunk(volumes_path, materials_path)
            
            log_info(f"Loaded volumes with shape {volumes.shape} and materials with shape {materials.shape}")
            
            # Plot some random samples
            if plot_samples > 0:
                self._plot_samples(volumes, materials, n_samples=plot_samples)
            
            # Process based on the selected method
            encoder_decoder = None
            encoded_data = None
            
            if self.processing_method == "OHE":
                # One-hot encoding
                log_action("Applying one-hot encoding")
                encoder_decoder = OneHotEncoderDecoder(materials, verbose=self.verbose)
                encoded_data, classes = encoder_decoder.ohe_encode()
                log_success(f"One-hot encoded with {len(classes)} classes: {classes}")
                
            elif self.processing_method == "BINARY":
                # Binary encoding
                log_action("Applying binary encoding")
                encoder_decoder = BinaryEncoderDecoder(materials, verbose=self.verbose)
                encoded_data, classes = encoder_decoder.binary_encode()
                log_success(f"Binary encoded with classes: {classes}")
                
            elif self.processing_method == "RGB":
                # RGB encoding
                log_action("Applying RGB encoding")
                encoder_decoder = RGBEncoderDecoder(materials, verbose=self.verbose)
                encoded_data, color_mapping = encoder_decoder.rgb_encode()
                log_success(f"RGB encoded with {len(color_mapping)} color mappings")
                
            else:
                log_error(f"Unknown processing method: {self.processing_method}")
                raise ValueError(f"Unknown processing method: {self.processing_method}")
            
            # Create TensorFlow dataset
            log_action("Creating TensorFlow dataset")
            
            # Determine how many samples to use
            if train_size is None or train_size >= encoded_data.shape[0]:
                train_size = encoded_data.shape[0]
            
            # Create dataset
            dataset = tf.data.Dataset.from_tensor_slices(encoded_data)
            
            # Shuffle and batch
            dataset = dataset.shuffle(buffer_size).take(train_size).batch(batch_size)
            
            log_success(f"Created dataset with {train_size} samples and batch size {batch_size}")
            end_section()
            
            return dataset, encoder_decoder
            
        except Exception as e:
            log_error(f"Error preprocessing collection: {str(e)}")
            end_section("Preprocessing failed")
            raise
    
    def _plot_samples(
        self,
        volumes: np.ndarray,
        materials: np.ndarray,
        n_samples: int = 3
    ):
        """
        Plot random samples from the loaded data.
        
        Args:
            volumes: Array of volumes
            materials: Array of materials
            n_samples: Number of samples to plot
        """
        begin_section(f"Plotting {n_samples} sample sculptures")
        
        try:
            # Get indices of random samples
            indices = random.sample(range(volumes.shape[0]), min(n_samples, volumes.shape[0]))
            
            # Plot each sample
            for i, idx in enumerate(indices):
                log_action(f"Plotting sample {i+1}/{n_samples} (index {idx})", is_last=(i==len(indices)-1))
                
                # Plot the sculpture
                self.visualizer.plot_sculpture(
                    volumes[idx],
                    materials[idx],
                    title=f"Sample {idx}",
                    hide_axis=True
                )
            
            log_success(f"Plotted {n_samples} samples")
            end_section()
            
        except Exception as e:
            log_error(f"Error plotting samples: {str(e)}")
            end_section("Sample plotting failed")
            raise
    
    def visualize_encoded(
        self,
        encoded_data: np.ndarray,
        encoder_decoder: Any,
        sample_index: int = 0,
        original_materials: Optional[np.ndarray] = None
    ):
        """
        Visualize an encoded sample and optionally compare to original.
        
        Args:
            encoded_data: Encoded data array
            encoder_decoder: EncoderDecoder instance used for encoding
            sample_index: Index of the sample to visualize
            original_materials: Original materials array for comparison
        """
        begin_section(f"Visualizing encoded sample {sample_index}")
        
        try:
            # Get the encoded sample
            encoded_sample = encoded_data[sample_index]
            
            # Decode the sample
            if isinstance(encoder_decoder, OneHotEncoderDecoder):
                # Decode the sample
                volumes, decoded_materials = encoder_decoder.ohe_decode(encoded_sample[np.newaxis, ...])
                title = "One-Hot Encoded Sample"
                
            elif isinstance(encoder_decoder, BinaryEncoderDecoder):
                # Decode the sample
                volumes, decoded_materials = encoder_decoder.binary_decode(encoded_sample[np.newaxis, ...])
                title = "Binary Encoded Sample"
                
            elif isinstance(encoder_decoder, RGBEncoderDecoder):
                # Decode the sample
                volumes, decoded_materials = encoder_decoder.rgb_decode(encoded_sample[np.newaxis, ...])
                title = "RGB Encoded Sample"
                
            else:
                log_error(f"Unknown encoder type: {type(encoder_decoder)}")
                raise ValueError(f"Unknown encoder type: {type(encoder_decoder)}")
            
            # Visualize the decoded sample (removing the batch dimension)
            self.visualizer.plot_sculpture(
                volumes[0],
                decoded_materials[0],
                title=title,
                hide_axis=True
            )
            
            # Plot the original for comparison if provided
            if original_materials is not None:
                original_sample = original_materials[sample_index]
                
                # Create a volume array for the original
                original_volume = np.zeros(original_sample.shape, dtype=np.int8)
                for idx in np.ndindex(original_sample.shape):
                    if original_sample[idx] is not None:
                        original_volume[idx] = 1
                
                # Visualize the original sample
                self.visualizer.plot_sculpture(
                    original_volume,
                    original_sample,
                    title="Original Sample",
                    hide_axis=True
                )
            
            log_success(f"Visualized encoded sample {sample_index}")
            end_section()
            
        except Exception as e:
            log_error(f"Error visualizing encoded sample: {str(e)}")
            end_section("Visualization failed")
            raise

# Example usage
if __name__ == "__main__":
    # Define paths to test data
    import os
    
    # Check if the data directory and file exists
    data_dir = "data"
    timestamp = datetime.now().strftime("%Y-%m-%d")
    volumes_path = os.path.join(data_dir, f"volume_data[{timestamp}]chunk[1].npy")
    materials_path = os.path.join(data_dir, f"material_data[{timestamp}]chunk[1].npy")
    
    # If the files don't exist, create some test data
    if not os.path.exists(volumes_path) or not os.path.exists(materials_path):
        # Create a collector for test data
        collector = Collector(
            void_dim=20,
            edges=(2, 0.2, 0.5),
            planes=(1, 0.3, 0.5),
            pipes=(1, 0.3, 0.5),
            directory=data_dir,
            chunk_size=10,
            n_chunks=1,
            verbose=True
        )
        
        # Generate a small collection
        volumes, materials = collector.create_collection()
        
        # Update paths to the actual files
        for filename in os.listdir(data_dir):
            if filename.startswith("volume_data") and filename.endswith(".npy"):
                volumes_path = os.path.join(data_dir, filename)
            elif filename.startswith("material_data") and filename.endswith(".npy"):
                materials_path = os.path.join(data_dir, filename)
    
    # Create curators with different processing methods
    log_action("Testing One-Hot Encoding")
    curator_ohe = Curator(processing_method="OHE", verbose=True)
    dataset_ohe, encoder_ohe = curator_ohe.preprocess_collection(
        volumes_path,
        materials_path,
        plot_samples=1,
        batch_size=4
    )
    
    log_action("Testing Binary Encoding")
    curator_binary = Curator(processing_method="BINARY", verbose=True)
    dataset_binary, encoder_binary = curator_binary.preprocess_collection(
        volumes_path,
        materials_path,
        plot_samples=1,
        batch_size=4
    )
    
    log_action("Testing RGB Encoding")
    curator_rgb = Curator(processing_method="RGB", verbose=True)
    dataset_rgb, encoder_rgb = curator_rgb.preprocess_collection(
        volumes_path,
        materials_path,
        plot_samples=1,
        batch_size=4
    )
    
    log_success("All encoding methods tested successfully")