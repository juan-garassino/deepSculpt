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

Terminology:
- structure: 3D numpy array representing the sculpture shape (formerly "volume")
- colors: 3D numpy array with color information (formerly "material")
"""

import os
import time
import random
import glob
import numpy as np
import matplotlib.colors as mcolors
from typing import List, Tuple, Dict, Any, Optional, Union, Set
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import tensorflow as tf
from tensorflow.data import Dataset

from logger import (
    begin_section,
    end_section,
    log_action,
    log_success,
    log_error,
    log_info,
    log_warning,
)
from visualization import Visualizer


class EncoderDecoder:
    """Base class for encoding and decoding methods."""

    def __init__(self, colors: np.ndarray, verbose: bool = False):
        """
        Initialize the EncoderDecoder.

        Args:
            colors: Array of colors to encode
            verbose: Whether to print detailed information
        """
        self.colors = colors
        self.verbose = verbose
        self.unique_colors = self._get_unique_colors()

    def _get_unique_colors(self) -> Set[Any]:
        """
        Extract all unique colors from the colors array.

        Returns:
            Set of unique color values
        """
        unique_colors = set()
        # Flatten the array and add each unique color to the set
        for color in np.ndarray.flatten(self.colors):
            unique_colors.add(color)

        if self.verbose:
            log_info(f"Found {len(unique_colors)} unique colors: {unique_colors}")

        return unique_colors


class OneHotEncoderDecoder(EncoderDecoder):
    """
    Class for one-hot encoding and decoding color values using sklearn's OneHotEncoder.
    """

    def __init__(
        self,
        colors_array: np.ndarray,
        color_list: Optional[List[Any]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the OneHotEncoderDecoder.

        Args:
            colors_array: Array of colors to encode
            color_list: List of all possible color values (if None, extracted from data)
            verbose: Whether to print detailed information
        """
        super().__init__(colors_array, verbose)
        self.colors_array = colors_array
        self.void_dim = self.colors_array.shape[1]
        self.n_samples = self.colors_array.shape[0]
        self.n_classes = None
        self.classes = None

        # Use provided color list or determine from data
        if color_list is not None:
            self.color_list = color_list
        else:
            self.color_list = sorted(list(self.unique_colors), key=lambda x: str(x))

        # Initialize the one-hot encoder
        self.one_hot_encoder = OneHotEncoder(
            categories=[self.color_list], handle_unknown="ignore"
        )

    def ohe_encode(self) -> Tuple[np.ndarray, List[Any]]:
        """
        Encode colors using one-hot encoding.

        Returns:
            Tuple of (encoded_array, color_classes)
        """
        begin_section("One-Hot Encoding Colors")

        try:
            if not self.color_list:
                raise ValueError("The list of colors cannot be empty.")

            # Reshape the array for encoding
            flat_colors = self.colors_array.reshape(-1, 1)

            # Fit and transform using sklearn's OneHotEncoder
            encoded_array = self.one_hot_encoder.fit_transform(flat_colors)

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
                    self.n_classes,
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
        self, one_hot_encoded_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode one-hot encoded colors back to original values.

        Args:
            one_hot_encoded_array: One-hot encoded array

        Returns:
            Tuple of (structures_array, colors_array) where:
              - structures_array has 1s where material exists and 0s elsewhere
              - colors_array contains the original color values
        """
        begin_section("One-Hot Decoding Colors")

        try:
            self.n_samples = one_hot_encoded_array.shape[0]

            # Reshape the array for decoding
            flat_encoded = one_hot_encoded_array.reshape(
                (
                    self.n_samples * self.void_dim * self.void_dim * self.void_dim,
                    self.n_classes,
                )
            )

            # Inverse transform using the encoder
            decoded_colors = self.one_hot_encoder.inverse_transform(flat_encoded)

            # Create a binary structure array (1 where material exists, 0 elsewhere)
            decoded_structures = np.where(decoded_colors == None, 0, 1)

            # Reshape back to original dimensions
            decoded_structures_reshaped = decoded_structures.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )

            decoded_colors_reshaped = decoded_colors.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )

            log_success(
                f"Decoded to shapes: structures {decoded_structures_reshaped.shape}, colors {decoded_colors_reshaped.shape}"
            )
            end_section()

            return decoded_structures_reshaped, decoded_colors_reshaped

        except Exception as e:
            log_error(f"Error during one-hot decoding: {str(e)}")
            end_section("One-hot decoding failed")
            raise


class BinaryEncoderDecoder(EncoderDecoder):
    """
    Class for binary encoding and decoding color values using sklearn's LabelEncoder.
    """

    def __init__(self, colors_array: np.ndarray, verbose: bool = False):
        """
        Initialize the BinaryEncoderDecoder.

        Args:
            colors_array: Array of colors to encode
            verbose: Whether to print detailed information
        """
        super().__init__(colors_array, verbose)
        self.colors_array = colors_array
        self.void_dim = self.colors_array.shape[1]
        self.n_samples = self.colors_array.shape[0]
        self.classes = None
        self.n_bit = None
        self.binarizer_encoder = LabelEncoder()

    def binary_encode(self) -> Tuple[np.ndarray, List[Any]]:
        """
        Encode colors using binary encoding.

        Returns:
            Tuple of (encoded_array, color_classes)
        """
        begin_section("Binary Encoding Colors")

        try:
            # Flatten the array for label encoding
            flat_colors = self.colors_array.reshape(-1)

            # Transform using sklearn's LabelEncoder
            label_encoded = self.binarizer_encoder.fit_transform(flat_colors)

            # Get the classes from the encoder
            self.classes = self.binarizer_encoder.classes_

            # Calculate how many bits we need to represent all classes
            self.n_bit = int(np.ceil(np.log2(len(self.classes))))

            log_info(f"Encoding {len(self.classes)} classes using {self.n_bit} bits")

            # Convert each label to its binary representation
            binary_format = f"{{:0{self.n_bit}b}}"

            # Convert to binary and split into individual bits
            binary_encoded = np.array(
                [
                    [int(bit) for bit in binary_format.format(label)]
                    for label in label_encoded
                ],
                dtype=float,
            )

            # Reshape to original dimensions plus bit dimension
            binary_encoded_reshaped = binary_encoded.reshape(
                (
                    self.n_samples,
                    self.void_dim,
                    self.void_dim,
                    self.void_dim,
                    self.n_bit,
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
        self, binary_encoded_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode binary encoded colors back to original values.

        Args:
            binary_encoded_array: Binary encoded array

        Returns:
            Tuple of (structures_array, colors_array) where:
              - structures_array has 1s where material exists and 0s elsewhere
              - colors_array contains the original color values
        """
        begin_section("Binary Decoding Colors")

        try:
            self.n_samples = binary_encoded_array.shape[0]

            # Reshape the array for decoding
            flat_encoded = binary_encoded_array.reshape(
                (self.n_samples * self.void_dim**3, self.n_bit)
            )

            # Convert each binary vector back to an integer
            # First, convert to strings of 0s and 1s
            binary_strings = [
                "".join(str(int(bit)) for bit in vector) for vector in flat_encoded
            ]

            # Convert binary strings to integers
            label_indices = [int(binary, 2) for binary in binary_strings]

            # Convert integer labels back to original classes
            decoded_colors = self.binarizer_encoder.inverse_transform(label_indices)

            # Create a binary structure array (1 where material exists, 0 elsewhere)
            decoded_structures = np.where(decoded_colors == None, 0, 1)

            # Reshape back to original dimensions
            decoded_structures_reshaped = decoded_structures.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )

            decoded_colors_reshaped = decoded_colors.reshape(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim)
            )

            log_success(
                f"Decoded to shapes: structures {decoded_structures_reshaped.shape}, colors {decoded_colors_reshaped.shape}"
            )
            end_section()

            return decoded_structures_reshaped, decoded_colors_reshaped

        except Exception as e:
            log_error(f"Error during binary decoding: {str(e)}")
            end_section("Binary decoding failed")
            raise


class RGBEncoderDecoder(EncoderDecoder):
    """
    Class for RGB encoding and decoding color values.
    """

    def __init__(
        self,
        colors_array: Optional[np.ndarray] = None,
        color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None,
        verbose: bool = False,
    ):
        """
        Initialize the RGBEncoderDecoder.

        Args:
            colors_array: Array of colors to encode (optional)
            color_dict: Dictionary mapping color names to RGB tuples (optional)
            verbose: Whether to print detailed information
        """
        if colors_array is not None:
            super().__init__(colors_array, verbose)
            self.colors_array = colors_array
            self.void_dim = colors_array.shape[1] if len(colors_array.shape) > 1 else 0
            self.n_samples = colors_array.shape[0] if len(colors_array.shape) > 0 else 0
        else:
            self.verbose = verbose
            self.colors_array = None
            self.void_dim = 0
            self.n_samples = 0

        # Initialize color dictionary
        self.color_dict = self._initialize_color_dict(color_dict)

    def _initialize_color_dict(
        self, color_dict: Optional[Dict[Any, Tuple[int, int, int]]] = None
    ) -> Dict[Any, Tuple[int, int, int]]:
        """
        Initialize the color dictionary with defaults if not provided.

        Args:
            color_dict: Dictionary mapping color names to RGB tuples

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
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "yellow": (255, 255, 0),
            "white": (255, 255, 255),
            "black": (0, 0, 0),
        }

        for name, rgb in common_colors.items():
            if name not in result_dict:
                result_dict[name] = rgb

        if self.verbose:
            log_info(f"Initialized color dictionary with {len(result_dict)} colors")

        return result_dict

    def rgb_encode(self) -> Tuple[np.ndarray, Dict[Any, Tuple[int, int, int]]]:
        """
        Encode colors array using RGB values.

        Returns:
            Tuple of (encoded_array, color_mapping)
        """
        begin_section("RGB Encoding Colors")

        try:
            if self.colors_array is None:
                raise ValueError("Colors array not provided")

            # Initialize output array with RGB channels
            rgb_array = np.zeros(
                (self.n_samples, self.void_dim, self.void_dim, self.void_dim, 3),
                dtype=np.uint8,
            )

            # Convert each color to its RGB value
            for s in range(self.n_samples):
                for i in range(self.void_dim):
                    for j in range(self.void_dim):
                        for k in range(self.void_dim):
                            color = self.colors_array[s, i, j, k]
                            if color in self.color_dict:
                                rgb_array[s, i, j, k] = self.color_dict[color]
                            elif color is not None:
                                # Assign a default gray for unknown colors
                                rgb_array[s, i, j, k] = (128, 128, 128)

            log_success(f"RGB encoded to shape {rgb_array.shape}")
            end_section()

            return rgb_array, self.color_dict

        except Exception as e:
            log_error(f"Error during RGB encoding: {str(e)}")
            end_section("RGB encoding failed")
            raise

    def rgb_decode(
        self, rgb_array: np.ndarray, threshold: float = 10.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decode RGB encoded array back to colors.

        Args:
            rgb_array: RGB encoded array
            threshold: Threshold for color matching (Euclidean distance)

        Returns:
            Tuple of (structures_array, colors_array) where:
              - structures_array has 1s where material exists and 0s elsewhere
              - colors_array contains the original color values
        """
        begin_section("RGB Decoding Colors")

        try:
            n_samples = rgb_array.shape[0]
            void_dim = rgb_array.shape[1]

            # Prepare output arrays
            colors_array = np.empty(
                (n_samples, void_dim, void_dim, void_dim), dtype=object
            )
            structures_array = np.zeros(
                (n_samples, void_dim, void_dim, void_dim), dtype=np.uint8
            )

            # Invert the color dictionary for lookup
            rgb_to_color = {rgb: color for color, rgb in self.color_dict.items()}

            # Process each voxel
            for s in range(n_samples):
                for i in range(void_dim):
                    for j in range(void_dim):
                        for k in range(void_dim):
                            rgb = tuple(rgb_array[s, i, j, k])

                            # Check for empty/black voxels
                            if rgb == (0, 0, 0):
                                colors_array[s, i, j, k] = None
                                continue

                            # Try exact match first
                            if rgb in rgb_to_color:
                                colors_array[s, i, j, k] = rgb_to_color[rgb]
                                structures_array[s, i, j, k] = 1
                            else:
                                # Find closest color within threshold
                                min_dist = float("inf")
                                closest_color = None

                                for known_rgb, color in rgb_to_color.items():
                                    if known_rgb == (0, 0, 0):  # Skip None/black
                                        continue

                                    # Calculate Euclidean distance in RGB space
                                    dist = np.sqrt(
                                        sum(
                                            (a - b) ** 2 for a, b in zip(rgb, known_rgb)
                                        )
                                    )

                                    if dist < min_dist:
                                        min_dist = dist
                                        closest_color = color

                                # Assign if within threshold, otherwise None
                                if min_dist <= threshold and closest_color is not None:
                                    colors_array[s, i, j, k] = closest_color
                                    structures_array[s, i, j, k] = 1
                                else:
                                    colors_array[s, i, j, k] = None

            log_success(
                f"RGB decoded to shapes: structures {structures_array.shape}, colors {colors_array.shape}"
            )
            end_section()

            return structures_array, colors_array

        except Exception as e:
            log_error(f"Error during RGB decoding: {str(e)}")
            end_section("RGB decoding failed")
            raise


class Curator:
    """Class for preprocessing sculpture data for machine learning."""

    def __init__(
        self,
        processing_method: str = "OHE",
        output_dir: str = "processed_data",
        verbose: bool = False,
    ):
        """
        Initialize the Curator instance.

        Args:
            processing_method: Type of encoding to use ('OHE', 'BINARY', or 'RGB')
            output_dir: Directory to save processed data
            verbose: Whether to print detailed information
        """
        self.processing_method = processing_method
        self.output_dir = output_dir
        self.verbose = verbose
        self.visualizer = Visualizer(figsize=15, dpi=100)

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_samples_from_collection(
        self, collection_dir: str, limit: Optional[int] = None, shuffle: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load samples from a collection directory.

        Args:
            collection_dir: Path to the collection (date) directory
            limit: Maximum number of samples to load (None for all)
            shuffle: Whether to shuffle the samples

        Returns:
            Tuple of (structures, colors) arrays
        """
        begin_section(
            f"Loading samples from collection {os.path.basename(collection_dir)}"
        )

        try:
            # Path to samples directory
            samples_dir = os.path.join(collection_dir, "samples")

            if not os.path.exists(samples_dir):
                raise ValueError(f"Samples directory not found: {samples_dir}")

            # Check if we have the new directory structure or the old one
            structures_dir = os.path.join(samples_dir, "structures")
            colors_dir = os.path.join(samples_dir, "colors")

            # List to hold all structure and color files
            structure_files = []
            colors_files = []

            # Try the new directory structure first
            if os.path.exists(structures_dir) and os.path.exists(colors_dir):
                # Try new naming pattern
                struct_files_new = glob.glob(
                    os.path.join(structures_dir, "structure_*.npy")
                )
                if struct_files_new:
                    structure_files.extend(struct_files_new)

                # Try old naming pattern in new directory
                struct_files_old = glob.glob(
                    os.path.join(structures_dir, "volume_*.npy")
                )
                if struct_files_old:
                    structure_files.extend(struct_files_old)

                # Look for color files - new naming
                color_files_new = glob.glob(os.path.join(colors_dir, "colors_*.npy"))
                if color_files_new:
                    colors_files.extend(color_files_new)

                # Look for color files - old naming
                color_files_old = glob.glob(os.path.join(colors_dir, "material_*.npy"))
                if color_files_old:
                    colors_files.extend(color_files_old)

            # If no files found in subdirectories or they don't exist,
            # look in the main samples directory
            if not structure_files:
                # Try all possible naming patterns directly in samples dir
                struct_files_main = glob.glob(os.path.join(samples_dir, "volume_*.npy"))
                if struct_files_main:
                    structure_files.extend(struct_files_main)

                struct_files_main_new = glob.glob(
                    os.path.join(samples_dir, "structure_*.npy")
                )
                if struct_files_main_new:
                    structure_files.extend(struct_files_main_new)

                # Look for color files in main dir
                color_files_main = glob.glob(
                    os.path.join(samples_dir, "material_*.npy")
                )
                if color_files_main:
                    colors_files.extend(color_files_main)

                color_files_main_new = glob.glob(
                    os.path.join(samples_dir, "colors_*.npy")
                )
                if color_files_main_new:
                    colors_files.extend(color_files_main_new)

            # Log what we found
            log_info(
                f"Found {len(structure_files)} structure files and {len(colors_files)} color files"
            )

            if not structure_files:
                raise ValueError(f"No structure files found in {samples_dir}")

            if not colors_files:
                raise ValueError(f"No color files found in {samples_dir}")

            # Sort the files to ensure consistent ordering
            structure_files = sorted(structure_files)
            colors_files = sorted(colors_files)

            # Try to match files based on sample numbers
            paired_files = []

            # First, create a mapping of sample numbers to color files
            color_file_map = {}
            for color_file in colors_files:
                basename = os.path.basename(color_file)
                # Extract sample number from different naming patterns
                if "colors_" in basename:
                    sample_num = basename.replace("colors_", "").replace(".npy", "")
                elif "material_" in basename:
                    sample_num = basename.replace("material_", "").replace(".npy", "")
                else:
                    continue

                color_file_map[sample_num] = color_file

            # Now match structure files with color files
            for struct_file in structure_files:
                basename = os.path.basename(struct_file)
                # Extract sample number
                if "structure_" in basename:
                    sample_num = basename.replace("structure_", "").replace(".npy", "")
                elif "volume_" in basename:
                    sample_num = basename.replace("volume_", "").replace(".npy", "")
                else:
                    continue

                # Look for a matching color file
                if sample_num in color_file_map:
                    paired_files.append((struct_file, color_file_map[sample_num]))

            if not paired_files:
                log_warning("Could not find matching pairs using sample numbers")

                # If we have equal numbers of files, just pair them by index
                if len(structure_files) == len(colors_files):
                    log_info("Using index-based pairing since file counts match")
                    paired_files = list(zip(structure_files, colors_files))

            if not paired_files:
                raise ValueError(
                    f"Could not match structure and color files in {samples_dir}"
                )

            log_info(f"Successfully paired {len(paired_files)} files")

            # Update our file lists with the paired files
            structure_files = [pair[0] for pair in paired_files]
            colors_files = [pair[1] for pair in paired_files]

            # Shuffle if requested
            if shuffle:
                indices = list(range(len(structure_files)))
                random.shuffle(indices)
                structure_files = [structure_files[i] for i in indices]
                colors_files = [colors_files[i] for i in indices]

            # Apply limit if specified
            if limit is not None and limit > 0:
                structure_files = structure_files[:limit]
                colors_files = colors_files[:limit]

            log_info(f"Loading {len(structure_files)} samples")

            # Load all samples
            structures = []
            colors_list = []

            for i, (struct_file, color_file) in enumerate(
                zip(structure_files, colors_files)
            ):
                try:
                    structure = np.load(struct_file, allow_pickle=True)
                    colors = np.load(color_file, allow_pickle=True)

                    structures.append(structure)
                    colors_list.append(colors)

                    if self.verbose and (i + 1) % 20 == 0:
                        log_info(f"Loaded {i+1}/{len(structure_files)} samples")

                except Exception as e:
                    log_warning(f"Error loading sample {struct_file}: {str(e)}")

            if not structures:
                raise ValueError(f"Failed to load any samples from {samples_dir}")

            # Convert to arrays
            structures_array = np.array(structures)
            colors_array = np.array(colors_list, dtype=object)

            log_success(
                f"Loaded {len(structures)} samples with shapes: structures {structures_array.shape}, colors {colors_array.shape}"
            )
            end_section()

            return structures_array, colors_array

        except Exception as e:
            log_error(f"Error loading samples: {str(e)}")
            end_section("Sample loading failed")
            raise

    def preprocess_collection(
        self,
        collection_dir: str,
        batch_size: int = 32,
        buffer_size: int = 1000,
        train_size: Optional[int] = None,
        validation_split: float = 0.2,
        plot_samples: int = 3,
    ) -> Dict[str, Any]:
        """
        Preprocess a collection for machine learning.

        Args:
            collection_dir: Path to the collection directory
            batch_size: Batch size for the dataset
            buffer_size: Buffer size for dataset shuffling
            train_size: Number of samples to take for training (None for all)
            validation_split: Fraction of data to use for validation
            plot_samples: Number of random samples to plot

        Returns:
            Dictionary with processed datasets and encoders
        """
        begin_section(f"Preprocessing Collection with {self.processing_method}")

        try:
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            collection_name = os.path.basename(collection_dir)
            output_dir = os.path.join(
                self.output_dir,
                f"{collection_name}_{self.processing_method}_{timestamp}",
            )
            os.makedirs(output_dir, exist_ok=True)

            # Create subdirectories
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            # Load the data
            log_action("Loading sculpture data")
            structures, colors = self.load_samples_from_collection(
                collection_dir=collection_dir, limit=train_size, shuffle=True
            )

            # Calculate actual train size
            total_samples = structures.shape[0]
            if train_size is None or train_size > total_samples:
                train_size = total_samples

            log_info(f"Using {train_size} samples for processing")

            # Plot some random samples - two methods
            if plot_samples > 0:
                # Method 1: Use our helper method
                self._plot_samples(
                    structures=structures,
                    colors=colors,
                    n_samples=min(plot_samples, total_samples),
                    output_dir=plots_dir,
                )

                # Method 2: Also visualize randomly from directory for comparison
                samples_dir = os.path.join(collection_dir, "samples")
                dir_viz_path = os.path.join(plots_dir, "from_directory")
                os.makedirs(dir_viz_path, exist_ok=True)

                self.visualizer.visualize_samples_from_directory(
                    directory=samples_dir,
                    n_samples=min(plot_samples, total_samples),
                    output_dir=dir_viz_path,
                    angles=[0, 1, 2, 3],  # Show all four angles
                )

            # Process based on the selected method
            encoder_decoder = None
            encoded_data = None

            if self.processing_method == "OHE":
                # One-hot encoding
                log_action("Applying one-hot encoding")
                encoder_decoder = OneHotEncoderDecoder(
                    colors_array=colors, verbose=self.verbose
                )
                encoded_data, classes = encoder_decoder.ohe_encode()
                log_success(f"One-hot encoded with {len(classes)} classes: {classes}")

            elif self.processing_method == "BINARY":
                # Binary encoding
                log_action("Applying binary encoding")
                encoder_decoder = BinaryEncoderDecoder(
                    colors_array=colors, verbose=self.verbose
                )
                encoded_data, classes = encoder_decoder.binary_encode()
                log_success(f"Binary encoded with classes: {classes}")

            elif self.processing_method == "RGB":
                # RGB encoding
                log_action("Applying RGB encoding")
                encoder_decoder = RGBEncoderDecoder(
                    colors_array=colors, verbose=self.verbose
                )
                encoded_data, color_mapping = encoder_decoder.rgb_encode()
                log_success(f"RGB encoded with {len(color_mapping)} color mappings")

            else:
                log_error(f"Unknown processing method: {self.processing_method}")
                raise ValueError(f"Unknown processing method: {self.processing_method}")

            # Calculate validation split
            val_size = int(train_size * validation_split)
            train_size = train_size - val_size

            log_info(f"Creating training set with {train_size} samples")
            log_info(f"Creating validation set with {val_size} samples")

            # Create TensorFlow datasets
            log_action("Creating TensorFlow datasets")

            # Split the data for training and validation
            train_data = encoded_data[:train_size]
            val_data = encoded_data[train_size : train_size + val_size]

            # Create datasets
            train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
            train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)

            val_dataset = tf.data.Dataset.from_tensor_slices(val_data)
            val_dataset = val_dataset.batch(batch_size)

            # Save information about the preprocessing
            self._save_metadata(
                output_dir=output_dir,
                collection_dir=collection_dir,
                train_size=train_size,
                val_size=val_size,
                batch_size=batch_size,
                buffer_size=buffer_size,
                encoded_shape=encoded_data.shape,
                processing_method=self.processing_method,
            )

            # Create result dictionary
            result = {
                "train_dataset": train_dataset,
                "val_dataset": val_dataset,
                "encoder_decoder": encoder_decoder,
                "output_dir": output_dir,
                "train_size": train_size,
                "val_size": val_size,
                "encoded_shape": encoded_data.shape,
            }

            log_success(
                f"Created datasets with {train_size} training and {val_size} validation samples"
            )
            log_info(f"Processed data saved to {output_dir}")
            end_section()

            return result

        except Exception as e:
            log_error(f"Error preprocessing collection: {str(e)}")
            end_section("Preprocessing failed")
            raise

    def _save_metadata(
        self,
        output_dir: str,
        collection_dir: str,
        train_size: int,
        val_size: int,
        batch_size: int,
        buffer_size: int,
        encoded_shape: Tuple[int, ...],
        processing_method: str,
    ):
        """Save metadata about the preprocessing"""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "collection_dir": collection_dir,
            "train_size": train_size,
            "val_size": val_size,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "encoded_shape": [int(dim) for dim in encoded_shape],
            "processing_method": processing_method,
        }

        # Convert to pretty JSON string
        import json

        metadata_str = json.dumps(metadata, indent=4)

        # Save to file
        metadata_path = os.path.join(output_dir, "preprocessing_metadata.json")
        with open(metadata_path, "w") as f:
            f.write(metadata_str)

        log_success(f"Saved preprocessing metadata to {metadata_path}")

    def _plot_samples(
        self,
        structures: np.ndarray,
        colors: np.ndarray,
        n_samples: int = 3,
        output_dir: Optional[str] = None,
    ):
        """
        Plot random samples from the loaded data.

        Args:
            structures: Array of structures
            colors: Array of colors
            n_samples: Number of samples to plot
            output_dir: Directory to save plots (optional)
        """
        begin_section(f"Plotting {n_samples} sample sculptures")

        try:
            # Get indices of random samples
            indices = random.sample(
                range(structures.shape[0]), min(n_samples, structures.shape[0])
            )

            # Plot each sample
            for i, idx in enumerate(indices):
                log_action(
                    f"Plotting sample {i+1}/{n_samples} (index {idx})",
                    is_last=(i == len(indices) - 1),
                )

                # Create output path if saving
                save_path = None
                if output_dir:
                    save_path = os.path.join(output_dir, f"sample_{idx:05d}.png")

                # Plot the sculpture
                self.visualizer.plot_sculpture(
                    structure=structures[idx],
                    colors=colors[idx],
                    title=f"Sample {idx}",
                    hide_axis=True,
                    save_path=save_path,
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
        original_colors: Optional[np.ndarray] = None,
        save_dir: Optional[str] = None,
    ):
        """
        Visualize an encoded sample and optionally compare to original.

        Args:
            encoded_data: Encoded data array
            encoder_decoder: EncoderDecoder instance used for encoding
            sample_index: Index of the sample to visualize
            original_colors: Original colors array for comparison
            save_dir: Directory to save visualizations (optional)
        """
        begin_section(f"Visualizing encoded sample {sample_index}")

        try:
            # Get the encoded sample
            encoded_sample = encoded_data[sample_index]

            # Create save paths if needed
            decoded_save_path = None
            original_save_path = None

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                decoded_save_path = os.path.join(
                    save_dir, f"decoded_sample_{sample_index:05d}.png"
                )
                original_save_path = os.path.join(
                    save_dir, f"original_sample_{sample_index:05d}.png"
                )

            # Decode the sample
            if isinstance(encoder_decoder, OneHotEncoderDecoder):
                # Decode the sample
                structures, decoded_colors = encoder_decoder.ohe_decode(
                    encoded_sample[np.newaxis, ...]
                )
                title = "One-Hot Encoded Sample"

            elif isinstance(encoder_decoder, BinaryEncoderDecoder):
                # Decode the sample
                structures, decoded_colors = encoder_decoder.binary_decode(
                    encoded_sample[np.newaxis, ...]
                )
                title = "Binary Encoded Sample"

            elif isinstance(encoder_decoder, RGBEncoderDecoder):
                # Decode the sample
                structures, decoded_colors = encoder_decoder.rgb_decode(
                    encoded_sample[np.newaxis, ...]
                )
                title = "RGB Encoded Sample"

            else:
                log_error(f"Unknown encoder type: {type(encoder_decoder)}")
                raise ValueError(f"Unknown encoder type: {type(encoder_decoder)}")

            # Visualize the decoded sample (removing the batch dimension)
            self.visualizer.plot_sculpture(
                structure=structures[0],
                colors=decoded_colors[0],
                title=title,
                hide_axis=True,
                save_path=decoded_save_path,
            )

            # Plot the original for comparison if provided
            if original_colors is not None:
                original_sample = original_colors[sample_index]

                # Create a structure array for the original
                original_structure = np.zeros(original_sample.shape, dtype=np.int8)
                for idx in np.ndindex(original_sample.shape):
                    if original_sample[idx] is not None:
                        original_structure[idx] = 1

                # Visualize the original sample
                self.visualizer.plot_sculpture(
                    structure=original_structure,
                    colors=original_sample,
                    title="Original Sample",
                    hide_axis=True,
                    save_path=original_save_path,
                )

            log_success(f"Visualized encoded sample {sample_index}")
            end_section()

        except Exception as e:
            log_error(f"Error visualizing encoded sample: {str(e)}")
            end_section("Visualization failed")
            raise


# Example usage
if __name__ == "__main__":
    # Find the most recent collection
    import os
    from collector import Collector

    # List available collections
    base_dir = "data"
    collections = Collector.list_available_collections(base_dir)

    if not collections:
        print("No collections found. Please generate some samples first.")
    else:
        # Use the most recent collection
        latest_collection = collections[-1]
        collection_path = os.path.join(base_dir, latest_collection)
        print(f"Using collection: {latest_collection}")

        # First demonstrate the new visualizer directly
        from visualization import Visualizer

        samples_dir = os.path.join(collection_path, "samples")
        print(f"\nVisualizing random samples from {samples_dir}")

        visualizer = Visualizer(figsize=10, dpi=100)
        visualizer.visualize_samples_from_directory(
            directory=samples_dir,
            n_samples=3,
            output_dir=os.path.join(
                collection_path, "visualizations", "random_showcase"
            ),
            angles=[0, 1, 2, 3],  # Show all four angles
        )

        # Create a curator with each encoding method
        for method in ["OHE", "BINARY", "RGB"]:
            print(f"\nTesting {method} encoding")
            curator = Curator(processing_method=method, verbose=True)

            # Process the collection
            result = curator.preprocess_collection(
                collection_dir=collection_path,
                batch_size=8,
                train_size=20,  # Use a small subset for testing
                plot_samples=2,
            )

            print(
                f"Created datasets with {result['train_size']} training and {result['val_size']} validation samples"
            )
            print(f"Output directory: {result['output_dir']}")
