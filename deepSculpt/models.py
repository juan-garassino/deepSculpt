"""
Model Architectures for DeepSculpt 3D Generation

This module provides various neural network architectures for 3D shape generation:
1. GAN models with multiple architecture variants
2. Autoencoders for 3D shape encoding and reconstruction
3. Utilities for model creation, customization, and visualization

The module supports creating models for both raw volumetric data and preprocessed/encoded data
from the curator module.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Conv3D, Conv3DTranspose,
    BatchNormalization, LeakyReLU, Dropout, ReLU, Activation,
    ThresholdedReLU, Concatenate, ZeroPadding3D, MaxPooling3D,
    UpSampling3D, GlobalAveragePooling3D, Layer
)
from tensorflow.keras.regularizers import l2
from typing import Dict, List, Optional, Union, Tuple, Any, Callable


class ModelFactory:
    """Factory class for creating different 3D generation models."""
    
    @staticmethod
    def create_generator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1,
                        dropout_rate=0.0, use_attention=False, alpha=0.2):
        """
        Create a generator model based on the specified type.
        
        Args:
            model_type: Type of model architecture
                - 'simple': Basic 3D transposed convolution generator
                - 'complex': Generator with more filters and layers
                - 'skip': Generator with skip connections (U-Net style)
                - 'monochrome': Generator for grayscale outputs
                - 'autoencoder': Decoder for autoencoder architecture
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: Output channels (0 for monochrome, 1 for color)
            dropout_rate: Rate for dropout regularization
            use_attention: Whether to add self-attention layers
            alpha: LeakyReLU negative slope parameter
            
        Returns:
            Generator model
        """
        factory_map = {
            "simple": ModelFactory._create_simple_generator,
            "complex": ModelFactory._create_complex_generator,
            "skip": ModelFactory._create_skip_generator,
            "monochrome": ModelFactory._create_monochrome_generator,
            "autoencoder": ModelFactory._create_autoencoder_generator,
            "residual": ModelFactory._create_residual_generator
        }
        
        if model_type not in factory_map:
            print(f"Unknown model type: {model_type}, defaulting to skip")
            model_type = "skip"
        
        model = factory_map[model_type](
            void_dim=void_dim, 
            noise_dim=noise_dim, 
            color_mode=color_mode,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            alpha=alpha
        )
        
        # Print summary of model architecture
        print(f"Created {model_type} generator with {model.count_params():,} parameters")
        
        return model
    
    @staticmethod
    def create_discriminator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1,
                           dropout_rate=0.3, use_attention=False, alpha=0.2):
        """
        Create a discriminator model based on the specified type.
        
        Args:
            model_type: Type of model architecture
                - 'simple': Basic 3D convolution discriminator
                - 'complex': Discriminator with more filters and layers
                - 'skip': Discriminator with skip connections
                - 'monochrome': Discriminator for grayscale inputs
                - 'autoencoder': Discriminator for autoencoder latent space
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: Input channels (0 for monochrome, 1 for color)
            dropout_rate: Rate for dropout regularization
            use_attention: Whether to add self-attention layers
            alpha: LeakyReLU negative slope parameter
            
        Returns:
            Discriminator model
        """
        factory_map = {
            "simple": ModelFactory._create_simple_discriminator,
            "complex": ModelFactory._create_complex_discriminator,
            "skip": ModelFactory._create_skip_discriminator,
            "monochrome": ModelFactory._create_monochrome_discriminator,
            "autoencoder": ModelFactory._create_autoencoder_discriminator,
            "residual": ModelFactory._create_residual_discriminator,
            "patch": ModelFactory._create_patch_discriminator
        }
        
        if model_type not in factory_map:
            print(f"Unknown model type: {model_type}, defaulting to skip")
            model_type = "skip"
        
        model = factory_map[model_type](
            void_dim=void_dim, 
            noise_dim=noise_dim, 
            color_mode=color_mode,
            dropout_rate=dropout_rate,
            use_attention=use_attention,
            alpha=alpha
        )
        
        # Print summary of model architecture
        print(f"Created {model_type} discriminator with {model.count_params():,} parameters")
        
        return model
    
    @staticmethod
    def _create_simple_generator(void_dim, noise_dim, color_mode, dropout_rate=0.0, 
                               use_attention=False, alpha=0.2):
        """Create a simple generator model."""
        model = Sequential()
        
        # Calculate output channels based on color mode
        output_channels = 6 if color_mode == 1 else 3
        
        # Initial dense layer and reshape
        model.add(Dense(
            (void_dim // 8) ** 3 * noise_dim,
            use_bias=False,
            input_shape=(noise_dim,)
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Reshape((void_dim // 8, void_dim // 8, void_dim // 8, noise_dim)))
        
        # Apply dropout if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # First transposed conv block
        model.add(Conv3DTranspose(
            noise_dim,
            (3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        
        # Second transposed conv block
        model.add(Conv3DTranspose(
            noise_dim // 2,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        
        # Apply dropout if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Third transposed conv block
        model.add(Conv3DTranspose(
            noise_dim // 4,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        
        # Add attention layer if requested
        if use_attention:
            model.add(SelfAttention3D(noise_dim // 4))
        
        # Final transposed conv block with channels based on color mode
        model.add(Conv3DTranspose(
            output_channels,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="softmax"
        ))
        model.add(ThresholdedReLU(theta=0.0))
        model.add(Reshape((void_dim, void_dim, void_dim, output_channels)))
        
        return model
    
    @staticmethod
    def _create_simple_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                 use_attention=False, alpha=0.2):
        """Create a simple discriminator model."""
        output_channels = 6 if color_mode == 1 else 3
        
        model = Sequential()
        model.add(Conv3D(
            noise_dim // 8,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            input_shape=[void_dim, void_dim, void_dim, output_channels]
        ))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv3D(
            noise_dim // 4,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same"
        ))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))
        
        model.add(Conv3D(
            noise_dim // 2,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same"
        ))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))
        
        # Add attention layer if requested
        if use_attention:
            model.add(SelfAttention3D(noise_dim // 2))
        
        model.add(Conv3D(
            noise_dim,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same"
        ))
        model.add(LeakyReLU(alpha=alpha))
        model.add(Dropout(dropout_rate))
        
        model.add(Flatten())
        model.add(Dense(1))
        
        return model
    
    @staticmethod
    def _create_complex_generator(void_dim, noise_dim, color_mode, dropout_rate=0.0, 
                                use_attention=False, alpha=0.2):
        """Create a more complex generator model with deeper architecture."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(noise_dim,))
        
        # Initial dense layer
        x = Dense((void_dim // 16) ** 3 * noise_dim * 2, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Reshape to 3D volume
        x = Reshape((void_dim // 16, void_dim // 16, void_dim // 16, noise_dim * 2))(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # First transposed conv block (void_dim/16 -> void_dim/8)
        x = Conv3DTranspose(
            noise_dim * 2, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same", 
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Second transposed conv block (void_dim/8 -> void_dim/4)
        x = Conv3DTranspose(
            noise_dim, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same", 
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Add attention layer if requested
        if use_attention:
            x = SelfAttention3D(noise_dim)(x)
        
        # Third transposed conv block (void_dim/4 -> void_dim/2)
        x = Conv3DTranspose(
            noise_dim // 2, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same", 
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Fourth transposed conv block (void_dim/2 -> void_dim)
        x = Conv3DTranspose(
            noise_dim // 4, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same", 
            use_bias=False
        )(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Final output layer
        outputs = Conv3DTranspose(
            output_channels, 
            (4, 4, 4), 
            strides=(1, 1, 1), 
            padding="same", 
            activation="tanh"
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    @staticmethod
    def _create_complex_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                    use_attention=False, alpha=0.2):
        """Create a more complex discriminator model with deeper architecture."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(void_dim, void_dim, void_dim, output_channels))
        
        # First conv block
        x = Conv3D(
            noise_dim // 8, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same"
        )(inputs)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(dropout_rate)(x)
        
        # Second conv block
        x = Conv3D(
            noise_dim // 4, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same"
        )(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(dropout_rate)(x)
        
        # Add attention layer if requested
        if use_attention:
            x = SelfAttention3D(noise_dim // 4)(x)
        
        # Third conv block
        x = Conv3D(
            noise_dim // 2, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same"
        )(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(dropout_rate)(x)
        
        # Fourth conv block
        x = Conv3D(
            noise_dim, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same"
        )(x)
        x = LeakyReLU(alpha=alpha)(x)
        x = Dropout(dropout_rate)(x)
        
        # Fifth conv block for deeper architecture
        x = Conv3D(
            noise_dim * 2, 
            (4, 4, 4), 
            strides=(2, 2, 2), 
            padding="same"
        )(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Flatten and output
        x = Flatten()(x)
        x = Dense(noise_dim, activation="relu")(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    @staticmethod
    def _create_skip_generator(void_dim, noise_dim, color_mode, dropout_rate=0.0,
                            use_attention=False, alpha=0.2):
        """Create a generator with skip connections (U-Net style)."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(noise_dim,))
        
        # Initial dense layer and reshape
        x = Dense(
            (void_dim // 8) ** 3 * noise_dim,
            use_bias=False
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Reshape((void_dim // 8, void_dim // 8, void_dim // 8, noise_dim))(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Keep track of skip connections
        skip_connections = []
        filters = [noise_dim, noise_dim // 2, noise_dim // 4, output_channels]
        strides = [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
        
        for i in range(4):
            if i > 0:
                x = Concatenate()([x, skip_connections[-1]])
            
            x = Conv3DTranspose(
                filters[i],
                (3, 3, 3),
                strides=strides[i],
                padding="same",
                use_bias=False
            )(x)
            x = BatchNormalization()(x)
            
            if i != 3:  # Not the output layer
                x = ReLU()(x)
                skip_connections.append(x)
                
                # Add attention layer if requested
                if use_attention and i == 1:  # Add in the middle
                    x = SelfAttention3D(filters[i])(x)
                
                # Apply dropout if specified
                if dropout_rate > 0:
                    x = Dropout(dropout_rate)(x)
            else:
                x = ThresholdedReLU(theta=0.0)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=x)
        
        return model
    
    @staticmethod
    def _create_skip_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                use_attention=False, alpha=0.2):
        """Create a discriminator for the skip connection model."""
        # For compatibility, we'll use the same discriminator as the simple model
        return ModelFactory._create_simple_discriminator(
            void_dim, noise_dim, color_mode, dropout_rate, use_attention, alpha
        )
    
    @staticmethod
    def _create_monochrome_generator(void_dim, noise_dim, color_mode, dropout_rate=0.0,
                                   use_attention=False, alpha=0.2):
        """Create a monochrome generator model."""
        model = Sequential()
        
        # Initial dense layer and reshape
        model.add(Dense(
            (void_dim // 8) ** 3 * noise_dim,
            use_bias=False,
            input_shape=(noise_dim,)
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        model.add(Reshape((void_dim // 8, void_dim // 8, void_dim // 8, noise_dim)))
        
        # Apply dropout if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # First transposed conv block
        model.add(Conv3DTranspose(
            noise_dim,
            (3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        
        # Second transposed conv block
        model.add(Conv3DTranspose(
            noise_dim // 2,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        
        # Apply dropout if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Add attention layer if requested
        if use_attention:
            model.add(SelfAttention3D(noise_dim // 2))
        
        # Third transposed conv block
        model.add(Conv3DTranspose(
            noise_dim // 4,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(BatchNormalization())
        model.add(ReLU())
        
        # Final transposed conv block for monochrome output
        model.add(Conv3DTranspose(
            3,  # Always 3 channels for monochrome
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            use_bias=False,
            activation="relu"
        ))
        model.add(ThresholdedReLU(theta=0.0))
        model.add(Reshape((void_dim, void_dim, void_dim, 3)))
        
        return model
    
    @staticmethod
    def _create_monochrome_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                       use_attention=False, alpha=0.2):
        """Create a discriminator for monochrome models."""
        # Same as simple discriminator but with 3 input channels
        return ModelFactory._create_simple_discriminator(
            void_dim, noise_dim, 0, dropout_rate, use_attention, alpha
        )
    
    @staticmethod
    def _create_autoencoder_generator(void_dim, noise_dim, color_mode, dropout_rate=0.0,
                                    use_attention=False, alpha=0.2):
        """Create generator based on autoencoder architecture (decoder)."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(noise_dim,))
        
        # First dense layer to expand latent space
        x = Dense(4*4*4*64, activation="relu")(inputs)
        x = Reshape((4, 4, 4, 64))(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Upsampling blocks for 3D decoder
        # Block 1: 4x4x4 -> 8x8x8
        x = Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Block 2: 8x8x8 -> 16x16x16
        x = Conv3DTranspose(32, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Add attention layer if requested
        if use_attention:
            x = SelfAttention3D(32)(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Block 3: 16x16x16 -> 32x32x32
        x = Conv3DTranspose(16, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Block 4: 32x32x32 -> 64x64x64 (if void_dim is 64)
        if void_dim >= 64:
            x = Conv3DTranspose(8, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(alpha=alpha)(x)
        
        # Output layer
        outputs = Conv3DTranspose(
            output_channels, (3, 3, 3), padding="same", activation="tanh"
        )(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    @staticmethod
    def _create_autoencoder_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                        use_attention=False, alpha=0.2):
        """Create discriminator for autoencoder architecture (for latent space)."""
        # This is a simple fully connected discriminator for the latent space
        model = Sequential()
        model.add(Dense(256, activation="relu", input_dim=noise_dim))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(128, activation="relu"))
        
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
            
        model.add(Dense(64, activation="relu"))
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    @staticmethod
    def _create_residual_generator(void_dim, noise_dim, color_mode, dropout_rate=0.0,
                                 use_attention=False, alpha=0.2):
        """Create generator with residual blocks."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(noise_dim,))
        
        # Initial dense layer and reshape
        x = Dense((void_dim // 8) ** 3 * 64, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Reshape((void_dim // 8, void_dim // 8, void_dim // 8, 64))(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # First residual up-sampling block
        x = residual_block_3d_up(x, 64, kernel_size=3, strides=(1, 1, 1))
        
        # Second residual up-sampling block
        x = residual_block_3d_up(x, 32, kernel_size=3, strides=(2, 2, 2))
        
        # Add attention layer if requested
        if use_attention:
            x = SelfAttention3D(32)(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Third residual up-sampling block
        x = residual_block_3d_up(x, 16, kernel_size=3, strides=(2, 2, 2))
        
        # Final up-sampling and output
        x = Conv3DTranspose(output_channels, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
        outputs = Activation("tanh")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    @staticmethod
    def _create_residual_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                     use_attention=False, alpha=0.2):
        """Create discriminator with residual blocks."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(void_dim, void_dim, void_dim, output_channels))
        
        # Initial convolution
        x = Conv3D(16, (3, 3, 3), strides=(2, 2, 2), padding="same")(inputs)
        x = LeakyReLU(alpha=alpha)(x)
        
        # First residual down-sampling block
        x = residual_block_3d_down(x, 32, kernel_size=3, strides=(2, 2, 2))
        
        # Add attention layer if requested
        if use_attention:
            x = SelfAttention3D(32)(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Second residual down-sampling block
        x = residual_block_3d_down(x, 64, kernel_size=3, strides=(2, 2, 2))
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Global average pooling and output
        x = GlobalAveragePooling3D()(x)
        x = Dense(noise_dim, activation="relu")(x)
        outputs = Dense(1)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model
    
    @staticmethod
    def _create_patch_discriminator(void_dim, noise_dim, color_mode, dropout_rate=0.3,
                                  use_attention=False, alpha=0.2):
        """Create a PatchGAN-style discriminator for 3D volumes."""
        output_channels = 6 if color_mode == 1 else 3
        
        inputs = Input(shape=(void_dim, void_dim, void_dim, output_channels))
        
        # Layer 1: First convolution - no batchnorm
        x = Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding="same")(inputs)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Layer 2
        x = Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Add attention layer if requested
        if use_attention:
            x = SelfAttention3D(128)(x)
        
        # Layer 3
        x = Conv3D(256, (4, 4, 4), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Apply dropout if specified
        if dropout_rate > 0:
            x = Dropout(dropout_rate)(x)
        
        # Layer 4
        x = Conv3D(512, (4, 4, 4), strides=(2, 2, 2), padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)
        
        # Output layer
        outputs = Conv3D(1, (3, 3, 3), strides=(1, 1, 1), padding="same")(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        return model


class SelfAttention3D(Layer):
    """Self-Attention mechanism for 3D volumes."""
    
    def __init__(self, channels, reduction_ratio=8, name=None):
        """
        Initialize the Self-Attention layer.
        
        Args:
            channels: Number of input channels
            reduction_ratio: Ratio for the dimension reduction
            name: Name for the layer
        """
        super(SelfAttention3D, self).__init__(name=name)
        self.channels = channels
        self.reduction_ratio = reduction_ratio
        self.f = self.g = self.h = None
        
    def build(self, input_shape):
        """Build the layer."""
        self.f = self._build_conv(self.channels // self.reduction_ratio, 1)
        self.g = self._build_conv(self.channels // self.reduction_ratio, 1)
        self.h = self._build_conv(self.channels, 1)
        super(SelfAttention3D, self).build(input_shape)
    
    def _build_conv(self, filters, kernel_size):
        """Build a convolutional layer."""
        return Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            padding='same',
            activation=None,
            use_bias=False
        )
    
    def call(self, inputs, **kwargs):
        """Apply self-attention mechanism."""
        batch_size, height, width, depth, channels = inputs.shape
        
        f = self.f(inputs)
        g = self.g(inputs)
        h = self.h(inputs)
        
        # Reshape for matrix multiplication
        f_flat = tf.reshape(f, [batch_size, height * width * depth, self.channels // self.reduction_ratio])
        g_flat = tf.reshape(g, [batch_size, height * width * depth, self.channels // self.reduction_ratio])
        h_flat = tf.reshape(h, [batch_size, height * width * depth, self.channels])
        
        # Calculate attention maps
        s = tf.matmul(g_flat, f_flat, transpose_b=True)  # [batch, hw, hw]
        beta = tf.nn.softmax(s, axis=-1)  # Attention map
        
        # Apply attention
        o = tf.matmul(beta, h_flat)  # [batch, hw, channels]
        o = tf.reshape(o, [batch_size, height, width, depth, self.channels])  # [batch, h, w, d, channels]
        
        # Add residual connection
        gamma = self.add_weight(name='gamma', shape=(), initializer='zeros', trainable=True)
        return gamma * o + inputs
    
    def compute_output_shape(self, input_shape):
        """Compute the output shape."""
        return input_shape


def create_encoder(latent_dim, input_shape=(32, 32, 32, 1)):
    """
    Create an encoder network for adversarial autoencoder.
    
    Args:
        latent_dim: Dimension of the latent space
        input_shape: Shape of the input data
        
    Returns:
        Encoder model
    """
    inputs = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Second convolutional block
    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Third convolutional block
    x = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(256, activation="relu")(x)
    
    # Output layer for latent space
    outputs = Dense(latent_dim)(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs, name="encoder")
    
    return model


def create_variational_encoder(latent_dim, input_shape=(32, 32, 32, 1), beta=1.0):
    """
    Create a variational encoder for VAE architecture.
    
    Args:
        latent_dim: Dimension of the latent space
        input_shape: Shape of the input data
        beta: Weight for KL divergence term (beta-VAE)
        
    Returns:
        Tuple of (encoder model, z_mean, z_log_var, sampling layer)
    """
    inputs = Input(shape=input_shape)
    
    # Convolutional blocks
    x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding="same")(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Flatten
    x = Flatten()(x)
    
    # Dense layers for latent space
    x = Dense(256, activation="relu")(x)
    
    # Mean and variance for latent distribution
    z_mean = Dense(latent_dim, name="z_mean")(x)
    z_log_var = Dense(latent_dim, name="z_log_var")(x)
    
    # Define sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon * beta
    
    # Sampling layer
    z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean, z_log_var])
    
    # Create model
    encoder = Model(inputs=inputs, outputs=z, name="variational_encoder")
    
    return encoder, z_mean, z_log_var, z


def residual_block_3d_up(x, filters, kernel_size=3, strides=(1, 1, 1)):
    """
    Create a 3D residual block with upsampling.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of convolutional kernel
        strides: Strides for the upsampling
        
    Returns:
        Output tensor
    """
    # Store input
    inputs = x
    
    # First convolution
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(filters, kernel_size, strides=strides, padding="same")(x)
    
    # Second convolution
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv3DTranspose(filters, kernel_size, padding="same")(x)
    
    # Shortcut connection with upsampling if needed
    if strides != (1, 1, 1) or inputs.shape[-1] != filters:
        inputs = Conv3DTranspose(filters, 1, strides=strides, padding="same")(inputs)
    
    # Add skip connection
    return tf.keras.layers.add([x, inputs])


def residual_block_3d_down(x, filters, kernel_size=3, strides=(1, 1, 1)):
    """
    Create a 3D residual block with downsampling.
    
    Args:
        x: Input tensor
        filters: Number of filters
        kernel_size: Size of convolutional kernel
        strides: Strides for the downsampling
        
    Returns:
        Output tensor
    """
    # Store input
    inputs = x
    
    # First convolution
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv3D(filters, kernel_size, strides=strides, padding="same")(x)
    
    # Second convolution
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv3D(filters, kernel_size, padding="same")(x)
    
    # Shortcut connection with downsampling if needed
    if strides != (1, 1, 1) or inputs.shape[-1] != filters:
        inputs = Conv3D(filters, 1, strides=strides, padding="same")(inputs)
    
    # Add skip connection
    return tf.keras.layers.add([x, inputs])


def add_regularization(model, dropout_rate=0.3):
    """
    Add dropout regularization to a model.
    
    Args:
        model: Keras model to regularize
        dropout_rate: Rate for dropout layers
        
    Returns:
        Regularized model
    """
    # Function to check if a layer is a Conv layer or BatchNorm layer
    def is_conv_or_bn(layer):
        return isinstance(layer, Conv3D) or isinstance(layer, Conv3DTranspose) or isinstance(layer, BatchNormalization)
    
    # Function to check if a layer has activation
    def has_activation(layer):
        return hasattr(layer, 'activation') and layer.activation is not None
    
    # Create a new Sequential model
    regularized_model = Sequential()
    
    # Add all layers from the original model, with dropout after Conv and activation
    for i, layer in enumerate(model.layers):
        regularized_model.add(layer)
        
        # Add dropout after convolution layers with activation
        if is_conv_or_bn(layer) and has_activation(layer):
            regularized_model.add(Dropout(dropout_rate))
    
    return regularized_model


def build_conditional_gan(void_dim=64, noise_dim=100, condition_dim=10, model_type="skip", 
                         color_mode=1, use_attention=False):
    """
    Build a conditional GAN model for 3D generation.
    
    Args:
        void_dim: Dimension of the void/volume space
        noise_dim: Dimension of the noise input vector
        condition_dim: Dimension of the condition vector
        model_type: Type of model architecture
        color_mode: Output channels (0 for monochrome, 1 for color)
        use_attention: Whether to add self-attention layers
        
    Returns:
        Tuple of (generator, discriminator) models
    """
    # Create inputs
    noise_input = Input(shape=(noise_dim,))
    condition_input = Input(shape=(condition_dim,))
    
    # Concatenate noise and condition
    combined_input = Concatenate()([noise_input, condition_input])
    
    # Create base generator
    base_generator = ModelFactory.create_generator(
        model_type=model_type,
        void_dim=void_dim,
        noise_dim=noise_dim + condition_dim,  # Adjusted for condition
        color_mode=color_mode,
        use_attention=use_attention
    )
    
    # Generate output from combined input
    generated_output = base_generator(combined_input)
    
    # Create conditional generator model
    generator = Model(
        inputs=[noise_input, condition_input],
        outputs=generated_output,
        name="conditional_generator"
    )
    
    # Create discriminator input
    volume_input = Input(shape=(void_dim, void_dim, void_dim, 6 if color_mode == 1 else 3))
    
    # Process volume with base discriminator
    base_discriminator = ModelFactory.create_discriminator(
        model_type=model_type,
        void_dim=void_dim,
        noise_dim=noise_dim,
        color_mode=color_mode,
        use_attention=use_attention
    )
    
    # Remove the final layer to get features
    disc_features = base_discriminator.layers[-2].output
    
    # Process condition
    condition_embedding = Dense(64, activation="relu")(condition_input)
    condition_embedding = Dense(32, activation="relu")(condition_embedding)
    
    # Combine volume features and condition
    combined_features = Concatenate()([disc_features, condition_embedding])
    
    # Final classification layer
    disc_output = Dense(1)(combined_features)
    
    # Create conditional discriminator model
    discriminator = Model(
        inputs=[volume_input, condition_input],
        outputs=disc_output,
        name="conditional_discriminator"
    )
    
    return generator, discriminator


def save_model_diagrams(model, filename, show_shapes=True):
    """
    Save model architecture diagrams.
    
    Args:
        model: Keras model to visualize
        filename: Base filename for the output (without extension)
        show_shapes: Whether to show tensor shapes in the diagram
    """
    try:
        from tensorflow.keras.utils import plot_model
        
        # Save model diagram with shapes
        plot_model(
            model, 
            to_file=f"{filename}.png", 
            show_shapes=show_shapes,
            show_layer_names=True,
            expand_nested=True
        )
        
        print(f"Model diagram saved to {filename}.png")
        
    except ImportError as e:
        print(f"Could not save model diagram: {e}")
        print("Try installing pydot and graphviz to enable model visualization")

    
# Wrapper functions for backward compatibility
def create_generator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1):
    return ModelFactory.create_generator(model_type, void_dim, noise_dim, color_mode)

def create_discriminator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1):
    return ModelFactory.create_discriminator(model_type, void_dim, noise_dim, color_mode)


# Example usage when run as main script
if __name__ == "__main__":
    import argparse
    import os
    import time
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create and visualize DeepSculpt models")
    parser.add_argument("--model-type", type=str, default="skip",
                        choices=["simple", "complex", "skip", "monochrome", "autoencoder", "residual", "conditional"],
                        help="Type of model to create")
    parser.add_argument("--void-dim", type=int, default=64,
                        help="Dimension of the void space")
    parser.add_argument("--noise-dim", type=int, default=100,
                        help="Dimension of the noise vector")
    parser.add_argument("--color-mode", type=int, default=1,
                        help="Color mode (0 for monochrome, 1 for color)")
    parser.add_argument("--output-dir", type=str, default="./model_diagrams",
                        help="Directory for output files")
    parser.add_argument("--attention", action="store_true",
                        help="Add self-attention to models")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print model information
    print(f"Creating {args.model_type} models with void_dim={args.void_dim}, noise_dim={args.noise_dim}")
    
    # Time model creation
    start_time = time.time()
    
    if args.model_type == "conditional":
        # Create conditional GAN models
        generator, discriminator = build_conditional_gan(
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            condition_dim=10,  # Example condition dimension
            model_type="skip",  # Base model type
            color_mode=args.color_mode,
            use_attention=args.attention
        )
        
    elif args.model_type == "autoencoder":
        # Create encoder model for visualization
        encoder = create_encoder(
            latent_dim=args.noise_dim, 
            input_shape=(args.void_dim, args.void_dim, args.void_dim, 6 if args.color_mode == 1 else 3)
        )
        
        # Create decoder (generator) model
        generator = ModelFactory.create_generator(
            model_type=args.model_type,
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            color_mode=args.color_mode,
            use_attention=args.attention
        )
        
        # Create discriminator for latent space
        discriminator = ModelFactory.create_discriminator(
            model_type=args.model_type,
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            color_mode=args.color_mode,
            use_attention=args.attention
        )
        
        # Save encoder diagram
        save_model_diagrams(
            encoder,
            os.path.join(args.output_dir, f"{args.model_type}_encoder")
        )
        
    else:
        # Create standard GAN models
        generator = ModelFactory.create_generator(
            model_type=args.model_type,
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            color_mode=args.color_mode,
            use_attention=args.attention
        )
        
        discriminator = ModelFactory.create_discriminator(
            model_type=args.model_type,
            void_dim=args.void_dim,
            noise_dim=args.noise_dim,
            color_mode=args.color_mode,
            use_attention=args.attention
        )
    
    # Record elapsed time
    elapsed_time = time.time() - start_time
    
    # Print model summaries
    print("\nGenerator Summary:")
    generator.summary()
    
    print("\nDiscriminator Summary:")
    discriminator.summary()
    
    print(f"\nModel creation took {elapsed_time:.2f} seconds")
    
    # Save model diagrams
    save_model_diagrams(
        generator,
        os.path.join(args.output_dir, f"{args.model_type}_generator")
    )
    
    save_model_diagrams(
        discriminator,
        os.path.join(args.output_dir, f"{args.model_type}_discriminator")
    )
    
    print(f"Model diagrams saved to {args.output_dir}")