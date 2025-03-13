"""
Consolidated model definitions for DeepSculpt.
This file contains all model architectures used in the DeepSculpt project.
"""

import os
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import (
    Input, Dense, Reshape, Flatten, Conv3D, Conv3DTranspose,
    BatchNormalization, LeakyReLU, Dropout, ReLU, Activation,
    ThresholdedReLU, Concatenate
)


class ModelFactory:
    """Factory class for creating different 3D generation models."""
    
    @staticmethod
    def create_generator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1):
        """
        Create a generator model based on the specified type.
        
        Args:
            model_type: Type of model ('simple', 'complex', 'skip', 'monochrome')
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: 0 for monochrome, 1 for color
            
        Returns:
            Generator model
        """
        factory_map = {
            "simple": ModelFactory._create_simple_generator,
            "complex": ModelFactory._create_complex_generator,
            "skip": ModelFactory._create_skip_generator,
            "monochrome": ModelFactory._create_monochrome_generator,
            "autoencoder": ModelFactory._create_autoencoder_generator
        }
        
        if model_type not in factory_map:
            print(f"Unknown model type: {model_type}, defaulting to skip")
            model_type = "skip"
        
        return factory_map[model_type](void_dim, noise_dim, color_mode)
    
    @staticmethod
    def create_discriminator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1):
        """
        Create a discriminator model based on the specified type.
        
        Args:
            model_type: Type of model ('simple', 'complex', 'skip', 'monochrome')
            void_dim: Dimension of the void/volume space
            noise_dim: Dimension of the noise input vector
            color_mode: 0 for monochrome, 1 for color
            
        Returns:
            Discriminator model
        """
        factory_map = {
            "simple": ModelFactory._create_simple_discriminator,
            "complex": ModelFactory._create_complex_discriminator,
            "skip": ModelFactory._create_skip_discriminator,
            "monochrome": ModelFactory._create_monochrome_discriminator,
            "autoencoder": ModelFactory._create_autoencoder_discriminator
        }
        
        if model_type not in factory_map:
            print(f"Unknown model type: {model_type}, defaulting to skip")
            model_type = "skip"
        
        return factory_map[model_type](void_dim, noise_dim, color_mode)
    
    @staticmethod
    def _create_simple_generator(void_dim, noise_dim, color_mode):
        """Create a simple generator model."""
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
        
        # Final transposed conv block with channels based on color mode
        output_channels = 6 if color_mode == 1 else 3
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
    def _create_simple_discriminator(void_dim, noise_dim, color_mode):
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
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        
        model.add(Conv3D(
            noise_dim // 4,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu"
        ))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        
        model.add(Conv3D(
            noise_dim // 2,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu"
        ))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        
        model.add(Conv3D(
            noise_dim,
            (3, 3, 3),
            strides=(2, 2, 2),
            padding="same",
            activation="relu"
        ))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))
        
        model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))
        
        return model
    
    @staticmethod
    def _create_complex_generator(void_dim, noise_dim, color_mode):
        """Create a complex generator model with skip connections."""
        inputs = Input(shape=(noise_dim,))
        x = Dense((void_dim // 8) ** 3 * noise_dim, use_bias=False)(inputs)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        
        x = Reshape((void_dim // 8, void_dim // 8, void_dim // 8, noise_dim))(x)
        
        skip_connections = []
        filters = [noise_dim, noise_dim // 2, noise_dim // 4, 6 if color_mode == 1 else 3]
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
            x = LeakyReLU(alpha=0.2)(x)
            
            if i != 3:
                skip_connections.append(x)
            else:
                x = Activation("tanh")(x)
        
        x = Reshape((void_dim, void_dim, void_dim, 6 if color_mode == 1 else 3))(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model
    
    @staticmethod
    def _create_complex_discriminator(void_dim, noise_dim, color_mode):
        """Create a complex discriminator model."""
        # This is the same as the simple discriminator for now
        return ModelFactory._create_simple_discriminator(void_dim, noise_dim, color_mode)
    
    @staticmethod
    def _create_skip_generator(void_dim, noise_dim, color_mode):
        """Create a generator with skip connections."""
        inputs = Input(shape=(noise_dim,))
        x = Dense(
            (void_dim // 8) ** 3 * noise_dim,
            use_bias=False
        )(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
        x = Reshape((void_dim // 8, void_dim // 8, void_dim // 8, noise_dim))(x)
        
        skip_connections = []
        filters = [noise_dim, noise_dim // 2, noise_dim // 4, 6 if color_mode == 1 else 3]
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
            x = ReLU()(x)
            
            if i != 3:
                skip_connections.append(x)
            else:
                x = ThresholdedReLU(theta=0.0)(x)
        
        x = Reshape((void_dim, void_dim, void_dim, 6 if color_mode == 1 else 3))(x)
        
        model = Model(inputs=inputs, outputs=x)
        return model
    
    @staticmethod
    def _create_skip_discriminator(void_dim, noise_dim, color_mode):
        """Create a discriminator for the skip connection model."""
        # Same as simple discriminator for now
        return ModelFactory._create_simple_discriminator(void_dim, noise_dim, color_mode)
    
    @staticmethod
    def _create_monochrome_generator(void_dim, noise_dim, color_mode):
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
    def _create_monochrome_discriminator(void_dim, noise_dim, color_mode):
        """Create a discriminator for monochrome models."""
        # Same as simple discriminator but with 3 input channels
        return ModelFactory._create_simple_discriminator(void_dim, noise_dim, 0)
    
    @staticmethod
    def _create_autoencoder_generator(void_dim, noise_dim, color_mode):
        """Create generator based on autoencoder architecture."""
        model = Sequential()
        
        # Dense layer to expand the latent dimension
        model.add(Dense(256, activation="relu", input_dim=noise_dim))
        model.add(Reshape((8, 8, 8, 16)))
        
        # Upsampling layers
        model.add(Conv3DTranspose(
            128, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        ))
        model.add(Conv3DTranspose(
            64, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        ))
        model.add(Conv3DTranspose(
            32, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="sigmoid"
        ))
        
        return model
    
    @staticmethod
    def _create_autoencoder_discriminator(void_dim, noise_dim, color_mode):
        """Create discriminator for autoencoder architecture."""
        model = Sequential()
        model.add(Dense(256, activation="relu", input_dim=noise_dim))
        model.add(Dense(1, activation="sigmoid"))
        return model


# Wrapper functions for backward compatibility
def create_generator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1):
    return ModelFactory.create_generator(model_type, void_dim, noise_dim, color_mode)

def create_discriminator(model_type="skip", void_dim=64, noise_dim=100, color_mode=1):
    return ModelFactory.create_discriminator(model_type, void_dim, noise_dim, color_mode)


# Example encoder from adversarial autoencoder
def create_encoder(latent_dim, input_shape=(32, 32, 32, 1)):
    """Create an encoder network for adversarial autoencoder."""
    model = Sequential()
    model.add(
        Conv3D(
            32,
            (5, 5, 5),
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
            input_shape=input_shape,
        )
    )
    model.add(
        Conv3D(
            64, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )
    )
    model.add(
        Conv3D(
            128, (5, 5, 5), strides=(2, 2, 2), padding="same", activation="relu"
        )
    )
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(latent_dim, activation=None))
    return model


# Function to add regularization to a model
def add_regularization(model, dropout_rate=0.3):
    """Add dropout regularization to Conv3DTranspose layers."""
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Conv3DTranspose):
            # Can't modify in place, so we need to rebuild
            pass
    
    # Instead, we can wrap the entire model in a new Sequential with dropout
    new_model = Sequential()
    for layer in model.layers:
        new_model.add(layer)
        if isinstance(layer, Conv3DTranspose):
            new_model.add(Dropout(dropout_rate))
    
    return new_model