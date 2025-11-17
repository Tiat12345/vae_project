

#neural_networks.py

#This file creates all the small neural networks used inside the VAE.
# There are four networks total:

# - BW encoder (MLP)
# - BW decoder (MLP)
# - Color encoder (CNN)
# - Color decoder (CNN)



import numpy as np
from tensorflow.keras import layers, Sequential



# MLP for BW images
def get_encoder_mlp(input_shape=(28*28,), units=400, latent_dim=20, activation='relu'):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(2 * latent_dim))   # mu and log_var together
    return model




def get_decoder_mlp(latent_dim=20, output_dim=28*28, units=400, activation='relu'):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(latent_dim,)))
    model.add(layers.Dense(units, activation=activation))
    model.add(layers.Dense(output_dim))
    return model



# CNN for Color images
def get_encoder_conv(input_shape=(28, 28, 3),
                     filters=32,
                     kernel_size=3,
                     strides=2,
                     latent_dim=50,
                     activation='relu'):
   
    model = Sequential()
    model.add(layers.InputLayer(input_shape=input_shape))

    model.add(layers.Conv2D(filters,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=activation))

    model.add(layers.Conv2D(filters*2,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=activation))

    model.add(layers.Conv2D(filters*4,
                            kernel_size=kernel_size,
                            strides=strides,
                            padding='same',
                            activation=activation))

    model.add(layers.Flatten())
    model.add(layers.Dense(2 * latent_dim))   # (mu, log_var)

    return model




def get_decoder_conv(latent_dim=50, activation='relu'):
    

    model = Sequential()
    model.add(layers.InputLayer(input_shape=(latent_dim,)))

    
    model.add(layers.Dense(4 * 4 * 128, activation=activation))
    model.add(layers.Reshape((4, 4, 128)))

    
    model.add(layers.Conv2DTranspose(
        64,
        kernel_size=4,
        strides=1,
        padding='valid',
        activation=activation
    ))

   
    model.add(layers.Conv2DTranspose(
        32,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=activation
    ))

  
    model.add(layers.Conv2DTranspose(
        3,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=None   
    ))

    
    model.add(layers.Activation("sigmoid"))

    return model
