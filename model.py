

#model.py

#This file defines the VAE class itself. It connects the
#encoder and decoder and also provides the KL-loss function.


import tensorflow as tf
import numpy as np


from neural_networks import (
    get_encoder_mlp,
    get_decoder_mlp,
    get_encoder_conv,
    get_decoder_conv
)




from abc import ABC, abstractmethod

# Abstract base class for encoder and decoder

class BiCoder(tf.keras.layers.Layer, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def call(self, x):
        pass


# implementations for encoder and decoder for BW and Color images

class EncoderBW(BiCoder):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = get_encoder_mlp(latent_dim=latent_dim)

    def call(self, x):
        b = tf.shape(x)[0]
        x_flat = tf.reshape(x, (b, -1))
        out = self.net(x_flat)
        mu = out[:, :self.latent_dim]
        logv = out[:, self.latent_dim:]
        return mu, logv




class DecoderBW(BiCoder):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = get_decoder_mlp(latent_dim=latent_dim)

    def call(self, z):
        out = self.net(z)
        out = tf.reshape(out, (tf.shape(z)[0], 28, 28, 1))
        out = tf.sigmoid(out)
        return out




class EncoderColor(BiCoder):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = get_encoder_conv(latent_dim=latent_dim)

    def call(self, x):
        out = self.net(x)
        mu = out[:, :self.latent_dim]
        logv = out[:, self.latent_dim:]
        return mu, logv




class DecoderColor(BiCoder):
    def __init__(self, latent_dim=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = get_decoder_conv(latent_dim=latent_dim)

    def call(self, z):
        return self.net(z)


# VAE class connecting encoder and decoder

class VAE(tf.keras.Model):
    def __init__(self, mode="bw", latent_dim=None):
        super().__init__()

        self.mode = mode

        if mode == "bw":
            self.latent_dim = 20 if latent_dim is None else latent_dim
            self.encoder = EncoderBW(self.latent_dim)
            self.decoder = DecoderBW(self.latent_dim)
        else:
            self.latent_dim = 50 if latent_dim is None else latent_dim
            self.encoder = EncoderColor(self.latent_dim)
            self.decoder = DecoderColor(self.latent_dim)

    def reparameterize(self, mu, logv):
        eps = tf.random.normal(tf.shape(mu))
        return mu + eps * tf.exp(0.5 * logv)

    def call(self, x):
        mu, logv = self.encoder(x)
        z = self.reparameterize(mu, logv)
        x_hat = self.decoder(z)
        return x_hat, mu, logv

    def kl_loss(self, mu, logv):
        return -0.5 * tf.reduce_sum(1 + logv - tf.square(mu) - tf.exp(logv), axis=1)
