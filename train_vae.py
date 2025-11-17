

#train_vae.py

# It trains either BW or COLOR
#depending on command line arguments.


import argparse
import numpy as np
import tensorflow as tf
import os

from model import VAE
from data_loader import MNISTBWLoader, MNISTColorLoader

# Helper functions to create datasets
def make_dataset_color(loader, version, batch_size):
    x = loader.load_data("train", version)
    x = x.astype("float32")
    if x.max() > 1:
        x = x / 255.0
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.shuffle(10000).batch(batch_size)
    return ds


def make_dataset_bw(loader, batch_size):
    x = loader.load_data("train")
    x = x.astype("float32") / 255.0
    x = np.expand_dims(x, -1)
    ds = tf.data.Dataset.from_tensor_slices(x)
    ds = ds.shuffle(10000).batch(batch_size)
    return ds

# Training step function
@tf.function
def train_step(model, optimizer, batch):
    with tf.GradientTape() as tape:
        x_hat, mu, logv = model(batch)
        bce = tf.keras.losses.binary_crossentropy(batch, x_hat)
        if len(bce.shape) == 3:
            rec = tf.reduce_sum(bce, axis=[1, 2])
        else:
            rec = tf.reduce_sum(bce, axis=[1, 2, 3])
        kl = model.kl_loss(mu, logv)
        loss = tf.reduce_mean(rec + kl)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def main(args):
    if args.dset == "bw":
        loader = MNISTBWLoader("./data")
        ds = make_dataset_bw(loader, args.batch)
        model = VAE("bw", args.latent)
    else:
        loader = MNISTColorLoader("./data")
        ds = make_dataset_color(loader, args.version, args.batch)
        model = VAE("color", args.latent)
    # Ensure model is built
    dummy_shape = (1, 28, 28, 1) if args.dset == "bw" else (1, 28, 28, 3)
    model(tf.zeros(dummy_shape))
    # Create optimizer
    optimizer = tf.keras.optimizers.legacy.Adam(args.lr)

    save_dir = "saved_models/bw" if args.dset == "bw" else f"saved_models/vae_color_{args.version}"
    os.makedirs(save_dir, exist_ok=True)
    # Training loop
    for ep in range(args.epochs):
        print("\nEpoch", ep+1)
        for step, batch in enumerate(ds):
            loss = train_step(model, optimizer, batch)
            if step % 100 == 0:
                print(" step", step, "loss =", float(loss))

    out = os.path.join(save_dir, "vae_weights.h5")
    model.save_weights(out)
    print("Saved weights to:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dset", type=str, required=True, help="bw or color")
    p.add_argument("--version", type=str, default="m0")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--latent", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-4)
    main(p.parse_args())
