

#run_vae_color.py

#This script loads a trained COLOR VAE and then:
# - samples from prior
# - generates samples from posterior



import argparse
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from model import VAE
from data_loader import MNISTColorLoader

# Helper function to save image grid
def save_grid(imgs, rows, out_path):
    N = imgs.shape[0]
    cols = int(np.ceil(N / rows))
    fig, ax = plt.subplots(rows, cols, figsize=(cols, rows))
    ax = np.array(ax).reshape(-1)
    for i in range(rows * cols):
        ax[i].axis("off")
        if i < N:
            ax[i].imshow(imgs[i])
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    loader = MNISTColorLoader("./data")
    model = VAE("color", args.latent)
    model(tf.zeros((1, 28, 28, 3)))

    print("Loading weights:", args.weights)
    model.load_weights(args.weights)

    decoder = model.decoder

    # PRIOR
    if args.generate_from_prior:
        z = np.random.randn(args.n_samples, args.latent).astype("float32")
        x_hat = decoder(z).numpy()
        out = os.path.join(args.out_dir, "samples_from_prior.png")
        save_grid(x_hat, int(np.sqrt(args.n_samples)), out)
        print("Saved:", out)

    

    # POSTERIOR
    if args.generate_from_posterior:
        x_te = loader.load_data("test")
        N = min(args.n_samples, x_te.shape[0])
        x_small = x_te[:N]
        mu, logv = model.encoder(x_small)
        mu = mu.numpy()
        logv = logv.numpy()
        eps = np.random.randn(*mu.shape).astype("float32")
        z = mu + eps * np.exp(0.5 * logv)
        x_hat = decoder(z).numpy()
        out = os.path.join(args.out_dir, "generated_from_posterior.png")
        save_grid(x_hat, int(np.sqrt(N)), out)
        print("Saved:", out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--latent", type=int, default=50)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_recon", type=int, default=16)
    p.add_argument("--out_dir", type=str, default="results/color")
    p.add_argument("--generate_from_prior", action="store_true")
    p.add_argument("--generate_from_posterior", action="store_true")
    p.add_argument("--version", type=str, default="m0")
    main(p.parse_args())
