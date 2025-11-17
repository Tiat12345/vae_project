
#run_vae_bw.py 

#Generates:
#- samples from prior
#- posterior samples



import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import MNISTBWLoader
from model import VAE

# Helper function to ensure model is built
def ensure_built(model):
    dummy = tf.zeros((1,28,28,1))
    model(dummy)

# Helper function to save image grid
def save_grid(imgs, nrow, outp):
    print("Saving image grid to:", outp)
    N = imgs.shape[0]
    H = imgs.shape[1]
    W = imgs.shape[2]
    fig, axs = plt.subplots(nrow, int(np.ceil(N/nrow)), figsize=(10,10))
    axs = np.array(axs).reshape(-1)

    for i in range(len(axs)):
        axs[i].axis("off")
        if i < N:
            axs[i].imshow(imgs[i].squeeze(), cmap="gray")

    plt.tight_layout()
    plt.savefig(outp)
    plt.close()


def main(args):
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    model = VAE(mode="bw", latent_dim=args.latent)
    ensure_built(model)
    model.load_weights(args.weights)

    loader = MNISTBWLoader("./data")

    # Generate prior samples
    if args.generate_from_prior:
        print("Generating prior samples...")
        z = np.random.randn(args.n_samples, args.latent).astype("float32")
        xh = model.decoder(z).numpy()
        save_grid(xh, int(np.sqrt(args.n_samples)), os.path.join(args.out_dir, "samples_from_prior.png"))

   

    # Generate posterior samples
    if args.generate_from_posterior:
        print("Generating posterior samples...")
        x = loader.load_data("test").astype("float32") / 255.0
        x = np.expand_dims(x, -1)
        x_in = x[:args.n_samples]
        mu, logv = model.encoder(x_in)
        mu = mu.numpy()
        logv = logv.numpy()
        z = mu + np.random.randn(*mu.shape) * np.exp(0.5 * logv)
        xh = model.decoder(z).numpy()
        save_grid(xh, int(np.sqrt(args.n_samples)), os.path.join(args.out_dir, "generated_from_posterior.png"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--latent", type=int, default=20)
    p.add_argument("--n_samples", type=int, default=64)
    p.add_argument("--n_recon", type=int, default=16)
    p.add_argument("--out_dir", default="results/bw")
    p.add_argument("--generate_from_prior", action="store_true")
    p.add_argument("--generate_from_posterior", action="store_true")
    main(p.parse_args())
