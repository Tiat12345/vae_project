
#visualize_latent.py

#latent visualizer for:
#- BW VAE
#- COLOR VAE 

#Output:
# PCA + TSNE PDF files.


import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from data_loader import MNISTBWLoader, MNISTColorLoader
from model import VAE

# Helper function to ensure model is built
def ensure_built(model, dset):
    if dset == "bw":
        model(tf.zeros((1,28,28,1)))
    else:
        model(tf.zeros((1,28,28,3)))

# Helper function to plot 2D scatter
def plot_2d(z2, labels, path, ttl):
    print("saving:", path)
    plt.figure(figsize=(8,8))
    plt.scatter(z2[:,0], z2[:,1], c=labels, cmap="tab10", s=5)
    plt.title(ttl)
    plt.savefig(path)
    plt.close()


def main(args):
    if not os.path.exists("results"):
        os.makedirs("results")

    # load model
    model = VAE(mode=args.dset, latent_dim=args.latent)
    ensure_built(model, args.dset)
    model.load_weights(args.weights)

    # load test data
    if args.dset == "bw":
        loader = MNISTBWLoader("./data")
        x = loader.load_data("test").astype("float32") / 255.0
        x = np.expand_dims(x, -1)
        labels = loader.load_data("labels")
    else:
        loader = MNISTColorLoader("./data")
        x = loader.load_data("test")
        labels = loader.load_data("labels")

    # encode
    all_mu = []
    batch = 256
    for i in range(0, len(x), batch):
        mu, _ = model.encoder(x[i:i+batch])
        all_mu.append(mu.numpy())
    Z = np.concatenate(all_mu, axis=0)

    # PCA
    pca2 = PCA(n_components=2).fit_transform(Z)
    if args.dset == "bw":
        ppath = "results/latent_bw_pca.pdf"
    else:
        ppath = f"results/latent_color_{args.version}_pca.pdf"
    plot_2d(pca2, labels, ppath, "PCA")

    # TSNE
    ts = TSNE(n_components=2, init="pca", perplexity=30).fit_transform(Z)
    if args.dset == "bw":
        tpath = "results/latent_bw_tsne.pdf"
    else:
        tpath = f"results/latent_color_{args.version}_tsne.pdf"
    plot_2d(ts, labels, tpath, "TSNE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dset", required=True)
    p.add_argument("--version", default="m0")
    p.add_argument("--weights", required=True)
    p.add_argument("--latent", type=int, default=20)
    main(p.parse_args())
