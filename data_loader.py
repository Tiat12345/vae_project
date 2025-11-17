
#data_loader.py 
#This file loads the MNIST BW and MNIST COLOR datasets.


import os
import numpy as np
import pickle
import subprocess


# Helper function to download a file 
def robust_wget(url, output_path):
    print("[wget] starting download to:", output_path)
    print("  from url:", url)

    
    command = [
        "wget",
        "--no-check-certificate",
        "--content-disposition",
        "--tries=5",
        "--timeout=20",
        "-O", output_path,
        url
    ]

    
    subprocess.run(command, check=True)

    
    with open(output_path, "rb") as fp:
        beginning = fp.read(30)
        if b"<html" in beginning.lower():
            print("ERROR: file was html instead of data, deleting it")
            os.remove(output_path)
            raise RuntimeError("HTML downloaded instead of dataset")


# Loader class for MNIST BW dataset
class MNISTBWLoader:
    TRAIN_URL = 'https://www.dropbox.com/scl/fi/fjye8km5530t9981ulrll/mnist_bw.npy?rlkey=ou7nt8t88wx1z38nodjjx6lch&st=5swdpnbr&dl=0'
    TEST_URL  = 'https://www.dropbox.com/scl/fi/dj8vbkfpf5ey523z6ro43/mnist_bw_te.npy?rlkey=5msedqw3dhv0s8za976qlaoir&st=nmu00cvk&dl=0'
    LABEL_URL = 'https://www.dropbox.com/scl/fi/8kmcsy9otcxg8dbi5cqd4/mnist_bw_y_te.npy?rlkey=atou1x07fnna5sgu6vrrgt9j1&st=m05mfkwb&dl=0'

    def __init__(self, data_path="./data"):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        print("MNISTBWLoader ready, data path:", self.data_path)

    def _download(self, url, filename):
        full_path = os.path.join(self.data_path, filename)
        if not os.path.exists(full_path):
            print("Downloading:", filename)
            robust_wget(url, full_path)
        return full_path

    def load_data(self, which="train"):
        if which == "train":
            p = self._download(self.TRAIN_URL, "mnist_bw.npy")
            data = np.load(p, allow_pickle=True)
            return data
        elif which == "test":
            p = self._download(self.TEST_URL, "mnist_bw_te.npy")
            data = np.load(p, allow_pickle=True)
            return data
        elif which == "labels":
            p = self._download(self.LABEL_URL, "mnist_bw_y_te.npy")
            data = np.load(p, allow_pickle=True)
            return data
        else:
            raise ValueError("which must be train/test/labels")


# Loader class for MNIST COLOR dataset
class MNISTColorLoader:
    TRAIN_URL  = 'https://www.dropbox.com/scl/fi/w7hjg8ucehnjfv1re5wzm/mnist_color.pkl?rlkey=ya9cpgr2chxt017c4lg52yqs9&st=ev984mfc&dl=0'
    TEST_URL   = 'https://www.dropbox.com/scl/fi/w08xctj7iou6lqvdkdtzh/mnist_color_te.pkl?rlkey=xntuty30shu76kazwhb440abj&st=u0hd2nym&dl=0'
    LABELS_URL = 'https://www.dropbox.com/scl/fi/fkf20sjci5ojhuftc0ro0/mnist_color_y_te.npy?rlkey=fshs83hd5pvo81ag3z209tf6v&st=99z1o18q&dl=0'

    def __init__(self, data_path="./data"):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
        print("MNISTColorLoader ready:", self.data_path)

    def _local(self, filename):
        return os.path.join(self.data_path, filename)

    def _download_if_needed(self, url, filename):
        p = self._local(filename)
        if os.path.exists(p):
            return p
        robust_wget(url, p)
        return p

    def _load_pickle(self, p):
        with open(p, "rb") as f:
            return pickle.load(f, encoding="latin1")

    def load_data(self, which="train", version="m0"):
        if which == "train":
            p = self._local("mnist_color.pkl")
            if not os.path.exists(p):
                p = self._download_if_needed(self.TRAIN_URL, "mnist_color.pkl")
            data = self._load_pickle(p)

            if isinstance(data, dict):
                images = data[version]
            else:
                images = np.array(data)

            images = images.astype("float32")
            if images.max() > 1.0:
                images = images / 255.0
            return images

        elif which == "test":
            p = self._local("mnist_color_te.pkl")
            if not os.path.exists(p):
                p = self._download_if_needed(self.TEST_URL, "mnist_color_te.pkl")
            data = self._load_pickle(p)
            if isinstance(data, dict):
                images = data["m4"]    # always test on m4
            else:
                images = np.array(data)
            images = images.astype("float32")
            if images.max() > 1.0:
                images = images / 255.0
            return images

        elif which == "labels":
            p = self._local("mnist_color_y_te.npy")
            if not os.path.exists(p):
                p = self._download_if_needed(self.LABELS_URL, "mnist_color_y_te.npy")
            return np.load(p)

        else:
            raise ValueError("which must be train/test/labels")
